from __future__ import annotations
import os, json, math, time, argparse, struct
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import sentencepiece as spm
except Exception:
    spm = None

# --------------------------
# Tokenizer helpers (SPM BPE)
# --------------------------

def train_sentencepiece(text_path: str, out_dir: str, vocab_size: int = 32000, character_coverage: float = 0.9995):
    assert spm is not None, "pip install sentencepiece"
    os.makedirs(out_dir, exist_ok=True)
    model_prefix = os.path.join(out_dir, "tokenizer")
    spm.SentencePieceTrainer.Train(
        input=text_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=character_coverage,
        bos_id=1, eos_id=2, unk_id=0, pad_id=3
    )
    return model_prefix + ".model"

class PackedDataset(Dataset):
    """Memory-mapped int32 token dataset packed into fixed-length blocks."""
    def __init__(self, bin_path: str, seq_len: int):
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.int32, mode='r')
        if len(self.data) <= seq_len:
            raise ValueError("Dataset shorter than seq_len.")

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.data[idx:idx+self.seq_len], dtype=np.int32))
        # Convert target tensor to torch.long
        y = torch.from_numpy(np.array(self.data[idx+1:idx+self.seq_len+1], dtype=np.int32)).long()
        return x, y


def pack_dataset(spm_model: str, text_path: str, out_bin: str, chunk_size: int = 4_000_000):
    """Encode text to ids and write contiguous int32 .bin for fast training."""
    assert spm is not None, "pip install sentencepiece"
    sp = spm.SentencePieceProcessor()
    sp.load(spm_model)

    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    # Write int32 in chunks
    with open(text_path, 'r', encoding='utf-8') as fin, open(out_bin, 'wb') as fout:
        buf = []
        for line in fin:
            line = line.strip()
            if not line:
                continue
            ids = sp.encode(line, out_type=int)
            # add eos
            ids.append(sp.eos_id())
            buf.extend(ids)
            if len(buf) >= chunk_size:
                arr = np.array(buf, dtype=np.int32)
                fout.write(arr.tobytes())
                buf = []
        if buf:
            arr = np.array(buf, dtype=np.int32)
            fout.write(arr.tobytes())

# --------------------------
# Model components
# --------------------------

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        # x: [B, T, C]
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * norm_x

class RotaryEmbedding(nn.Module):
    """RoPE helper with precomputed cos/sin caches."""
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum('i,j->ij', t, inv_freq)  # [T, dim/2]
        self.register_buffer('cos', torch.cos(freqs), persistent=False)
        self.register_buffer('sin', torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor):
        # x: [B, T, H, D], positions: [T]
        cos = self.cos[positions][:, None, :]  # [T,1,D]
        sin = self.sin[positions][:, None, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot.flatten(-2)

class MHA(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, rope: RotaryEmbedding):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope = rope

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B,T,3C]
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape to heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # [B,H,T,D]
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # positions
        pos = torch.arange(T, device=x.device)
        q = self.rope(q.transpose(1,2), pos).transpose(1,2)  # back to [B,H,T,D]
        k = self.rope(k.transpose(1,2), pos).transpose(1,2)

        # scaled dot-product attention (causal)
        if hasattr(F, 'scaled_dot_product_attention'):
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        else:
            # manual attention
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,T,T]
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
            attn = torch.softmax(scores, dim=-1) @ v
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(attn))

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_mult: float, dropout: float):
        super().__init__()
        hidden = int(hidden_mult * d_model)
        self.w1 = nn.Linear(d_model, hidden, bias=False)  # gate
        self.w2 = nn.Linear(d_model, hidden, bias=False)  # up
        self.w3 = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, rope: RotaryEmbedding, ffn_mult: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MHA(d_model, n_head, dropout, rope)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, ffn_mult, dropout)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    seq_len: int = 1024
    ffn_mult: float = 4.0
    dropout: float = 0.0
    rope_base: float = 10000.0

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.rope = RotaryEmbedding(config.d_model // config.n_head, max_position_embeddings=config.seq_len, base=config.rope_base)
        self.blocks = nn.ModuleList([
            Block(config.d_model, config.n_head, config.dropout, self.rope, config.ffn_mult)
            for _ in range(config.n_layer)
        ])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        # idx: [B,T]
        x = self.tok_emb(idx)  # [B,T,C]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.lm_head(x)  # [B,T,V]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int = 50, temperature: float = 1.0, top_k: Optional[int] = None):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.seq_len:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_id], dim=1)
        return idx

# --------------------------
# Training utilities
# --------------------------

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

class TokenBinDataset(Dataset):
    def __init__(self, path: str, seq_len: int):
        self.mm = np.memmap(path, dtype=np.int32, mode='r')
        self.seq_len = seq_len
        if len(self.mm) < seq_len + 1:
            raise ValueError("token bin too small")
    def __len__(self):
        return len(self.mm) // self.seq_len - 1
    def __getitem__(self, i):
        start = i * self.seq_len
        x = torch.from_numpy(np.array(self.mm[start:start+self.seq_len], dtype=np.int32))
        # Convert target tensor to torch.long
        y = torch.from_numpy(np.array(self.mm[start+1:start+self.seq_len+1], dtype=np.int32)).long()
        return x, y


def save_hf_compatible(model: GPT, out_dir: str, spm_model: str):
    os.makedirs(out_dir, exist_ok=True)
    ckpt = os.path.join(out_dir, 'pytorch_model.bin')
    torch.save(model.state_dict(), ckpt)
    cfg = {
        "architectures": ["GPT"],
        "vocab_size": model.config.vocab_size,
        "hidden_size": model.config.d_model,
        "num_attention_heads": model.config.n_head,
        "num_hidden_layers": model.config.n_layer,
        "max_position_embeddings": model.config.seq_len,
        "ffn_mult": model.config.ffn_mult,
        "model_type": "gptx-min",
    }
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)
    # copy tokenizer model
    if spm_model and os.path.isfile(spm_model):
        import shutil
        shutil.copy(spm_model, os.path.join(out_dir, 'tokenizer.model'))


def train_loop(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load tokenizer to get vocab size
    assert os.path.isfile(os.path.join(args.data_dir, 'tokenizer.model')), "tokenizer.model not found"
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(args.data_dir, 'tokenizer.model'))
    vocab_size = sp.get_piece_size()

    cfg = GPTConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        seq_len=args.seq_len,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(device)

    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    print(f"Model params: {count_params(model)/1e6:.2f}M")

    train_bin = os.path.join(args.data_dir, 'train.bin')
    val_bin = os.path.join(args.data_dir, 'val.bin') if os.path.isfile(os.path.join(args.data_dir, 'val.bin')) else train_bin

    train_ds = TokenBinDataset(train_bin, args.seq_len)
    val_ds = TokenBinDataset(val_bin, args.seq_len)

    def collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate)

    # Optimizer
    fused_ok = torch.cuda.is_available() and hasattr(torch.optim, 'AdamW')
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay, fused=fused_ok)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def cosine_lr(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    best_val = float('inf')
    global_step = 0

    for epoch in range(10**9):  # loop until max_steps
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                logits, loss = model(xb, yb)
            scaler.scale(loss).backward()
            # grad clip
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            # LR schedule
            for pg in optim.param_groups:
                pg['lr'] = args.lr * cosine_lr(global_step)

            global_step += 1
            pbar.set_postfix(loss=float(loss), lr=optim.param_groups[0]['lr'])
            if global_step % args.eval_interval == 0 or global_step == args.max_steps:
                val_loss = evaluate(model, val_loader, device)
                print(f"\nstep {global_step} val_loss {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_ckpt(model, optim, scaler, os.path.join(args.out_dir, 'ckpt_best.pt'), spm_model=os.path.join(args.data_dir, 'tokenizer.model'))
            if global_step % args.ckpt_interval == 0:
                save_ckpt(model, optim, scaler, os.path.join(args.out_dir, 'ckpt_last.pt'), spm_model=os.path.join(args.data_dir, 'tokenizer.model'))
            if global_step >= args.max_steps:
                save_ckpt(model, optim, scaler, os.path.join(args.out_dir, 'ckpt_last.pt'), spm_model=os.path.join(args.data_dir, 'tokenizer.model'))
                return


def evaluate(model: GPT, val_loader: DataLoader, device: str) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    return float(np.mean(losses))


def save_ckpt(model: GPT, optim, scaler, path: str, spm_model: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'scaler': scaler.state_dict(),
        'config': asdict(model.config),
    }, path)
    # also dump HF-compatible folder next to it
    save_hf_compatible(model, os.path.dirname(path), spm_model)


def load_ckpt(path: str, device: str = 'cpu') -> Tuple[GPT, dict]:
    ckpt = torch.load(path, map_location=device)
    cfg = GPTConfig(**ckpt['config'])
    model = GPT(cfg).to(device)

    # Clean the state dict keys by removing the '_orig_mod.' prefix
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    return model, ckpt

# --------------------------
# CLI
# --------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    pprep = sub.add_parser('prep')
    pprep.add_argument('--text_path', type=str, required=True)
    pprep.add_argument('--out_dir', type=str, required=True)
    pprep.add_argument('--vocab_size', type=int, default=32000)
    pprep.add_argument('--val_split', type=float, default=0.01)

    ptrain = sub.add_parser('train')
    ptrain.add_argument('--out_dir', type=str, required=True)
    ptrain.add_argument('--data_dir', type=str, required=True)
    ptrain.add_argument('--seq_len', type=int, default=1024)
    ptrain.add_argument('--batch_size', type=int, default=16)
    ptrain.add_argument('--grad_accum_steps', type=int, default=1)
    ptrain.add_argument('--n_layer', type=int, default=12)
    ptrain.add_argument('--n_head', type=int, default=12)
    ptrain.add_argument('--d_model', type=int, default=768)
    ptrain.add_argument('--ffn_mult', type=float, default=4.0)
    ptrain.add_argument('--dropout', type=float, default=0.0)
    ptrain.add_argument('--lr', type=float, default=6e-4)
    ptrain.add_argument('--weight_decay', type=float, default=0.1)
    ptrain.add_argument('--max_steps', type=int, default=200000)
    ptrain.add_argument('--warmup_steps', type=int, default=2000)
    ptrain.add_argument('--eval_interval', type=int, default=1000)
    ptrain.add_argument('--ckpt_interval', type=int, default=5000)
    ptrain.add_argument('--compile', type=int, default=1)

    pgen = sub.add_parser('generate')
    pgen.add_argument('--ckpt', type=str, required=True)
    pgen.add_argument('--spm', type=str, required=True)
    pgen.add_argument('--prompt', type=str, required=True)
    pgen.add_argument('--max_new_tokens', type=int, default=50)
    pgen.add_argument('--temperature', type=float, default=0.8)
    pgen.add_argument('--top_k', type=int, default=50)

    args = parser.parse_args()

    if args.cmd == 'prep':
        os.makedirs(args.out_dir, exist_ok=True)
        print("Training tokenizer...")
        spm_model = train_sentencepiece(args.text_path, args.out_dir, args.vocab_size)
        print(f"Saved tokenizer: {spm_model}")
        # simple random split
        print("Packing dataset…")
        # pack full to train.bin then split tail to val.bin
        train_bin = os.path.join(args.out_dir, 'train.bin')
        pack_dataset(spm_model, args.text_path, train_bin)
        # create a small val set by splitting the tail
        data = np.memmap(train_bin, dtype=np.int32, mode='r')
        n = len(data)
        val_n = int(n * args.val_split)
        if val_n > 0:
            val_bin = os.path.join(args.out_dir, 'val.bin')
            with open(val_bin, 'wb') as f:
                f.write(np.array(data[-val_n:], dtype=np.int32).tobytes())
            with open(train_bin, 'rb') as f:
                buf = f.read((n - val_n) * 4)
            with open(train_bin, 'wb') as f:
                f.write(buf)
        print("Done.")

    elif args.cmd == 'train':
        os.makedirs(args.out_dir, exist_ok=True)
        train_loop(args)

    elif args.cmd == 'generate':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, _ = load_ckpt(args.ckpt, device)
        sp = spm.SentencePieceProcessor(); sp.load(args.spm)
        ids = torch.tensor([sp.encode(args.prompt, out_type=int)], device=device)
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        text = sp.decode(out[0].tolist())
        print(text)

if __name__ == '__main__':
    main()
