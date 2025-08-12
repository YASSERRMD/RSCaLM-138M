# Model Architecture

**Type:** decoder-only Transformer (causal LM)
**Blocks:** RMSNorm → Multi-Head Self-Attention (RoPE) → RMSNorm → SwiGLU MLP
**Weight tying:** `lm_head.weight = tok_emb.weight`
**Tokenizer:** SentencePiece BPE (ids: `unk=0, bos=1, eos=2, pad=3`)

## Diagram (per layer)

```
input ids → Embedding ───────────────────────────────────────────────────────┐
                                       ┌─ RMSNorm ── MHA (causal, RoPE) ──┐ │
hidden ────────────────────────────────┤            + residual             ├─┤
                                       └─ RMSNorm ── SwiGLU MLP ──────────┘ │
                                                                            ▼
                                 ... repeat N layers ...                   RMSNorm
                                                                            ▼
                                                                  Linear (lm_head, tied)
                                                                            ▼
                                                                  token logits
```

## Shapes

* `B`: batch, `T`: sequence length, `C=d_model`, `H=n_head`, `D=C/H`, `V=vocab_size`
* Embedding: `tok_emb: [V, C]`, hidden: `[B, T, C]`
* Q/K/V: `[B, H, T, D]`, attention scores: `[B, H, T, T]`
* MLP hidden: `H_ff = ffn_mult * C`

## Attention (causal + RoPE)

* Query/Key/Value projections (bias-free):

  * `Q, K, V = xW_q, xW_k, xW_v`, with RoPE applied to Q,K per position
* Causal SDPA (PyTorch): `scaled_dot_product_attention(q, k, v, is_causal=True)`
* Output projection: `x = x + Attn(RMSNorm(x))`; then `x = x + MLP(RMSNorm(x))`

**RoPE:** rotary embeddings rotate Q,K in each head to encode relative positions (base `rope_base=10000`), enabling better length extrapolation than absolute positions.

## Normalization & MLP

* **RMSNorm** (epsilon `1e-6`):
  `RMSNorm(x) = x / sqrt(mean(x^2) + eps) * γ`
* **SwiGLU** (bias-free):
  `MLP(x) = W3[ SiLU(W1 x) ⊙ (W2 x) ]`, with `W1,W2: [C→H_ff]`, `W3: [H_ff→C]`

## Loss & Training

* Next-token cross-entropy on logits `[B, T, V]`
* Teacher forcing; label shift by 1
* Optimizer: AdamW; LR: warmup → cosine decay
* Grad clip: `1.0`, AMP (fp16/bf16) enabled

## Complexity

* Time: `O(N * T * C^2)` (dominated by attention/MLP)
* Memory (activations): `O(N * T * C)` + attention `O(N * H * T^2)` (reduced by SDPA kernels)

## Config → Implementation Map

| Config field | Meaning                             | Code path / state\_dict prefix                     |
| ------------ | ----------------------------------- | -------------------------------------------------- |
| `vocab_size` | tokenizer vocabulary                | `tok_emb.weight`, `lm_head.weight` (tied)          |
| `d_model`    | hidden size `C`                     | everywhere                                         |
| `n_head`     | attention heads `H`                 | `blocks.*.attn.*`                                  |
| `n_layer`    | number of transformer blocks        | `blocks.{i}.*`                                     |
| `seq_len`    | max context for RoPE cache          | `RotaryEmbedding(max_position_embeddings=seq_len)` |
| `ffn_mult`   | MLP expansion `H_ff = ffn_mult * C` | `blocks.*.mlp.w[1..3]`                             |
| `rope_base`  | RoPE θ (default 10000)              | `RotaryEmbedding(base=rope_base)`                  |
| `dropout`    | dropout prob (attn/MLP)             | `MHA.drop`, `SwiGLU.drop`                          |

## Parameter Count (approx.)

Let `C=d_model`, `H_ff=ffn_mult*C`, `L=n_layer`, `V=vocab_size`.

* **Embeddings (tied with head):** `V*C` (counted once)
* **Per layer:**

  * Attention: `3*C*C` (qkv) + `C*C` (o\_proj) = `4C²`
  * MLP (SwiGLU): `2*C*H_ff + H_ff*C = 3*C*H_ff`
  * Norms: \~`2*C` (negligible)
* **Total:**
  `Params ≈ V*C + L*(4*C² + 3*C*H_ff) + C`
  With `H_ff = ffn_mult*C`, this becomes
  `≈ V*C + L*C²*(4 + 3*ffn_mult) + C`

(For `C=768, L=12, ffn_mult=4, V≈32k` → \~**138M**.)

## Generation

Autoregressive loop with top-k/p sampling and optional repetition control:

```python
out = model.generate(
    ids,
    max_new_tokens=80,
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.3,
    no_repeat_ngram_size=4,
)
```

## File/Key Layout (your checkpoint)

* **Files:** `ckpt_last.pt`/`ckpt_best.pt` → `model` (state\_dict), `config`, `optim`, `scaler`
* **Important keys:**

  * `tok_emb.weight`, `lm_head.weight` (tied)
  * `blocks.{i}.norm1.weight`, `blocks.{i}.attn.qkv.weight`, `blocks.{i}.attn.proj.weight`
  * `blocks.{i}.norm2.weight`, `blocks.{i}.mlp.w1.weight`, `w2.weight`, `w3.weight`
  * `norm.weight` (final RMSNorm)

## LLaMA Conversion (optional)

Weights can be mapped to `LlamaForCausalLM` by splitting `attn.qkv.weight` → `q_proj/k_proj/v_proj` and copying MLP gate/up/down + norms. See `convert_to_llama.py`.

---




# How to Train

## 0) Environment

```bash
# Python 3.10+
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch sentencepiece tqdm numpy
# (Optional) PyTorch nightly for torch.compile speedups
```

> GPU: any recent NVIDIA w/ CUDA works. Mixed precision (AMP) is used automatically.

---

## 1) Prepare Text Data

The training script expects **one UTF-8 text file** with **one sample per line**.
If your corpus is JSON/Parquet/etc., convert it to `.txt` first (each document → one line).

```bash
# Example: concatenate many small files
cat data_raw/*.txt > corpus.txt
```

**Quality tips**: deduplicate, strip HTML/boilerplate, normalize whitespace.
Big wins come from clean text.

---

## 2) Train a Tokenizer + Pack Dataset

```bash
python standalone_transformer_lm.py prep \
  --text_path corpus.txt \
  --out_dir ./data \
  --vocab_size 32000
```

This writes:

* `./data/tokenizer.model` (SentencePiece BPE, ids: unk=0, bos=1, eos=2, pad=3)
* `./data/train.bin` (+ `val.bin` split from tail)

---

## 3) Choose a Model Size

Common presets (RoPE + RMSNorm + SwiGLU):

|   Size | Layers | Heads | d\_model | ffn\_mult | Seq Len |
| -----: | -----: | ----: | -------: | --------: | ------: |
| \~138M |     12 |    12 |      768 |       4.0 |    1024 |
| \~350M |     24 |    20 |     1024 |       4.0 |    1024 |
| \~700M |     32 |    24 |     1280 |       4.0 |    1024 |

> Rule of thumb for data: target **\~20× tokens per parameter** (in billions).

---

## 4) Train (single-GPU example)

```bash
python standalone_transformer_lm.py train \
  --out_dir ./runs/lm138 \
  --data_dir ./data \
  --seq_len 1024 \
  --batch_size 16 \
  --grad_accum_steps 4 \
  --n_layer 12 --n_head 12 --d_model 768 --ffn_mult 4 \
  --max_steps 200000 --warmup_steps 2000 --lr 6e-4 \
  --weight_decay 0.1 --dropout 0.0 --compile 1
```

**Flags you’ll tune most:**

* `--batch_size`, `--grad_accum_steps` → fit memory; effective tokens/step = `batch_size * seq_len * grad_accum_steps`.
* `--max_steps` → total optimizer steps.
* `--dropout` → try `0.1` if you see overfitting.
* `--seq_len` → increase only if you really need longer context.

---

## 5) Train (multi-GPU, torchrun)

```bash
# Example: 2 GPUs
torchrun --nproc_per_node=2 standalone_transformer_lm.py train \
  --out_dir ./runs/lm138_ddp \
  --data_dir ./data \
  --seq_len 1024 \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --n_layer 12 --n_head 12 --d_model 768 --ffn_mult 4 \
  --max_steps 200000 --warmup_steps 2000 --lr 6e-4 \
  --weight_decay 0.1 --dropout 0.0 --compile 1
```

> Scale `batch_size` and/or `grad_accum_steps` per GPU. Keep the **effective** tokens/step similar when comparing runs.

---

## 6) Resume / Continue Training

```bash
# Point --out_dir to the same folder; the script will pick up ckpt_last.pt
python standalone_transformer_lm.py train \
  --out_dir ./runs/lm138 \
  --data_dir ./data \
  --seq_len 1024 \
  --batch_size 16 --grad_accum_steps 4 \
  --n_layer 12 --n_head 12 --d_model 768 --ffn_mult 4 \
  --max_steps 300000 --warmup_steps 2000 --lr 6e-4 \
  --weight_decay 0.1 --dropout 0.1 --compile 1
```

---

## 7) Generate (quick sanity)

```bash
python standalone_transformer_lm.py generate \
  --ckpt ./runs/lm138/ckpt_last.pt \
  --spm ./data/tokenizer.model \
  --prompt "The sun is" \
  --max_new_tokens 50
```

> If you see repetition, try smaller `max_new_tokens`, and at training time consider longer runs, better data, or `--dropout 0.1`.

---

## 8) Export Weights (HF-style files)

Each checkpoint saves:

* `pytorch_model.bin` (state\_dict)
* `config.json` (minimal metadata)
* copies `tokenizer.model`

These are enough to **archive** or **convert** later.

---

## 9) (Optional) Convert to LLaMA Format

Use the provided `convert_to_llama.py` to create a folder loadable by 🤗:

```bash
python convert_to_llama.py \
  --ckpt ./runs/lm138/ckpt_last.pt \
  --spm ./data/tokenizer.model \
  --out ./RSCaLM-138M-LLaMA
```

Then test:

```bash
python test_llama_model.py  # loads from ./RSCaLM-138M-LLaMA
```

---

## 10) Practical Tips

* **Memory:** if you OOM, reduce `batch_size` first, then `seq_len`, or add `grad_accum_steps`.
* **Throughput:** `--compile 1` can help on recent PyTorch; disable if you see compile stalls.
* **Data wins:** cleaning, dedup, and domain matching can matter more than fancy hyperparams.
* **Scaling:** for real quality on small models, plan **billions** of clean tokens.

---
