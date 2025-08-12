#!/usr/bin/env python
# convert_to_llama.py
# Convert your custom RoPE+RMSNorm+SwiGLU causal LM checkpoint to LlamaForCausalLM.

import os, json, shutil, argparse, sys
import torch
from transformers import LlamaConfig, LlamaForCausalLM

def load_ckpt(path):
    ckpt = torch.load(path, map_location="cpu")
    if "model" not in ckpt or "config" not in ckpt:
        print("ERROR: checkpoint must contain 'model' (state_dict) and 'config' dict.", file=sys.stderr)
        sys.exit(1)

    # --- ADD THIS CLEANING LOGIC ---
    state_dict = ckpt["model"]
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    # --- END OF ADDED LOGIC ---

    # Return the cleaned state_dict
    return state_dict, ckpt["config"]

def map_to_llama(sd, cfg):
    """Return a new state_dict mapped to LLaMA naming."""
    new_sd = {}

    def cp(dst, src):
        if src not in sd:
            raise KeyError(f"Missing key in source state_dict: {src}")
        new_sd[dst] = sd[src]

    # --- embeddings + tied head ---
    cp("model.embed_tokens.weight", "tok_emb.weight")
    # tie lm_head to embeddings
    new_sd["lm_head.weight"] = sd["tok_emb.weight"]

    n_layer = cfg["n_layer"]
    d_model = cfg["d_model"]

    # detect attention style
    packed_qkv = f"blocks.0.attn.qkv.weight" in sd

    for i in range(n_layer):
        # norms
        cp(f"model.layers.{i}.input_layernorm.weight",          f"blocks.{i}.norm1.weight")
        cp(f"model.layers.{i}.post_attention_layernorm.weight", f"blocks.{i}.norm2.weight")

        # attention projections
        if packed_qkv:
            qkv_key = f"blocks.{i}.attn.qkv.weight"
            if qkv_key not in sd:
                raise KeyError(f"Expected {qkv_key} in state_dict")
            qkv = sd[qkv_key]  # [3*d_model, d_model]
            D = qkv.shape[1]
            assert qkv.shape[0] == 3 * D, f"qkv first dim must be 3*d_model, got {qkv.shape}"
            q, k, v = torch.split(qkv, D, dim=0)
            new_sd[f"model.layers.{i}.self_attn.q_proj.weight"] = q
            new_sd[f"model.layers.{i}.self_attn.k_proj.weight"] = k
            new_sd[f"model.layers.{i}.self_attn.v_proj.weight"] = v
        else:
            # separate q/k/v weights (expected keys)
            cp(f"model.layers.{i}.self_attn.q_proj.weight", f"blocks.{i}.attn.q_proj.weight")
            cp(f"model.layers.{i}.self_attn.k_proj.weight", f"blocks.{i}.attn.k_proj.weight")
            cp(f"model.layers.{i}.self_attn.v_proj.weight", f"blocks.{i}.attn.v_proj.weight")

        # output proj
        cp(f"model.layers.{i}.self_attn.o_proj.weight", f"blocks.{i}.attn.proj.weight")

        # MLP (SwiGLU gate/up/down)
        cp(f"model.layers.{i}.mlp.gate_proj.weight", f"blocks.{i}.mlp.w1.weight")
        cp(f"model.layers.{i}.mlp.up_proj.weight",   f"blocks.{i}.mlp.w2.weight")
        cp(f"model.layers.{i}.mlp.down_proj.weight", f"blocks.{i}.mlp.w3.weight")

    # final norm
    cp("model.norm.weight", "norm.weight")

    return new_sd

def main():
    ap = argparse.ArgumentParser(description="Convert custom causal LM checkpoint to LLaMA format.")
    ap.add_argument("--ckpt", required=True, help="Path to ckpt_last.pt / ckpt_best.pt")
    ap.add_argument("--spm", required=True, help="Path to tokenizer.model (SentencePiece)")
    ap.add_argument("--out", default="llama_converted", help="Output folder for LLaMA model")
    ap.add_argument("--rope_theta", type=float, default=None, help="Override rope theta (default: use cfg.rope_base or 10000)")
    ap.add_argument("--private_ids", action="store_true", help="If your special token IDs differ, we won’t write tokenizer_config.json")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    sd, cfg = load_ckpt(args.ckpt)

    # Build LLaMA config
    llama_cfg = LlamaConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["d_model"],
        num_hidden_layers=cfg["n_layer"],
        num_attention_heads=cfg["n_head"],
        intermediate_size=int(cfg["ffn_mult"] * cfg["d_model"]),
        max_position_embeddings=cfg["seq_len"],
        rope_theta=args.rope_theta if args.rope_theta is not None else cfg.get("rope_base", 10000.0),
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
        hidden_act="silu"
    )

    model = LlamaForCausalLM(llama_cfg)
    new_sd = map_to_llama(sd, cfg)

    # Load mapped weights and report discrepancies
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(">>> load_state_dict report")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    # Save transformers model (safetensors + config.json)
    model.save_pretrained(args.out, safe_serialization=True)

    # Copy tokenizer
    shutil.copy(args.spm, os.path.join(args.out, "tokenizer.model"))

    # Write tokenizer_config.json (IDs from your training; change if different)
    if not args.private_ids:
        with open(os.path.join(args.out, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump({
                "model_max_length": cfg["seq_len"],
                "bos_token_id": 1,
                "eos_token_id": 2,
                "unk_token_id": 0,
                "pad_token_id": 3
            }, f, indent=2)

    print(f"Saved LLaMA folder -> {args.out}")

if __name__ == "__main__":
    main()
