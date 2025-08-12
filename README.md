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
