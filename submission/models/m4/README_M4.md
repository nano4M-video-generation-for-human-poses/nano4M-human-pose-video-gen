# M4 Data Processing Notebook

## 🛠️ What Changed vs. M3
- **Bug fix** ✅ – the joint-coordinate ( `coords` / `coords_next` ) quantisation bug present in M3 is **fully corrected**. This eliminates the systematic offset that previously degraded pose quality.
- **Input / target curriculum** 🎚️ – new asymmetric Dirichlet alphas and wider token-length ranges encourage stronger cross-modal generalisation.
- **Faster schedule** 🏃 – model is trained for **3 B tokens** instead of 5 B and converges in **14 h 34 m 45 s**.

Everything else (tokenisers, frame selection, folder structure) is identical to M3, so only the differences are highlighted below.

## 🧩 Tokenization Overview (unchanged)
| Modality          | What it represents                                      | How it is tokenised |
|-------------------|---------------------------------------------------------|---------------------|
| `tok_rgb`         | Tokens for the **reference RGB frame**.                 | 1) Resize shorter side to **256 px**.<br>2) Map to \[-1, 1\].<br>3) Cosmos-0.1 DI16×16 image tokenizer → 16 × 16 = **1024 tokens**. |
| `tok_rgb_next`    | Tokens for the **first visually distinct successor**.    | Same pipeline as `tok_rgb`. |
| `coords`          | **13-joint skeleton** in the reference frame.           | Scale to 256 × 256, **quantise to 8 192 bins** per axis, plus visibility bit → *(3 × 13)* tokens. |
| `coords_next`     | Skeleton for the successor frame.                       | Same as `coords`. |
| `captions`        | **BLIP-generated caption** of the reference frame.      | Stored as raw text. |

> **Note** – the coordinate bug fix preserves fine-grained limb geometry and yields noticeably crisper pose reconstructions.

## 🔀 Data Splitting (unchanged)
`split_output_dataset()` still produces:
- **80 %** → `train/`
- **10 %** → `eval/`
- **10 %** → `test/`

Every split contains all five modalities so that each sample is self-contained.

## ⚙️ Training Configuration

### ✨ Modalities
```yaml
modalities: ["tok_rgb", "coords", "tok_rgb_next", "coords_next"]
vocab_sizes: [65536, 65536, 65536, 65536]
max_seq_lens: [256, 39, 256, 39]
input_alphas: [1.5, 1.5, 0.5, 0.5]      # emphasise vision inputs
target_alphas: [0.5, 0.5, 1.5, 1.5]     # emphasise future prediction
input_tokens_range: [96, 256]
target_tokens_range: [39, 256]
```
Values taken from the **`M4-config.yaml`** run-time file.

### 🏋️ Training Hyperparameters
```yaml
batch_size: 96                # per-GPU (≈512 global)
total_tokens: 3000            # in millions
token_warmup: 500             # in millions
num_tokens_per_sample: 256    # 128 in + 128 out
lr: 0.002                     # cosine schedule → 1 e-7
weight_decay: 0.05
clip_grad: 1.0
dtype: fp32
training_time: 14:34:45       # wall-clock
```
Key deltas w.r.t. M3:
- **Higher LR (0.002 ↗︎)** to compensate for shorter schedule.
- **Smaller total tokens (3 B ↘︎)** without sacrificing final loss.

## 📂 Repository Layout
```
.|-- README_M4.md           # This file
|-- M4-data_processing.ipynb
|-- M4.ipynb               # Training notebook / lightning script
|-- M4-config.yaml         # Full Hydra config
```

---
### 🔗 References
- **Config:** `M4-config.yaml`
- **Baseline README (M3):** `README_M3.md`
