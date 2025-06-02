# M2 Data Processing & Training Note

---

## 🛠️ What Changed vs. M1
* **+ Caption modality** – BLIP-generated sentence per key frame (tokenised as plain text).
* **+ Future frame targets** – `tok_rgb_next`, `coords_next` now predicted rather than merely encoded.
* **Dirichlet Curriculum** – asymmetric alphas bias the network toward vision on the input side and text on the output side.

---

## 🧩 Tokenisation Overview
| Modality          | Max len | Description                                           |
|-------------------|:------:|-------------------------------------------------------|
| `tok_rgb`         | 256    | 256×256 frame → DI16×16 tokenizer ⇒ **1024 tokens**, truncated to 256. |
| `coords`          | 39     | 13 joints × xyz + vis flag; binned to 8 192 levels. |
| `captions`        | 1536   | BLIP caption, raw text tokens (GPT‑2 BPE). |
| `tok_rgb_next`    | 256    | Same pipeline as `tok_rgb`. |
| `coords_next`     | 234    | 6× intermediate skeletons between frames. |

See full spec in **`M2-config.yaml`**.

---

## ⚙️ Training Configuration
```yaml
batch_size: 128                # per-GPU (512 global)
total_tokens: 5000             # million ⇒ ≈5 B
warmup_tokens: 500             # million
token_warmup: 500              # synonym in older Hydra
num_tokens_per_sample: 256     # 128 in + 128 out
lr: 0.0006 → 1e-6             # cosine decay
weight_decay: 0.05
clip_grad: 1.0
dtype: fp32
```
training time : ~12h

## 🧪 Evaluation & Qualitative Results
- Quantitative metrics were the best among all other models M3/M4 (eval CE ≈ 1.88). However, the data used for training was far beyond ~2000 inputs (Overfitting problem), than can be easly avoid by extending the dataset.
- M2 introduced **video roll‑outs**:
```
results/
├── pose_future_original.gif
├── pose_future_predicted.gif
└── ...
```
Each GIF shows the skeleton movement for **7‑frame sequences**.

---

## 📂 Files
```
.|-- README_M2.md               # This file
|-- M2-data_processing.ipynb
|-- M2.ipynb                   # Training notebook
|-- M2-config.yaml             # Full Hydra config
```

---
