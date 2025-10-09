# M1 Data Processing & Training Report

## 📓 Notebook Index
| Notebook / File | Purpose |
|-----------------|---------|
| **data_penn_view.ipynb** | Quick sanity‑check: loads a *masked* batch through the custom dataloader and verifies that everything compiles on GPU. |
| **data_split.ipynb** | Performs the 70 / 15 / 15 split (train / eval / test) and moves samples into their respective folders with a fixed seed for reproducibility. |
| **data_tokenize_rgb.ipynb** | Demonstrates Cosmos tokenisation for **one RGB frame** (resize → tensor → Cosmos‑0.1 DI16×16). |
| **data_tokenize_caption.ipynb** | Uses **Salesforce BLIP (base)** to caption a single frame and write the result to JSON. |
| **data_tokenize_human‑poses.ipynb** | Explores two strategies for human‑pose tokens (canvas vs. quantised). After TA feedback the **quantised‑coords** route was chosen. |
| **data_tokenize_all_example.ipynb** | End‑to‑end pipeline on **one example** to confirm that all three modalities line up correctly. |
| **data_tokenize_all.ipynb** | Full dataset processing: for **every frame of every video** it writes `tok_rgb`, `coords`, and `captions` so each sample is self‑contained. |
| **m1-config.yaml** (renamed from *multi_penn_action.yaml*) | Complete training configuration used in this run. |
| **M1.ipynb** | Post‑training metrics, qualitative sampling, and loss curves. |

---

## 🧩 Tokenisation Overview

| Modality   | What it represents                         | How it is tokenised |
|------------|--------------------------------------------|---------------------|
| `tok_rgb`  | **RGB frame tokens** (reference & targets) | 1) Resize shorter side to **256 px**.<br>2) Scale to [-1, 1].<br>3) **Cosmos‑0.1 DI16×16** → 1024 discrete tokens. |
| `coords`   | **13‑joint human pose** of the frame       | Normalise each joint to the 256 × 256 canvas and **quantise (0–8191)** both *x* and *y*. Visibility is kept as a binary token → (3 × 13) output. |
| `captions` | **Natural‑language caption**               | Generated with **BLIP (base)** and stored as raw text. |

---

## 🔀 Data Splitting
The dataset is shuffled with `seed = 42` and split **70 % train / 15 % eval / 15 % test** using `data_split.ipynb`.  
Every sample contains the three modalities above so models can be trained *causally* or *multitask* without extra I/O.

---

## ⚙️ Training Configuration (see `m1-config.yaml`)
```yaml
# Core modal settings
modalities: ["tok_rgb", "coords", "captions"]
vocab_sizes: [65536, 8192, 65536]
max_seq_lens: [256, 39, 64]   # 13 joints ×3 ≈39, caption truncated at 64 tokens
input_alphas:  [1, 1, 1]
target_alphas: [1, 1, 1]

# Selected optimisation hyper‑params
batch_size:  ${global_vars.batch_size}
total_tokens: 5000M
warmup_tokens: 500M
num_tokens_per_sample: 192     # 128 in + 64 out  (captions shorter)
lr: 6e-4
min_lr: 1e-6
weight_decay: 0.05
clip_grad: 1.0
dtype: fp32
training_time: 12:29:13
```

---

## 📉 Losses (final epoch)
| Component | Loss |
|-----------|------|
| **Total** | **3.70** |
| Captions  | 0.03 |
| Coords    | 2.42 |
| tok_rgb   | 8.89 |

---

## 🧪 Analysis
- **Captions:** Near‑perfect generation (loss 0.03); text descriptions are coherent and relevant to the visual content.  
- **Coords:** Solid (loss 2.42); predicted joint trajectories follow the ground‑truth motion convincingly.  
- **RGB Frames:** Performance is **poor** – attempts to reconstruct frames from pose → RGB produce mostly a **black box artefact** with faint silhouettes.  
  - Indicates that the *image space* objective dominates the total loss but fails to converge.  
  - Likely caused by **information asymmetry** (coords ≪ RGB token count) and insufficient decoder capacity.

### Next Steps
1. **Revise data pipeline** – experiment with *frame‑to‑frame* targets (`tok_rgb_next`) instead of pose‑to‑RGB to reduce modality gap.  
2. **Curriculum learning** – start with lower‑resolution DI32×32 tokens and progressively unfreeze higher frequencies.  
3. **Loss re‑balancing** – scale the RGB term down or introduce *perceptual* / *VQGAN* losses.  
4. **Augment captions** – add action verbs (BLIP × ActionBank) to help align motion semantics.

---
