# M3 Data Processing Notebook

## 🧩 Tokenization Overview
This notebook tokenizes the dataset into distinct modalities for multimodal modeling. In particular:

| Modality | What it represents | How it is tokenized |
|----------|-------------------|---------------------|
| `tok_rgb` | Tokens for the **reference RGB frame**. | 1) Resize frame so the shorter side is **256 px**.<br>2) Convert to a tensor in \[-1, 1\].<br>3) Pass through the **Cosmos-0.1 DI16×16 image tokenizer**, producing a 16 × 16 grid of discrete tokens (1024 tokens per frame). |
| `tok_rgb_next` | Tokens for the **first visually distinct successor frame** (chosen where SSIM < 0.89). | Identical Cosmos pipeline as `tok_rgb`. |
| `coords` | **Joint coordinates** for the reference frame. | For each of 13 joints: scale (x, y) to the 256 × 256 canvas, then **quantise into 8 192 bins** (0 – 8191). Visibility is kept as a binary token. Output shape: *(3 × 13)* (`x_tok`, `y_tok`, `vis`). |
| `coords_next` | Joint coordinates for the successor frame. | Same quantisation as `coords`. |
| `captions` | **Natural-language caption** of the reference frame. | Generated with **Salesforce BLIP** (base) and stored as raw text in a JSON file. |


/!\ We take one frame that is not as similar as the first reference (6 previously) from each videos and then we take as much as possible from each videos.  /!\

## 🔀 Data Splitting
After preprocessing, every `(reference frame, successor frame)` pair is shuffled with a fixed seed = 42 and moved into dedicated folders via `split_output_dataset()`:

- **80 %** → `train/`
- **10 %** → `eval/`
- **10 %** → `test/`

Each split contains **all five modalities** (`tok_rgb`, `tok_rgb_next`, `coords`, `coords_next`, `captions`) so that every training or evaluation sample is complete and self-contained.
# M3 Data Processing Notebook

## 🧩 Tokenization Overview
This notebook tokenizes the dataset into distinct modalities for multimodal modeling. In particular:

| Modality         | What it represents                                | How it is tokenized |
|------------------|---------------------------------------------------|---------------------|
| `tok_rgb`        | Tokens for the **reference RGB frame**.           | 1) Resize frame so the shorter side is **256 px**.<br>2) Convert to a tensor in \[-1, 1\].<br>3) Pass through the **Cosmos-0.1 DI16×16 image tokenizer**, producing a 16 × 16 grid of discrete tokens (1024 tokens per frame). |
| `tok_rgb_next`   | Tokens for the **first visually distinct successor frame** (chosen where SSIM < 0.89). | Identical Cosmos pipeline as `tok_rgb`. |
| `coords`         | **Joint coordinates** for the reference frame.     | For each of 13 joints: scale (x, y) to the 256 × 256 canvas, then **quantise into 8 192 bins** (0 – 8191). Visibility is kept as a binary token. Output shape: *(3 × 13)* (`x_tok`, `y_tok`, `vis`). |
| `coords_next`    | Joint coordinates for the successor frame.         | Same quantisation as `coords`. |
| `captions`       | **Natural-language caption** of the reference frame. | Generated with **Salesforce BLIP** (base) and stored as raw text in a JSON file. |

## 🔀 Data Splitting
After preprocessing, every `(reference frame, successor frame)` pair is shuffled with a fixed seed = 42 and moved into dedicated folders via `split_output_dataset()`:

- **80 %** → `train/`
- **10 %** → `eval/`
- **10 %** → `test/`

Each split contains **all five modalities** (`tok_rgb`, `tok_rgb_next`, `coords`, `coords_next`, `captions`) so that every training or evaluation sample is complete and self-contained.

## ⚙️ Training Configuration

### ✨ Modalities
```yaml
modalities: ["tok_rgb", "coords", "tok_rgb_next", "coords_next"]  # Input and output modalities
vocab_sizes: [65536, 65536, 65536, 65536]                          # Vocab sizes for each modality
max_seq_lens: [256, 39, 256, 39]                                   # Max sequence lengths for each modality
input_alphas: [1, 1, 1, 1]                                         # Input Dirichlet alpha values
target_alphas: [1, 1, 1, 1]                                        # Target Dirichlet alpha values
input_tokens_range: [32, 128]                                      # Min and max encoder tokens
target_tokens_range: [32, 128]                                     # Min and max decoder tokens
```

### 🏋️ Training Hyperparameters
```yaml
batch_size: ${global_vars.batch_size}
total_tokens: 5000             # in millions of tokens
warmup_tokens: 500             # in millions of tokens
num_tokens_per_sample: 256     # 128 input + 128 output tokens per sample
lr: 0.0006                     # Max learning rate
min_lr: 0.000001               # Min learning rate (after cosine decay)
weight_decay: 0.05             # AdamW weight decay
clip_grad: 1.0                 # Gradient clipping norm
dtype: fp32                   # Precision (use fp16 on V100s or bf16 on A100s if stable)
training_time: 12:13:08        # Total training time
```
## 🧪 Analysis

- A bug in the `coords` tokenization affected generation quality for joint-based predictions.
- Despite this, the model was able to generate **6 consecutive `tok_rgb_next` frames**.
- These predicted frames captured **essential scene elements** such as the **background** and the **start of human movement**.
- The generated sequence shows that **motion execution is partially preserved**, although finer details like the human skeleton is not always consistently reconstructed.