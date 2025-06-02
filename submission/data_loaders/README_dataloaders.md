# Data-Loader Suite

This directory wraps every flavour of the **Penn-Action** dataset we use across the M‑series models.

```
├── create_penn_action_masked_dataloader.py        # baseline factory
├── create_penn_action_masked_dataloader_opt.py    # optimised factory
├── M1/                                            # ⇢ loader for M1
├── M4/                                            # ⇢  loader for M4
├── M3/                                            # ⇢ loader for M3
└── M2/                                            # ⇢ loader for M2
```

---

## 1 ↗︎ `create_penn_action_masked_dataloader.py`
A **drop‑in utility** that turns any split (train/eval/test) into a masked, batched PyTorch iterator.

**Pipeline**
1. Composes a `SimpleMultimodalMasking` transform to inject `[MASK]` tokens.
2. Instantiates `PennActionMultimodalDataset` – one file‑per‑frame layout.
3. Adds a `DistributedSampler` when `distributed=True`.
4. Returns a vanilla `torch.utils.data.DataLoader`.

---

## 2 🚀 `create_penn_action_masked_dataloader_opt.py`
A **performance‑oriented** variant with multiple quality‑of‑life tweaks:

| Feature | Why it matters |
|---------|----------------|
| **DataLoader2 backend (opt‑in)** | Better IO overlap & fault tolerance under distributed training. |
| **`prefetch_factor`, `persistent_workers`** | Keeps worker pools warm, reducing epoch‑to‑epoch stalls. |
| **Optional CUDA pre‑fetch** | Asynchronously moves the *next* batch to GPU, hiding host→device latency. |
| **Infinite iterator wrapper** | Supports streamer‑style training loops that never reset epoch counters. |

---

## 3 Model‑specific folders (M1, M2, M3, M4)
Each sub‑directory contains a thin adapter that *only* tweaks path conventions and modality lists so the shared factories above work with that model’s preprocessing scheme.

---
