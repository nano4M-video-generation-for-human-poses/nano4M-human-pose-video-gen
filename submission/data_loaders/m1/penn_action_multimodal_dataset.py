# Copyright 2025 EPFL
# Apache-2.0

import os, json
from pathlib import Path
from typing import List, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PennActionMultimodalDataset(Dataset):
    """
    Loads pre-tokenised Penn-Action data produced by the Cosmos-+-BLIP pipeline.

    Directory assumed:
      {root_dir}/{split}/{modality}/{video}/{frame_stem}.{ext}

    Example modalities:
      "tok_rgb", "tok_pose", "coords", "captions"
    """

    def __init__(self,
                 root_dir: str,
                 split: str,
                 modalities: List[str],
                 transforms: Optional[Callable] = None,
                 sample_from_k_augmentations: int = 1,
                 text_tokenizer_path: str = "gpt2",
                 text_max_length: int = 256,
                 verify_alignment: bool = True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.modalities = modalities
        self.transforms = transforms
        self.K = sample_from_k_augmentations

        # ---- file enumeration (video + frame) -----------------------------
        self.file_stems = self._collect_stems()        # e.g. "0001/00001"
        self.mod_ext    = self._infer_extensions()

        if verify_alignment:
            self._check_alignment()

        # ---- text tokenizer ----------------------------------------------
        self.text_tok = self._build_text_tokenizer(text_tokenizer_path,
                                                   text_max_length)
        self.text_max_len = text_max_length

    # ---------------------------------------------------------------------
    # helpers
    # ---------------------------------------------------------------------
    def _collect_stems(self):
        """
        Walk the first modality folder and list <video>/<frame_stem> for
        every file it finds.  Assumes zero-padded names; natural sort applied.
        """
        mod0_dir = self.root_dir / self.split / self.modalities[0]
        stems = []
        for video_dir in sorted(d for d in mod0_dir.iterdir() if d.is_dir()):
            for file_path in sorted(video_dir.iterdir()):
                if file_path.is_file():
                    stems.append(f"{video_dir.name}/{file_path.stem}")
        if not stems:
            raise RuntimeError(f"No data found under {mod0_dir}")
        return stems
    def _infer_extensions(self):
        """Look at one real file inside each modality/video and store its suffix."""
        exts = {}
        for m in self.modalities:
            mod_dir = self.root_dir / self.split / m
            # --- pick the first actual file ---
            first_file = next((p for p in mod_dir.rglob("*") if p.is_file()), None)
            if first_file is None:
                raise RuntimeError(f"No files under {mod_dir}")
            exts[m] = first_file.suffix        # '.npy' or '.json'
        return exts

    def _check_alignment(self):
        """
        Verify every modality actually has the exact same set of stems.
        Runs once at construction – raises if something is missing.
        """
        ref = set(self.file_stems)
        for m in self.modalities[1:]:
            stems = {f"{p.parent.name}/{p.stem}"
                     for p in (self.root_dir/self.split/m).rglob("*.*")}
            missing = ref - stems
            if missing:
                raise RuntimeError(f"{m} missing {len(missing)} items, "
                                   f"e.g. {next(iter(missing))}")

    def _build_text_tokenizer(self, name_or_path, max_len):
        tok = AutoTokenizer.from_pretrained(name_or_path)
        tok.add_special_tokens({'pad_token': '[PAD]',
                                'bos_token': '[SOS]',
                                'eos_token': '[EOS]'})
        tok._tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            special_tokens=[('[EOS]', tok.eos_token_id),
                            ('[SOS]', tok.bos_token_id)]
        )
        tok.model_max_length = max_len
        return tok

    # ---------------------------------------------------------------------
    # Dataset interface
    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self.file_stems)

    def __getitem__(self, idx):
        stem = self.file_stems[idx]          # e.g. '0001/00023'
        k    = np.random.randint(0, self.K)

        sample = {}
        for m in self.modalities:
            ext = self.mod_ext[m]
            path = self.root_dir / self.split / m / f"{stem}{ext}"

            if "tok_rgb" == m:
                arr = np.load(path, mmap_mode="r")          # shapes:
                                                            # (H,W)  = 40×40   or
                                                            # (K,H,W)= K×40×40
                # -------- pick the right augmentation slice ----------
                if arr.ndim == 3:          # (K, H, W)
                    kk  = k if k < arr.shape[0] else 0
                    tok = arr[kk]
                else:                      # (H, W)
                    tok = arr
            
                # -------- flatten to 1-D sequence of 1600 codes -------
                tok = tok.flatten()        # (1600,)
            
                sample[m] = torch.as_tensor(tok, dtype=torch.long)

            elif m == "coords":
                # ── load (K,3,13) or (3,13) ─────────────────────────────────────────
                arr = np.load(path, mmap_mode="r")
                kk  = k if arr.ndim == 3 else 0
                data = arr[kk] if arr.ndim == 3 else arr       # shape (3, 13)
            
                # -------------------------------------------------------------------
                # split into   x (13) | y (13) | vis (13)   then tokenise each part
                # -------------------------------------------------------------------
            
                # 1) coordinates 
                xy = data[:2].reshape(-1)          # 26 numbers in [0, 1]
                # map 0 → 1 , 1 → 8190   (leave 0 for PAD, 8191 for "vis = 1")
                xy_tok = np.clip((xy * 8189 + 1).astype(np.int64), 1, 8190)
            
                # 2) visibility flags 
                vis = data[2].reshape(-1)          # 13 values ∈ {0., 1.}
                vis_tok = np.where(vis > 0.5, 8191, 0).astype(np.int64)
            
                # 3) concatenate & ship off  ───────────────────────────────────────
                tokens = np.concatenate([xy_tok, vis_tok])    # (39,)
                sample[m] = torch.as_tensor(tokens, dtype=torch.long)
            elif m in ("captions"):
                caps_json = json.load(open(path, "r"))
            
                # Accept *either* list-of-strings *or* dict with single caption
                if isinstance(caps_json, list):
                    cap_text = caps_json[k] if k < len(caps_json) else caps_json[0]
                elif isinstance(caps_json, dict):
                    cap_text = caps_json.get("caption", "")
                else:
                    raise ValueError(f"Unsupported caption format in {path}")
            
                ids = self.text_tok(
                    cap_text,
                    max_length=self.text_max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )["input_ids"][0]
                sample[m] = ids.long()

            else:
                raise ValueError(f"Unknown modality '{m}'")

        if self.transforms:
            sample = self.transforms(sample)
        return sample
