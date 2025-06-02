# Copyright 2025 EPFL
# Apache-2.0

import json
from pathlib import Path
from typing import List, Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

# disable parallelism warning from tokenizers
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
XY_MAX, VIS_TOK = 8190, 8191          # keep in sync with CoordTokenizer
TOK_RGB_FRAME_TOKS = 40 * 40          # 1600 codes per RGB frame
COORDS_FRAME_TOKS  = 39               # 26 XY + 13 vis tokens
NEXT_LEN           = 1              # “_next” modalities contain 6 frames

class PennDataset(Dataset):
    """
    Loads pre-tokenised data organized as:
      {root_dir}/{split}/{modality}/{pid}/{file}
    where each `pid` folder contains exactly one file whose
    name is arbitrary but consistent per modality.
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        modalities: List[str],
        transforms: Optional[Callable] = None,
        sample_from_k_augmentations: int = 1,
        text_tokenizer_path: str = "gpt2",
        text_max_length: int = 256,
        verify_alignment: bool = True,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.modalities = modalities
        self.transforms = transforms
        self.K = sample_from_k_augmentations

        # collect all PID names from the first modality directory
        self.pids = self._collect_pids()
        # figure out each modality's file extension (.npy, .json, ...)
        self.mod_ext = self._infer_extensions()

        if verify_alignment:
            self._check_alignment()

        # build text tokenizer if captions are in use
        if "captions" in modalities:
            self.text_tok = self._build_text_tokenizer(
                text_tokenizer_path, text_max_length
            )
            self.text_max_len = text_max_length

    def _collect_pids(self) -> List[str]:
        base = self.root_dir / self.split / self.modalities[0]
        if not base.exists():
            raise RuntimeError(f"No such directory: {base}")
        pids = sorted([d.name for d in base.iterdir() if d.is_dir()])
        if not pids:
            raise RuntimeError(f"No PID folders under {base}")
        return pids

    def _infer_extensions(self):
        exts = {}
        for m in self.modalities:
            mod_dir = self.root_dir / self.split / m
            # find first file under any PID folder
            first = next(
                (p for pid in mod_dir.iterdir() if pid.is_dir()
                      for p in pid.iterdir() if p.is_file()),
                None
            )
            if first is None:
                raise RuntimeError(f"No files found under {mod_dir}")
            exts[m] = first.suffix
        return exts

    def _check_alignment(self):
        # ensure every modality has exactly the same set of pids
        ref = set(self.pids)
        for m in self.modalities[1:]:
            found = {
                d.name
                for d in (self.root_dir / self.split / m).iterdir()
                if d.is_dir()
            }
            missing = ref - found
            if missing:
                raise RuntimeError(f"Modality '{m}' missing PIDs: {sorted(missing)}")
    def _quantise_coords(self,frame: np.ndarray) -> np.ndarray:
        """
        frame: (3, 13) float32  – two XY rows in [0,1] and one visibility row 0/1
        returns: (39,) int64    – 26 XY tokens (1 … 8190) + 13 vis tokens (0 / 8191)
        """
        xy  = frame[:2]                                         # (2,13)
        vis = frame[2]                                          # (13,)
    
        xy_tok  = np.rint(xy * (XY_MAX - 1) + 1).astype(np.int64)
        vis_tok = (vis > 0.5).astype(np.int64) * VIS_TOK
    
        return np.concatenate([xy_tok.flatten(), vis_tok])
    def _build_text_tokenizer(self, name_or_path, max_len):
        tok = AutoTokenizer.from_pretrained(name_or_path)
        tok.add_special_tokens({
            'pad_token': '[PAD]',
            'bos_token': '[SOS]',
            'eos_token': '[EOS]'
        })
        tok._tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            special_tokens=[
                ('[SOS]', tok.bos_token_id),
                ('[EOS]', tok.eos_token_id),
            ]
        )
        tok.model_max_length = max_len
        return tok
    def _is_tokenised(self, arr: np.ndarray) -> bool:
        """Return True if `arr` looks like integer tokens, not floats."""
        return np.issubdtype(arr.dtype, np.integer) or arr.max() > 1.0
    def _looks_like_raw_xy(self, arr: np.ndarray) -> bool:
        """True ⇢ values in [0,1] and a float dtype."""
        return arr.dtype.kind == "f" and arr.max() <= 1.001
    def __len__(self):
        return len(self.pids)
    def __getitem__(self, idx: int):
        pid   = self.pids[idx]
        k     = np.random.randint(0, self.K)
        sample = {}
    
        for m in self.modalities:
            fdir = self.root_dir / self.split / m / pid
            ext  = self.mod_ext[m]
    
            # ------------------------------------------------ tok_rgb ---------
            if m == "tok_rgb":
                path = next(fdir.glob(f"*{ext}"))
                arr  = np.load(path, mmap_mode="r", allow_pickle=True)   # (K,H,W) or (H,W)
                if arr.ndim == 3:
                    arr = arr[k % arr.shape[0]]
                sample[m] = torch.from_numpy(arr.flatten()).long()
    
            # --------------------------------------------- tok_rgb_next -------
            elif m == "tok_rgb_next":
                frames = []
                for p in sorted(fdir.glob(f"*{ext}")):                   # 6 files
                    arr = np.load(p, mmap_mode="r", allow_pickle=True)   # (K,H,W) or (H,W)
                    if arr.ndim == 3:
                        arr = arr[k % arr.shape[0]]
                    frames.append(arr.reshape(-1))                       # 1600
                sample[m] = torch.from_numpy(np.concatenate(frames)).long()  # 9600
    
            # -------------------------------------------------- coords --------
            elif m == "coords":
                path = next(fdir.glob(f"*{ext}"))
                arr  = np.load(path, mmap_mode="r", allow_pickle=True)
            
                if arr.ndim >= 2 and arr.shape[-2:] == (3, 13):
                    # matrix → flatten
                    frame  = arr[k % arr.shape[0]] if arr.ndim == 3 else arr
                    tokens = frame.reshape(-1).astype(np.int64)              # (39,)
                else:
                    if self._looks_like_raw_xy(arr):                        # raw floats
                        frame  = arr[k] if arr.ndim == 3 else arr
                        tokens = self._quantise_coords(frame)               # (39,)
                    else:                                                   # already 39 ints
                        tokens = (arr[k] if arr.ndim == 2 else arr).reshape(-1).astype(np.int64)
            
                sample[m] = torch.from_numpy(tokens)
            
            # ───────────────────────────── coords_next (one frame) ─────────────────────
            elif m == "coords_next":
                frame_tokens = []
                for p in sorted(fdir.glob(f"*{ext}")):                      # normally 1 file
                    arr = np.load(p, mmap_mode="r", allow_pickle=True)
            
                    if arr.ndim >= 2 and arr.shape[-2:] == (3, 13):
                        frame  = arr[k % arr.shape[0]] if arr.ndim == 3 else arr
                        tokens = frame.reshape(-1).astype(np.int64)
                    else:
                        if self._looks_like_raw_xy(arr):
                            frame  = arr[k] if arr.ndim == 3 else arr
                            tokens = self._quantise_coords(frame)
                        else:
                            tokens = (arr[k] if arr.ndim == 2 else arr).reshape(-1).astype(np.int64)
            
                    frame_tokens.append(tokens)
            
                sample[m] = torch.from_numpy(np.concatenate(frame_tokens))   # (39,) or (T·39,)
    
            # ------------------------------------------------ captions --------
            elif m == "captions":
                path = next(fdir.glob(f"*{ext}"))
                caps = json.load(open(path, "r"))
                text = caps[k % len(caps)] if isinstance(caps, list) \
                       else caps.get("caption", "")
                ids = self.text_tok(text,
                                    max_length=self.text_max_len,
                                    padding="max_length",
                                    truncation=True,
                                    return_tensors="pt")["input_ids"][0]
                sample[m] = ids.long()
    
            else:
                raise ValueError(f"Unknown modality {m}")
    
        return self.transforms(sample) if self.transforms else sample
    