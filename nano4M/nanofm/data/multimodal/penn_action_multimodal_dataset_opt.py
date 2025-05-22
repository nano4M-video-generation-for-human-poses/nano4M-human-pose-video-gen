# %% ---------------------------- Cell · PennLoader ---------------------------
"""
Five-modality Penn-Action dataloader
-----------------------------------
Put this cell in your notebook *after* the preprocessing cells have run.
"""

import os, json, re
from pathlib import Path
from typing import Optional, Callable, Dict

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# %% ───────────────────────────── The PennDataset class ───────────────────────
class PennDataset(Dataset):
    MODS = ["tok_rgb", "coords", "captions", "tok_rgb_next", "coords_next"]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        text_tok: str = "gpt2",
        text_max: int = 256,
    ):
        self.root   = Path(root) / split
        self.trans  = transforms
        self.text   = self._build_text_tok(text_tok, text_max)

        self._ext   = {m: self._guess_ext(m) for m in self.MODS}
        self.stems  = self._list_refs()                      # 0001/00001 …

    # --- torch Dataset interface ---------------------------------------------------
    def __len__(self): return len(self.stems)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem   = self.stems[idx]
        sample = {}

        # reference RGB ------------------------------------------------------------
        rgb   = np.load(self._p("tok_rgb", stem)).flatten()
        sample["tok_rgb"] = torch.from_numpy(rgb).long()

        # successor RGB ------------------------------------------------------------
        nxt = self._load_successors("tok_rgb_next", stem)
        sample["tok_rgb_next"] = torch.from_numpy(nxt).long()

        # reference coords ---------------------------------------------------------
        c_ref = np.load(self._p("coords", stem))
        sample["coords"] = torch.from_numpy(self._coords_to_tok(c_ref)).long()

        # successor coords ---------------------------------------------------------
        c_next = self._load_successors("coords_next", stem,
                                       convert=self._coords_to_tok)
        sample["coords_next"] = torch.from_numpy(c_next).long()

        # caption ------------------------------------------------------------------
        text_json = json.load(open(self._p("captions", stem)))
        text = text_json.get("caption", "") if isinstance(text_json, dict) else text_json[0]
        ids  = self.text(text,
                         padding="max_length",
                         truncation=True,
                         max_length=self.text.model_max_length,
                         return_tensors="pt")["input_ids"][0]
        sample["captions"] = ids.long()

        if self.trans:
            sample = self.trans(sample)
        return sample

    # --- private helpers ----------------------------------------------------------
    def _p(self, mod: str, s: str) -> Path:
        return self.root / mod / f"{s}{self._ext[mod]}"

    def _guess_ext(self, mod):
        folder = self.root / mod
        return next(p.suffix for p in folder.rglob("*.*") if p.is_file())

    def _list_refs(self):
        stems = []
        for vd in sorted((self.root/"tok_rgb").iterdir()):
            for f in sorted(vd.glob("*.npy")):
                if not re.search(r"_n\\d+$", f.stem):        # skip successors
                    stems.append(f"{vd.name}/{f.stem}")
        return stems

    # ---- coords helper ----------------------------------------------------------
    @staticmethod
    def _coords_to_tok(arr: np.ndarray) -> np.ndarray:
        xy  = arr[:2].reshape(-1).astype(np.int64)     # 26
        vis = (8191 * (arr[2] > 0)).astype(np.int64)   # 13 (0 or 8191)
        return np.concatenate([xy, vis])               # (39,)

    def _load_successors(self, mod, stem, convert=None):
        vid, fr = stem.split("/")
        files = sorted((self.root/mod/vid).glob(f"{fr}_n*.{self._ext[mod][1:]}"))
        chunks = []
        for p in files:
            data = np.load(p) if p.suffix == ".npy" else json.load(open(p))
            if convert is not None:
                data = convert(data)
            chunks.append(data.reshape(-1))
        return np.concatenate(chunks).astype(np.int64) if chunks else np.zeros(0, dtype=np.int64)

    # ---- text tokenizer ---------------------------------------------------------
    @staticmethod
    def _build_text_tok(path, max_len):
        tok = AutoTokenizer.from_pretrained(path)
        tok.add_special_tokens({'pad_token': '[PAD]','bos_token': '[SOS]','eos_token': '[EOS]'})
        from tokenizers.processors import TemplateProcessing
        tok._tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            special_tokens=[('[EOS]', tok.eos_token_id),('[SOS]', tok.bos_token_id)]
        )
        tok.model_max_length = max_len
        return tok
