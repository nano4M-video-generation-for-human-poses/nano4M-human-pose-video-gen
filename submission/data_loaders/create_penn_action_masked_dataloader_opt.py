# %% ──────────────────────────── Optimised dataloader factory ───────────────────────────
"""
Optimised Penn-Action 4-Modality dataloader factory.

Adds:
  • persistent_workers & prefetch_factor
  • optional CUDA pre-fetch (moves next batch to GPU on a side-stream)
  • optional DataLoader2 backend
"""

from functools import partial
from typing import Optional

import torch
from torch.utils.data import DataLoader, DistributedSampler
try:                                 # DataLoader2 is nicer but optional
    from torchdata.dataloader2 import DataLoader2
    from torchdata.dataloader2.reading_service import (
        DistributedReadingService, MultiProcessingReadingService
    )
    _HAS_DL2 = True
except ModuleNotFoundError:
    _HAS_DL2 = False

from nanofm.data.multimodal.penn_action_multimodal_dataset_opt_v2 import (
    PennDataset,
)
from nanofm.data.multimodal.masking import SimpleMultimodalMasking


# ─────────────────────────────────── helpers ──────────────────────────────────
def _move_to_device(batch, device: torch.device):
    """Recursively move a (nested) batch dict/list/tuple to `device`."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(_move_to_device(v, device) for v in batch)
    return batch


def _cuda_prefetch_dataloader(loader: DataLoader, device: torch.device):
    """
    Wrap a DL so that *next* batch is moved to `device` on a separate CUDA stream
    before the caller asks for it (one-element look-ahead).
    """
    stream = torch.cuda.Stream(device)
    first  = True
    next_batch = None

    for batch in loader:
        # enqueue copy of *future* batch on side-stream
        with torch.cuda.stream(stream):
            next_batch = _move_to_device(batch, device)
        # on first iteration we only launch the copy
        if first:
            first = False
        else:
            # make current stream wait until previous copy finished
            torch.cuda.current_stream(device).wait_stream(stream)
            yield prev_batch
        prev_batch = next_batch

    # yield last batch
    torch.cuda.current_stream(device).wait_stream(stream)
    yield next_batch


# ───────────────────────────────── factory ───────────────────────────────────
def create_penn_action_masked_dataloader_opt(
        root_dir: str,
        split: str,
        *,
        modalities,
        vocab_sizes,
        max_seq_lens,
        overlap_vocab: bool,
        overlap_posembs: bool,
        input_alphas,
        target_alphas,
        input_tokens_range,
        target_tokens_range,
        sample_from_k_augmentations: int = 1,
        text_tokenizer_path: str = "gpt2",
        text_max_length: int = 256,
        # loader knobs ---------------------------------------------------------
        batch_size: int = 128,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        shuffle: bool = True,
        distributed: bool = True,
        backend_dl2: Optional[bool] = None,        # None → auto (use DL2 if avail)
        # extra features -------------------------------------------------------
        infinite: bool = False,
        device_prefetch: Optional[str] = None,     # e.g. "cuda:0"
):
    """
    Create an optimised Penn-Action dataloader.

    Parameters
    ----------
    backend_dl2 : bool | None
        True  → force DataLoader2,  False → classic DataLoader,
        None  → auto-select DataLoader2 if installed.
    device_prefetch : str | None
        If a torch device is given (e.g. "cuda:0"), batches are moved there
        asynchronously *one iteration ahead*.
    """
    # 1) masking transform ----------------------------------------------------
    mask_transform = SimpleMultimodalMasking(
        modalities              = modalities,
        vocab_sizes             = vocab_sizes,
        max_seq_lens            = max_seq_lens,
        input_alphas            = input_alphas,
        target_alphas           = target_alphas,
        input_tokens_range      = tuple(input_tokens_range),
        target_tokens_range     = tuple(target_tokens_range),
        overlap_vocab           = overlap_vocab,
        overlap_posembs         = overlap_posembs,
        include_unmasked_data_dict = True,
    )

    # 2) dataset --------------------------------------------------------------
    dataset = PennDataset(
        root_dir                    = root_dir,
        split                       = split,
        modalities                  = modalities,
        transforms                  = mask_transform
    )

    # 3) sampler --------------------------------------------------------------
    sampler = (DistributedSampler(dataset,
                                  drop_last = drop_last,
                                  shuffle   = shuffle)
               if distributed else None)

    # 4) choose backend -------------------------------------------------------
    use_dl2 = (_HAS_DL2 if backend_dl2 is None else backend_dl2)
    if use_dl2 and not _HAS_DL2:
        raise RuntimeError("DataLoader2 requested but torchdata not installed.")

    # 5) build loader ---------------------------------------------------------
    if use_dl2:
        # DataLoader2 uses "reading services" to handle multiprocessing / dist.
        reading_svc = []
        if distributed:
            reading_svc.append(DistributedReadingService())
        if num_workers > 0:
            reading_svc.append(MultiProcessingReadingService(num_workers))
        reading_svc = reading_svc or None

        loader = DataLoader2(
            dataset,
            batch_size        = batch_size,
            datapipe_adapter_fn = lambda dp: dp,    # no special transforms
            reading_service   = reading_svc,
            drop_last         = drop_last,
            pin_memory        = pin_memory,
            prefetch_factor   = prefetch_factor,
            persistent_workers= persistent_workers,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size        = batch_size,
            sampler           = sampler,
            shuffle           = (sampler is None and shuffle),
            num_workers       = num_workers,
            pin_memory        = pin_memory,
            drop_last         = drop_last,
            persistent_workers= persistent_workers,
            prefetch_factor   = prefetch_factor,
        )

    # 6) wrap into infinite iterator if requested ----------------------------
    if infinite:
        from nanofm.data.utils import infinite_iterator
        loader = infinite_iterator(loader)

    # 7) optional CUDA pre-fetch ---------------------------------------------
    if device_prefetch is not None and torch.cuda.is_available():
        device = torch.device(device_prefetch)
        loader = _cuda_prefetch_dataloader(loader, device)

    return loader