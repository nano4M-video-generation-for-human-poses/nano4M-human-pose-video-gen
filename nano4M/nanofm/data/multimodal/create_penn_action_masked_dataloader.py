# Copyright 2025 EPFL – Apache-2.0
"""Factory that builds a Penn-Action dataloader with the 4M masking transform."""

from torch.utils.data import DataLoader, DistributedSampler
from nanofm.data.multimodal.penn_action_multimodal_dataset import (
    PennActionMultimodalDataset,
)
from nanofm.data.multimodal.masking import SimpleMultimodalMasking

def create_penn_action_masked_dataloader(
        root_dir: str,
        split: str,
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
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True,
        distributed: bool = True,
        infinite: bool = False,
):
    # 1) masking transform ---------------------------------------------------
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

    # 2) dataset -------------------------------------------------------------
    dataset = PennActionMultimodalDataset(
        root_dir                    = root_dir,
        split                       = split,
        modalities                  = modalities,
        transforms                  = mask_transform,
        sample_from_k_augmentations = sample_from_k_augmentations,
        text_tokenizer_path         = text_tokenizer_path,
        text_max_length             = text_max_length,
    )

    # 3) sampler -------------------------------------------------------------
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
    else:
        sampler = None

    # 4) dataloader ----------------------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size      = batch_size,
        sampler         = sampler,
        shuffle         = (sampler is None and shuffle),
        num_workers     = num_workers,
        pin_memory      = pin_memory,
        drop_last       = drop_last,
        persistent_workers = num_workers > 0,
    )

    # Optionally wrap into an infinite iterator (used during training)
    if infinite:
        from nanofm.data.utils import infinite_iterator
        loader = infinite_iterator(loader)

    return loader