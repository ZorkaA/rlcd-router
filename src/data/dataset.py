"""
Dataset module for MoE Speculative Router Calibration.

Handles:
1. Sequence-atomic 80/20 train/calibration partitioning with zero data leakage.
2. Multi-horizon lookahead target alignment for deep layers (5-24) at T+1, T+2, T+3.
3. Sequence boundary token masking (L-3..L-1) to isolate sequence contexts.
4. Safetensors serialization and deserialization with metadata headers.
5. PyTorch MoECalibrationDataset supporting in-memory and memory-mapped execution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset

from src.data.stream_extractor import ExtractionBatch


class DatasetItem(NamedTuple):
    """Container for a single calibration dataset sample."""
    hidden_states: torch.Tensor          # (d_model,)
    target_router_logits: torch.Tensor   # (horizons, num_deep_layers, num_experts)
    target_top4_indices: torch.Tensor    # (horizons, num_deep_layers, top_k)
    valid_mask: torch.Tensor             # (horizons,)


@dataclass
class AlignedSequence:
    """Container for aligned sequence tensors."""
    hidden_states: torch.Tensor          # (N, d_model)
    target_router_logits: torch.Tensor   # (N, H, num_deep_layers, num_experts)
    target_top4_indices: torch.Tensor    # (N, H, num_deep_layers, top_k)
    valid_mask: torch.Tensor             # (N, H)

    @classmethod
    def empty(
        cls,
        d_model: int,
        num_horizons: int,
        num_deep_layers: int,
        num_experts: int,
        top_k: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu"),
    ) -> AlignedSequence:
        return cls(
            hidden_states=torch.empty((0, d_model), dtype=dtype, device=device),
            target_router_logits=torch.empty(
                (0, num_horizons, num_deep_layers, num_experts), dtype=dtype, device=device
            ),
            target_top4_indices=torch.empty(
                (0, num_horizons, num_deep_layers, top_k), dtype=torch.int64, device=device
            ),
            valid_mask=torch.empty((0, num_horizons), dtype=torch.bool, device=device),
        )


@dataclass
class DatasetSplitConfig:
    """Configuration parameters for dataset splitting and alignment."""
    train_ratio: float = 0.8
    calib_ratio: Optional[float] = None
    deep_layer_start: int = 5    # 1-indexed (Layer 5)
    deep_layer_end: int = 24     # 1-indexed (Layer 24)
    horizons: Tuple[int, ...] = (1, 2, 3)
    top_k: int = 4
    drop_boundary_tokens: bool = False
    shuffle_sequences: bool = False
    seed: int = 42
    dtype: torch.dtype = torch.float16


def partition_sequence_indices(
    num_sequences: int,
    train_ratio: float = 0.8,
    calib_ratio: Optional[float] = None,
    shuffle: bool = False,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Partition sequence indices into strictly disjoint train and calibration splits.

    Guarantees sequence-atomic isolation with zero context contamination.
    """
    if num_sequences < 2:
        raise ValueError(
            f"At least 2 sequences required for train/calib partitioning, got {num_sequences}."
        )

    if calib_ratio is not None:
        if not (0.0 < calib_ratio < 1.0):
            raise ValueError(f"calib_ratio must be between 0 and 1, got {calib_ratio}")
        n_calib = max(1, int(round(num_sequences * calib_ratio)))
        n_train = num_sequences - n_calib
    else:
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
        n_train = max(1, int(round(num_sequences * train_ratio)))
        n_calib = num_sequences - n_train

    if n_train < 1 or n_calib < 1:
        raise ValueError(
            f"Invalid split configuration: resulted in {n_train} train and {n_calib} calib sequences."
        )

    if shuffle:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(num_sequences, generator=g).tolist()
        train_indices = sorted(perm[:n_train])
        calib_indices = sorted(perm[n_train:])
    else:
        train_indices = list(range(n_train))
        calib_indices = list(range(n_train, num_sequences))

    # Assert strict disjointness invariant
    assert set(train_indices).isdisjoint(set(calib_indices)), "Context contamination detected in partition!"
    assert len(train_indices) + len(calib_indices) == num_sequences
    return train_indices, calib_indices


def align_sequence_targets(
    hidden_states: torch.Tensor,
    router_logits: Union[torch.Tensor, Sequence[torch.Tensor]],
    deep_layer_start: int = 5,
    deep_layer_end: int = 24,
    horizons: Sequence[int] = (1, 2, 3),
    top_k: int = 4,
    drop_boundary_tokens: bool = False,
) -> AlignedSequence:
    """
    Align multi-horizon targets for deep layers and apply boundary token masking.

    Args:
        hidden_states: Tensor of shape (L, d_model) or (1, L, d_model) from tap layer (Layer 3).
        router_logits: Tensor of shape (L, num_layers, num_experts) or tuple of 2D layer tensors.
        deep_layer_start: First deep layer (1-indexed, inclusive, default 5).
        deep_layer_end: Last deep layer (1-indexed, inclusive, default 24).
        horizons: Sequence of lookahead step counts (default: (1, 2, 3)).
        top_k: Number of expert indices to extract (default: 4).
        drop_boundary_tokens: If True, drop trailing tokens (L-3..L-1). If False, keep and mask.

    Returns:
        AlignedSequence containing hidden states, target logits, top-4 indices, and valid_mask.
    """
    if hidden_states.dim() == 3 and hidden_states.shape[0] == 1:
        hidden_states = hidden_states.squeeze(0)
    L, d_model = hidden_states.shape

    if isinstance(router_logits, (list, tuple)):
        # Normalize list/tuple of 2D tensors to deep layers slice
        total_layers = len(router_logits)
        start_idx = max(0, deep_layer_start - 1)
        end_idx = min(total_layers, deep_layer_end)
        deep_logits_tensor = torch.stack(
            [router_logits[i].squeeze(0) if router_logits[i].dim() == 3 else router_logits[i]
             for i in range(start_idx, end_idx)],
            dim=1
        )
    else:
        if router_logits.dim() == 4 and router_logits.shape[0] == 1:
            router_logits = router_logits.squeeze(0)
        total_layers = router_logits.shape[1]
        start_idx = max(0, deep_layer_start - 1)
        end_idx = min(total_layers, deep_layer_end)
        deep_logits_tensor = router_logits[:, start_idx:end_idx, :]

    num_deep_layers = deep_logits_tensor.shape[1]
    num_experts = deep_logits_tensor.shape[2]
    H = len(horizons)
    max_h = max(horizons)
    k = min(top_k, num_experts)

    if drop_boundary_tokens:
        valid_len = max(0, L - max_h)
        if valid_len == 0:
            return AlignedSequence.empty(
                d_model, H, num_deep_layers, num_experts, k,
                dtype=hidden_states.dtype, device=hidden_states.device
            )
        out_hidden = hidden_states[:valid_len]
        out_logits = torch.zeros(
            (valid_len, H, num_deep_layers, num_experts),
            dtype=deep_logits_tensor.dtype,
            device=deep_logits_tensor.device,
        )
        valid_mask = torch.ones((valid_len, H), dtype=torch.bool, device=deep_logits_tensor.device)
        for h_idx, h in enumerate(horizons):
            out_logits[:, h_idx] = deep_logits_tensor[h : h + valid_len]
    else:
        out_hidden = hidden_states
        out_logits = torch.zeros(
            (L, H, num_deep_layers, num_experts),
            dtype=deep_logits_tensor.dtype,
            device=deep_logits_tensor.device,
        )
        valid_mask = torch.zeros((L, H), dtype=torch.bool, device=deep_logits_tensor.device)
        for h_idx, h in enumerate(horizons):
            if h < L:
                valid_len = L - h
                out_logits[:valid_len, h_idx] = deep_logits_tensor[h:]
                valid_mask[:valid_len, h_idx] = True

    out_top4 = torch.topk(out_logits, k=k, dim=-1).indices
    return AlignedSequence(
        hidden_states=out_hidden,
        target_router_logits=out_logits,
        target_top4_indices=out_top4,
        valid_mask=valid_mask,
    )


def save_dataset_safetensors(
    tensors: Dict[str, torch.Tensor],
    filepath: Union[str, Path],
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    """
    Save calibration dataset tensors to a safetensors file with validation.

    Ensures all tensors are contiguous and conform to the contract schema.
    """
    required_keys = {"hidden_states", "target_router_logits", "target_top4_indices", "valid_mask"}
    missing = required_keys - set(tensors.keys())
    if missing:
        raise KeyError(f"Missing required contract tensors: {missing}")

    # Validate shape consistency
    n_samples = tensors["hidden_states"].shape[0]
    for key in required_keys:
        if tensors[key].shape[0] != n_samples:
            raise ValueError(
                f"Sample count mismatch for tensor '{key}': expected {n_samples}, got {tensors[key].shape[0]}"
            )

    # Validate contract dimensions
    if tensors["hidden_states"].dim() != 2:
        raise ValueError(f"hidden_states must be 2D (num_samples, d_model), got {tensors['hidden_states'].shape}")
    if tensors["target_router_logits"].dim() != 4:
        raise ValueError(f"target_router_logits must be 4D (N, H, layers, experts), got {tensors['target_router_logits'].shape}")
    if tensors["target_top4_indices"].dim() != 4:
        raise ValueError(f"target_top4_indices must be 4D (N, H, layers, top_k), got {tensors['target_top4_indices'].shape}")
    if tensors["valid_mask"].dim() != 2:
        raise ValueError(f"valid_mask must be 2D (N, H), got {tensors['valid_mask'].shape}")

    # Enforce contiguous layout
    contiguous_tensors = {k: v.contiguous() for k, v in tensors.items()}

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "num_samples": str(n_samples),
        "d_model": str(tensors["hidden_states"].shape[1]),
        "num_horizons": str(tensors["target_router_logits"].shape[1]),
        "num_deep_layers": str(tensors["target_router_logits"].shape[2]),
        "num_experts": str(tensors["target_router_logits"].shape[3]),
        "top_k": str(tensors["target_top4_indices"].shape[3]),
    }
    if metadata:
        meta.update(metadata)

    save_file(contiguous_tensors, str(filepath), metadata=meta)


def load_dataset_safetensors(
    filepath: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, torch.Tensor]:
    """Load calibration dataset tensors from a safetensors file into memory."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Safetensors file not found at: {filepath}")
    loaded = load_file(str(filepath), device=str(device))
    if "valid_mask" in loaded and loaded["valid_mask"].dtype != torch.bool:
        loaded["valid_mask"] = loaded["valid_mask"].bool()
    return loaded


class MoECalibrationDataset(Dataset):
    """
    PyTorch Dataset for MoE Speculative Router Calibration.

    Provides high-throughput batching for Medusa speculative head training,
    grid temperature scaling optimization, and targeted gating evaluation.
    """

    def __init__(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        target_router_logits: Optional[torch.Tensor] = None,
        target_top4_indices: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        filepath: Optional[Union[str, Path]] = None,
        in_memory: bool = True,
        metadata: Optional[Dict[str, str]] = None,
    ):
        if filepath is not None:
            self.filepath = str(filepath)
            self.in_memory = in_memory
            if in_memory:
                loaded = load_dataset_safetensors(self.filepath)
                self.hidden_states = loaded["hidden_states"]
                self.target_router_logits = loaded["target_router_logits"]
                self.target_top4_indices = loaded["target_top4_indices"]
                self.valid_mask = loaded["valid_mask"]
                with safe_open(self.filepath, framework="pt", device="cpu") as f:
                    self.metadata = f.metadata() or {}
                self._safe_handle = None
                self._length = self.hidden_states.shape[0]
            else:
                self.hidden_states = None
                self.target_router_logits = None
                self.target_top4_indices = None
                self.valid_mask = None
                with safe_open(self.filepath, framework="pt", device="cpu") as f:
                    self._length = f.get_slice("hidden_states").get_shape()[0]
                    self.metadata = f.metadata() or {}
                self._safe_handle = None
        else:
            if (
                hidden_states is None
                or target_router_logits is None
                or target_top4_indices is None
                or valid_mask is None
            ):
                raise ValueError("All four contract tensors must be supplied when filepath is None.")

            n_samples = hidden_states.shape[0]
            if not (
                target_router_logits.shape[0] == n_samples
                and target_top4_indices.shape[0] == n_samples
                and valid_mask.shape[0] == n_samples
            ):
                raise ValueError("Sample count mismatch across provided tensors.")

            self.filepath = None
            self.in_memory = True
            self.hidden_states = hidden_states
            self.target_router_logits = target_router_logits
            self.target_top4_indices = target_top4_indices
            self.valid_mask = valid_mask if valid_mask.dtype == torch.bool else valid_mask.bool()
            self.metadata = metadata or {}
            self._safe_handle = None
            self._length = n_samples

    def _get_handle(self):
        if self._safe_handle is None:
            self._safe_handle = safe_open(self.filepath, framework="pt", device="cpu")
        return self._safe_handle

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: Union[int, slice]) -> Union[DatasetItem, MoECalibrationDataset]:
        if isinstance(idx, slice):
            if not self.in_memory:
                raise RuntimeError("Slice indexing requires in-memory dataset.")
            return MoECalibrationDataset(
                hidden_states=self.hidden_states[idx],
                target_router_logits=self.target_router_logits[idx],
                target_top4_indices=self.target_top4_indices[idx],
                valid_mask=self.valid_mask[idx],
                metadata=self.metadata,
            )

        if self.in_memory:
            return DatasetItem(
                self.hidden_states[idx],
                self.target_router_logits[idx],
                self.target_top4_indices[idx],
                self.valid_mask[idx],
            )
        else:
            h = self._get_handle()
            return DatasetItem(
                h.get_slice("hidden_states")[idx],
                h.get_slice("target_router_logits")[idx],
                h.get_slice("target_top4_indices")[idx],
                h.get_slice("valid_mask")[idx].bool(),
            )

    @property
    def num_samples(self) -> int:
        return self._length

    @property
    def d_model(self) -> int:
        if self.in_memory:
            return self.hidden_states.shape[-1]
        return self._get_handle().get_slice("hidden_states").get_shape()[-1]

    @property
    def num_horizons(self) -> int:
        if self.in_memory:
            return self.target_router_logits.shape[1]
        return self._get_handle().get_slice("target_router_logits").get_shape()[1]

    @property
    def num_deep_layers(self) -> int:
        if self.in_memory:
            return self.target_router_logits.shape[2]
        return self._get_handle().get_slice("target_router_logits").get_shape()[2]

    @property
    def num_experts(self) -> int:
        if self.in_memory:
            return self.target_router_logits.shape[3]
        return self._get_handle().get_slice("target_router_logits").get_shape()[3]

    def filter_valid(self) -> MoECalibrationDataset:
        """Return a new dataset containing only samples where all horizons are valid."""
        if not self.in_memory:
            raise RuntimeError("filter_valid requires an in-memory dataset.")
        all_valid = self.valid_mask.all(dim=-1)
        return MoECalibrationDataset(
            hidden_states=self.hidden_states[all_valid],
            target_router_logits=self.target_router_logits[all_valid],
            target_top4_indices=self.target_top4_indices[all_valid],
            valid_mask=self.valid_mask[all_valid],
            metadata=self.metadata,
        )

    def select_layers(self, layer_indices: Sequence[int]) -> MoECalibrationDataset:
        """Return a new dataset slicing a subset of deep layers."""
        if not self.in_memory:
            raise RuntimeError("select_layers requires an in-memory dataset.")
        indices = torch.tensor(layer_indices, dtype=torch.long)
        return MoECalibrationDataset(
            hidden_states=self.hidden_states,
            target_router_logits=self.target_router_logits[:, :, indices, :],
            target_top4_indices=self.target_top4_indices[:, :, indices, :],
            valid_mask=self.valid_mask,
            metadata=self.metadata,
        )

    def select_horizon(self, horizon_idx: int) -> MoECalibrationDataset:
        """Return a new dataset slicing a single horizon."""
        if not self.in_memory:
            raise RuntimeError("select_horizon requires an in-memory dataset.")
        return MoECalibrationDataset(
            hidden_states=self.hidden_states,
            target_router_logits=self.target_router_logits[:, horizon_idx:horizon_idx+1, :, :],
            target_top4_indices=self.target_top4_indices[:, horizon_idx:horizon_idx+1, :, :],
            valid_mask=self.valid_mask[:, horizon_idx:horizon_idx+1],
            metadata=self.metadata,
        )

    def save(self, filepath: Union[str, Path], metadata: Optional[Dict[str, str]] = None) -> None:
        """Save the dataset to a safetensors file."""
        if not self.in_memory:
            raise RuntimeError("Cannot save a memory-mapped dataset without loading tensors.")
        meta = self.metadata.copy()
        if metadata:
            meta.update(metadata)
        save_dataset_safetensors(
            tensors={
                "hidden_states": self.hidden_states,
                "target_router_logits": self.target_router_logits,
                "target_top4_indices": self.target_top4_indices,
                "valid_mask": self.valid_mask,
            },
            filepath=filepath,
            metadata=meta,
        )

    @classmethod
    def from_safetensors(
        cls,
        filepath: Union[str, Path],
        in_memory: bool = True,
    ) -> MoECalibrationDataset:
        """Factory constructor to load dataset from a safetensors file."""
        return cls(filepath=filepath, in_memory=in_memory)


def build_and_split_calibration_datasets(
    sequences: Sequence[Any],
    output_dir: Optional[Union[str, Path]] = None,
    config: Optional[DatasetSplitConfig] = None,
) -> Tuple[MoECalibrationDataset, MoECalibrationDataset]:
    """
    High-level orchestrator: partitions sequences, aligns multi-horizon targets,
    and constructs/saves train and calibration datasets.

    Args:
        sequences: List of (hidden_states, router_logits) tuples or ExtractionBatch objects.
        output_dir: Optional directory to save train_data.safetensors and calib_data.safetensors.
        config: Optional DatasetSplitConfig instance.

    Returns:
        Tuple of (train_dataset, calib_dataset).
    """
    cfg = config or DatasetSplitConfig()
    num_sequences = len(sequences)

    # 1. Sequence-atomic partition
    train_indices, calib_indices = partition_sequence_indices(
        num_sequences=num_sequences,
        train_ratio=cfg.train_ratio,
        calib_ratio=cfg.calib_ratio,
        shuffle=cfg.shuffle_sequences,
        seed=cfg.seed,
    )

    def process_split_sequences(indices: List[int], split_name: str) -> MoECalibrationDataset:
        aligned_list: List[AlignedSequence] = []
        for idx in indices:
            item = sequences[idx]
            if isinstance(item, ExtractionBatch):
                h_seq = item.single_hidden_state
                r_seq = item.single_router_logits
            elif isinstance(item, (tuple, list)):
                h_seq, r_seq = item[0], item[1]
            else:
                raise TypeError(f"Unsupported sequence item type: {type(item)}")

            aligned = align_sequence_targets(
                hidden_states=h_seq,
                router_logits=r_seq,
                deep_layer_start=cfg.deep_layer_start,
                deep_layer_end=cfg.deep_layer_end,
                horizons=cfg.horizons,
                top_k=cfg.top_k,
                drop_boundary_tokens=cfg.drop_boundary_tokens,
            )
            aligned_list.append(aligned)

        # Concatenate samples across sequences
        cat_hidden = torch.cat([a.hidden_states for a in aligned_list], dim=0)
        cat_logits = torch.cat([a.target_router_logits for a in aligned_list], dim=0)
        cat_top4 = torch.cat([a.target_top4_indices for a in aligned_list], dim=0)
        cat_mask = torch.cat([a.valid_mask for a in aligned_list], dim=0)

        metadata = {
            "split": split_name,
            "num_sequences": str(len(indices)),
            "sequence_indices": ",".join(map(str, indices)),
            "drop_boundary_tokens": str(cfg.drop_boundary_tokens),
            "deep_layer_start": str(cfg.deep_layer_start),
            "deep_layer_end": str(cfg.deep_layer_end),
        }

        dataset = MoECalibrationDataset(
            hidden_states=cat_hidden,
            target_router_logits=cat_logits,
            target_top4_indices=cat_top4,
            valid_mask=cat_mask,
            metadata=metadata,
        )
        return dataset

    train_ds = process_split_sequences(train_indices, "train")
    calib_ds = process_split_sequences(calib_indices, "calib")

    # 2. Optional disk persistence
    if output_dir is not None:
        save_calibration_datasets(train_ds, calib_ds, output_dir)

    return train_ds, calib_ds


def partition_and_align(
    sequences: Sequence[Any],
    config: Optional[DatasetSplitConfig] = None,
) -> Tuple[MoECalibrationDataset, MoECalibrationDataset]:
    """Convenience function to partition and align sequences into train and calib datasets."""
    return build_and_split_calibration_datasets(sequences, output_dir=None, config=config)


def save_calibration_datasets(
    train_dataset: MoECalibrationDataset,
    calib_dataset: MoECalibrationDataset,
    output_dir: Union[str, Path],
) -> Tuple[Path, Path]:
    """Persist train and calibration datasets to safetensors files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    train_path = out_path / "train_data.safetensors"
    calib_path = out_path / "calib_data.safetensors"
    train_dataset.save(train_path)
    calib_dataset.save(calib_path)
    return train_path, calib_path


def load_calibration_dataset(
    filepath: Union[str, Path],
    in_memory: bool = True,
) -> MoECalibrationDataset:
    """Load a calibration dataset from safetensors file."""
    return MoECalibrationDataset.from_safetensors(filepath, in_memory=in_memory)


# Test and architecture aliases
split_dataset = build_and_split_calibration_datasets
SequenceDataset = MoECalibrationDataset
CalibrationDataset = MoECalibrationDataset
