"""End-to-end training loop for the Medusa Speculative Head.

Loads training data from safetensors, instantiates ``MedusaSpeculativeHead``,
trains with AdamW optimiser, logs per-epoch CE/MMCE/total loss, and saves a
checkpoint to ``checkpoints/speculative_head.pt``.

This module supports both file-based and in-memory (tensor-based) data
sources for flexibility in production and testing contexts.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    CHECKPOINTS_DIR,
    DEFAULT_BATCH_SIZE_TRAIN,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MMCE_LAMBDA,
    DEFAULT_MMCE_SIGMA,
    DEFAULT_TRAIN_EPOCHS,
    DEFAULT_WEIGHT_DECAY,
    SPEC_HEAD_CHECKPOINT,
    SPEC_HEAD_INPUT_DIM,
    NUM_DEEP_LAYERS,
    NUM_EXPERTS,
    NUM_HORIZONS,
    TRAIN_DATA_FILE,
    TrainingConfig,
)
from src.models.medusa_head import MedusaSpeculativeHead
from src.training.mmce_loss import combined_loss

logger = logging.getLogger(__name__)


class Trainer:
    """End-to-end trainer for the Medusa Speculative Head.

    Orchestrates the full training loop: data loading, model instantiation,
    AdamW optimisation with CE + λ·MMCE loss, per-epoch logging, and
    checkpoint saving.

    Args:
        config: Optional ``TrainingConfig`` dataclass.  If ``None``, defaults
            from ``src.config`` are used.
        device: Target device string or ``torch.device``.
        train_data_path: Path to the training safetensors file.  If ``None``,
            uses the default ``TRAIN_DATA_FILE``.
        hidden_states: Optional pre-loaded hidden states tensor for in-memory
            training (shape ``(N, input_dim)``).
        target_router_logits: Optional pre-loaded target logits tensor
            (shape ``(N, H, L, E)``).
        valid_mask: Optional pre-loaded validity mask (shape ``(N, H)``).

    Example::

        >>> trainer = Trainer(device="cpu")
        >>> history = trainer.train()  # returns list of loss dicts per epoch
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        train_data_path: Optional[Union[str, Path]] = None,
        hidden_states: Optional[torch.Tensor] = None,
        target_router_logits: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.train_data_path = Path(train_data_path) if train_data_path else TRAIN_DATA_FILE

        # In-memory data (bypasses file loading)
        self._hidden_states = hidden_states
        self._target_router_logits = target_router_logits
        self._valid_mask = valid_mask

        # Will be set in train()
        self.model: Optional[MedusaSpeculativeHead] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.history: List[Dict[str, float]] = []

        logger.info(
            "Trainer initialised: device=%s, epochs=%d, lr=%.1e, "
            "batch_size=%d, mmce_lambda=%.2f, mmce_sigma=%.2f",
            self.device,
            self.config.epochs,
            self.config.learning_rate,
            self.config.batch_size,
            self.config.mmce_lambda,
            self.config.mmce_sigma,
        )

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load training data from safetensors or use in-memory tensors.

        Returns:
            Tuple of (hidden_states, target_router_logits, valid_mask) tensors,
            all on CPU.

        Raises:
            FileNotFoundError: If the safetensors file does not exist and no
                in-memory data was provided.
            ValueError: If loaded tensors have incompatible shapes.
        """
        if self._hidden_states is not None and self._target_router_logits is not None:
            logger.info("Using in-memory training data (%d samples)", self._hidden_states.shape[0])
            hs = self._hidden_states.float()
            trl = self._target_router_logits.float()
            vm = self._valid_mask if self._valid_mask is not None else torch.ones(
                hs.shape[0], self.config.num_horizons, dtype=torch.bool
            )
            return hs, trl, vm

        logger.info("Loading training data from %s", self.train_data_path)
        if not self.train_data_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {self.train_data_path}. "
                f"Run Milestone 1 data generation first, or provide in-memory tensors."
            )

        from src.data.dataset import load_dataset_safetensors
        data = load_dataset_safetensors(self.train_data_path)

        hs = data["hidden_states"].float()
        trl = data["target_router_logits"].float()
        vm = data["valid_mask"]
        if vm.dtype != torch.bool:
            vm = vm.bool()

        logger.info(
            "Loaded training data: hidden_states=%s, target_router_logits=%s, valid_mask=%s",
            list(hs.shape),
            list(trl.shape),
            list(vm.shape),
        )

        # Validate shapes
        if hs.shape[0] != trl.shape[0]:
            raise ValueError(
                f"Sample count mismatch: hidden_states has {hs.shape[0]} "
                f"but target_router_logits has {trl.shape[0]}"
            )
        if hs.dim() != 2:
            raise ValueError(f"hidden_states must be 2-D, got {hs.dim()}-D")
        if trl.dim() != 4:
            raise ValueError(f"target_router_logits must be 4-D, got {trl.dim()}-D")

        return hs, trl, vm

    # ------------------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------------------

    def train(self) -> List[Dict[str, float]]:
        """Execute the full training loop.

        Returns:
            A list of per-epoch loss dictionaries with keys
            ``{'epoch', 'ce_loss', 'mmce_loss', 'total_loss', 'time_s'}``.
        """
        # 1. Load data
        hidden_states, target_logits, valid_mask = self._load_data()
        num_samples = hidden_states.shape[0]

        # Infer architecture dimensions from data
        input_dim = hidden_states.shape[1]
        num_horizons = target_logits.shape[1]
        num_deep_layers = target_logits.shape[2]
        num_experts = target_logits.shape[3]

        logger.info(
            "Training data: %d samples, input_dim=%d, horizons=%d, "
            "deep_layers=%d, experts=%d",
            num_samples, input_dim, num_horizons, num_deep_layers, num_experts,
        )

        # 2. Instantiate model
        self.model = MedusaSpeculativeHead(
            input_dim=input_dim,
            num_deep_layers=num_deep_layers,
            num_experts=num_experts,
            num_horizons=num_horizons,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "Model: %d total parameters (%d trainable)", total_params, trainable_params
        )

        # 3. Create DataLoader
        dataset = TensorDataset(hidden_states, target_logits, valid_mask.float())
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # 4. Optimiser
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # 5. Training loop
        self.history = []
        self.model.train()

        logger.info("Starting training for %d epochs", self.config.epochs)

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            epoch_ce = 0.0
            epoch_mmce = 0.0
            epoch_total = 0.0
            num_batches = 0

            for batch_hs, batch_trl, batch_vm in dataloader:
                batch_hs = batch_hs.to(self.device)
                batch_trl = batch_trl.to(self.device)
                batch_vm = batch_vm.to(self.device).bool()

                # Forward pass
                pred_logits = self.model(batch_hs)  # (B, H, L, E)

                # Compute combined loss
                total, ce, mmce = combined_loss(
                    pred_logits,
                    batch_trl,
                    valid_mask=batch_vm,
                    mmce_lambda=self.config.mmce_lambda,
                    mmce_sigma=self.config.mmce_sigma,
                )

                # Backward pass
                self.optimizer.zero_grad()
                total.backward()
                self.optimizer.step()

                epoch_ce += ce.item()
                epoch_mmce += mmce.item()
                epoch_total += total.item()
                num_batches += 1

            # Average over batches
            avg_ce = epoch_ce / max(num_batches, 1)
            avg_mmce = epoch_mmce / max(num_batches, 1)
            avg_total = epoch_total / max(num_batches, 1)
            elapsed = time.time() - epoch_start

            epoch_record = {
                "epoch": epoch,
                "ce_loss": avg_ce,
                "mmce_loss": avg_mmce,
                "total_loss": avg_total,
                "time_s": elapsed,
            }
            self.history.append(epoch_record)

            logger.info(
                "Epoch %d/%d — CE: %.4f | MMCE: %.4f | Total: %.4f | Time: %.2fs",
                epoch,
                self.config.epochs,
                avg_ce,
                avg_mmce,
                avg_total,
                elapsed,
            )

        # 6. Save checkpoint
        checkpoint_path = self.config.checkpoint_path
        self._save_checkpoint(checkpoint_path)

        logger.info("Training complete. %d epochs recorded.", len(self.history))
        return self.history

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Save the trained model checkpoint.

        Args:
            filepath: Destination path for the ``.pt`` file.
        """
        if self.model is None:
            raise RuntimeError("No model to save — call train() first.")

        self.model.save_checkpoint(str(filepath))
        logger.info("Checkpoint saved to %s", filepath)

    def get_model(self) -> MedusaSpeculativeHead:
        """Return the trained model (after ``train()`` has been called).

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self.model is None:
            raise RuntimeError("Model not available — call train() first.")
        return self.model
