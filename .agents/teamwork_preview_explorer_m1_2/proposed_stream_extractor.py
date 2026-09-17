"""Zero-OOM Streaming Activation and Router Logits Extractor for Qwen MoE.

This module streams a 100k-token corpus through `Qwen/Qwen1.5-MoE-A2.7B` (or synthetic fixture)
with batch size B=1 and sequence length L=1024 (98 sequences total). It harvests intermediate
Layer N hidden states (default Layer 3) and 24-layer router logits, computes native top-4 expert
assignments and probabilities, enforces strict memory hygiene (inference_mode, immediate CPU detach,
periodic MPS cache flush), and provides progress reporting and memory tracking hooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import logging
import math
import os
import time
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import psutil
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =====================================================================
# 1. Data Structures & Value Containers
# =====================================================================

@dataclass
class MemorySnapshot:
    """Point-in-time snapshot of process and device memory utilization."""
    step: int
    timestamp: float
    rss_bytes: int
    vms_bytes: int
    mps_allocated_bytes: int = 0
    mps_driver_bytes: int = 0
    cuda_allocated_bytes: int = 0
    cuda_reserved_bytes: int = 0
    num_python_tensors: int = 0

    @property
    def rss_mb(self) -> float:
        return self.rss_bytes / (1024 * 1024)

    @property
    def mps_allocated_mb(self) -> float:
        return self.mps_allocated_bytes / (1024 * 1024)

    @property
    def mps_driver_mb(self) -> float:
        return self.mps_driver_bytes / (1024 * 1024)

    @property
    def cuda_allocated_mb(self) -> float:
        return self.cuda_allocated_bytes / (1024 * 1024)


@dataclass
class ExtractionBatch:
    """Container holding harvested tensors for a single extracted sequence.
    
    All harvested tensors reside on CPU in FP16/Int64, detached from any autograd graph.
    """
    sequence_idx: int
    input_ids: torch.Tensor          # (B, L) int64 CPU
    hidden_states: torch.Tensor      # (B, L, d_model) float16 CPU
    router_logits: torch.Tensor      # (B, L, num_layers, num_experts) float16 CPU
    top4_indices: torch.Tensor       # (B, L, num_layers, 4) int64 CPU
    top4_probs: torch.Tensor         # (B, L, num_layers, 4) float16 CPU
    seq_len: int
    batch_size: int = 1

    @property
    def single_hidden_state(self) -> torch.Tensor:
        """Return (L, d_model) tensor for B=1 sequences."""
        return self.hidden_states.squeeze(0)

    @property
    def single_router_logits(self) -> torch.Tensor:
        """Return (L, num_layers, num_experts) tensor for B=1 sequences."""
        return self.router_logits.squeeze(0)

    @property
    def single_top4_indices(self) -> torch.Tensor:
        """Return (L, num_layers, 4) tensor for B=1 sequences."""
        return self.top4_indices.squeeze(0)

    @property
    def single_top4_probs(self) -> torch.Tensor:
        """Return (L, num_layers, 4) tensor for B=1 sequences."""
        return self.top4_probs.squeeze(0)


@dataclass
class ExtractorStats:
    """Summary metrics of an extraction run."""
    total_sequences: int
    total_tokens: int
    duration_seconds: float
    throughput_tokens_per_sec: float
    initial_rss_mb: float
    peak_rss_mb: float
    final_rss_mb: float
    net_rss_delta_mb: float
    peak_mps_allocated_mb: float = 0.0
    memory_snapshots: List[MemorySnapshot] = field(default_factory=list)


@dataclass
class StreamExtractorConfig:
    """Configuration options for StreamExtractor."""
    tap_layer: int = 3                      # 1-indexed (index 3 in outputs.hidden_states)
    batch_size: int = 1                     # Strictly B=1 for Zero-OOM guarantee
    seq_len: int = 1024                     # Sequence length L
    target_tokens: int = 100_000            # Total tokens to extract
    cache_flush_interval: int = 10          # Flush cache and GC every N sequences
    top_k: int = 4                          # Number of native top-k experts to extract
    device: Optional[str] = None            # 'mps', 'cpu', 'cuda', or None (auto)
    dtype: torch.dtype = torch.float16      # Offloaded tensor precision
    drop_last: bool = True                  # Drop trailing tokens if < seq_len
    seed: int = 42                          # Random seed for synthetic generation

    @property
    def num_sequences(self) -> int:
        return math.ceil(self.target_tokens / self.seq_len)


# =====================================================================
# 2. Memory Tracking & Hooks System
# =====================================================================

class MemoryTracker:
    """Tracks host RSS and accelerator memory to detect leaks."""

    def __init__(self, device_type: Optional[str] = None):
        self.process = psutil.Process(os.getpid())
        self.device_type = device_type or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.snapshots: List[MemorySnapshot] = []

    def capture(self, step: int) -> MemorySnapshot:
        mem_info = self.process.memory_info()
        mps_alloc = 0
        mps_driver = 0
        cuda_alloc = 0
        cuda_res = 0

        if self.device_type == "mps" and torch.backends.mps.is_available():
            if hasattr(torch.mps, "current_allocated_memory"):
                mps_alloc = torch.mps.current_allocated_memory()
            if hasattr(torch.mps, "driver_allocated_memory"):
                mps_driver = torch.mps.driver_allocated_memory()
        elif self.device_type == "cuda" and torch.cuda.is_available():
            cuda_alloc = torch.cuda.memory_allocated()
            cuda_res = torch.cuda.memory_reserved()

        tensor_count = 0
        for obj in gc.get_objects():
            try:
                if isinstance(obj, torch.Tensor):
                    tensor_count += 1
            except Exception:
                pass

        snap = MemorySnapshot(
            step=step,
            timestamp=time.time(),
            rss_bytes=mem_info.rss,
            vms_bytes=mem_info.vms,
            mps_allocated_bytes=mps_alloc,
            mps_driver_bytes=mps_driver,
            cuda_allocated_bytes=cuda_alloc,
            cuda_reserved_bytes=cuda_res,
            num_python_tensors=tensor_count,
        )
        self.snapshots.append(snap)
        return snap

    def assert_no_leak(self, start_step: int = 20, max_growth_mb: float = 100.0) -> None:
        """Assert that memory growth between start_step and the end is bounded."""
        relevant = [s for s in self.snapshots if s.step >= start_step]
        if len(relevant) < 2:
            return
        delta = relevant[-1].rss_mb - relevant[0].rss_mb
        if delta > max_growth_mb:
            raise AssertionError(
                f"Memory leak detected! RSS grew by {delta:.2f} MB from step {start_step} "
                f"(limit: {max_growth_mb:.2f} MB)."
            )


class ExtractionHook:
    """Base callback hook protocol for stream extraction."""

    def on_extraction_start(self, total_sequences: int) -> None:
        pass

    def on_batch_start(self, seq_idx: int, total_sequences: int) -> None:
        pass

    def on_batch_end(
        self,
        seq_idx: int,
        total_sequences: int,
        batch: ExtractionBatch,
        snapshot: Optional[MemorySnapshot],
    ) -> None:
        pass

    def on_cache_flush(self, seq_idx: int, snapshot: Optional[MemorySnapshot]) -> None:
        pass

    def on_extraction_end(self, stats: ExtractorStats) -> None:
        pass


class LoggingHook(ExtractionHook):
    """Console and logger progress reporter."""

    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval

    def on_extraction_start(self, total_sequences: int) -> None:
        logger.info(f"Starting stream extraction: {total_sequences} sequences...")

    def on_batch_end(
        self,
        seq_idx: int,
        total_sequences: int,
        batch: ExtractionBatch,
        snapshot: Optional[MemorySnapshot],
    ) -> None:
        if (seq_idx + 1) % self.log_interval == 0 or seq_idx == total_sequences - 1:
            rss_str = f"{snapshot.rss_mb:.1f} MB" if snapshot else "N/A"
            mps_str = f", MPS: {snapshot.mps_allocated_mb:.1f} MB" if snapshot and snapshot.mps_allocated_bytes else ""
            logger.info(
                f"Extracted sequence [{seq_idx + 1:3d}/{total_sequences}] | "
                f"Tokens: {(seq_idx + 1) * batch.seq_len:6d} | RSS: {rss_str}{mps_str}"
            )

    def on_extraction_end(self, stats: ExtractorStats) -> None:
        logger.info(
            f"Extraction completed in {stats.duration_seconds:.2f}s "
            f"({stats.throughput_tokens_per_sec:.1f} tokens/s). "
            f"Initial RSS: {stats.initial_rss_mb:.1f} MB, Peak: {stats.peak_rss_mb:.1f} MB, "
            f"Net Delta: {stats.net_rss_delta_mb:+.1f} MB."
        )


# =====================================================================
# 3. Core Stream Extractor Implementation
# =====================================================================

class StreamExtractor:
    """Zero-OOM streaming extractor for Qwen MoE hidden states and router logits."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[StreamExtractorConfig] = None,
        hooks: Optional[Sequence[ExtractionHook]] = None,
    ):
        self.model = model
        self.config = config or StreamExtractorConfig()
        self.hooks = list(hooks) if hooks else [LoggingHook()]

        param = next(model.parameters(), None)
        self.device = torch.device(self.config.device) if self.config.device else (
            param.device if param is not None else torch.device("cpu")
        )
        self.memory_tracker = MemoryTracker(self.device.type)

    def extract_single_sequence(
        self,
        input_ids: torch.Tensor,
        seq_idx: int = 0,
    ) -> ExtractionBatch:
        """Extract Layer N hidden state and 24-layer router logits for one sequence.
        
        Guarantees zero autograd accumulation and immediate CPU offloading in FP16.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        batch_size, seq_len = input_ids.shape

        if input_ids.device != self.device:
            batch_device_ids = input_ids.to(self.device)
        else:
            batch_device_ids = input_ids

        with torch.inference_mode():
            outputs = self.model(
                input_ids=batch_device_ids,
                output_hidden_states=True,
                output_router_logits=True,
                use_cache=False,
            )

            # 1. Harvest Layer N hidden state (outputs.hidden_states[tap_layer])
            if self.config.tap_layer >= len(outputs.hidden_states):
                raise IndexError(
                    f"tap_layer={self.config.tap_layer} out of range for model with "
                    f"{len(outputs.hidden_states) - 1} layers."
                )
            h_N = outputs.hidden_states[self.config.tap_layer].detach().to("cpu", dtype=self.config.dtype)

            # 2. Harvest router logits across all layers
            reshaped_logits = [
                r.view(batch_size, seq_len, -1).detach().to("cpu", dtype=self.config.dtype)
                for r in outputs.router_logits
            ]
            # Stack along layer dimension: shape (B, L, num_layers, num_experts)
            stacked_logits = torch.stack(reshaped_logits, dim=2)

            # 3. Compute native top-k expert assignments and probabilities per layer
            num_experts = stacked_logits.shape[-1]
            k = min(self.config.top_k, num_experts)
            probs = torch.softmax(stacked_logits.float(), dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
            topk_probs = topk_probs.to(self.config.dtype)

            # 4. Explicit deletion of forward pass artifacts
            del outputs
            del reshaped_logits
            del probs
            if batch_device_ids is not input_ids:
                del batch_device_ids

        return ExtractionBatch(
            sequence_idx=seq_idx,
            input_ids=input_ids.detach().to("cpu", dtype=torch.int64),
            hidden_states=h_N,
            router_logits=stacked_logits,
            top4_indices=topk_indices,
            top4_probs=topk_probs,
            seq_len=seq_len,
            batch_size=batch_size,
        )

    def _flush_memory(self) -> None:
        """Hardware-aware memory synchronization and cache flushing."""
        if self.device.type == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    def stream_sequences(
        self,
        sequence_iterator: Iterable[torch.Tensor],
        total_sequences: Optional[int] = None,
    ) -> Iterator[ExtractionBatch]:
        """Stream input sequences through the model, yielding ExtractionBatch objects.
        
        Args:
            sequence_iterator: Iterable yielding (1, L) or (L,) token ID tensors.
            total_sequences: Expected total sequences (for progress hooks).
            
        Yields:
            ExtractionBatch for each processed sequence.
        """
        total = total_sequences or self.config.num_sequences
        for hook in self.hooks:
            hook.on_extraction_start(total)

        start_time = time.time()
        initial_snap = self.memory_tracker.capture(step=0)
        total_tokens = 0

        for seq_idx, seq_tensor in enumerate(sequence_iterator):
            if total_sequences is not None and seq_idx >= total_sequences:
                break

            for hook in self.hooks:
                hook.on_batch_start(seq_idx, total)

            batch = self.extract_single_sequence(seq_tensor, seq_idx=seq_idx)
            total_tokens += batch.seq_len * batch.batch_size

            should_flush = (
                (seq_idx + 1) % self.config.cache_flush_interval == 0
                or (seq_idx + 1) == total
            )
            if should_flush:
                self._flush_memory()
                snap = self.memory_tracker.capture(step=seq_idx + 1)
                for hook in self.hooks:
                    hook.on_cache_flush(seq_idx, snap)
            else:
                snap = None

            for hook in self.hooks:
                hook.on_batch_end(seq_idx, total, batch, snap)

            yield batch

        self._flush_memory()
        final_snap = self.memory_tracker.capture(step=total)
        duration = time.time() - start_time
        all_rss = [s.rss_mb for s in self.memory_tracker.snapshots]
        all_mps = [s.mps_allocated_mb for s in self.memory_tracker.snapshots]

        stats = ExtractorStats(
            total_sequences=total,
            total_tokens=total_tokens,
            duration_seconds=duration,
            throughput_tokens_per_sec=total_tokens / max(duration, 1e-6),
            initial_rss_mb=initial_snap.rss_mb,
            peak_rss_mb=max(all_rss) if all_rss else initial_snap.rss_mb,
            final_rss_mb=final_snap.rss_mb,
            net_rss_delta_mb=final_snap.rss_mb - initial_snap.rss_mb,
            peak_mps_allocated_mb=max(all_mps) if all_mps else 0.0,
            memory_snapshots=self.memory_tracker.snapshots,
        )

        for hook in self.hooks:
            hook.on_extraction_end(stats)

    def stream_token_tensor(
        self,
        token_ids: torch.Tensor,
        seq_len: Optional[int] = None,
        max_sequences: Optional[int] = None,
    ) -> Iterator[ExtractionBatch]:
        """Stream chunks from a contiguous 1D or 2D token tensor."""
        L = seq_len or self.config.seq_len
        if token_ids.dim() == 1:
            total_tokens = token_ids.size(0)
            n_seqs = total_tokens // L
            if max_sequences is not None:
                n_seqs = min(n_seqs, max_sequences)

            def chunk_generator():
                for i in range(n_seqs):
                    yield token_ids[i * L : (i + 1) * L].unsqueeze(0)

            return self.stream_sequences(chunk_generator(), total_sequences=n_seqs)

        elif token_ids.dim() == 2:
            n_seqs, actual_L = token_ids.shape
            if max_sequences is not None:
                n_seqs = min(n_seqs, max_sequences)

            def row_generator():
                for i in range(n_seqs):
                    yield token_ids[i : i + 1]

            return self.stream_sequences(row_generator(), total_sequences=n_seqs)
        else:
            raise ValueError(f"Expected 1D or 2D token_ids tensor, got {token_ids.shape}")

    def stream_synthetic(
        self,
        num_sequences: Optional[int] = None,
        seq_len: Optional[int] = None,
        vocab_size: int = 151936,
        seed: Optional[int] = None,
    ) -> Iterator[ExtractionBatch]:
        """Stream deterministic pseudo-random sequences for testing without corpus download."""
        n_seqs = num_sequences or self.config.num_sequences
        L = seq_len or self.config.seq_len
        rng = torch.Generator().manual_seed(seed or self.config.seed)

        def synthetic_generator():
            for _ in range(n_seqs):
                yield torch.randint(0, vocab_size, (1, L), generator=rng)

        return self.stream_sequences(synthetic_generator(), total_sequences=n_seqs)

    def stream_text_corpus(
        self,
        texts: Iterable[str],
        tokenizer: Any,
        seq_len: Optional[int] = None,
        max_sequences: Optional[int] = None,
    ) -> Iterator[ExtractionBatch]:
        """Stream raw text lines, tokenizing on the fly and yielding fixed-length batches."""
        L = seq_len or self.config.seq_len
        max_seqs = max_sequences or self.config.num_sequences

        def token_buffer_generator():
            buffer: List[int] = []
            seqs_emitted = 0

            for text in texts:
                if not text or not text.strip():
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                buffer.extend(ids)

                while len(buffer) >= L:
                    chunk = buffer[:L]
                    buffer = buffer[L:]
                    yield torch.tensor(chunk, dtype=torch.int64).unsqueeze(0)
                    seqs_emitted += 1
                    if seqs_emitted >= max_seqs:
                        return

        return self.stream_sequences(token_buffer_generator(), total_sequences=max_seqs)


# =====================================================================
# 4. High-Level Convenience Functional APIs
# =====================================================================

def extract_from_model(
    model: nn.Module,
    corpus_tokens: torch.Tensor,
    tap_layer: int = 3,
    seq_len: int = 1024,
    device: Optional[str] = None,
    cache_flush_interval: int = 10,
    hooks: Optional[Sequence[ExtractionHook]] = None,
) -> List[ExtractionBatch]:
    """Extract and collect all batches in memory (suitable for moderate corpus sizes)."""
    cfg = StreamExtractorConfig(
        tap_layer=tap_layer,
        seq_len=seq_len,
        device=device,
        cache_flush_interval=cache_flush_interval,
    )
    extractor = StreamExtractor(model, config=cfg, hooks=hooks)
    return list(extractor.stream_token_tensor(corpus_tokens))
