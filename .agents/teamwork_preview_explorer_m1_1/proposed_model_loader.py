"""Model Loading and Synthetic Test Fixtures for MoE Calibration Pipeline.

This module provides production-grade model and tokenizer loaders for
`Qwen/Qwen1.5-MoE-A2.7B`, automated device and precision resolution (MPS vs CPU),
and a zero-download synthetic MoE model fixture (`get_synthetic_model`) for fast unit tests.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2MoeConfig,
    Qwen2MoeForCausalLM,
)

# Relative or absolute config imports supported
try:
    from src.config import (
        MODEL_NAME,
        SYNTHETIC_HIDDEN_SIZE,
        SYNTHETIC_INTERMEDIATE_SIZE,
        SYNTHETIC_MOE_INTERMEDIATE_SIZE,
        SYNTHETIC_NUM_ATTENTION_HEADS,
        SYNTHETIC_NUM_EXPERTS,
        SYNTHETIC_NUM_EXPERTS_PER_TOK,
        SYNTHETIC_NUM_HIDDEN_LAYERS,
        SYNTHETIC_NUM_KEY_VALUE_HEADS,
        SYNTHETIC_SHARED_EXPERT_INTERMEDIATE_SIZE,
        SYNTHETIC_VOCAB_SIZE,
        resolve_device,
        resolve_dtype,
    )
except ImportError:
    from proposed_config import (
        MODEL_NAME,
        SYNTHETIC_HIDDEN_SIZE,
        SYNTHETIC_INTERMEDIATE_SIZE,
        SYNTHETIC_MOE_INTERMEDIATE_SIZE,
        SYNTHETIC_NUM_ATTENTION_HEADS,
        SYNTHETIC_NUM_EXPERTS,
        SYNTHETIC_NUM_EXPERTS_PER_TOK,
        SYNTHETIC_NUM_HIDDEN_LAYERS,
        SYNTHETIC_NUM_KEY_VALUE_HEADS,
        SYNTHETIC_SHARED_EXPERT_INTERMEDIATE_SIZE,
        SYNTHETIC_VOCAB_SIZE,
        resolve_device,
        resolve_dtype,
    )


# =====================================================================
# 1. Real Qwen1.5-MoE Model & Tokenizer Loaders
# =====================================================================
def load_model(
    model_name_or_path: str = MODEL_NAME,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    output_hidden_states: bool = True,
    output_router_logits: bool = True,
    use_cache: bool = False,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
) -> PreTrainedModel:
    """Load Qwen MoE model with strict device/dtype handling and extraction hooks.

    Args:
        model_name_or_path: HuggingFace model hub ID or local path.
        device: Target device ('mps', 'cpu', 'cuda', or None for auto-detect).
        dtype: Desired precision ('float16', 'float32', 'bfloat16', or None).
               Note: on MPS backend, bfloat16 will raise ValueError.
        output_hidden_states: Enable capturing hidden states across all layers.
        output_router_logits: Enable capturing raw gating router logits across all layers.
        use_cache: Disable KV caching for zero-OOM memory-efficient streaming.
        trust_remote_code: Allow executing custom code if required by model.
        local_files_only: Only search local cache without downloading.

    Returns:
        Loaded model in eval mode on target device with resolved dtype.
    """
    target_device = resolve_device(device)
    target_dtype = resolve_dtype(target_device, dtype)

    # Load pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=target_dtype,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        use_cache=use_cache,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )

    # Move to target device and set to eval mode
    model = model.to(target_device)
    model.eval()

    # Enforce forward output attributes on configuration explicitly
    model.config.output_hidden_states = output_hidden_states
    model.config.output_router_logits = output_router_logits
    model.config.use_cache = use_cache

    return model


def load_tokenizer(
    model_name_or_path: str = MODEL_NAME,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
) -> PreTrainedTokenizerBase:
    """Load pretrained tokenizer with pad token initialization.

    Args:
        model_name_or_path: HuggingFace model hub ID or local path.
        trust_remote_code: Allow remote code execution.
        local_files_only: Only search local cache.

    Returns:
        HuggingFace tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(
    model_name_or_path: str = MODEL_NAME,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    output_hidden_states: bool = True,
    output_router_logits: bool = True,
    use_cache: bool = False,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load both model and tokenizer with matching configuration."""
    tokenizer = load_tokenizer(
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model = load_model(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        use_cache=use_cache,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    return model, tokenizer


# =====================================================================
# 2. Lightweight Synthetic Tokenizer for Unit Testing & CI
# =====================================================================
class SyntheticTokenizer:
    """Fast, offline, deterministic mock tokenizer for testing without network/cache."""

    def __init__(self, vocab_size: int = SYNTHETIC_VOCAB_SIZE, model_max_length: int = 2048):
        self.vocab_size: int = vocab_size
        self.model_max_length: int = model_max_length
        self.pad_token_id: int = 0
        self.eos_token_id: int = 1
        self.bos_token_id: int = 2

    def __len__(self) -> int:
        return self.vocab_size

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation: bool = False,
        **kwargs: Any,
    ) -> Union[List[int], torch.Tensor]:
        tokens: List[int] = [
            (abs(hash(w)) % (self.vocab_size - 10)) + 10 for w in text.split()
        ]
        if not tokens:
            tokens = [self.bos_token_id]
        if max_length and truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        if return_tensors == "pt":
            return torch.tensor([tokens], dtype=torch.long)
        return tokens

    def __call__(
        self,
        text: Union[str, List[str]],
        return_tensors: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if isinstance(text, str):
            token_list = self.encode(text, return_tensors=None, max_length=max_length, truncation=truncation)
            if return_tensors == "pt":
                input_ids = torch.tensor([token_list], dtype=torch.long)
                attention_mask = torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attention_mask}
            return {"input_ids": token_list, "attention_mask": [1] * len(token_list)}
        else:
            encoded_batch = [
                self.encode(t, return_tensors=None, max_length=max_length, truncation=truncation)
                for t in text
            ]
            max_len = max(len(b) for b in encoded_batch)
            padded = [b + [self.pad_token_id] * (max_len - len(b)) for b in encoded_batch]
            masks = [[1] * len(b) + [0] * (max_len - len(b)) for b in encoded_batch]
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(padded, dtype=torch.long),
                    "attention_mask": torch.tensor(masks, dtype=torch.long),
                }
            return {"input_ids": padded, "attention_mask": masks}

    def decode(self, token_ids: Union[List[int], torch.Tensor], **kwargs: Any) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.squeeze().tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return " ".join(f"w_{t}" for t in token_ids)


# =====================================================================
# 3. Fast Zero-Download Synthetic MoE Model Factory
# =====================================================================
def get_synthetic_config(
    vocab_size: int = SYNTHETIC_VOCAB_SIZE,
    hidden_size: int = SYNTHETIC_HIDDEN_SIZE,
    intermediate_size: int = SYNTHETIC_INTERMEDIATE_SIZE,
    num_hidden_layers: int = SYNTHETIC_NUM_HIDDEN_LAYERS,
    num_attention_heads: int = SYNTHETIC_NUM_ATTENTION_HEADS,
    num_key_value_heads: int = SYNTHETIC_NUM_KEY_VALUE_HEADS,
    num_experts: int = SYNTHETIC_NUM_EXPERTS,
    num_experts_per_tok: int = SYNTHETIC_NUM_EXPERTS_PER_TOK,
    moe_intermediate_size: int = SYNTHETIC_MOE_INTERMEDIATE_SIZE,
    shared_expert_intermediate_size: int = SYNTHETIC_SHARED_EXPERT_INTERMEDIATE_SIZE,
    output_hidden_states: bool = True,
    output_router_logits: bool = True,
    use_cache: bool = False,
) -> Qwen2MoeConfig:
    """Create a scaled-down Qwen2MoeConfig with exact architectural parity.

    Default specifications:
    - hidden_size: 64
    - num_hidden_layers: 6
    - num_experts: 16
    - num_experts_per_tok: 4
    - Total parameters: ~1.56 Million (< 6 MB memory footprint).
    """
    return Qwen2MoeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        decoder_sparse_step=1,
        norm_topk_prob=False,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        use_cache=use_cache,
    )


def get_synthetic_model(
    vocab_size: int = SYNTHETIC_VOCAB_SIZE,
    hidden_size: int = SYNTHETIC_HIDDEN_SIZE,
    intermediate_size: int = SYNTHETIC_INTERMEDIATE_SIZE,
    num_hidden_layers: int = SYNTHETIC_NUM_HIDDEN_LAYERS,
    num_attention_heads: int = SYNTHETIC_NUM_ATTENTION_HEADS,
    num_key_value_heads: int = SYNTHETIC_NUM_KEY_VALUE_HEADS,
    num_experts: int = SYNTHETIC_NUM_EXPERTS,
    num_experts_per_tok: int = SYNTHETIC_NUM_EXPERTS_PER_TOK,
    moe_intermediate_size: int = SYNTHETIC_MOE_INTERMEDIATE_SIZE,
    shared_expert_intermediate_size: int = SYNTHETIC_SHARED_EXPERT_INTERMEDIATE_SIZE,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[Union[str, torch.dtype]] = None,
    output_hidden_states: bool = True,
    output_router_logits: bool = True,
    use_cache: bool = False,
) -> Qwen2MoeForCausalLM:
    """Instantiate a zero-download synthetic Qwen2Moe model for rapid testing.

    Args:
        device: Target device ('cpu', 'mps', 'cuda'). Defaults to 'cpu' for unit tests.
        dtype: Precision. Auto-selected based on device (FP32 on CPU, FP16 on MPS).
        output_hidden_states: Configure forward pass to output hidden states tuple.
        output_router_logits: Configure forward pass to output router logits tuple.
        use_cache: Set KV caching flag.

    Returns:
        Qwen2MoeForCausalLM in eval mode on target device.
    """
    target_device = resolve_device(device)
    target_dtype = resolve_dtype(target_device, dtype)

    config = get_synthetic_config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        use_cache=use_cache,
    )

    model = Qwen2MoeForCausalLM(config)
    model = model.to(device=target_device, dtype=target_dtype)
    model.eval()

    return model


def get_synthetic_tokenizer(
    vocab_size: int = SYNTHETIC_VOCAB_SIZE,
) -> SyntheticTokenizer:
    """Return a fast synthetic mock tokenizer for unit tests."""
    return SyntheticTokenizer(vocab_size=vocab_size)


def get_synthetic_model_and_tokenizer(
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[Union[str, torch.dtype]] = None,
    **kwargs: Any,
) -> Tuple[Qwen2MoeForCausalLM, SyntheticTokenizer]:
    """Return both synthetic model and synthetic tokenizer for end-to-end testing."""
    model = get_synthetic_model(device=device, dtype=dtype, **kwargs)
    tokenizer = get_synthetic_tokenizer(vocab_size=model.config.vocab_size)
    return model, tokenizer
