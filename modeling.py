import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union, Callable
from transformers.utils import logging
from transformers.processing_utils import Unpack
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from .compression import FocusKV, FocusKVV2, HeadwiseKV

try:
    from flash_attn import flash_attn_varlen_func
    _HAS_FLASH_VARLEN = True
except ImportError:
    _HAS_FLASH_VARLEN = False

KV_COMPRESSION_MAP = {
    "focuskv": FocusKV,
    "focuskv_v1": FocusKV,
    "focuskv_v2": FocusKVV2,
}

logger = logging.get_logger(__name__)


def _replace_kv_cache(cache, layer_idx, key_states, value_states):
    """Replace the KV cache at *layer_idx* in-place.

    Compatible with:
      - New layered DynamicCache  (cache.layers[i].keys / .values)
      - Old flat-list DynamicCache (cache.key_cache[i] / .value_cache[i])
    """
    if hasattr(cache, "layers") and layer_idx < len(cache.layers):
        lc = cache.layers[layer_idx]
        lc.keys = key_states
        lc.values = value_states
    elif hasattr(cache, "key_cache"):
        cache.key_cache[layer_idx] = key_states
        cache.value_cache[layer_idx] = value_states
    else:
        raise AttributeError(
            f"Cannot replace KV cache: {type(cache).__name__} has neither "
            "'layers' nor 'key_cache' attribute"
        )


# ═══════════ Focus-KV padding mask helpers ═══════════

def _get_kv_pad_lengths(past_key_values, layer_idx):
    """Retrieve per-KV-head padding lengths for a layer, or None."""
    if hasattr(past_key_values, "kv_pad_lengths"):
        return past_key_values.kv_pad_lengths.get(layer_idx, None)
    return None


def _store_kv_pad_lengths(kv_cluster, past_key_values, layer_idx):
    """Persist new padding lengths from compression to the cache object.

    Only acts when the compression method exposes ``_pad_lengths``
    When the active Focus-KV variant has no rectangular padding, this is a
    no-op.
    """
    pad_lengths = getattr(kv_cluster, "_pad_lengths", None)
    if pad_lengths is None:
        return
    if not hasattr(past_key_values, "kv_pad_lengths"):
        past_key_values.kv_pad_lengths = {}
    past_key_values.kv_pad_lengths[layer_idx] = pad_lengths.clone()
    kv_cluster._pad_lengths = None


def _build_kv_padding_mask(pad_lengths, kv_len, num_kv_groups, device, dtype):
    """Build an additive attention mask that blocks zero-padded KV positions.

    Padding is prepended (positions 0 .. pad_len-1 per head).

    Returns:
        (1, num_query_heads, 1, kv_len) additive mask with 0.0 for valid
        positions and ``dtype.min`` for padding.  Avoid a host-side ``.item()``
        check here: that would synchronize CUDA once per layer and decode step.
    """
    positions = torch.arange(kv_len, device=device)
    # (num_kv_heads, kv_len): True = valid token, False = padding
    valid = positions.unsqueeze(0) >= pad_lengths.unsqueeze(1).to(device)

    if num_kv_groups > 1:
        valid = valid.unsqueeze(1).expand(-1, num_kv_groups, -1)
        valid = valid.reshape(-1, kv_len)

    num_q_heads = valid.shape[0]
    mask = torch.zeros(1, num_q_heads, 1, kv_len, dtype=dtype, device=device)
    mask.masked_fill_(~valid.unsqueeze(0).unsqueeze(2), torch.finfo(dtype).min)
    return mask


def _align_attention_mask_to_kv_len(attention_mask, kv_len):
    """Match HF's shared attention mask to this layer's compressed KV length."""
    if attention_mask is None or attention_mask.shape[-1] == kv_len:
        return attention_mask

    cur_len = attention_mask.shape[-1]
    if cur_len > kv_len:
        return attention_mask[..., -kv_len:]

    pad_shape = (*attention_mask.shape[:-1], kv_len - cur_len)
    pad = torch.zeros(
        pad_shape,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([pad, attention_mask], dim=-1)


def _infer_batch_left_padding(attention_mask):
    """Infer per-example leading prompt padding from a 2D/4D HF mask.

    The value is only consumed before the first head-wise cache rewrite. After
    that rewrite every example has exactly the configured budget per KV head.
    """
    if attention_mask is None:
        return None
    if attention_mask.ndim == 2:
        valid = attention_mask.to(dtype=torch.bool)
    elif attention_mask.ndim == 4:
        last_query_mask = attention_mask[:, 0, -1, :]
        if last_query_mask.dtype == torch.bool:
            valid = last_query_mask
        else:
            # HF additive masks use 0 for visible and a large negative value
            # for masked positions.
            valid = last_query_mask >= 0
    else:
        return None

    # cumprod remains one only across the leading run of invalid positions.
    return (~valid).to(torch.long).cumprod(dim=-1).sum(dim=-1)


def _apply_kv_padding_mask(past_kv, layer_idx, key_states, query_states,
                           attention_mask, num_kv_groups,
                           align_to_kv_len=False):
    """Build and merge padding mask into *attention_mask* (in-place safe).

    Returns the (possibly updated) attention_mask.
    """
    if align_to_kv_len:
        attention_mask = _align_attention_mask_to_kv_len(
            attention_mask, key_states.shape[2],
        )
    pad_lengths = _get_kv_pad_lengths(past_kv, layer_idx)
    if pad_lengths is None:
        return attention_mask
    kv_pad_mask = _build_kv_padding_mask(
        pad_lengths, key_states.shape[2],
        num_kv_groups, key_states.device, query_states.dtype,
    )
    if kv_pad_mask is None:
        return attention_mask
    if attention_mask is not None:
        return attention_mask + kv_pad_mask
    return kv_pad_mask


# ═══════════ Focus-KV v1 head-wise cache helpers ═══════════

def _has_headwise_kv(cache, layer_idx):
    return hasattr(cache, "_focus_hkv") and layer_idx in cache._focus_hkv


def _get_headwise_kv(cache, layer_idx):
    return cache._focus_hkv[layer_idx]


def _store_headwise_kv(cache, layer_idx, hkv: HeadwiseKV):
    """Store HeadwiseKV and free the standard cache tensor for this layer."""
    if not hasattr(cache, "_focus_hkv"):
        cache._focus_hkv = {}
        cache._focus_seq_lens = {}
        _patch_cache_seq_length(cache)

    cache._focus_hkv[layer_idx] = hkv
    cache._focus_seq_lens[layer_idx] = hkv.max_seq_len()

    if hasattr(cache, "layers") and layer_idx < len(cache.layers):
        layer = cache.layers[layer_idx]
        reference = layer.keys
        layer.keys = torch.zeros(
            1, 1, 1, 1, device=reference.device, dtype=reference.dtype,
        )
        layer.values = torch.zeros_like(layer.keys)
    elif hasattr(cache, "key_cache") and layer_idx < len(cache.key_cache):
        dev = cache.key_cache[layer_idx].device
        dt = cache.key_cache[layer_idx].dtype
        cache.key_cache[layer_idx] = torch.zeros(1, 1, 1, 1, device=dev, dtype=dt)
        cache.value_cache[layer_idx] = torch.zeros(1, 1, 1, 1, device=dev, dtype=dt)


def _append_headwise_kv(cache, layer_idx, key_states, value_states):
    """Append new token(s) to an existing HeadwiseKV entry."""
    cache._focus_hkv[layer_idx].append(key_states, value_states)
    cache._focus_seq_lens[layer_idx] = cache._focus_hkv[layer_idx].max_seq_len()
    if layer_idx == 0 and hasattr(cache, "_seen_tokens"):
        cache._seen_tokens += key_states.shape[-2]


def _patch_cache_seq_length(cache):
    """Monkey-patch ``get_seq_length`` so position tracking stays correct.

    Uses ``weakref`` to avoid circular references that prevent timely
    garbage collection of the cache object (and its GPU tensors).
    """
    import weakref

    _cls_get_seq = type(cache).get_seq_length
    _ref = weakref.ref(cache)

    def _patched(layer_idx=0):
        c = _ref()
        if c is not None and hasattr(c, "_focus_seq_lens") and layer_idx in c._focus_seq_lens:
            return c._focus_seq_lens[layer_idx]
        if c is not None:
            return _cls_get_seq(c, layer_idx)
        return 0

    cache.get_seq_length = _patched

    if hasattr(cache, "get_usable_length"):
        _cls_get_usable = type(cache).get_usable_length if hasattr(type(cache), "get_usable_length") else None

        def _patched_usable(new_seq_length, layer_idx=0):
            c = _ref()
            if c is not None and hasattr(c, "_focus_seq_lens") and layer_idx in c._focus_seq_lens:
                return c._focus_seq_lens[layer_idx]
            if c is not None and _cls_get_usable is not None:
                return _cls_get_usable(c, new_seq_length, layer_idx)
            return new_seq_length

        cache.get_usable_length = _patched_usable


def _headwise_eager_attention(query_states, hkv, num_kv_groups, scaling):
    """Per-KV-head attention loop (decoding only, q_len typically 1)."""
    num_kv_heads = len(hkv.keys)
    outputs = []
    for h in range(num_kv_heads):
        q_start = h * num_kv_groups
        q_end = q_start + num_kv_groups
        q_h = query_states[:, q_start:q_end]        # [bsz, group, q_len, dim]
        k_h = hkv.keys[h]                            # [bsz, 1, kv_len_h, dim]
        v_h = hkv.values[h]                           # [bsz, 1, kv_len_h, dim]

        scores = torch.matmul(q_h, k_h.transpose(-1, -2)) * scaling
        attn_w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q_h.dtype)
        outputs.append(torch.matmul(attn_w, v_h))

    return torch.cat(outputs, dim=1), None


def _headwise_varlen_attention(query_states, hkv, scaling):
    """Varlen FlashAttention for FlatHeadwiseKV (decode only, bsz=1).

    Replaces the Python for-loop in ``_headwise_eager_attention`` with a
    single ``flash_attn_varlen_func`` kernel call.  GQA is handled by
    setting ``nheads_q = group_size`` and ``nheads_k = 1``.
    """
    bsz, num_q_heads, q_len, head_dim = query_states.shape
    assert bsz == 1 and q_len == 1, (
        f"varlen attention requires bsz=1, q_len=1; got {bsz}, {q_len}"
    )

    # Q: [1, num_q_heads, 1, dim] → [num_kv_heads, group_size, dim]
    q_flat = query_states[0, :, 0, :].view(
        hkv.num_kv_heads, hkv.group_size, head_dim,
    )

    # K, V: per-head lists → [total_k, 1, dim]
    k_flat, v_flat = hkv.get_flat_kv()
    k_flat = k_flat.unsqueeze(1)
    v_flat = v_flat.unsqueeze(1)

    attn_output = flash_attn_varlen_func(
        q_flat, k_flat, v_flat,
        hkv.cu_qlen, hkv.cu_klen,
        max_seqlen_q=1,
        max_seqlen_k=hkv.max_seqlen_k,
        softmax_scale=scaling,
        causal=False,
    )
    # [num_kv_heads, group_size, dim] → [1, num_q_heads, 1, dim]
    attn_output = attn_output.view(1, num_q_heads, 1, head_dim)
    return attn_output, None


def LlamaAttention_init(
    self, config: LlamaConfig, layer_idx: int, compression_config: dict
):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True

    self.q_proj = nn.Linear(
        config.hidden_size,
        config.num_attention_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.k_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.v_proj = nn.Linear(
        config.hidden_size,
        config.num_key_value_heads * self.head_dim,
        bias=config.attention_bias,
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim,
        config.hidden_size,
        bias=config.attention_bias,
    )

    # =============== New logic start ===============
    self.config.update(compression_config)
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        **compression_config["method_config"]
    )
    # =============== New logic end =================

def LlamaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        _rectangular_cache = getattr(
            self.kv_cluster, "_uses_rectangular_cache", False,
        )
        _existing_pad_lengths = _get_kv_pad_lengths(
            past_key_value, self.layer_idx,
        )
        _batch_pad_lengths = _infer_batch_left_padding(attention_mask)

        # =============== Enable Query Cache ============
        if not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}

        if self.layer_idx not in past_key_value.query_cache:
            # prefill stage
            bsz, n_heads, _, head_dim = query_states.shape
            past_key_value.query_cache[self.layer_idx] = torch.empty(
                bsz, n_heads, 0, head_dim
            )
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        else:
            # Add current query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]

            # Keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end =========

        # =============== decoding-time compression start ===============
        cached_queries = past_key_value.query_cache[self.layer_idx]
        if self.config.compression is None:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,
                value_states,
                layer_idx=self.layer_idx,
                pad_lengths=_existing_pad_lengths,
                batch_pad_lengths=_batch_pad_lengths,
            )

            if self.config.update_kv is True:
                past_key_value.update(
                    key_states_compress,
                    value_states_compress,
                    self.layer_idx,
                    cache_kwargs,
                )
                if _rectangular_cache:
                    key_states = key_states_compress
                    value_states = value_states_compress
                    _store_kv_pad_lengths(
                        self.kv_cluster, past_key_value, self.layer_idx,
                    )
            else:
                past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )

        elif self.config.compression is True:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,
                value_states,
                layer_idx=self.layer_idx,
                pad_lengths=_existing_pad_lengths,
                batch_pad_lengths=_batch_pad_lengths,
            )

            if self.config.update_kv is True:
                _replace_kv_cache(past_key_value, self.layer_idx,
                                  key_states_compress, value_states_compress)
                if _rectangular_cache:
                    key_states = key_states_compress
                    value_states = value_states_compress
                    _store_kv_pad_lengths(
                        self.kv_cluster, past_key_value, self.layer_idx,
                    )
        else:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        # =============== decoding-time compression end ===============

    if past_key_value is not None:
        attention_mask = _apply_kv_padding_mask(
            past_key_value, self.layer_idx, key_states, query_states,
            attention_mask, self.num_key_value_groups,
            align_to_kv_len=_rectangular_cache,
        )

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    if past_key_value is not None:
        _store_kv_pad_lengths(self.kv_cluster, past_key_value, self.layer_idx)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def Qwen2Attention_init(
    self, config: Qwen2Config, layer_idx: int, compression_config: dict
):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True
    self.q_proj = nn.Linear(
        config.hidden_size, config.num_attention_heads * self.head_dim, bias=True
    )
    self.k_proj = nn.Linear(
        config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
    )
    self.v_proj = nn.Linear(
        config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True
    )
    self.o_proj = nn.Linear(
        config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
    )

    # =============== New logic start ===============
    self.config.update(compression_config)
    self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
        **compression_config["method_config"]
    )
    # =============== New logic end =================

def Qwen2Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        _rectangular_cache = getattr(
            self.kv_cluster, "_uses_rectangular_cache", False,
        )
        _existing_pad_lengths = _get_kv_pad_lengths(
            past_key_value, self.layer_idx,
        )
        _batch_pad_lengths = _infer_batch_left_padding(attention_mask)

        # =============== Enable Query Cache ============
        if not hasattr(past_key_value, "query_cache"):
            past_key_value.query_cache = {}

        if self.layer_idx not in past_key_value.query_cache:
            # prefill stage
            bsz, n_heads, _, head_dim = query_states.shape
            past_key_value.query_cache[self.layer_idx] = torch.empty(
                bsz, n_heads, 0, head_dim
            )
            past_key_value.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        else:
            # Add current query to cache
            past_key_value.query_cache[self.layer_idx] = torch.cat(
                (past_key_value.query_cache[self.layer_idx], query_states), dim=2
            )  # [batch, n_q_heads, seq_len, head_dim]

            # Keep only window_size most recent queries
            window_size = self.config.method_config["window_size"]
            if past_key_value.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_value.query_cache[self.layer_idx] = past_key_value.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end ===============

        # =============== decoding-time compression start ===============
        cached_queries = past_key_value.query_cache[self.layer_idx]
        if self.config.compression is None:
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,
                value_states,
                layer_idx=self.layer_idx,
                pad_lengths=_existing_pad_lengths,
                batch_pad_lengths=_batch_pad_lengths,
            )

            if self.config.update_kv is True:
                past_key_value.update(
                    key_states_compress,
                    value_states_compress,
                    self.layer_idx,
                    cache_kwargs,
                )
                if _rectangular_cache:
                    key_states = key_states_compress
                    value_states = value_states_compress
                    _store_kv_pad_lengths(
                        self.kv_cluster, past_key_value, self.layer_idx,
                    )
            else:
                past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    cache_kwargs,
                )

        elif self.config.compression is True:
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                cached_queries,
                value_states,
                layer_idx=self.layer_idx,
                pad_lengths=_existing_pad_lengths,
                batch_pad_lengths=_batch_pad_lengths,
            )
            if self.config.update_kv is True:
                _replace_kv_cache(past_key_value, self.layer_idx,
                                  key_states_compress, value_states_compress)
                if _rectangular_cache:
                    key_states = key_states_compress
                    value_states = value_states_compress
                    _store_kv_pad_lengths(
                        self.kv_cluster, past_key_value, self.layer_idx,
                    )
        else:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )
        # =============== decoding-time compression end ===============

    if past_key_value is not None:
        attention_mask = _apply_kv_padding_mask(
            past_key_value, self.layer_idx, key_states, query_states,
            attention_mask, self.num_key_value_groups,
            align_to_kv_len=_rectangular_cache,
        )

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get(
            "output_attentions", False
        ):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    if past_key_value is not None:
        _store_kv_pad_lengths(self.kv_cluster, past_key_value, self.layer_idx)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def Qwen3Attention_init(
    self, config: Qwen3Config, layer_idx: int, compression_config: dict
):
        nn.Module.__init__(self)
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

        # =============== New logic start ===============
        self.config.update(compression_config)
        self.kv_cluster = KV_COMPRESSION_MAP[compression_config["method"]](
            **compression_config["method_config"]
        )
        # =============== New logic end =================

def Qwen3Attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # Back-compat: transformers <4.49 passed this as `past_key_values` (plural).
    # 4.49+ uses `past_key_value` (singular) at the Attention level.  Accept both
    # so the monkey-patch works across versions instead of silently skipping the
    # entire `if past_key_values is not None:` block (which causes complete loss
    # of KV history and degenerate `::::` output from token 1).
    # Use `is None` not `or`, because an empty DynamicCache is falsy at prefill.
    if past_key_value is None:
        past_key_value = kwargs.pop("past_key_values", None)
    past_key_values = past_key_value

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # =============== Enable Query Cache ============
        if not hasattr(past_key_values, "query_cache"):
            past_key_values.query_cache = {}

        if self.layer_idx not in past_key_values.query_cache:
            bsz, n_heads, _, head_dim = query_states.shape
            past_key_values.query_cache[self.layer_idx] = torch.empty(
                bsz, n_heads, 0, head_dim
            )
            past_key_values.query_cache[self.layer_idx] = query_states[
                :, :, -self.config.method_config["window_size"] :, :
            ]
        else:
            past_key_values.query_cache[self.layer_idx] = torch.cat(
                (past_key_values.query_cache[self.layer_idx], query_states), dim=2
            )

            window_size = self.config.method_config["window_size"]
            if past_key_values.query_cache[self.layer_idx].shape[-2] > window_size:
                past_key_values.query_cache[self.layer_idx] = past_key_values.query_cache[
                    self.layer_idx
                ][:, :, -window_size:, :]
        # =============== Enable Query Cache end =========

        # =============== decoding-time compression start ===============
        cached_queries = past_key_values.query_cache[self.layer_idx]
        _headwise = getattr(self.kv_cluster, "_uses_headwise_cache", False)
        _rectangular_cache = getattr(
            self.kv_cluster, "_uses_rectangular_cache", False,
        )
        _existing_pad_lengths = _get_kv_pad_lengths(
            past_key_values, self.layer_idx,
        )
        _batch_pad_lengths = _infer_batch_left_padding(attention_mask)
        _hkv_exists = _headwise and _has_headwise_kv(past_key_values, self.layer_idx)

        if _hkv_exists:
            # ---- v1 head-wise path: manage KV outside standard cache ----
            _append_headwise_kv(past_key_values, self.layer_idx,
                                key_states, value_states)
            if self.config.compression is True:
                hkv = _get_headwise_kv(past_key_values, self.layer_idx)
                result = self.kv_cluster.update_kv(
                    None, cached_queries, None,
                    layer_idx=self.layer_idx, headwise_kv=hkv,
                )
                _store_headwise_kv(past_key_values, self.layer_idx, result)

        elif self.config.compression is None:
            result = self.kv_cluster.update_kv(
                key_states, cached_queries, value_states,
                layer_idx=self.layer_idx,
                pad_lengths=_existing_pad_lengths,
                batch_pad_lengths=_batch_pad_lengths,
            )
            if _headwise and isinstance(result, HeadwiseKV):
                past_key_values.update(key_states, value_states, self.layer_idx)
                # Store after DynamicCache has created/updated the layer so the
                # now-redundant rectangular prompt tensor can be released.
                _store_headwise_kv(past_key_values, self.layer_idx, result)
            else:
                k_c, v_c = result
                past_key_values.update(k_c, v_c, self.layer_idx)
                if _rectangular_cache:
                    key_states, value_states = k_c, v_c
                    _store_kv_pad_lengths(
                        self.kv_cluster, past_key_values, self.layer_idx,
                    )

        elif self.config.compression is True:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx,
            )
            result = self.kv_cluster.update_kv(
                key_states, cached_queries, value_states,
                layer_idx=self.layer_idx,
                pad_lengths=_existing_pad_lengths,
                batch_pad_lengths=_batch_pad_lengths,
            )
            if _headwise and isinstance(result, HeadwiseKV):
                _store_headwise_kv(past_key_values, self.layer_idx, result)
            else:
                _replace_kv_cache(past_key_values, self.layer_idx,
                                  result[0], result[1])
                if _rectangular_cache:
                    key_states, value_states = result
                    _store_kv_pad_lengths(
                        self.kv_cluster, past_key_values, self.layer_idx,
                    )
        else:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx,
            )
        # =============== decoding-time compression end ===============

    # =============== attention computation ===============
    _use_headwise_attn = (
        past_key_values is not None
        and _has_headwise_kv(past_key_values, self.layer_idx)
    )

    if _use_headwise_attn:
        hkv = _get_headwise_kv(past_key_values, self.layer_idx)
        if _HAS_FLASH_VARLEN and hasattr(hkv, 'get_flat_kv'):
            attn_output, attn_weights = _headwise_varlen_attention(
                query_states, hkv, self.scaling,
            )
        else:
            attn_output, attn_weights = _headwise_eager_attention(
                query_states, hkv, self.num_key_value_groups, self.scaling,
            )
    else:
        if past_key_values is not None:
            attention_mask = _apply_kv_padding_mask(
                past_key_values, self.layer_idx, key_states, query_states,
                attention_mask, self.num_key_value_groups,
                align_to_kv_len=_rectangular_cache,
            )

        if hasattr(ALL_ATTENTION_FUNCTIONS, "get_interface"):
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS.get(
                self.config._attn_implementation, eager_attention_forward
            )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

    if past_key_values is not None and not _use_headwise_attn:
        _store_kv_pad_lengths(self.kv_cluster, past_key_values, self.layer_idx)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

def CausalLM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # sample-level statistics — detect prefill (first forward of a new sample).
    # New DynamicCache pre-creates layers so len() > 0 even when empty;
    # fall back to checking get_seq_length().
    _cache_empty = (
        past_key_values is None
        or (hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0)
        or (not hasattr(past_key_values, "get_seq_length") and len(past_key_values) == 0)
    )
    input_length = (
        input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
    )
    if _cache_empty:
        if self.config.compression_content == "think":
            self.after_think = False
        self.length = input_length
        self._decode_steps = 0
        # Prefill is the first/rectangular compression opportunity.  Resetting
        # this per request also prevents a previous generation's final flag
        # from changing the next prompt's cache path.
        for layer in self.model.layers:
            layer.self_attn.config.compression = None
    else:
        self.length = getattr(self, "length", 0) + input_length
        if self.config.divide_method == "step_length":
            interval = self.config.divide_length
            if interval <= 0:
                raise ValueError("divide_length must be positive")
            self._decode_steps = getattr(self, "_decode_steps", 0) + input_length
            compress_now = self._decode_steps % interval == 0
            if (
                self.config.compression_content == "think"
                and getattr(self, "after_think", False)
            ):
                compress_now = False
            # Step-based compression must be selected before the attention
            # forward so the N-th decode token performs the rewrite itself.
            for layer in self.model.layers:
                layer.self_attn.config.compression = compress_now

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    # =============== Step-level Compression logic start ===============
    # Only materialize an argmax token when content/newline scheduling needs
    # it. Fixed-interval throughput runs otherwise launch an unnecessary
    # vocabulary-wide reduction on every decode step.
    predicted_token_ids = None
    if (
        self.config.compression_content == "think"
        or self.config.divide_method == "newline"
    ):
        # assume non-batch input, shape: [1, logits_to_keep, vocab_size]
        predicted_token_ids = logits[:, -1, :].argmax(dim=-1)

    if self.config.compression_content == "think" and self.after_think == False:
        self.after_think = (
            predicted_token_ids[0].cpu().item() in self.after_think_token_ids
        )

    if self.config.divide_method == "newline":
        compress_next = predicted_token_ids[0].cpu().item() in self.newline_token_ids
    elif self.config.divide_method != "step_length":
        raise ValueError(f"Invalid divide_method: {self.config.divide_method}")

    # Newline-triggered compression applies to the next decode forward because
    # the predicted token is only known after this forward.  Step-length
    # compression was already applied above to the current forward.
    if self.config.divide_method == "newline":
        if self.config.compression_content == "think" and self.after_think == True:
            compress_next = False
        for layer in self.model.layers:
            layer.self_attn.config.compression = compress_next
    # =============== Step-level Compression logic end =================

    loss = None
    if labels is not None:
        loss = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            **kwargs,
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
