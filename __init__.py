"""Focus-KV: importance-guided KV cache compression for reasoning models.

The package provides a memory-oriented head-wise implementation
(:class:`FocusKV`) and a throughput-oriented rectangular implementation
(:class:`FocusKVV2`). Use :func:`replace_llama`, :func:`replace_qwen2`, or
:func:`replace_qwen3` to monkey-patch the HuggingFace Transformers attention
implementation in place.
"""

__version__ = "0.1.0"

from .monkeypatch import replace_llama, replace_qwen2, replace_qwen3

__all__ = ["replace_llama", "replace_qwen2", "replace_qwen3"]
