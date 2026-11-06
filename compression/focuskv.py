"""Focus-KV v1: importance-guided non-uniform per-head KV-cache compression.

Focus-KV allocates KV-cache budget heterogeneously across layers and KV heads
based on a pre-computed reasoning-head importance profile, then performs
variable-length per-head compression so the actual memory footprint equals
``sum(head_budgets)`` rather than ``num_heads * max_budget``.

The class :class:`HeadwiseKV` provides the variable-length KV container that
backs the per-head storage; :class:`FocusKV` is the compression strategy.

The importance-aware budget allocator (:func:`compute_head_budgets` and
helpers) loads a JSON profile produced offline by the reasoning-head
identification pipeline, aggregates query-head scores into KV-head scores
under grouped-query attention, then distributes a global token pool with
upper-bound capping and iterative redistribution.
"""

import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import cal_similarity, compute_attention_scores


# ======================================================================
# Importance loading & per-head budget allocation
# ======================================================================


def load_reasoning_importance(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def aggregate_to_kv_heads(
    query_head_importance: dict,
    num_layers: int,
    num_query_heads: int,
    num_kv_heads: int,
    aggregation: str = "max",
) -> torch.Tensor:
    """Reduce query-head importance scores to per-KV-head scores under GQA."""
    group_size = num_query_heads // num_kv_heads
    kv_importance = torch.zeros(num_layers, num_kv_heads, device="cpu")

    for layer_idx in range(num_layers):
        layer_key = str(layer_idx)
        if layer_key not in query_head_importance:
            continue
        layer_scores = query_head_importance[layer_key]

        for kv_idx in range(num_kv_heads):
            group_scores = []
            for offset in range(group_size):
                q_idx = kv_idx * group_size + offset
                q_key = str(q_idx)
                if q_key in layer_scores:
                    group_scores.append(layer_scores[q_key])

            if not group_scores:
                continue
            t = torch.tensor(group_scores, device="cpu")
            if aggregation == "max":
                kv_importance[layer_idx, kv_idx] = t.max()
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

    return kv_importance


def compute_head_budgets(
    importance_path: str,
    avg_budget: int,
    window_size: int,
    uniform_ratio: float = 0.2,
    aggregation: str = "max",
    max_budget_ratio: float = 2.0,
) -> torch.Tensor:
    """Compute per-head KV budget with global allocation and upper-bound capping.

    Distributes the reasoning budget globally across all layers and heads,
    capping each head at ``max_budget_ratio * avg_budget`` and iteratively
    redistributing the excess to the remaining (non-frozen) heads.

    Returns:
        ``LongTensor`` of shape ``(num_layers, num_kv_heads)`` giving the
        total per-head budget (``window_size + uniform_base + reasoning_alloc``).
    """
    data = load_reasoning_importance(importance_path)
    num_layers = data["num_layers"]
    num_query_heads = data["num_query_heads"]
    num_kv_heads = data["num_kv_heads"]

    kv_importance = aggregate_to_kv_heads(
        data["query_head_importance"],
        num_layers, num_query_heads, num_kv_heads,
        aggregation=aggregation,
    )

    positive_importance = kv_importance.clamp(min=0.0)

    allocatable = avg_budget - window_size
    assert allocatable > 0, "avg_budget must be greater than window_size"

    uniform_base = int(allocatable * uniform_ratio)
    reasoning_pool_per_head = allocatable - uniform_base
    total_reasoning_pool = reasoning_pool_per_head * num_kv_heads * num_layers

    max_budget = int(max_budget_ratio * avg_budget)
    max_reasoning = max(max_budget - window_size - uniform_base, 1)

    total_positive = positive_importance.sum()

    if total_positive > 0:
        reasoning_alloc = torch.zeros(
            num_layers, num_kv_heads, dtype=torch.long, device="cpu",
        )
        frozen = torch.zeros(
            num_layers, num_kv_heads, dtype=torch.bool, device="cpu",
        )

        for _ in range(num_layers * num_kv_heads):
            remaining = total_reasoning_pool - reasoning_alloc.sum().item()
            if remaining <= 0:
                break

            active_imp = positive_importance.clone()
            active_imp[frozen] = 0.0
            active_total = active_imp.sum()
            if active_total <= 0:
                break

            share = active_imp / active_total
            proposed = (share * remaining).floor().long()
            candidate = reasoning_alloc + proposed

            exceeded = candidate > max_reasoning
            if not exceeded.any():
                reasoning_alloc = candidate
                break

            newly_frozen = exceeded & ~frozen
            reasoning_alloc[newly_frozen] = max_reasoning
            frozen |= exceeded

        remainder = int(total_reasoning_pool - reasoning_alloc.sum().item())
        if remainder > 0:
            eligible_imp = positive_importance.clone()
            eligible_imp[frozen] = 0.0
            flat_imp = eligible_imp.flatten()
            n = min(remainder, int((~frozen).sum().item()))
            if n > 0:
                _, top_idx = flat_imp.topk(n)
                flat_alloc = reasoning_alloc.flatten()
                flat_alloc[top_idx] += 1
                reasoning_alloc = flat_alloc.view(num_layers, num_kv_heads)

        reasoning_alloc = reasoning_alloc.clamp(max=max_reasoning)
    else:
        per_head = min(reasoning_pool_per_head, max_reasoning)
        reasoning_alloc = torch.full(
            (num_layers, num_kv_heads), per_head,
            dtype=torch.long, device="cpu",
        )

    budgets = (window_size + uniform_base + reasoning_alloc).long()
    budgets = budgets.clamp(min=window_size + 1, max=max_budget)

    return budgets


# ======================================================================
# Variable-length per-head KV container
# ======================================================================


class HeadwiseKV:
    """Variable-length per-head KV cache with pre-allocated buffers.

    Each KV head stores its own tensor with potentially different ``seq_len``,
    eliminating the padding overhead of rectangular caches.  Total memory
    across all heads equals ``sum(head_budgets)`` — identical to a uniform
    allocator at the same ``avg_budget``.

    Unlike the naive ``torch.cat``-per-token approach, this implementation
    pre-allocates extra capacity so that :meth:`append` is an in-place write
    in the common case, drastically reducing CUDA memory fragmentation.
    """

    __slots__ = ("keys", "values", "_bufs_k", "_bufs_v", "_lens")

    _GROW = 256

    def __init__(self, keys: list, values: list):
        n = len(keys)
        self._lens = []
        self._bufs_k = []
        self._bufs_v = []
        new_keys = []
        new_values = []
        for h in range(n):
            L = keys[h].shape[2]
            self._lens.append(L)
            cap = L + self._GROW
            bsz, nh, _, dim = keys[h].shape
            bk = torch.empty(bsz, nh, cap, dim, dtype=keys[h].dtype,
                             device=keys[h].device)
            bv = torch.empty(bsz, nh, cap, dim, dtype=values[h].dtype,
                             device=values[h].device)
            bk[:, :, :L] = keys[h]
            bv[:, :, :L] = values[h]
            self._bufs_k.append(bk)
            self._bufs_v.append(bv)
            new_keys.append(bk[:, :, :L])
            new_values.append(bv[:, :, :L])
        self.keys = new_keys
        self.values = new_values

    def append(self, key_states: torch.Tensor, value_states: torch.Tensor):
        """Append new token(s) to every head (in-place when possible)."""
        n_new = key_states.shape[2]
        for h in range(len(self._bufs_k)):
            L = self._lens[h]
            cap = self._bufs_k[h].shape[2]
            if L + n_new > cap:
                new_cap = L + n_new + self._GROW
                bsz, nh, _, dim = self._bufs_k[h].shape
                new_bk = torch.empty(bsz, nh, new_cap, dim,
                                     dtype=self._bufs_k[h].dtype,
                                     device=self._bufs_k[h].device)
                new_bv = torch.empty(bsz, nh, new_cap, dim,
                                     dtype=self._bufs_v[h].dtype,
                                     device=self._bufs_v[h].device)
                new_bk[:, :, :L] = self._bufs_k[h][:, :, :L]
                new_bv[:, :, :L] = self._bufs_v[h][:, :, :L]
                self._bufs_k[h] = new_bk
                self._bufs_v[h] = new_bv
            self._bufs_k[h][:, :, L:L + n_new] = key_states[:, h : h + 1]
            self._bufs_v[h][:, :, L:L + n_new] = value_states[:, h : h + 1]
            self._lens[h] = L + n_new
            self.keys[h] = self._bufs_k[h][:, :, :self._lens[h]]
            self.values[h] = self._bufs_v[h][:, :, :self._lens[h]]

    def max_seq_len(self) -> int:
        return max(self._lens)


# ======================================================================
# Focus-KV v1 compression strategy
# ======================================================================


class FocusKV:
    """Non-uniform per-head budget allocation **without** padding overhead.

    The compressed KV is stored as per-head variable-length tensors
    (:class:`HeadwiseKV`).  Attention is computed per-KV-head during
    decoding, so no padding is required and the actual memory footprint
    equals ``sum(head_budgets)`` — the same total as a uniform allocator
    at the same ``avg_budget``.

    The ``max_budget_ratio`` parameter genuinely controls how much extra
    budget important heads receive (e.g. ``cap=2.0`` lets the most
    important head keep ``2 * avg_budget`` tokens) while the total is
    kept constant by giving less-important heads proportionally fewer
    tokens.
    """

    def __init__(
        self,
        budget=128,
        window_size=8,
        kernel_size=7,
        mix_lambda=0.07,
        retain_ratio=0.1,
        retain_direction="last",
        importance_path=None,
        uniform_ratio=0.2,
        aggregation="max",
        max_budget_ratio=2.0,
        **kwargs,
    ):
        self.avg_budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        self.importance_path = importance_path
        self.uniform_ratio = uniform_ratio
        self.aggregation = aggregation
        self.max_budget_ratio = max_budget_ratio

        self._head_budgets = None
        self._global_max_budget = None
        self._uses_headwise_cache = True
        self._pad_lengths = None

        if importance_path is not None:
            self._head_budgets = compute_head_budgets(
                importance_path=importance_path,
                avg_budget=budget,
                window_size=window_size,
                uniform_ratio=uniform_ratio,
                aggregation=aggregation,
                max_budget_ratio=max_budget_ratio,
            )
            self._global_max_budget = self._head_budgets.max().item()

    def _get_head_budget(self, layer_idx: int, head_idx: int) -> int:
        if self._head_budgets is None:
            return self.avg_budget
        return self._head_budgets[layer_idx, head_idx].item()

    # ------------------------------------------------------------------
    # Score computation (rectangular — batched over all heads)
    # ------------------------------------------------------------------

    def _compute_scores_rect(self, key_states, query_states):
        attn_weights = compute_attention_scores(query_states, key_states)
        attn_weights_sum = (
            F.softmax(
                attn_weights[:, :, -self.window_size :, : -self.window_size],
                dim=-1,
                dtype=torch.float32,
            )
            .mean(dim=-2)
            .to(query_states.dtype)
        )
        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )
        similarity_cos = cal_similarity(
            key_states,
            retain_ratio=self.retain_ratio,
            retain_direction=self.retain_direction,
        )[:, : -self.window_size]
        final_score = (
            attn_cache * self.mix_lambda
            - similarity_cos * (1 - self.mix_lambda)
        )
        return final_score

    # ------------------------------------------------------------------
    # Score computation (single head — for HeadwiseKV input)
    # ------------------------------------------------------------------

    def _compute_scores_single_head(self, k_h, q_group):
        """Compute token-selection scores for one KV head.

        Args:
            k_h:      [bsz, 1, kv_len, head_dim]
            q_group:  [bsz, group_size, window_size, head_dim]

        Returns:
            final_score: [bsz, 1, past_len]  or *None* if past_len <= 0
        """
        head_dim = k_h.shape[-1]
        kv_len = k_h.shape[2]
        past_len = kv_len - self.window_size
        if past_len <= 0:
            return None

        attn_weights = (
            torch.matmul(q_group, k_h.transpose(-1, -2))
            / math.sqrt(head_dim)
        )  # [bsz, group_size, window, kv_len]
        attn_weights = attn_weights.max(dim=1, keepdim=True).values

        attn_weights_sum = (
            F.softmax(
                attn_weights[:, :, :, :past_len],
                dim=-1,
                dtype=torch.float32,
            )
            .mean(dim=2)
            .to(k_h.dtype)
        )
        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

        k_vec = k_h[0, 0]
        k_norm = k_vec / (k_vec.norm(dim=-1, keepdim=True) + 1e-8)
        sim_cos = torch.mm(k_norm, k_norm.t())
        sim_cos.fill_diagonal_(0.0)
        sim_mask = sim_cos > 0.5

        seq_len = sim_mask.size(-1)
        k_retain = max(int(seq_len * self.retain_ratio), 1)

        indices = torch.where(
            sim_mask,
            torch.arange(seq_len, device=sim_mask.device),
            torch.zeros(seq_len, seq_len, dtype=torch.long, device=sim_mask.device),
        )

        if self.retain_direction == "last":
            sim_retain = indices.max(dim=-1).values
        elif self.retain_direction == "first":
            sim_retain = indices.min(dim=-1).values
        elif self.retain_direction == "last_percent":
            sim_retain = indices.topk(k=k_retain, dim=-1).values[:, 0]
        elif self.retain_direction == "first_percent":
            sim_retain = indices.topk(
                k=k_retain, dim=-1, largest=False,
            ).values[:, -1]
        else:
            sim_retain = indices.max(dim=-1).values

        row_idx = torch.arange(seq_len, device=sim_cos.device)
        sim_cos[row_idx, sim_retain] = 0.0

        similarity = sim_cos.mean(dim=0).softmax(dim=-1)
        similarity = similarity[:past_len].unsqueeze(0).unsqueeze(0)

        final_score = (
            attn_cache * self.mix_lambda
            - similarity * (1 - self.mix_lambda)
        )
        return final_score

    # ------------------------------------------------------------------
    # Uniform fallback used when no importance profile is provided
    # ------------------------------------------------------------------

    def _update_kv_uniform(self, key_states, query_states, value_states):
        self._pad_lengths = None
        head_dim = query_states.shape[-1]
        kv_cache_len = key_states.shape[-2]

        if kv_cache_len < self.avg_budget:
            return key_states, value_states

        final_score = self._compute_scores_rect(key_states, query_states)

        select_budget = self.avg_budget - self.window_size
        indices = final_score.topk(select_budget, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past = key_states[:, :, : -self.window_size, :].gather(2, indices)
        v_past = value_states[:, :, : -self.window_size, :].gather(2, indices)
        k_cur = key_states[:, :, -self.window_size :, :]
        v_cur = value_states[:, :, -self.window_size :, :]

        return (
            torch.cat([k_past, k_cur], dim=2),
            torch.cat([v_past, v_cur], dim=2),
        )

    # ------------------------------------------------------------------
    # First compression: rectangular tensor → HeadwiseKV
    # ------------------------------------------------------------------

    def _compress_rect_to_headwise(
        self, key_states, query_states, value_states, layer_idx,
    ):
        bsz, num_kv_heads, kv_len, head_dim = key_states.shape

        if kv_len <= self._global_max_budget:
            return key_states, value_states

        num_q_heads = query_states.shape[1]
        group_size = num_q_heads // num_kv_heads
        past_len = kv_len - self.window_size

        k_list, v_list = [], []
        for h in range(num_kv_heads):
            h_budget = min(self._get_head_budget(layer_idx, h), kv_len)

            if kv_len <= h_budget:
                k_list.append(key_states[:, h : h + 1])
                v_list.append(value_states[:, h : h + 1])
                continue

            select_count = h_budget - self.window_size
            if select_count <= 0:
                k_list.append(key_states[:, h : h + 1, -self.window_size :])
                v_list.append(value_states[:, h : h + 1, -self.window_size :])
                continue

            k_h = key_states[:, h : h + 1]
            q_start = h * group_size
            q_group = query_states[:, q_start : q_start + group_size]

            final_score = self._compute_scores_single_head(k_h, q_group)
            if final_score is None:
                k_list.append(k_h)
                v_list.append(value_states[:, h : h + 1])
                continue

            h_indices = final_score.topk(select_count, dim=-1).indices
            h_idx_exp = h_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            k_past_h = key_states[:, h : h + 1, :past_len].gather(
                2, h_idx_exp,
            )
            v_past_h = value_states[:, h : h + 1, :past_len].gather(
                2, h_idx_exp,
            )

            k_cur = key_states[:, h : h + 1, -self.window_size :]
            v_cur = value_states[:, h : h + 1, -self.window_size :]

            k_list.append(torch.cat([k_past_h, k_cur], dim=2))
            v_list.append(torch.cat([v_past_h, v_cur], dim=2))

        return HeadwiseKV(k_list, v_list)

    # ------------------------------------------------------------------
    # Subsequent compression: HeadwiseKV → HeadwiseKV
    # ------------------------------------------------------------------

    def _compress_headwise_to_headwise(
        self, headwise_kv, query_states, layer_idx,
    ):
        num_kv_heads = len(headwise_kv.keys)
        num_q_heads = query_states.shape[1]
        group_size = num_q_heads // num_kv_heads

        k_list, v_list = [], []
        for h in range(num_kv_heads):
            k_h = headwise_kv.keys[h]
            v_h = headwise_kv.values[h]
            kv_len_h = k_h.shape[2]
            head_dim = k_h.shape[-1]

            h_budget = min(self._get_head_budget(layer_idx, h), kv_len_h)

            if kv_len_h <= h_budget:
                k_list.append(k_h)
                v_list.append(v_h)
                continue

            select_count = h_budget - self.window_size
            if select_count <= 0:
                k_list.append(k_h[:, :, -self.window_size :])
                v_list.append(v_h[:, :, -self.window_size :])
                continue

            q_start = h * group_size
            q_group = query_states[:, q_start : q_start + group_size]

            final_score = self._compute_scores_single_head(k_h, q_group)
            if final_score is None:
                k_list.append(k_h)
                v_list.append(v_h)
                continue

            h_indices = final_score.topk(select_count, dim=-1).indices
            h_idx_exp = h_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            past_len_h = kv_len_h - self.window_size
            k_past = k_h[:, :, :past_len_h].gather(2, h_idx_exp)
            v_past = v_h[:, :, :past_len_h].gather(2, h_idx_exp)

            k_cur = k_h[:, :, -self.window_size :]
            v_cur = v_h[:, :, -self.window_size :]

            k_list.append(torch.cat([k_past, k_cur], dim=2))
            v_list.append(torch.cat([v_past, v_cur], dim=2))

        return HeadwiseKV(k_list, v_list)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def update_kv(self, key_states, query_states, value_states, **kwargs):
        layer_idx = kwargs.get("layer_idx")
        headwise_kv = kwargs.get("headwise_kv")

        if self._head_budgets is None or layer_idx is None:
            return self._update_kv_uniform(key_states, query_states, value_states)

        if headwise_kv is not None:
            return self._compress_headwise_to_headwise(
                headwise_kv, query_states, layer_idx,
            )
        return self._compress_rect_to_headwise(
            key_states, query_states, value_states, layer_idx,
        )
