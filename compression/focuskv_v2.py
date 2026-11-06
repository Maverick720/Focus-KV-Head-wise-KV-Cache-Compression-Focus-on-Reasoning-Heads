"""Focus-KV v2: throughput-oriented dense cache rewriting.

Focus-KV v1 minimizes physical KV memory by keeping a variable-length tensor
per KV head.  That is memory efficient, but in the HuggingFace path it forces
decode attention through a Python loop over heads unless a separate varlen
kernel path is available.

This implementation keeps the v1 allocation, scoring, and eviction rules and
changes the cache layout after each scheduled compression event.
Selected tokens for each head are rewritten into a rectangular
``[batch, kv_heads, max_live_len, head_dim]`` KV cache, with invalid slots
prepended and masked by ``_pad_lengths``.  The rewrite is amortized across
multiple decode steps and lets ordinary decode use one batched attention call
instead of a Python attention loop over KV heads.

The dense rewrite improves throughput without changing the per-head token
selection rule.
"""

import math

import torch
import torch.nn.functional as F

from .focuskv import FocusKV


class FocusKVV2(FocusKV):
    """Focus-KV with a rectangular rewritten cache for faster decoding.

    The retained token set is intentionally the same as v1 for a given head
    budget and score function.  Only the post-compression layout changes:

    * v1 stores a ``HeadwiseKV`` object and computes attention head by head.
    * v2 stores a padded rectangular KV tensor and exposes per-head pad
      lengths so ``modeling.py`` can mask padding and call the model's normal
      attention backend.

    This is most useful for throughput experiments where the extra padding up
    to the largest per-head budget is acceptable in exchange for fewer Python
    dispatches and larger fused attention kernels.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("uniform_ratio", 0.3)
        super().__init__(*args, **kwargs)
        self._uses_headwise_cache = False
        self._uses_rectangular_cache = True

    def _compute_scores_single_head(
        self, k_h, q_group, key_valid_mask=None,
    ):
        """Compute token-selection scores for one KV head."""
        head_dim = k_h.shape[-1]
        kv_len = k_h.shape[2]
        past_len = kv_len - self.window_size
        if past_len <= 0:
            return None

        attn_weights = (
            torch.matmul(q_group, k_h.transpose(-1, -2))
            / math.sqrt(head_dim)
        )
        attn_weights = attn_weights.max(dim=1, keepdim=True).values

        past_valid_mask = None
        if key_valid_mask is not None:
            key_valid_mask = key_valid_mask.to(
                device=k_h.device, dtype=torch.bool,
            )
            if key_valid_mask.shape != (k_h.shape[0], kv_len):
                raise ValueError(
                    "key_valid_mask must have shape "
                    f"{(k_h.shape[0], kv_len)}, got {tuple(key_valid_mask.shape)}"
                )
            past_valid_mask = key_valid_mask[:, :past_len]

        past_attn_weights = attn_weights[:, :, :, :past_len]
        if past_valid_mask is not None:
            past_attn_weights = past_attn_weights.masked_fill(
                ~past_valid_mask[:, None, None, :],
                torch.finfo(past_attn_weights.dtype).min,
            )

        attn_weights_sum = (
            F.softmax(past_attn_weights, dim=-1, dtype=torch.float32)
            .mean(dim=2)
            .to(k_h.dtype)
        )
        attn_cache = F.max_pool1d(
            attn_weights_sum,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        )

        k_vec = k_h[:, 0]
        k_norm = k_vec / (k_vec.norm(dim=-1, keepdim=True) + 1e-8)
        sim_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))
        diagonal = torch.eye(
            kv_len, dtype=torch.bool, device=sim_cos.device,
        ).unsqueeze(0)
        sim_cos.masked_fill_(diagonal, 0.0)
        if key_valid_mask is not None:
            pair_valid = (
                key_valid_mask.unsqueeze(1) & key_valid_mask.unsqueeze(2)
            )
            sim_cos.masked_fill_(~pair_valid, 0.0)
        sim_mask = sim_cos > 0.5

        seq_len = sim_mask.size(-1)
        k_retain = max(int(seq_len * self.retain_ratio), 1)
        indices = torch.where(
            sim_mask,
            torch.arange(seq_len, device=sim_mask.device).view(1, 1, -1),
            torch.zeros_like(sim_mask, dtype=torch.long),
        )

        if self.retain_direction == "last":
            sim_retain = indices.max(dim=-1).values
        elif self.retain_direction == "first":
            sim_retain = indices.min(dim=-1).values
        elif self.retain_direction == "last_percent":
            sim_retain = indices.topk(k=k_retain, dim=-1).values[:, :, 0]
        elif self.retain_direction == "first_percent":
            sim_retain = indices.topk(
                k=k_retain, dim=-1, largest=False,
            ).values[:, :, -1]
        else:
            sim_retain = indices.max(dim=-1).values

        sim_cos.scatter_(2, sim_retain.unsqueeze(-1), 0.0)

        if key_valid_mask is None:
            similarity_logits = sim_cos.mean(dim=1)
        else:
            valid_count = key_valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
            similarity_logits = sim_cos.sum(dim=1) / valid_count
            similarity_logits = similarity_logits.masked_fill(
                ~key_valid_mask, torch.finfo(similarity_logits.dtype).min,
            )
        similarity = similarity_logits.softmax(dim=-1)
        similarity = similarity[:, :past_len].unsqueeze(1)

        final_score = (
            attn_cache * self.mix_lambda
            - similarity * (1 - self.mix_lambda)
        )
        if past_valid_mask is not None:
            final_score = final_score.masked_fill(
                ~past_valid_mask.unsqueeze(1),
                torch.finfo(final_score.dtype).min,
            )
        return final_score

    @staticmethod
    def _current_valid_lengths(key_states, pad_lengths=None):
        """Return valid lengths for each head in the current rectangular cache."""
        num_kv_heads = key_states.shape[1]
        kv_len = key_states.shape[2]
        if pad_lengths is None:
            return torch.full(
                (num_kv_heads,), kv_len, dtype=torch.long, device=key_states.device,
            )
        pad = pad_lengths.to(device=key_states.device, dtype=torch.long)
        if pad.numel() != num_kv_heads:
            raise ValueError(
                f"pad_lengths has {pad.numel()} entries for {num_kv_heads} KV heads"
            )
        if bool((pad < 0).any().item()) or bool((pad > kv_len).any().item()):
            raise ValueError(
                f"invalid KV padding lengths for physical cache length {kv_len}: {pad}"
            )
        return kv_len - pad

    def _needs_rewrite(
        self, key_states, layer_idx, pad_lengths=None, batch_pad_lengths=None,
    ):
        if self._head_budgets is None or layer_idx is None:
            return False

        # Match v1's first-compression boundary exactly. Before a cache has
        # been rewritten, v1 keeps the rectangular prompt intact until it is
        # longer than the largest budget used anywhere in the model.
        if pad_lengths is None:
            if batch_pad_lengths is None:
                return key_states.shape[2] > self._global_max_budget
            batch_pad_lengths = batch_pad_lengths.to(
                device=key_states.device, dtype=torch.long,
            ).reshape(-1)
            if batch_pad_lengths.numel() != key_states.shape[0]:
                raise ValueError(
                    f"batch_pad_lengths has {batch_pad_lengths.numel()} entries "
                    f"for batch size {key_states.shape[0]}"
                )
            if bool((batch_pad_lengths < 0).any().item()) or bool(
                (batch_pad_lengths > key_states.shape[2]).any().item()
            ):
                raise ValueError("invalid batch padding lengths")
            min_valid_len = key_states.shape[2] - batch_pad_lengths.max()
            return bool((min_valid_len > self._global_max_budget).item())

        valid_lens = self._current_valid_lengths(key_states, pad_lengths)
        budgets = self._head_budgets[layer_idx].to(valid_lens.device)
        return bool((valid_lens > budgets).any().item())

    def _pad_and_stack(self, k_list, v_list, actual_lengths, key_states):
        """Pad selected per-head KV to a single rectangular cache tensor."""
        bsz, _, _, head_dim = key_states.shape
        max_len = max(actual_lengths)
        k_padded = []
        v_padded = []
        pad_lens = []

        for k_h, v_h, cur_len in zip(k_list, v_list, actual_lengths):
            pad_len = max_len - cur_len
            pad_lens.append(pad_len)
            if pad_len > 0:
                k_pad = torch.zeros(
                    bsz, 1, pad_len, head_dim,
                    device=k_h.device, dtype=k_h.dtype,
                )
                v_pad = torch.zeros(
                    bsz, 1, pad_len, head_dim,
                    device=v_h.device, dtype=v_h.dtype,
                )
                k_h = torch.cat([k_pad, k_h], dim=2)
                v_h = torch.cat([v_pad, v_h], dim=2)
            k_padded.append(k_h)
            v_padded.append(v_h)

        pad_lengths = torch.tensor(
            pad_lens, dtype=torch.long, device=key_states.device,
        )
        self._pad_lengths = pad_lengths
        return torch.cat(k_padded, dim=1), torch.cat(v_padded, dim=1)

    def _rewrite_rectangular_cache(
        self, key_states, query_states, value_states, layer_idx, pad_lengths=None,
        batch_pad_lengths=None,
    ):
        bsz, num_kv_heads, kv_len, _ = key_states.shape
        num_q_heads = query_states.shape[1]
        group_size = num_q_heads // num_kv_heads
        valid_lens = self._current_valid_lengths(key_states, pad_lengths)
        key_valid_mask = None
        if pad_lengths is None and batch_pad_lengths is not None:
            batch_pad_lengths = batch_pad_lengths.to(
                device=key_states.device, dtype=torch.long,
            ).reshape(-1)
            if batch_pad_lengths.numel() != bsz:
                raise ValueError(
                    f"batch_pad_lengths has {batch_pad_lengths.numel()} entries "
                    f"for batch size {bsz}"
                )
            key_valid_mask = (
                torch.arange(kv_len, device=key_states.device).unsqueeze(0)
                >= batch_pad_lengths.unsqueeze(1)
            )
        k_list = []
        v_list = []
        actual_lengths = []

        for h in range(num_kv_heads):
            pad_len = kv_len - int(valid_lens[h].item())
            k_h = key_states[:, h : h + 1, pad_len:, :]
            v_h = value_states[:, h : h + 1, pad_len:, :]
            kv_len_h = k_h.shape[2]
            head_dim = k_h.shape[-1]

            h_budget = min(self._get_head_budget(layer_idx, h), kv_len_h)
            if kv_len_h <= h_budget:
                k_list.append(k_h)
                v_list.append(v_h)
                actual_lengths.append(kv_len_h)
                continue

            select_count = h_budget - self.window_size
            if select_count <= 0:
                k_list.append(k_h[:, :, -self.window_size :, :])
                v_list.append(v_h[:, :, -self.window_size :, :])
                actual_lengths.append(self.window_size)
                continue

            q_start = h * group_size
            q_group = query_states[:, q_start : q_start + group_size]
            final_score = self._compute_scores_single_head(
                k_h, q_group, key_valid_mask=key_valid_mask,
            )
            if final_score is None:
                k_list.append(k_h)
                v_list.append(v_h)
                actual_lengths.append(kv_len_h)
                continue

            h_indices = final_score.topk(select_count, dim=-1).indices
            h_idx_exp = h_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            past_len_h = kv_len_h - self.window_size
            k_past = k_h[:, :, :past_len_h, :].gather(2, h_idx_exp)
            v_past = v_h[:, :, :past_len_h, :].gather(2, h_idx_exp)

            k_cur = k_h[:, :, -self.window_size :, :]
            v_cur = v_h[:, :, -self.window_size :, :]
            k_list.append(torch.cat([k_past, k_cur], dim=2))
            v_list.append(torch.cat([v_past, v_cur], dim=2))
            actual_lengths.append(h_budget)

        return self._pad_and_stack(k_list, v_list, actual_lengths, key_states)

    def update_kv(self, key_states, query_states, value_states, **kwargs):
        layer_idx = kwargs.get("layer_idx")
        pad_lengths = kwargs.get("pad_lengths")
        batch_pad_lengths = kwargs.get("batch_pad_lengths")

        if self._head_budgets is None or layer_idx is None:
            self._pad_lengths = None
            return self._update_kv_uniform(key_states, query_states, value_states)

        self._pad_lengths = None
        if not self._needs_rewrite(
            key_states, layer_idx, pad_lengths, batch_pad_lengths,
        ):
            return key_states, value_states

        return self._rewrite_rectangular_cache(
            key_states, query_states, value_states, layer_idx, pad_lengths,
            batch_pad_lengths,
        )
