#!/usr/bin/env python3
"""
find_reasoning_heads_multigpu.py — Multi-GPU ablation scan for reasoning heads.

Parallelizes the single-GPU find_reasoning_heads.py by distributing layers
across available GPUs. Each GPU loads a full model replica and independently
ablates its assigned subset of layers. Partial results are checkpointed per-GPU
and merged into a single importance matrix after all GPUs finish.

Usage:
    python find_reasoning_heads_multigpu.py \
        --model_name_or_path Qwen/Qwen3-8B \
        --probe_data_path probeset/probeset.jsonl \
        --output_dir head_importance/qwen3_8b \
        --batch_size 1 --max_length 16384 --max_samples 100 \
        --eval_steps 100 --num_gpus 4

The output ``query_head_importance.json`` is written under ``--output_dir``
and can then be passed directly to Focus-KV via the ``importance_path``
argument (e.g. ``head_importance/qwen3_8b/query_head_importance.json``).
"""

import argparse
import ast
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ═══════════════════════════════ Dataset ═══════════════════════════════


class ProbeDataset(Dataset):
    """Probe dataset for reasoning-head discovery."""

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 16384,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

        print(f"Loaded {len(self.samples)} probe samples from {data_path}")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _parse_message(m):
        if isinstance(m, dict):
            return m
        try:
            return ast.literal_eval(m)
        except (ValueError, SyntaxError):
            return json.loads(m)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        if not ("messages" in sample and sample["messages"]):
            raise ValueError(
                f"Sample {idx} has no 'messages' field — cannot build chat input."
            )

        try:
            return self._build_from_messages(sample)
        except Exception as e:
            print(
                f"WARNING: sample {idx} failed to build from messages: {e}"
            )
            raise

    def _build_from_messages(self, sample) -> Dict[str, torch.Tensor]:
        messages = [self._parse_message(m) for m in sample["messages"]]

        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.tokenizer.apply_chat_template(
            [messages[0]], tokenize=False, add_generation_prompt=True
        )
        return self._tokenize_with_label_mask(full_text, prompt_text)

    def _tokenize_with_label_mask(
        self, full_text: str, prompt_text: str
    ) -> Dict[str, torch.Tensor]:
        tok_full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        input_ids = tok_full.input_ids.squeeze(0)

        tok_prompt = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        prompt_len = tok_prompt.input_ids.shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        actual_len = len(input_ids)
        if actual_len < self.max_length:
            pad_len = self.max_length - actual_len
            pad_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            )
            input_ids = torch.cat(
                [input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)]
            )
            labels = torch.cat(
                [labels, torch.full((pad_len,), -100, dtype=torch.long)]
            )

        attention_mask = torch.zeros_like(input_ids)
        attention_mask[:actual_len] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ═══════════════════════════════ Ablation Helpers ═══════════════════════


def ablate_head(model, layer_idx: int, head_idx: int, num_query_heads: int):
    """
    Scale a single query head's Q output by ε ≈ 1e-9 (Undifferentiated Attention).
    Returns a restore callable that must be invoked in a finally block.
    """
    epsilon = 1e-9
    attn_module = model.model.layers[layer_idx].self_attn
    q_proj = attn_module.q_proj
    original_forward = q_proj.forward
    n_heads = num_query_heads

    def masked_q_forward(hidden_states):
        query_states = original_forward(hidden_states)
        bsz, q_len, _ = query_states.shape
        head_dim = query_states.shape[-1] // n_heads
        query_states = query_states.view(bsz, q_len, n_heads, head_dim)
        query_states[:, :, head_idx, :] *= epsilon
        return query_states.view(bsz, q_len, -1)

    q_proj.forward = masked_q_forward

    def restore():
        q_proj.forward = original_forward

    return restore


@torch.no_grad()
def compute_loss(
    model, dataloader: DataLoader, num_steps: int, device: str, desc: str = ""
) -> float:
    """Compute average loss over num_steps batches."""
    losses: List[float] = []
    pbar = tqdm(
        enumerate(dataloader),
        total=min(num_steps, len(dataloader)),
        desc=desc or "Forward",
        unit="batch",
        leave=False,
    )
    for i, batch in pbar:
        if i >= num_steps:
            break
        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss_val = outputs.loss.item()
            losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}")
        except Exception as e:
            print(f"  Warning: batch {i} forward error: {e}")
            continue
    pbar.close()
    if not losses:
        raise ValueError("No loss computed — check data and eval_steps.")
    return float(np.mean(losses))


# ═══════════════════════════════ GPU Worker ═══════════════════════════════


def gpu_worker(
    rank: int,
    gpu_id: int,
    assigned_layers: List[int],
    model_name_or_path: str,
    probe_data_path: str,
    output_dir: str,
    batch_size: int,
    max_length: int,
    max_samples: int,
    eval_steps: int,
    dtype_str: str,
):
    """
    Worker process for a single GPU.
    Loads the model, computes baseline, ablates assigned layers, saves partial results.
    """
    device = f"cuda:{gpu_id}"
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_str]
    ckpt_path = os.path.join(output_dir, f"partial_gpu{rank}.json")

    print(f"[GPU {rank} | cuda:{gpu_id}] Starting, assigned layers: {assigned_layers}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, trust_remote_code=True, torch_dtype=dtype
    ).to(device)
    model.eval()

    num_query_heads = model.config.num_attention_heads

    dataset = ProbeDataset(
        data_path=probe_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # ── Resume from partial checkpoint ──
    computed: Dict[str, float] = {}
    baseline_loss: Optional[float] = None

    if os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            ckpt = json.load(f)
        baseline_loss = ckpt.get("baseline_loss")
        computed = ckpt.get("scores", {})
        print(f"[GPU {rank}] Resumed: {len(computed)} heads already computed")

    # ── Baseline ──
    if baseline_loss is None:
        print(f"[GPU {rank}] Computing baseline loss ...")
        baseline_loss = compute_loss(
            model, dataloader, eval_steps, device,
            desc=f"[GPU {rank}] Baseline",
        )
        print(f"[GPU {rank}] Baseline loss: {baseline_loss:.6f}")
        with open(ckpt_path, "w") as f:
            json.dump({"baseline_loss": baseline_loss, "scores": computed}, f)

    # ── Ablation scan ──
    total_heads = len(assigned_layers) * num_query_heads
    done = len(computed)
    initial_done = done
    t0 = time.time()

    for layer_idx in assigned_layers:
        layer_remaining = sum(
            1 for h in range(num_query_heads)
            if f"{layer_idx},{h}" not in computed
        )
        if layer_remaining == 0:
            continue
        print(
            f"[GPU {rank}] ── Layer {layer_idx} "
            f"({num_query_heads - layer_remaining}/{num_query_heads} heads done) ──"
        )

        for head_idx in range(num_query_heads):
            key = f"{layer_idx},{head_idx}"
            if key in computed:
                continue

            restore = ablate_head(model, layer_idx, head_idx, num_query_heads)
            try:
                abl_loss = compute_loss(
                    model, dataloader, eval_steps, device,
                    desc=f"[GPU {rank}] L{layer_idx}H{head_idx}",
                )
                importance = abl_loss - baseline_loss
            except Exception as e:
                print(f"[GPU {rank}] Error at L{layer_idx}H{head_idx}: {e}")
                importance = 0.0
            finally:
                restore()

            computed[key] = float(importance)
            done += 1

            with open(ckpt_path, "w") as f:
                json.dump({"baseline_loss": baseline_loss, "scores": computed}, f)

            elapsed = time.time() - t0
            heads_this_run = done - initial_done
            rate = heads_this_run / elapsed if elapsed > 0 else 0
            eta = (total_heads - done) / rate if rate > 0 else float("inf")
            print(
                f"[GPU {rank}] L{layer_idx}H{head_idx}  "
                f"imp={importance:+.6f}  "
                f"({done}/{total_heads})  "
                f"ETA {eta / 60:.1f}min"
            )

    total_time = (time.time() - t0) / 60
    print(f"[GPU {rank}] Finished — {done} heads in {total_time:.1f} min")


# ═══════════════════════════ Distribution & Merge ═══════════════════════


def distribute_layers(num_layers: int, num_gpus: int) -> List[List[int]]:
    """Split layers into contiguous blocks across GPUs, as evenly as possible."""
    assignments: List[List[int]] = []
    base = num_layers // num_gpus
    remainder = num_layers % num_gpus
    start = 0
    for rank in range(num_gpus):
        count = base + (1 if rank < remainder else 0)
        assignments.append(list(range(start, start + count)))
        start += count
    return assignments


def merge_partial_results(
    output_dir: str,
    num_gpus: int,
    num_layers: int,
    num_query_heads: int,
    num_kv_heads: int,
    model_name: str,
) -> Tuple[np.ndarray, float]:
    """Read all partial_gpu*.json files and merge into final query_head_importance.json."""
    print("\n═══ Merging partial results ═══")

    query_importance = np.zeros((num_layers, num_query_heads))
    baseline_losses: List[float] = []
    total_computed = 0

    for rank in range(num_gpus):
        ckpt_path = os.path.join(output_dir, f"partial_gpu{rank}.json")
        if not os.path.exists(ckpt_path):
            print(f"  WARNING: missing partial results for GPU {rank}")
            continue
        with open(ckpt_path, "r") as f:
            ckpt = json.load(f)
        baseline_losses.append(ckpt["baseline_loss"])
        scores = ckpt.get("scores", {})
        for key, val in scores.items():
            l, h = map(int, key.split(","))
            query_importance[l, h] = val
        total_computed += len(scores)
        print(
            f"  GPU {rank}: {len(scores)} heads, "
            f"baseline={ckpt['baseline_loss']:.6f}"
        )

    expected = num_layers * num_query_heads
    print(f"  Total: {total_computed}/{expected} heads merged")
    if total_computed < expected:
        print(f"  WARNING: {expected - total_computed} heads missing!")

    baseline_loss = float(np.mean(baseline_losses)) if baseline_losses else 0.0

    result = {
        "model_name": model_name,
        "num_layers": num_layers,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "baseline_loss": baseline_loss,
        "query_head_importance": {
            str(l): {
                str(h): float(query_importance[l, h])
                for h in range(num_query_heads)
            }
            for l in range(num_layers)
        },
    }
    out_path = os.path.join(output_dir, "query_head_importance.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Final results saved → {out_path}")

    return query_importance, baseline_loss


# ═══════════════════════════════ CLI ═══════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Multi-GPU ablation scan for reasoning head discovery. "
            "Distributes layers across GPUs for parallel ablation."
        )
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace model id or local path",
    )
    parser.add_argument(
        "--probe_data_path",
        type=str,
        required=True,
        help="Path to probe dataset (.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="head_importance/run",
        help=(
            "Directory to save all outputs (per-GPU partials + final "
            "query_head_importance.json). Recommended layout: "
            "head_importance/<model_short_name>/"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--max_length", type=int, default=16384, help="Max sequence length"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Max number of probe samples to use",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Number of eval batches per ablation",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs (e.g. '0,2,4,6'). Overrides --num_gpus.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--merge_only",
        action="store_true",
        help="Skip ablation, only merge existing partial_gpu*.json files.",
    )
    args = parser.parse_args()

    # ── Resolve GPU list ──
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        n = args.num_gpus or torch.cuda.device_count()
        gpu_ids = list(range(n))

    num_gpus = len(gpu_ids)
    if num_gpus < 1:
        raise RuntimeError("No CUDA GPUs available.")

    # ── Load config (lightweight, no model weights) ──
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    num_layers = config.num_hidden_layers
    num_query_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_query_heads)

    print(
        f"Architecture: {num_layers} layers, "
        f"{num_query_heads} query heads, "
        f"{num_kv_heads} KV heads "
        f"(group size {num_query_heads // num_kv_heads})"
    )

    if num_gpus > num_layers:
        num_gpus = num_layers
        gpu_ids = gpu_ids[:num_gpus]
        print(f"  Capped to {num_gpus} GPUs (one per layer)")

    print(f"Using {num_gpus} GPU(s): {gpu_ids}")

    # ── Distribute layers ──
    layer_assignments = distribute_layers(num_layers, num_gpus)
    for rank, layers in enumerate(layer_assignments):
        print(
            f"  GPU {rank} (cuda:{gpu_ids[rank]}): "
            f"layers {layers[0]}–{layers[-1]} ({len(layers)} layers, "
            f"{len(layers) * num_query_heads} heads)"
        )

    total_heads = num_layers * num_query_heads
    print(f"Total ablation experiments: {total_heads}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Ablation ──
    if not args.merge_only:
        mp.set_start_method("spawn", force=True)
        processes: List[mp.Process] = []
        t0 = time.time()

        for rank in range(num_gpus):
            p = mp.Process(
                target=gpu_worker,
                args=(
                    rank,
                    gpu_ids[rank],
                    layer_assignments[rank],
                    args.model_name_or_path,
                    args.probe_data_path,
                    args.output_dir,
                    args.batch_size,
                    args.max_length,
                    args.max_samples,
                    args.eval_steps,
                    args.dtype,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        elapsed = time.time() - t0
        print(f"\nAll GPUs finished in {elapsed / 60:.1f} min")

        failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            print(
                f"WARNING: GPU(s) {failed} exited with errors. "
                f"Partial results may be incomplete. "
                f"Re-run to resume from checkpoints."
            )

    # ── Merge ──
    merge_partial_results(
        output_dir=args.output_dir,
        num_gpus=num_gpus,
        num_layers=num_layers,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        model_name=args.model_name_or_path,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
