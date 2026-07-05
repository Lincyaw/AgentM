"""Convert train_filtered.jsonl to AReaL SFT format (input_ids + loss_mask).

Usage:
    python datasets/prepare_sft.py \
        --input datasets/train_filtered.jsonl \
        --output datasets/trajectory_index_sft \
        --model Qwen/Qwen3.5-2B \
        --max-length 8192
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


def build_chatml(messages: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    """Build ChatML token sequence and loss mask (no <think> tags)."""
    parts: list[tuple[str, bool]] = []  # (text, trainable)
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        header = f"<|im_start|>{role}\n"
        footer = "<|im_end|>\n"
        if role == "assistant":
            parts.append((header, False))
            parts.append((content, True))
            parts.append((footer, True))
        else:
            parts.append((header + content + footer, False))

    input_ids: list[int] = []
    loss_mask: list[int] = []
    for text, trainable in parts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        input_ids.extend(ids)
        loss_mask.extend([int(trainable)] * len(ids))

    return input_ids, loss_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-2B")
    parser.add_argument("--max-length", type=int, default=8192)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    with args.input.open() as f:
        raw = [json.loads(line) for line in f]

    records = []
    skipped = 0
    for item in raw:
        input_ids, loss_mask = build_chatml(item["messages"], tokenizer)
        if len(input_ids) > args.max_length:
            skipped += 1
            continue
        records.append({"input_ids": input_ids, "loss_mask": loss_mask})

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(str(args.output / "train"))

    print(f"Examples: {len(records)} (skipped {skipped} > {args.max_length} tokens)")
    lens = [len(r["input_ids"]) for r in records]
    train_tokens = sum(sum(r["loss_mask"]) for r in records)
    print(f"Tokens: avg={sum(lens)//len(lens)} max={max(lens)} trainable={train_tokens:,}")


if __name__ == "__main__":
    main()
