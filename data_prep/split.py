"""Seeded shuffle + train/val/test split for object-detection pairs.

We do not stratify: object detection labels are multi-label per image, so a clean
stratified split is non-trivial. We shuffle with a fixed seed and log the per-split
class distribution in dataset.meta.json so imbalance is visible.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .validate import Pair


@dataclass
class SplitResult:
    train: list[Pair]
    val: list[Pair]
    test: list[Pair]


def split_pairs(
    pairs: list[Pair],
    ratios: tuple[float, float, float],
    seed: int,
) -> SplitResult:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {sum(ratios)}")
    if any(r < 0 for r in ratios):
        raise ValueError(f"ratios must be non-negative, got {ratios}")

    ordered = sorted(pairs, key=lambda p: p.stem)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n = len(ordered)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return SplitResult(
        train=ordered[:n_train],
        val=ordered[n_train : n_train + n_val],
        test=ordered[n_train + n_val :],
    )
