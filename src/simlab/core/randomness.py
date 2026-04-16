"""Helpers for reproducible randomness."""

from __future__ import annotations

import random


def derive_seed(base_seed: int, *parts: int) -> int:
    """Derive a deterministic child seed from a base seed and integer tags."""
    seed = base_seed
    for part in parts:
        seed = (seed * 31 + part) % (2**32)
    return seed


def python_rng(seed: int) -> random.Random:
    """Return a Python RNG configured with a fixed seed."""
    return random.Random(seed)
