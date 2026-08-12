"""Load cover stories from the TinyStories corpus."""

from __future__ import annotations

import random
import re
from typing import Literal

from datasets import load_dataset

_DIGIT_RE = re.compile(r"\d")
DATASET_ID = "roneneldan/TinyStories"


def _word_count(text: str) -> int:
    return len(text.split())


def load_stories(
    *,
    split: Literal["train", "validation"] = "train",
    num_stories: int,
    seed: int = 0,
    min_words: int = 80,
    max_words: int = 180,
) -> list[str]:
    """Return `num_stories` digit-free TinyStories in the word-length band.

    Streams the Hub dataset (memory-light), oversamples matches, then shuffles
    locally so a fixed seed is reproducible without a huge shuffle buffer.
    """
    if num_stories < 1:
        raise ValueError("num_stories must be >= 1")
    if not 1 <= min_words <= max_words:
        raise ValueError("need 1 <= min_words <= max_words")

    ds = load_dataset(DATASET_ID, split=split, streaming=True)
    # Seed-dependent offset so different seeds see different regions of the stream.
    if seed:
        ds = ds.skip(seed * 997)

    # Oversample then shuffle: avoids HF streaming shuffle's large buffer cost.
    target_pool = max(num_stories * 8, num_stories)
    pool: list[str] = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if not text or _DIGIT_RE.search(text):
            continue
        n_words = _word_count(text)
        if n_words < min_words or n_words > max_words:
            continue
        pool.append(text)
        if len(pool) >= target_pool:
            break

    if len(pool) < num_stories:
        raise RuntimeError(
            f"only found {len(pool)} TinyStories matching filters "
            f"(split={split}, words=[{min_words}, {max_words}], no digits); "
            f"wanted {num_stories}"
        )

    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:num_stories]
