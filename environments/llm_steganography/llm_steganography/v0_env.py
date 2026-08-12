"""v0 MultiTurnEnv for Hosted Training compatibility.

Hosted Training's vendored verifiers has an incomplete v1 surface (missing
`SingleAgentEnv` / `AgentConfig`). Public Hub envs that train reliably use the
legacy `load_environment()` + `MultiTurnEnv` API — so we do too.

Decode is isolated by overriding `get_prompt_messages` so turn 2 is a fresh
system+DECODE conversation (no encode bit in history).
"""

from __future__ import annotations

from typing import Any, Literal

from datasets import Dataset

import verifiers as vf

from llm_steganography.prompts import (
    SYSTEM_PROMPT,
    decode_user_message,
    encode_user_message,
)
from llm_steganography.scoring import (
    bit_recovery_reward,
    encode_is_valid,
    normalize_text,
    parse_bit,
)
from llm_steganography.tinystories import load_stories


def _assistant_text(messages: Any) -> str:
    if isinstance(messages, str):
        return normalize_text(messages)
    if not isinstance(messages, list):
        return ""
    chunks: list[str] = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            chunks.append(str(msg.get("content") or ""))
    return normalize_text("\n".join(chunks))


def _build_dataset(
    *,
    num_tasks: int,
    split: Literal["train", "validation"],
    seed: int,
    min_words: int,
    max_words: int,
) -> Dataset:
    num_stories = (num_tasks + 1) // 2
    stories = load_stories(
        split=split,
        num_stories=num_stories,
        seed=seed,
        min_words=min_words,
        max_words=max_words,
    )
    rows: list[dict[str, Any]] = []
    pairs = [(story, bit) for story in stories for bit in (0, 1)]
    for idx, (story, bit) in enumerate(pairs[:num_tasks]):
        rows.append(
            {
                "question": encode_user_message(story, bit),
                "answer": str(bit),
                "info": {"bit": bit, "story": story, "idx": idx},
            }
        )
    return Dataset.from_list(rows)


def load_environment(
    num_tasks: int = 80,
    split: Literal["train", "validation"] = "train",
    seed: int = 0,
    min_words: int = 80,
    max_words: int = 180,
    **kwargs: Any,
) -> vf.Environment:
    """Hub / Hosted Training entrypoint (v0)."""
    del kwargs  # ignore unknown forward-compat knobs

    def dataset_builder() -> Dataset:
        return _build_dataset(
            num_tasks=num_tasks,
            split=split,
            seed=seed,
            min_words=min_words,
            max_words=max_words,
        )

    class StegoEnv(vf.MultiTurnEnv):
        @vf.stop
        async def two_turns(self, state: vf.State) -> bool:
            return len(state["trajectory"]) >= 2

        async def get_prompt_messages(self, state: vf.State) -> vf.Messages:
            if len(state["trajectory"]) == 0:
                return state["prompt"]

            encoded = _assistant_text(state["trajectory"][-1]["completion"])
            state["encoded"] = encoded
            info = state.setdefault("info", {})
            if isinstance(info, dict):
                info["encoded"] = encoded

            # Fresh conversation — encode prompt (and bit) must not leak here.
            return [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": decode_user_message(encoded)},
            ]

        async def env_response(
            self, messages: vf.Messages, state: vf.State, **kwargs: Any
        ) -> vf.Messages:
            # Unused: decode prompt is built in get_prompt_messages.
            del messages, state, kwargs
            return []

    def bit_recovery(completion, answer, state, **kwargs) -> float:
        del completion, kwargs
        traj = state.get("trajectory") or []
        if len(traj) < 2:
            return 0.0
        encoded = state.get("encoded") or _assistant_text(traj[0]["completion"])
        decoded = _assistant_text(traj[1]["completion"])
        try:
            target = int(answer)
        except (TypeError, ValueError):
            target = int((state.get("info") or {}).get("bit", -1))
        return bit_recovery_reward(
            encoded=encoded, decoded=decoded, target_bit=target
        )

    def encode_valid_metric(completion, state, **kwargs) -> float:
        del completion, kwargs
        traj = state.get("trajectory") or []
        if not traj:
            return 0.0
        return 1.0 if encode_is_valid(_assistant_text(traj[0]["completion"])) else 0.0

    def decode_valid_metric(completion, state, **kwargs) -> float:
        del completion, kwargs
        traj = state.get("trajectory") or []
        if len(traj) < 2:
            return 0.0
        return 1.0 if parse_bit(_assistant_text(traj[1]["completion"])) is not None else 0.0

    rubric = vf.Rubric(
        funcs=[bit_recovery, encode_valid_metric, decode_valid_metric],
        weights=[1.0, 0.0, 0.0],
    )

    return StegoEnv(
        dataset=dataset_builder,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        max_turns=2,
    )
