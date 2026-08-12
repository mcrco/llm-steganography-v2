"""Encode→decode linguistic steganography taskset.

Each episode is two isolated single-turn runs of the same agent:
1. encode — rewrite a TinyStories story to carry a hidden bit
2. decode — recover that bit from the encoded story only

Decode runs in a fresh conversation so the encode prompt (and bit) are not
visible. Reward is bit recovery under hard format constraints.
"""

from __future__ import annotations

from typing import Literal

import verifiers.v1 as vf

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


class EncodeData(vf.TaskData):
    story: str
    """TinyStories cover text the model must rewrite while hiding the bit."""

    bit: int
    """Ground-truth hidden bit in {0, 1}."""


class DecodeData(vf.TaskData):
    encoded: str
    """Encode-phase output; decode sees this text, not the original bit prompt."""

    bit: int
    """Ground truth for scoring only — never placed in the decode prompt."""


class EncodeTask(vf.Task[EncodeData]):
    @vf.stop
    async def one_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.metric
    async def encode_valid(self, trace: vf.Trace) -> float:
        return 1.0 if encode_is_valid(trace.last_reply) else 0.0


class DecodeTask(vf.Task[DecodeData]):
    @classmethod
    def from_encode(cls, encode_task: EncodeTask, encoded: str) -> DecodeTask:
        """Mint a decode task whose prompt cannot see the encode-side bit."""
        return cls(
            DecodeData(
                idx=encode_task.data.idx,
                prompt=decode_user_message(encoded),
                system_prompt=SYSTEM_PROMPT,
                encoded=encoded,
                bit=encode_task.data.bit,
            ),
            encode_task.config,
        )

    @vf.stop
    async def one_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def bit_recovery(self, trace: vf.Trace) -> float:
        return bit_recovery_reward(
            encoded=self.data.encoded,
            decoded=trace.last_reply,
            target_bit=self.data.bit,
        )

    @vf.metric
    async def decode_valid(self, trace: vf.Trace) -> float:
        return 1.0 if parse_bit(trace.last_reply) is not None else 0.0


class LlmSteganographyEnv(vf.SingleAgentEnv):
    """Encode, then decode in a fresh conversation (same agent, no history leak)."""

    async def run(self, task: EncodeTask, agents: vf.Agents) -> None:
        async with agents.agent.interaction(task) as interaction:
            encode_segment = await interaction.turn(
                encode_user_message(task.data.story, task.data.bit)
            )
            if encode_segment.terminated:
                return
            encoded = normalize_text(encode_segment.last_reply)

        # New rollout: transcript is only system + decode prompt.
        await agents.agent.run(DecodeTask.from_encode(task, encoded))

    async def finalize(self, task: EncodeTask, episode: vf.Episode) -> None:
        # Mirror decode reward onto the encode trace so both phases train.
        encode_trace = next(
            (t for t in episode.traces if t.task.type == "EncodeTask"), None
        )
        decode_trace = next(
            (t for t in episode.traces if t.task.type == "DecodeTask"), None
        )
        if encode_trace is None or decode_trace is None:
            return
        reward = decode_trace.rewards.get("bit_recovery")
        if reward is not None:
            encode_trace.record_reward("bit_recovery", reward.score, reward.weight)


class LlmSteganographyConfig(vf.TasksetConfig):
    num_tasks: int = 80
    """How many story×bit pairs to load (each story is used with both bits)."""

    split: Literal["train", "validation"] = "train"
    """TinyStories split to sample cover stories from."""

    seed: int = 0
    """Shuffle seed for reproducible story sampling."""

    min_words: int = 80
    """Reject shorter stories (keeps covers story-shaped, not tiny scraps)."""

    max_words: int = 180
    """Reject longer stories so encode generations stay bounded."""


class LlmSteganographyTaskset(vf.Taskset[EncodeTask, LlmSteganographyConfig]):
    def load(self) -> list[EncodeTask]:
        c = self.config
        # Both bits per story → need ceil(num_tasks / 2) unique stories.
        num_stories = (c.num_tasks + 1) // 2
        stories = load_stories(
            split=c.split,
            num_stories=num_stories,
            seed=c.seed,
            min_words=c.min_words,
            max_words=c.max_words,
        )

        tasks: list[EncodeTask] = []
        idx = 0
        pairs = [(story, bit) for story in stories for bit in (0, 1)]
        for story, bit in pairs[: c.num_tasks]:
            tasks.append(
                EncodeTask(
                    EncodeData(
                        idx=idx,
                        prompt=None,
                        system_prompt=SYSTEM_PROMPT,
                        story=story,
                        bit=bit,
                    ),
                    c.task,
                )
            )
            idx += 1
        return tasks
