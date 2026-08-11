"""Encode→decode linguistic steganography taskset.

Each episode is a two-turn scripted interaction:
1. encode — rewrite a buffer sentence to carry a hidden bit
2. decode — recover that bit from the encoded sentence

Reward is bit recovery under hard format constraints. Similarity scoring can
layer on later without changing the interaction protocol.
"""

from __future__ import annotations

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
from llm_steganography.smoke_data import SMOKE_BUFFERS


class LlmSteganographyData(vf.TaskData):
    buffer: str
    """Cover sentence the model must rewrite while hiding the bit."""

    bit: int
    """Ground-truth hidden bit in {0, 1}."""


class LlmSteganographyTask(vf.Task[LlmSteganographyData]):
    @vf.stop
    async def two_turns(self, trace: vf.Trace) -> bool:
        # Protocol is exactly encode then decode.
        return trace.num_turns >= 2

    @vf.reward(weight=1.0)
    async def bit_recovery(self, trace: vf.Trace) -> float:
        replies = [m.content for m in trace.assistant_messages]
        if len(replies) < 2:
            return 0.0
        return bit_recovery_reward(
            encoded=replies[0],
            decoded=replies[1],
            target_bit=self.data.bit,
        )

    @vf.metric
    async def encode_valid(self, trace: vf.Trace) -> float:
        replies = [m.content for m in trace.assistant_messages]
        if not replies:
            return 0.0
        return 1.0 if encode_is_valid(replies[0]) else 0.0

    @vf.metric
    async def decode_valid(self, trace: vf.Trace) -> float:
        replies = [m.content for m in trace.assistant_messages]
        if len(replies) < 2:
            return 0.0
        return 1.0 if parse_bit(replies[1]) is not None else 0.0


class LlmSteganographyEnv(vf.SingleAgentEnv):
    """Drives encode, then decode using the model's own encode output."""

    async def run(self, task: LlmSteganographyTask, agents: vf.Agents) -> None:
        # prompt=None on the row so this scripted user opens the conversation.
        async with agents.agent.interaction(task) as interaction:
            encode_segment = await interaction.turn(
                encode_user_message(task.data.buffer, task.data.bit)
            )
            if encode_segment.terminated:
                return
            encoded = normalize_text(encode_segment.last_reply)
            await interaction.turn(decode_user_message(encoded))


class LlmSteganographyConfig(vf.TasksetConfig):
    num_tasks: int = 16
    """Smoke size: each buffer is paired with both bits (8 × 2)."""


class LlmSteganographyTaskset(
    vf.Taskset[LlmSteganographyTask, LlmSteganographyConfig]
):
    def load(self) -> list[LlmSteganographyTask]:
        tasks: list[LlmSteganographyTask] = []
        idx = 0
        # Round-robin over buffers × bits until num_tasks is filled.
        pairs = [(buffer, bit) for buffer in SMOKE_BUFFERS for bit in (0, 1)]
        for buffer, bit in pairs[: self.config.num_tasks]:
            tasks.append(
                LlmSteganographyTask(
                    LlmSteganographyData(
                        idx=idx,
                        prompt=None,
                        system_prompt=SYSTEM_PROMPT,
                        buffer=buffer,
                        bit=bit,
                    ),
                    self.config.task,
                )
            )
            idx += 1
        return tasks
