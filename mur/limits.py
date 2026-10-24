"""Whole-stage budget and context guards shared by model-facing commands.

The guards are checked *before* every model call using the maximum number of
tokens that call is allowed to generate.  Actual output lengths are accounted
after the call.  A token or wall exhaustion therefore stops the whole stage,
never just one inner loop or trajectory.
"""
from __future__ import annotations

import time


class StageBudgetExceeded(RuntimeError):
    """Hard stage budget reached; the caller must abort the whole stage."""


class ContextLimit(RuntimeError):
    """The model context cannot hold the prompt plus a usable continuation."""


class StageBudget:
    def __init__(self, max_generated: int = 1_000_000,
                 max_wall_seconds: float = 3600,
                 max_model_len: int = 16384,
                 clock=time.monotonic):
        if max_generated < 0 or max_wall_seconds <= 0 or max_model_len <= 0:
            raise ValueError("invalid stage budget")
        self.max_generated = int(max_generated)
        self.max_wall_seconds = float(max_wall_seconds)
        self.max_model_len = int(max_model_len)
        self._clock = clock
        self._started = clock()
        self.generated = 0
        self.context_capped_calls = 0

    @property
    def elapsed(self) -> float:
        return self._clock() - self._started

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_generated - self.generated)

    @property
    def remaining_wall(self) -> float:
        return max(0.0, self.max_wall_seconds - self.elapsed)

    def check_wall(self, label: str):
        if self.elapsed >= self.max_wall_seconds:
            raise StageBudgetExceeded(
                f"wall budget exhausted before {label}: "
                f"{self.elapsed:.1f}s >= {self.max_wall_seconds:.1f}s")

    def fit(self, prompt_tokens: int, desired_max_tokens: int, label: str) -> int:
        """Return the largest legal output cap, or raise ContextLimit.

        This never truncates the prompt.  It only caps the continuation to the
        remaining model context and records that the cap was reduced.
        """
        self.check_wall(label)
        if prompt_tokens < 0 or desired_max_tokens <= 0:
            raise ValueError("invalid context request")
        available = self.max_model_len - prompt_tokens
        if available <= 0:
            raise ContextLimit(
                f"{label}: prompt of {prompt_tokens} tokens leaves no room in "
                f"context {self.max_model_len}")
        effective = min(int(desired_max_tokens), available)
        if effective < int(desired_max_tokens):
            self.context_capped_calls += 1
        return effective

    def reserve(self, max_tokens_per_sequence, label: str):
        """Reject a call/batch unless its reserved maximum output fits."""
        self.check_wall(label)
        if isinstance(max_tokens_per_sequence, int):
            reserved = max_tokens_per_sequence
        else:
            reserved = sum(int(x) for x in max_tokens_per_sequence)
        if reserved < 0:
            raise ValueError("negative token reservation")
        if self.generated + reserved > self.max_generated:
            raise StageBudgetExceeded(
                f"generated-token budget would be exceeded before {label}: "
                f"used={self.generated} reserved={reserved} "
                f"max={self.max_generated}")
        return reserved

    def account(self, role: str, output_tokens: int):
        if role not in ("policy", "critic", "verifier"):
            raise ValueError("unknown model role")
        if type(output_tokens) is not int or output_tokens < 0:
            raise ValueError("output token count must be a nonnegative integer")
        self.generated += output_tokens
