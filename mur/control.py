"""Backend-independent detector and generated-token accounting."""
from dataclasses import dataclass, field
import math


@dataclass
class Momentum:
    alpha: float = 0.9
    gamma: float = 0.9
    bias_correction: bool = True
    raw: float = 0.0
    count: int = 0

    def __post_init__(self):
        if not 0 < self.alpha < 1 or not 0 < self.gamma <= 1:
            raise ValueError("alpha and gamma out of range")

    @property
    def history(self):
        if self.count == 0:
            return None
        return self.raw / (1 - self.alpha ** self.count) if self.bias_correction else self.raw

    def trigger(self, proposal_nll):
        self._valid(proposal_nll)
        return self.count > 0 and proposal_nll > self.history - math.log(self.gamma)

    def accept(self, accepted_nll):
        self._valid(accepted_nll)
        self.raw = self.alpha * self.raw + (1 - self.alpha) * accepted_nll
        self.count += 1

    @staticmethod
    def _valid(nll):
        if not math.isfinite(nll) or nll < 0:
            raise ValueError("invalid mean negative log probability")


@dataclass
class Ledger:
    calls: list = field(default_factory=list)

    def record(self, role, phase, input_tokens, output_tokens):
        if role not in ("policy", "critic", "verifier"):
            raise ValueError("unknown model role")
        if any(type(x) is not int or x < 0 for x in (input_tokens, output_tokens)):
            raise ValueError("token counts must be nonnegative integers")
        self.calls.append(dict(role=role, phase=phase, input_tokens=input_tokens,
                               output_tokens=output_tokens))

    def total(self, role=None, kind="output_tokens"):
        if kind not in ("output_tokens", "input_tokens"):
            raise ValueError("unknown token field")
        roles = (role,) if role else ("policy", "critic")
        return sum(c[kind] for c in self.calls if c["role"] in roles)


def select_judgment(judgments):
    """Entries: {label: Yes/No/None, probability: actual judgment-token probability}."""
    for j in judgments:
        if j["label"] in ("Yes", "No"):
            p = j["probability"]
            if p is None or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("invalid judgment probability")
    yes = [(i, j["probability"]) for i, j in enumerate(judgments) if j["label"] == "Yes"]
    if yes:
        return max(yes, key=lambda x: x[1])[0], False
    no = [(i, j["probability"]) for i, j in enumerate(judgments) if j["label"] == "No"]
    if len(no) == len(judgments) and no:
        return min(no, key=lambda x: x[1])[0], False
    if not judgments:
        raise ValueError("no candidates")
    return 0, True
