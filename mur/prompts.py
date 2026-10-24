"""Prompt construction and parsing frozen from PROTOCOL.md and upstream Appendix B."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable


def question_key(text: str) -> str:
    """Normalization used only for train/test deduplication, not for evaluation."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"\\left|\\right|\\,|\\!|\\;|\\ ", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def request_seed(seed: int, qid: str, step: int, role: str, candidate: int = 0) -> int:
    payload = f"{seed}|{qid}|{step}|{role}|{candidate}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31 - 1)


def policy_prompt(tokenizer, question: str, trajectory: list[str], step_idx: int) -> str:
    """Appendix B step-proposal prompt, Qwen non-thinking template, one user turn."""
    content = (
        f"Q: {question}\n"
        "Always end your solution with the phrase 'the answer is' followed by your final answer. "
        f"Start your solution with 'Step{step_idx}:'\n"
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        enable_thinking=False,
        add_generation_prompt=True,
    )
    text = text.strip()
    text += "\n" + "\n".join(trajectory)
    text += f"\nStep{step_idx}:"
    return text


def finalize_prompt(tokenizer, question: str, trajectory: list[str]) -> str:
    """Identical final fallback prompt for every method, only used at the step cap."""
    content = (
        f"Q: {question}\n"
        "Always end your solution with the phrase 'the answer is' followed by your final answer. "
        "Finish the solution now and state the final answer.\n"
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        enable_thinking=False,
        add_generation_prompt=True,
    )
    return text.strip() + "\n" + "\n".join(trajectory) + "\n"


def critic_prompt(question: str, trajectory: list[str], candidate: str) -> dict:
    """Appendix B / upstream GenPRM critique prompt for MATH."""
    system = (
        "You are a math teacher. Your task is to review and critique the paragraphs "
        "in solution directly. Output your judgement in the format of `\\boxed{Yes}` "
        "if the paragraph is correct, or `\\boxed{No}` if the paragraph is incorrect."
    )
    user = f"[Math Problem]\n\n{question}\n\n[Solution]\n\n"
    parts = list(trajectory) + [candidate]
    if parts and "the reasoning steps are:" in parts[0].lower():
        parts = parts[1:]
    for i, para in enumerate(parts):
        user += f"<paragraph_{i}>\n{para}\n</paragraph_{i}>\n\n"
    return {"system": system, "user": user}


def analyze_prompt(tokenizer, critic: dict, step_idx: int) -> str:
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": critic["system"]},
         {"role": "user", "content": critic["user"]}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return text.strip() + f"<analyze>\nLet's analyze the paragraph {step_idx} step by step: "


def judgment_prompt(analyze_input: str, analyze_text: str) -> str:
    return analyze_input + analyze_text + "<output>\n**Judgment**: $\\boxed"


def verifier_prompt(question: str, ground_truth: str, student_answer: str) -> str:
    return (
        f"User: ### Question: {question}\n\n"
        f"### Ground Truth Answer: {ground_truth}\n\n"
        f"### Student Answer: {student_answer}\n\n"
        "For the above question, please verify if the student's answer is equivalent to the ground truth answer.\n"
        "Do not solve the question by yourself; just check if the student's answer is equivalent to the ground truth answer.\n"
        'If the student\'s answer is correct, output "Final Decision: Yes". '
        'If the student\'s answer is incorrect, output "Final Decision: No". Assistant:'
    )


_BOX = re.compile(r"\\boxed\s*\{")


def boxed_values(text: str) -> list[str]:
    """Brace-matched \\boxed{...} extraction; returns interior text."""
    out = []
    for m in _BOX.finditer(text or ""):
        i = m.end()
        depth = 1
        j = i
        while j < len(text) and depth:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            out.append(text[i:j - 1].strip())
    return out


def extract_answer(text: str) -> str:
    """Prefer the frozen final-answer marker; fall back to the last boxed value."""
    markers = list(re.finditer(r"(?i)(?<![\"'])\bthe answer is\b(?![\"'])", text or ""))
    if markers:
        ans = text[markers[-1].end():]
        ans = re.split(r"\nStep\s*\d+\s*:", ans, maxsplit=1)[0]
        return ans.strip().strip("$").strip().rstrip(".").strip()
    boxes = boxed_values(text or "")
    return boxes[-1] if boxes else ""


def has_final_answer(text: str) -> bool:
    markers = list(re.finditer(r"(?i)(?<![\"'])\bthe answer is\b(?![\"'])", text or ""))
    return bool(markers and text[markers[-1].end():].strip())


def parse_decision(text: str) -> tuple[str | None, bool]:
    """Return (Yes/No/None, ambiguous). Strictly targets the Final Decision field."""
    matches = re.findall(r"Final\s+Decision\s*:\s*(Yes|No)", text or "", flags=re.I)
    if not matches:
        return None, True
    labels = {m.capitalize() for m in matches}
    return (None, True) if len(labels) > 1 else (matches[-1].capitalize(), False)


def strip_step_prefix(step: str) -> str:
    return re.sub(r"^Step\s*\d+\s*:\s*", "", step or "")


def to_step_string(step_idx: int, text: str) -> str:
    return f"Step{step_idx}: {(text or '').strip()}"
