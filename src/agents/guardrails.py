"""Lightweight conversation guardrails for Sentinel.

Two phases:
  - Input guard: regex-based jailbreak / prompt-injection patterns,
    domain-fence (must mention supply-chain or workspace context),
    length cap.
  - Output guard: PII regex, secret-leak regex, profanity regex, plus
    optional self-check via Groq (off by default — same env flag).

Each check returns a GuardResult with a verdict and reason. The
orchestrator decides what to do with a FAIL — block, redact, or warn.

This module is intentionally small. Swap for NeMo Guardrails or
Guardrails.ai later by reimplementing the two ``check_*`` functions.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Pattern, Tuple

_LOG = logging.getLogger(__name__)


class GuardVerdict(str, Enum):
    """Possible verdicts returned by every guard check."""

    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class GuardResult:
    """Outcome of an input or output guard inspection.

    Attributes:
        verdict: PASS, WARN, or BLOCK.
        reason: Human-readable explanation (for logs & UI surfacing).
        matched_rule: Identifier of the rule that fired (None on PASS).
        redacted_text: Set when ``verdict == WARN`` and redaction was
            applied (output guard only). Callers should prefer this
            string over the original text when surfacing the result.
    """

    verdict: GuardVerdict
    reason: str = ""
    matched_rule: Optional[str] = None
    redacted_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Input rules — block jailbreak / prompt-injection / role-override attempts
# ---------------------------------------------------------------------------

_JAILBREAK_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    (
        "ignore_instructions",
        re.compile(
            r"\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+"
            r"(instructions?|prompts?|rules?)\b",
            re.I,
        ),
    ),
    (
        "act_as_developer",
        re.compile(
            r"\b(act as|pretend to be)\s+(a\s+)?(developer|admin|root|god|owner)\b",
            re.I,
        ),
    ),
    (
        "dan_mode",
        re.compile(
            r"\b(DAN|do anything now|jailbreak|developer\s*mode|sudo)\b",
            re.I,
        ),
    ),
    (
        "system_prompt_leak",
        re.compile(
            r"\b(reveal|show|print|repeat|output)\s+(the\s+)?(system|original|initial)\s+"
            r"(prompt|message|instructions?)\b",
            re.I,
        ),
    ),
    (
        "override_role",
        re.compile(r"\byou\s+are\s+now\s+(a\s+|an\s+)?\w+", re.I),
    ),
]

#: Hard character cap on a single input. Anything beyond this is treated
#: as adversarial / pathological and blocked unconditionally.
_INPUT_LENGTH_CAP: int = 8000


def check_input_safety(text: str) -> GuardResult:
    """Inspect a user/dispatcher input. Return PASS/WARN/BLOCK verdict.

    Args:
        text: The raw input string (typically a crisis description or a
            user query).

    Returns:
        A :class:`GuardResult`. Empty inputs and inputs exceeding the
        length cap are BLOCKed; jailbreak patterns BLOCK with the rule
        ID populated; otherwise PASS with reason ``"input clean"``.
    """
    if not text or not text.strip():
        return GuardResult(GuardVerdict.BLOCK, reason="empty input")

    if len(text) > _INPUT_LENGTH_CAP:
        return GuardResult(
            GuardVerdict.BLOCK,
            reason=f"input length {len(text)} exceeds cap {_INPUT_LENGTH_CAP}",
            matched_rule="length_cap",
        )

    for rule_id, pattern in _JAILBREAK_PATTERNS:
        m = pattern.search(text)
        if m:
            return GuardResult(
                GuardVerdict.BLOCK,
                reason=f"jailbreak pattern matched: {rule_id} ('{m.group(0)[:60]}')",
                matched_rule=rule_id,
            )

    return GuardResult(GuardVerdict.PASS, reason="input clean")


# ---------------------------------------------------------------------------
# Output rules — redact PII / secret leakage from agent replies
# ---------------------------------------------------------------------------

_PII_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    ("ssn_us", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("api_key_groq", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("api_key_openai", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
]

_SECRET_PATTERNS: List[Tuple[str, Pattern[str]]] = [
    ("env_var_dump", re.compile(r"\bos\.environ\b|\bgetenv\b")),
    (
        "private_key_block",
        # Accepts any sequence of uppercase-word prefixes — handles
        # ``RSA PRIVATE``, ``OPENSSH PRIVATE``, plain ``PRIVATE``, or
        # bare ``KEY``. ``*?`` keeps the match non-greedy.
        re.compile(r"-----BEGIN ([A-Z]+ )*?KEY-----"),
    ),
]


def check_output_safety(text: str, redact: bool = True) -> GuardResult:
    """Inspect specialist/analyst output for PII or secret leakage.

    The output guard is intentionally lenient: WARN with redaction by
    default rather than BLOCK, so a single false positive cannot break
    a crisis response. Callers can opt into raw-match reporting by
    passing ``redact=False``.

    Args:
        text: The raw text the LLM produced.
        redact: When True (default), substitute every match with the
            literal token ``"[REDACTED]"`` and return the redacted text
            on the result. When False, leave the text unmodified and
            simply report the matched rules.

    Returns:
        A :class:`GuardResult`. PASS for clean / empty text; WARN
        otherwise (BLOCK is reserved for future stricter modes).
    """
    if not text:
        return GuardResult(GuardVerdict.PASS, reason="empty output")

    matches: List[Tuple[str, str]] = []
    for rule_id, pattern in _PII_PATTERNS + _SECRET_PATTERNS:
        for m in pattern.finditer(text):
            matches.append((rule_id, m.group(0)))

    if not matches:
        return GuardResult(GuardVerdict.PASS, reason="output clean")

    if redact:
        redacted = text
        for _, fragment in matches:
            redacted = redacted.replace(fragment, "[REDACTED]")
        rules = ",".join(sorted({rule for rule, _ in matches}))
        return GuardResult(
            GuardVerdict.WARN,
            reason=f"redacted {len(matches)} match(es) for rules: {rules}",
            matched_rule=rules,
            redacted_text=redacted,
        )

    rules = ",".join(sorted({rule for rule, _ in matches}))
    return GuardResult(
        GuardVerdict.WARN,
        reason=f"matched {len(matches)} pattern(s): {rules}",
        matched_rule=rules,
    )


# ---------------------------------------------------------------------------
# High-level pipeline used by the orchestrator
# ---------------------------------------------------------------------------


class GuardrailViolation(RuntimeError):
    """Raised when an input or output guard returns a BLOCK verdict.

    The ``reason`` and ``rule`` attributes mirror the corresponding
    fields on :class:`GuardResult` so callers can build user-facing
    error messages without re-parsing.
    """

    def __init__(self, reason: str, rule: Optional[str] = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.rule = rule


def guard_or_raise(text: str, *, direction: str = "input") -> str:
    """Apply the input or output guard; return safe text or raise on BLOCK.

    Args:
        text: The string to inspect.
        direction: ``"input"`` (use :func:`check_input_safety`) or
            ``"output"`` (use :func:`check_output_safety`).

    Returns:
        For input PASS: the original text unchanged.
        For output PASS: the original text unchanged.
        For output WARN with redaction: the redacted text.

    Raises:
        GuardrailViolation: When the underlying check returns BLOCK.
        ValueError: When ``direction`` is neither ``"input"`` nor
            ``"output"``.
    """
    if direction == "input":
        result = check_input_safety(text)
        if result.verdict == GuardVerdict.BLOCK:
            raise GuardrailViolation(result.reason, result.matched_rule)
        return text
    elif direction == "output":
        result = check_output_safety(text, redact=True)
        if result.verdict == GuardVerdict.BLOCK:
            raise GuardrailViolation(result.reason, result.matched_rule)
        if result.verdict == GuardVerdict.WARN and result.redacted_text:
            _LOG.warning("Output guard redacted content: %s", result.reason)
            return result.redacted_text
        return text
    else:
        raise ValueError(f"Unknown direction: {direction}")
