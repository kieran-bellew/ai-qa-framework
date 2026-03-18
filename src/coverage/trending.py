"""Test result trending — computes flaky scores, reliability, and trend detection."""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from src.models.coverage import CoverageRegistry, SignatureRecord, TestResultSummary

logger = logging.getLogger(__name__)

_MIN_HISTORY_FOR_TREND = 3


class TrendingReport(BaseModel):
    flaky_tests: list[tuple[str, str, float]] = Field(default_factory=list)  # (page_id, signature, flaky_score)
    unreliable_tests: list[tuple[str, str, float]] = Field(default_factory=list)  # (page_id, signature, reliability)
    regressions: list[tuple[str, str]] = Field(default_factory=list)
    improving: list[tuple[str, str]] = Field(default_factory=list)


def compute_flaky_score(history: list[TestResultSummary]) -> float:
    """Fraction of consecutive result transitions (pass→fail or fail→pass)."""
    if len(history) < _MIN_HISTORY_FOR_TREND:
        return 0.0
    transitions = 0
    for i in range(1, len(history)):
        if history[i].result != history[i - 1].result:
            transitions += 1
    return round(transitions / (len(history) - 1), 3)


def compute_reliability_score(history: list[TestResultSummary], window: int = 10) -> float:
    """Pass rate over the last `window` runs."""
    recent = history[-window:]
    if not recent:
        return 1.0
    passed = sum(1 for h in recent if h.result == "pass")
    return round(passed / len(recent), 3)


def detect_trend(history: list[TestResultSummary]) -> str:
    """Classify the trend for a signature's history."""
    if len(history) < _MIN_HISTORY_FOR_TREND:
        return "stable"

    recent = [h.result for h in history[-4:]]

    # Regression: was passing, now failing
    if len(recent) >= 4 and all(r == "pass" for r in recent[:2]) and all(r == "fail" for r in recent[2:]):
        return "regression"

    # Improving: was failing, now passing
    if len(recent) >= 4 and all(r == "fail" for r in recent[:2]) and all(r == "pass" for r in recent[2:]):
        return "improving"

    # Flaky: high transition rate
    flaky = compute_flaky_score(history)
    if flaky > 0.3:
        return "flaky"

    # Degrading: reliability declining
    if len(history) >= 6:
        first_half = compute_reliability_score(history[:len(history) // 2])
        second_half = compute_reliability_score(history[len(history) // 2:])
        if first_half - second_half > 0.2:
            return "degrading"

    return "stable"


def update_signature_trends(registry: CoverageRegistry) -> None:
    """Compute and store trending metrics on all signature records."""
    for page_cov in registry.pages.values():
        for cat_cov in page_cov.categories.values():
            for sig in cat_cov.signatures_tested:
                if len(sig.history) < _MIN_HISTORY_FOR_TREND:
                    continue
                sig.flaky_score = compute_flaky_score(sig.history)
                sig.reliability_score = compute_reliability_score(sig.history)
                sig.trend = detect_trend(sig.history)


def generate_trending_report(registry: CoverageRegistry) -> TrendingReport:
    """Generate a trending report across all signatures."""
    report = TrendingReport()

    for page_id, page_cov in registry.pages.items():
        for cat_cov in page_cov.categories.values():
            for sig in cat_cov.signatures_tested:
                if len(sig.history) < _MIN_HISTORY_FOR_TREND:
                    continue

                flaky = compute_flaky_score(sig.history)
                reliability = compute_reliability_score(sig.history)
                trend = detect_trend(sig.history)

                if flaky > 0.3:
                    report.flaky_tests.append((page_id, sig.signature, flaky))
                if reliability < 0.7:
                    report.unreliable_tests.append((page_id, sig.signature, reliability))
                if trend == "regression":
                    report.regressions.append((page_id, sig.signature))
                elif trend == "improving":
                    report.improving.append((page_id, sig.signature))

    # Sort by severity
    report.flaky_tests.sort(key=lambda x: x[2], reverse=True)
    report.unreliable_tests.sort(key=lambda x: x[2])

    return report
