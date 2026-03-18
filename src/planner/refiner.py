"""Test case refiner — sends failed tests back to AI for improvement."""

from __future__ import annotations

import json
import logging
import uuid

from src.ai.client import AIClient
from src.models.config import FrameworkConfig
from src.models.test_plan import Action, Assertion, TestCase
from src.models.test_result import RunResult, TestResult

logger = logging.getLogger(__name__)

_REFINE_SYSTEM_PROMPT = """You are a QA test refinement expert. A test case failed during execution. Your job is to analyze the failure and generate an improved version of the test.

CRITICAL: Return ONLY valid JSON. No markdown, no code fences.

Return exactly:
{
  "test_cases": [
    {
      "test_id": "string",
      "name": "string",
      "description": "string",
      "category": "string",
      "priority": 1-5,
      "target_page_id": "string",
      "coverage_signature": "string",
      "requires_auth": true,
      "preconditions": [],
      "steps": [{"action_type": "string", "selector": "string or null", "value": "string or null", "description": "string"}],
      "assertions": [{"assertion_type": "string", "selector": "string or null", "expected_value": "string or null", "description": "string"}],
      "timeout_seconds": 30
    }
  ]
}

When fixing tests:
- If a selector timed out, use a more resilient selector: role=, text=, label=, or data-testid
- If an assertion failed, make the assertion more flexible (use ai_evaluate or url_matches instead of text_equals)
- If the test has no steps, add proper navigation + interaction steps
- If an Angular Material dropdown was involved, use click to open + click_text for the option
- Keep the same coverage_signature so coverage tracking works across versions
- For SPA apps, use spa_navigate instead of navigate for in-app transitions
"""


class TestRefiner:
    """Refines failed tests using AI analysis of failure details."""

    def __init__(self, config: FrameworkConfig, ai_client: AIClient):
        self.config = config
        self.ai_client = ai_client

    def refine_failures(
        self,
        run_result: RunResult,
        original_tests: list[TestCase],
        max_refinements: int = 10,
    ) -> list[TestCase]:
        """Generate improved versions of failed test cases.

        Returns a list of refined TestCase objects that replace the originals.
        """
        failed_results = [
            tr for tr in run_result.test_results
            if tr.result in ("fail", "error")
        ]
        if not failed_results:
            return []

        # Map test_id -> original test case
        test_lookup = {tc.test_id: tc for tc in original_tests}

        # Batch failures for efficiency (up to max_refinements)
        to_refine = failed_results[:max_refinements]
        logger.info("Refining %d failed test cases", len(to_refine))

        failure_descriptions = []
        for tr in to_refine:
            tc = test_lookup.get(tr.test_id)
            if not tc:
                continue

            desc = {
                "test_id": tr.test_id,
                "name": tr.test_name,
                "category": tr.category,
                "target_page_id": tr.target_page_id,
                "coverage_signature": tr.coverage_signature,
                "failure_reason": tr.failure_reason or "",
                "original_steps": [
                    {"action_type": a.action_type, "selector": a.selector, "value": a.value, "description": a.description}
                    for a in tc.steps
                ],
                "original_assertions": [
                    {"assertion_type": a.assertion_type, "selector": a.selector, "expected_value": a.expected_value, "description": a.description}
                    for a in tc.assertions
                ],
                "step_failures": [
                    {"step": s.step_index, "action": s.action_type, "selector": s.selector, "error": s.error_message}
                    for s in (tr.step_results or [])
                    if s.status != "pass"
                ],
                "assertion_failures": [
                    {"type": a.assertion_type, "message": a.message}
                    for a in (tr.assertion_results or [])
                    if not a.passed
                ],
            }
            failure_descriptions.append(desc)

        if not failure_descriptions:
            return []

        user_msg = (
            f"These {len(failure_descriptions)} test cases failed. "
            f"Generate improved versions that fix the failures.\n\n"
            f"```json\n{json.dumps(failure_descriptions, indent=2)}\n```\n\n"
            f"Return improved test cases as JSON."
        )

        try:
            result = self.ai_client.complete_json(
                system_prompt=_REFINE_SYSTEM_PROMPT,
                user_message=user_msg,
                max_tokens=self.config.ai_max_planning_tokens,
            )

            refined = []
            for tc_data in result.get("test_cases", []):
                try:
                    preconditions = [Action(**a) for a in tc_data.get("preconditions", [])]
                    steps = [Action(**a) for a in tc_data.get("steps", [])]
                    assertions = [Assertion(**a) for a in tc_data.get("assertions", [])]

                    if not steps and preconditions:
                        steps = preconditions
                        preconditions = []

                    tc = TestCase(
                        test_id=tc_data.get("test_id", f"ref_{uuid.uuid4().hex[:6]}"),
                        name=tc_data.get("name", "Refined test"),
                        description=tc_data.get("description", ""),
                        category=tc_data.get("category", "functional"),
                        priority=tc_data.get("priority", 2),
                        target_page_id=tc_data.get("target_page_id", ""),
                        coverage_signature=tc_data.get("coverage_signature", ""),
                        requires_auth=tc_data.get("requires_auth", True),
                        preconditions=preconditions,
                        steps=steps,
                        assertions=assertions,
                        timeout_seconds=tc_data.get("timeout_seconds", 30),
                    )
                    refined.append(tc)
                except Exception as e:
                    logger.debug("Skipping invalid refined test: %s", e)

            logger.info("AI refined %d test cases", len(refined))
            return refined

        except Exception as e:
            logger.warning("Test refinement failed: %s", e)
            return []
