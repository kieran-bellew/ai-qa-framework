"""AI-driven security test generation — separate planning pass for security tests."""

from __future__ import annotations

import json
import logging
import math
import time
import uuid

from src.ai.client import AIClient
from src.ai.prompts.security import SECURITY_SYSTEM_PROMPT, build_security_prompt
from src.models.config import FrameworkConfig
from src.models.site_model import SiteModel
from src.models.test_plan import Action, Assertion, TestCase

logger = logging.getLogger(__name__)


class SecurityPlanner:
    """Generates targeted security tests using AI analysis."""

    def __init__(self, config: FrameworkConfig, ai_client: AIClient):
        self.config = config
        self.ai_client = ai_client

    def generate_security_tests(
        self,
        site_model: SiteModel,
        budget: int,
    ) -> list[TestCase]:
        """Generate security test cases for the site."""
        if budget <= 0 or "security" not in self.config.categories:
            return []

        logger.info("Generating AI-driven security tests (budget=%d)", budget)

        # Build security-focused summary
        summary = self._build_security_summary(site_model)
        summary_json = json.dumps(summary, indent=2)

        user_message = build_security_prompt(
            site_model_json=summary_json,
            max_tests=budget,
            hints=self.config.hints,
        )

        try:
            plan_data = self.ai_client.complete_json(
                system_prompt=SECURITY_SYSTEM_PROMPT,
                user_message=user_message,
                max_tokens=self.config.ai_max_planning_tokens,
            )
        except Exception as e:
            logger.error("Security planning AI call failed: %s", e)
            return self._generate_fallback_security_tests(site_model, budget)

        try:
            test_cases = []
            for tc_data in plan_data.get("test_cases", []):
                try:
                    preconditions = [Action(**a) for a in tc_data.get("preconditions", [])]
                    steps = [Action(**a) for a in tc_data.get("steps", [])]
                    assertions = [Assertion(**a) for a in tc_data.get("assertions", [])]

                    tc = TestCase(
                        test_id=tc_data.get("test_id", f"sec_{uuid.uuid4().hex[:6]}"),
                        name=tc_data.get("name", "Security test"),
                        description=tc_data.get("description", ""),
                        category="security",
                        priority=tc_data.get("priority", 2),
                        target_page_id=tc_data.get("target_page_id", ""),
                        coverage_signature=tc_data.get("coverage_signature", ""),
                        requires_auth=tc_data.get("requires_auth", True),
                        preconditions=preconditions,
                        steps=steps,
                        assertions=assertions,
                        timeout_seconds=tc_data.get("timeout_seconds", 30),
                    )
                    test_cases.append(tc)
                except Exception as e:
                    logger.debug("Skipping invalid security test case: %s", e)

            logger.info("AI generated %d security tests", len(test_cases))
            return test_cases[:budget]
        except Exception as e:
            logger.error("Failed to parse security plan: %s", e)
            return self._generate_fallback_security_tests(site_model, budget)

    def _build_security_summary(self, site_model: SiteModel) -> dict:
        """Build a security-focused site summary."""
        forms = []
        for page in site_model.pages:
            for form in page.forms:
                form_info = {
                    "page_url": page.url,
                    "page_id": page.page_id,
                    "method": form.method,
                    "auth_required": page.auth_required,
                    "fields": [
                        {
                            "name": f.name,
                            "type": f.field_type,
                            "required": f.required,
                            "selector": f.selector,
                            "interaction_pattern": f.interaction_pattern,
                        }
                        for f in form.fields[:10]
                    ],
                    "submit_selector": form.submit_selector,
                }
                forms.append(form_info)

        api_endpoints = [
            {"url": ep.url, "method": ep.method, "status_codes": ep.status_codes_seen}
            for ep in site_model.api_endpoints[:20]
        ]

        protected_pages = [
            {"url": p.url, "page_id": p.page_id, "title": p.title}
            for p in site_model.pages
            if p.auth_required is True
        ]

        return {
            "base_url": site_model.base_url,
            "has_auth": site_model.auth_flow is not None,
            "forms": forms,
            "api_endpoints": api_endpoints,
            "protected_pages": protected_pages[:10],
            "page_count": len(site_model.pages),
        }

    def _generate_fallback_security_tests(
        self, site_model: SiteModel, budget: int,
    ) -> list[TestCase]:
        """Generate basic security tests without AI."""
        tests: list[TestCase] = []
        payloads = self.config.security_xss_payloads[:3]
        tc_num = 0

        for page in site_model.pages:
            if tc_num >= budget:
                break

            for form in page.forms:
                text_fields = [
                    f for f in form.fields
                    if f.field_type in ("text", "email", "textarea", "search")
                    and f.selector
                ]
                if not text_fields:
                    continue

                # XSS test for the first text field
                field = text_fields[0]
                payload = payloads[tc_num % len(payloads)] if payloads else "<script>alert(1)</script>"
                tc_num += 1

                steps = [
                    Action(action_type="navigate", value=page.url, description=f"Go to {page.url}"),
                    Action(action_type="fill", selector=field.selector, value=payload,
                           description=f"Inject XSS payload into {field.name}"),
                ]
                if form.submit_selector:
                    steps.append(Action(
                        action_type="click", selector=form.submit_selector,
                        description="Submit form",
                    ))

                tests.append(TestCase(
                    test_id=f"sec_fallback_{tc_num:03d}",
                    name=f"XSS test: {field.name} on {page.title or page.url}",
                    category="security",
                    priority=2,
                    target_page_id=page.page_id,
                    coverage_signature=f"xss_{field.selector}",
                    steps=steps,
                    assertions=[
                        Assertion(
                            assertion_type="ai_evaluate",
                            expected_value="The injected script tag should NOT be rendered as active HTML. The payload should be escaped or sanitized.",
                            description="Verify XSS payload is not reflected",
                        ),
                    ],
                ))

        return tests[:budget]
