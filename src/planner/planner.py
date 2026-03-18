"""AI-driven test plan generation."""

from __future__ import annotations

import json
import logging
import time
import uuid

from src.ai.client import AIClient
from src.ai.prompts.planning import PLANNING_SYSTEM_PROMPT, build_planning_prompt
from src.models.config import FrameworkConfig
from src.models.coverage import CoverageGapReport, CoverageRegistry
from src.models.site_model import SiteModel
from src.models.test_plan import Action, Assertion, TestCase, TestPlan

from .schema_validator import validate_test_plan

logger = logging.getLogger(__name__)

# Well-known placeholder tokens for credential injection.
# The LLM is instructed to use these in Action.value fields;
# _inject_credentials() replaces them with real config values after parsing.
AUTH_PLACEHOLDER_USERNAME = "{{auth_username}}"
AUTH_PLACEHOLDER_PASSWORD = "{{auth_password}}"
AUTH_PLACEHOLDER_LOGIN_URL = "{{auth_login_url}}"


class Planner:
    """Generates test plans using AI analysis of the site model and coverage gaps."""

    def __init__(self, config: FrameworkConfig, ai_client: AIClient):
        self.config = config
        self.ai_client = ai_client

    def generate_plan(
        self,
        site_model: SiteModel,
        coverage_registry: CoverageRegistry | None = None,
        gap_report: CoverageGapReport | None = None,
        git_context_data: dict[str, str] | None = None,
    ) -> TestPlan:
        """Generate a test plan from the site model and coverage data.

        For large sites (>15 pages), splits into chunks and makes separate
        AI calls per chunk, then merges the results.
        """
        from .chunker import chunk_site_model, allocate_test_budget

        logger.info("Generating test plan for %s (%d pages)",
                    site_model.base_url, len(site_model.pages))

        chunk_threshold = 15
        chunks = chunk_site_model(site_model, max_pages_per_chunk=chunk_threshold)

        if len(chunks) <= 1:
            # Small site — single AI call
            plan = self._generate_plan_single(
                site_model, gap_report, git_context_data,
                max_tests=self.config.max_tests_per_run,
            )
            plan = self._append_security_tests(site_model, plan)
            return plan

        # Large site — chunked planning (parallel AI calls)
        budgets = allocate_test_budget(chunks, self.config.max_tests_per_run)
        logger.info("Chunked planning: %d chunks, budgets=%s", len(chunks), budgets)

        # Prepare chunk inputs
        chunk_inputs = []
        for i, (chunk, budget) in enumerate(zip(chunks, budgets)):
            chunk_page_ids = {p.page_id for p in chunk}
            chunk_graph = {
                k: [t for t in v if t["target_state_id"] in chunk_page_ids]
                for k, v in site_model.state_graph.items()
                if k in chunk_page_ids
            }
            chunk_model = SiteModel(
                base_url=site_model.base_url,
                pages=chunk,
                api_endpoints=site_model.api_endpoints,
                auth_flow=site_model.auth_flow,
                state_graph=chunk_graph,
            )

            sibling_summary = []
            for j, other_chunk in enumerate(chunks):
                if j == i:
                    continue
                titles = [p.title or p.url for p in other_chunk[:5]]
                sibling_summary.append(f"Chunk {j + 1}: {', '.join(titles)}")

            extra_hints = list(self.config.hints)
            if sibling_summary:
                extra_hints.append(
                    f"This is chunk {i + 1}/{len(chunks)}. "
                    f"Other app areas: {'; '.join(sibling_summary)}"
                )
            chunk_inputs.append((i, chunk_model, budget, extra_hints))

        # Run chunk planning in parallel using threads (AI client is sync)
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_test_cases: list[TestCase] = []
        seen_signatures: set[str] = set()
        max_workers = min(len(chunk_inputs), 4)

        def _plan_chunk(args):
            idx, model, budget, hints = args
            logger.info("Planning chunk %d/%d: %d pages, budget=%d",
                        idx + 1, len(chunks), len(model.pages), budget)
            try:
                return self._generate_plan_single(
                    model, gap_report, git_context_data,
                    max_tests=budget, extra_hints=hints,
                )
            except Exception as e:
                logger.warning("Chunk %d planning failed: %s, using fallback", idx + 1, e)
                return self._generate_fallback_plan(model)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_plan_chunk, ci): ci[0] for ci in chunk_inputs}
            for future in as_completed(futures):
                chunk_idx = futures[future]
                try:
                    plan = future.result()
                    for tc in plan.test_cases:
                        sig = tc.coverage_signature
                        if sig not in seen_signatures:
                            seen_signatures.add(sig)
                            all_test_cases.append(tc)
                except Exception as e:
                    logger.error("Chunk %d future failed: %s", chunk_idx + 1, e)

        # Re-number test IDs
        for idx, tc in enumerate(all_test_cases):
            tc.test_id = f"tc_{idx + 1:03d}"

        plan = TestPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            target_url=site_model.base_url,
            test_cases=all_test_cases[:self.config.max_tests_per_run],
            estimated_duration_seconds=len(all_test_cases) * 15,
        )
        plan = self._inject_credentials(plan)
        plan = self._append_security_tests(site_model, plan)
        logger.info("Chunked plan: %d test cases from %d chunks", len(plan.test_cases), len(chunks))
        return plan

    def _generate_plan_single(
        self,
        site_model: SiteModel,
        gap_report: CoverageGapReport | None = None,
        git_context_data: dict[str, str] | None = None,
        max_tests: int = 20,
        extra_hints: list[str] | None = None,
    ) -> TestPlan:
        """Generate a plan for a single chunk of pages."""
        site_summary = self._summarize_site_model(site_model)

        gaps_summary = "{}"
        if gap_report:
            gaps_summary = gap_report.model_dump_json(indent=2)

        config_summary = (
            f"Categories: {', '.join(self.config.categories)}\n"
            f"Max tests: {max_tests}\n"
            f"Visual diff tolerance: {self.config.visual_diff_tolerance}\n"
            f"Viewports: {json.dumps([v.model_dump() for v in self.config.viewports])}\n"
        )

        hints = list(self.config.hints)
        if extra_hints:
            hints.extend(extra_hints)

        user_message = build_planning_prompt(
            site_model_json=site_summary,
            coverage_gaps_json=gaps_summary,
            config_summary=config_summary,
            hints=hints,
            max_tests=max_tests,
            git_context_data=git_context_data,
        )

        try:
            plan_data = self.ai_client.complete_json(
                system_prompt=PLANNING_SYSTEM_PROMPT,
                user_message=user_message,
                max_tokens=self.config.ai_max_planning_tokens,
            )
        except Exception as e:
            logger.error("AI planning failed: %s. Generating fallback plan.", e)
            return self._inject_credentials(self._generate_fallback_plan(site_model))

        try:
            plan = self._parse_plan(plan_data, site_model)
            errors = validate_test_plan(plan)
            if errors:
                logger.warning("Plan validation warnings: %s", errors)
                plan.test_cases = [
                    tc for tc in plan.test_cases
                    if not any(tc.test_id in err for err in errors)
                ]
            plan = self._inject_credentials(plan)
            return plan
        except Exception as e:
            logger.error("Failed to parse AI plan: %s. Using fallback.", e)
            return self._inject_credentials(self._generate_fallback_plan(site_model))

    def _summarize_site_model(self, site_model: SiteModel) -> str:
        """Create a condensed version of the site model for the AI prompt."""
        summary = {
            "base_url": site_model.base_url,
            "pages": [],
            "api_endpoints_count": len(site_model.api_endpoints),
            "has_auth": site_model.auth_flow is not None,
        }

        for page in site_model.pages[:30]:  # Limit pages
            page_summary = {
                "page_id": page.page_id,
                "url": page.url,
                "page_type": page.page_type,
                "title": page.title,
                "auth_required": page.auth_required,
                "interactive_elements_count": sum(1 for e in page.elements if e.is_interactive),
                "forms": [
                    {
                        "form_id": f.form_id,
                        "method": f.method,
                        "form_pattern": f.form_pattern,
                        **({"wizard_steps": f.wizard_steps} if f.wizard_steps else {}),
                        "fields": [
                            {
                                "name": ff.name,
                                "type": ff.field_type,
                                "required": ff.required,
                                **({"interaction_pattern": ff.interaction_pattern} if ff.interaction_pattern != "standard" else {}),
                                **({"interaction_steps": ff.interaction_steps} if ff.interaction_steps else {}),
                                **({"validation_rules": ff.validation_rules} if ff.validation_rules else {}),
                            }
                            for ff in f.fields
                        ],
                        "submit_selector": f.submit_selector,
                    }
                    for f in page.forms
                ],
                "key_elements": [
                    {
                        "selector": e.selector,
                        "type": e.element_type,
                        "text": e.text_content[:50],
                    }
                    for e in page.elements[:15]
                    if e.is_interactive
                ],
            }
            if page.fingerprint:
                page_summary["fingerprint"] = page.fingerprint
                page_summary["parent_page_id"] = page.parent_page_id
                # Only include selector and description, not full action dict
                if page.trigger_action:
                    page_summary["trigger_action"] = {
                        "selector": page.trigger_action.get("selector", ""),
                        "description": page.trigger_action.get("description", "")[:60],
                    }

            summary["pages"].append(page_summary)

        # Include a condensed state graph (just state IDs and descriptions)
        if site_model.state_graph:
            condensed_graph = {}
            for src, transitions in site_model.state_graph.items():
                condensed_graph[src] = [
                    {
                        "target": t["target_state_id"],
                        "action": t.get("action", {}).get("description", "")[:60],
                    }
                    for t in transitions[:10]  # Limit transitions per state
                ]
            summary["state_graph"] = condensed_graph

        return json.dumps(summary, indent=2)

    def _parse_plan(self, data: dict, site_model: SiteModel) -> TestPlan:
        """Parse raw AI output into a TestPlan model."""
        plan_id = data.get("plan_id", f"plan_{uuid.uuid4().hex[:8]}")
        generated_at = data.get("generated_at", time.strftime("%Y-%m-%dT%H:%M:%SZ"))

        test_cases = []
        for tc_data in data.get("test_cases", []):
            try:
                preconditions = [Action(**a) for a in tc_data.get("preconditions", [])]
                steps = [Action(**a) for a in tc_data.get("steps", [])]
                assertions = [Assertion(**a) for a in tc_data.get("assertions", [])]

                tc = TestCase(
                    test_id=tc_data.get("test_id", f"tc_{uuid.uuid4().hex[:6]}"),
                    name=tc_data.get("name", "Unnamed test"),
                    description=tc_data.get("description", ""),
                    category=tc_data.get("category", "functional"),
                    priority=tc_data.get("priority", 3),
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
                logger.warning("Skipping invalid test case: %s", e)

        return TestPlan(
            plan_id=plan_id,
            generated_at=generated_at,
            target_url=site_model.base_url,
            test_cases=test_cases,
            estimated_duration_seconds=data.get("estimated_duration_seconds", len(test_cases) * 15),
            coverage_intent=data.get("coverage_intent", {}),
        )

    def _generate_fallback_plan(self, site_model: SiteModel) -> TestPlan:
        """Generate a basic test plan without AI when API is unavailable."""
        logger.info("Generating fallback plan (no AI)")
        test_cases = []
        tc_num = 0

        for page in site_model.pages:
            # Basic navigation test for each page
            tc_num += 1
            test_cases.append(TestCase(
                test_id=f"tc_fallback_{tc_num:03d}",
                name=f"Navigate to {page.title or page.url}",
                description=f"Verify {page.url} loads successfully",
                category="functional",
                priority=3,
                target_page_id=page.page_id,
                coverage_signature=f"navigate_{page.page_id}",
                steps=[Action(action_type="navigate", value=page.url, description=f"Go to {page.url}")],
                assertions=[
                    Assertion(assertion_type="url_matches", expected_value=page.url, description="URL loaded"),
                    Assertion(assertion_type="no_console_errors", description="No console errors"),
                ],
            ))

            # Visual baseline test
            if "visual" in self.config.categories:
                tc_num += 1
                test_cases.append(TestCase(
                    test_id=f"tc_fallback_{tc_num:03d}",
                    name=f"Visual check: {page.title or page.url}",
                    category="visual",
                    priority=4,
                    target_page_id=page.page_id,
                    coverage_signature=f"visual_{page.page_id}",
                    steps=[
                        Action(action_type="navigate", value=page.url, description=f"Go to {page.url}"),
                        Action(action_type="screenshot", description="Capture page"),
                    ],
                    assertions=[
                        Assertion(
                            assertion_type="screenshot_diff",
                            tolerance=self.config.visual_diff_tolerance,
                            description="Compare against baseline",
                        ),
                    ],
                ))

            # Form tests
            for form in page.forms:
                tc_num += 1
                steps = [
                    Action(action_type="navigate", value=page.url, description=f"Go to {page.url}"),
                ]
                for field in form.fields:
                    if field.field_type in ("text", "email", "password", "textarea"):
                        val = _test_value_for_type(field.field_type, field.name)
                        steps.append(Action(
                            action_type="fill", selector=field.selector,
                            value=val, description=f"Fill {field.name}",
                        ))
                    elif field.field_type == "select" and field.options:
                        steps.append(Action(
                            action_type="select", selector=field.selector,
                            value=field.options[0], description=f"Select {field.name}",
                        ))
                    elif field.field_type == "checkbox":
                        steps.append(Action(
                            action_type="click", selector=field.selector,
                            description=f"Check {field.name}",
                        ))

                if form.submit_selector:
                    steps.append(Action(
                        action_type="click", selector=form.submit_selector,
                        description="Submit form",
                    ))

                test_cases.append(TestCase(
                    test_id=f"tc_fallback_{tc_num:03d}",
                    name=f"Submit form on {page.title or page.url}",
                    category="functional",
                    priority=2,
                    target_page_id=page.page_id,
                    coverage_signature=f"form_submit_{form.form_id}",
                    steps=steps,
                    assertions=[Assertion(
                        assertion_type="no_console_errors",
                        description="No errors after submission",
                    )],
                ))

        # State navigation tests — for each interactive state, build steps
        # that replay the full action chain from the root URL to reach it.
        if site_model.state_graph:
            page_lookup = {p.page_id: p for p in site_model.pages}

            for page in site_model.pages:
                if not page.trigger_action or not page.parent_page_id:
                    continue

                # Build the action chain from root to this state
                steps = self._build_state_navigation_steps(
                    page, page_lookup, site_model.base_url,
                )
                if not steps:
                    continue

                tc_num += 1
                desc = page.trigger_action.get("description", page.page_id)[:60]
                test_cases.append(TestCase(
                    test_id=f"tc_fallback_{tc_num:03d}",
                    name=f"Navigate to {desc}",
                    description=f"Navigate to interactive state: {desc}",
                    category="functional",
                    priority=2,
                    target_page_id=page.page_id,
                    coverage_signature=f"journey:{page.parent_page_id}->{page.page_id}",
                    steps=steps,
                    assertions=[
                        Assertion(
                            assertion_type="page_loaded",
                            description="Page loaded after navigation",
                        ),
                    ],
                ))

        return TestPlan(
            plan_id=f"plan_fallback_{uuid.uuid4().hex[:8]}",
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            target_url=site_model.base_url,
            test_cases=test_cases[:self.config.max_tests_per_run],
            estimated_duration_seconds=len(test_cases) * 10,
        )

    def _append_security_tests(self, site_model: SiteModel, plan: TestPlan) -> TestPlan:
        """Run a separate security planning pass and append results."""
        if "security" not in self.config.categories:
            return plan

        # Allocate ~20% of budget for security, minimum 2
        existing_security = sum(1 for tc in plan.test_cases if tc.category == "security")
        remaining_budget = self.config.max_tests_per_run - len(plan.test_cases)
        security_budget = max(2, min(remaining_budget, len(plan.test_cases) // 4))

        if security_budget <= existing_security:
            return plan  # Already have enough security tests

        try:
            from .security_planner import SecurityPlanner

            sp = SecurityPlanner(self.config, self.ai_client)
            security_tests = sp.generate_security_tests(
                site_model, budget=security_budget - existing_security,
            )
            if security_tests:
                # Re-number security test IDs
                start_idx = len(plan.test_cases) + 1
                for i, tc in enumerate(security_tests):
                    tc.test_id = f"tc_{start_idx + i:03d}"
                plan.test_cases.extend(security_tests)
                logger.info("Appended %d security tests", len(security_tests))
        except Exception as e:
            logger.warning("Security planning failed: %s", e)

        return plan

    @staticmethod
    def _build_state_navigation_steps(
        target_page, page_lookup: dict, base_url: str,
    ) -> list[Action]:
        """Build the full step sequence to navigate to an interactive state.

        Walks the parent_page_id chain from the target state back to a root
        page (one with no trigger_action), then reverses to get the forward
        path. Each step in the chain replays the trigger_action click.
        """
        # Walk back to root, collecting (page, trigger_action) pairs
        chain = []
        current = target_page
        visited = set()
        while current and current.page_id not in visited:
            visited.add(current.page_id)
            if current.trigger_action:
                chain.append(current)
            if not current.parent_page_id:
                break
            current = page_lookup.get(current.parent_page_id)

        if not chain:
            return []

        # Reverse to get root-first order
        chain.reverse()

        # Start URL: the root page's URL (or base_url)
        root_parent = page_lookup.get(chain[0].parent_page_id)
        start_url = root_parent.url if root_parent else base_url

        steps = [
            Action(
                action_type="navigate",
                value=start_url,
                description=f"Go to {start_url}",
            ),
        ]

        for state in chain:
            action = state.trigger_action
            if action and action.get("selector"):
                steps.append(Action(
                    action_type=action.get("action_type", "click"),
                    selector=action["selector"],
                    description=action.get("description", action["selector"])[:80],
                ))
                # Add a small wait after each interaction for SPA transitions
                steps.append(Action(
                    action_type="wait",
                    value="2000",
                    description="Wait for state transition",
                ))

        return steps

    @staticmethod
    def _has_auth_placeholders(tc: TestCase) -> bool:
        """Check if a test case contains any unresolved auth placeholder tokens."""
        tokens = (AUTH_PLACEHOLDER_USERNAME, AUTH_PLACEHOLDER_PASSWORD, AUTH_PLACEHOLDER_LOGIN_URL)
        for action in tc.preconditions + tc.steps:
            if action.value and any(t in action.value for t in tokens):
                return True
        for assertion in tc.assertions:
            if assertion.expected_value and any(t in assertion.expected_value for t in tokens):
                return True
        return False

    def _inject_credentials(self, plan: TestPlan) -> TestPlan:
        """Replace auth placeholder tokens in the plan with real credentials.

        Walks all Action.value fields in preconditions and steps, and
        Assertion.expected_value fields, substituting well-known placeholder
        tokens with actual credentials from self.config.auth.

        When auth is not configured, any test cases that still contain
        auth placeholders are removed from the plan as a safety net.
        """
        auth = self.config.auth
        if not auth:
            original_count = len(plan.test_cases)
            plan.test_cases = [
                tc for tc in plan.test_cases
                if not self._has_auth_placeholders(tc)
            ]
            removed = original_count - len(plan.test_cases)
            if removed:
                logger.info(
                    "Removed %d test case(s) with auth placeholders (no auth configured)",
                    removed,
                )
            return plan

        substitutions = {
            AUTH_PLACEHOLDER_USERNAME: auth.username,
            AUTH_PLACEHOLDER_PASSWORD: auth.password,
            AUTH_PLACEHOLDER_LOGIN_URL: auth.login_url,
        }

        sub_count = 0

        for tc in plan.test_cases:
            for action in tc.preconditions + tc.steps:
                if action.value:
                    new_value = action.value
                    for token, real_value in substitutions.items():
                        if token in new_value:
                            new_value = new_value.replace(token, real_value)
                    if new_value != action.value:
                        masked = new_value
                        if auth.password in masked:
                            masked = masked.replace(auth.password, "***")
                        logger.debug(
                            "Credential injection [%s]: '%s' -> '%s'",
                            tc.test_id, action.value, masked,
                        )
                        action.value = new_value
                        sub_count += 1

            for assertion in tc.assertions:
                if assertion.expected_value:
                    new_ev = assertion.expected_value
                    for token, real_value in substitutions.items():
                        if token in new_ev:
                            new_ev = new_ev.replace(token, real_value)
                    if new_ev != assertion.expected_value:
                        masked = new_ev
                        if auth.password in masked:
                            masked = masked.replace(auth.password, "***")
                        logger.debug(
                            "Credential injection [%s assertion]: '%s' -> '%s'",
                            tc.test_id, assertion.expected_value, masked,
                        )
                        assertion.expected_value = new_ev
                        sub_count += 1

        if sub_count > 0:
            logger.info("Injected credentials into %d action/assertion fields", sub_count)
        else:
            logger.debug("No credential placeholders found in plan")

        return plan


def _test_value_for_type(field_type: str, name: str) -> str:
    """Generate realistic test data based on field type/name."""
    name_lower = name.lower()
    if field_type == "email" or "email" in name_lower:
        return "test@example.com"
    if field_type == "password" or "password" in name_lower:
        return "TestP@ssw0rd123"
    if "phone" in name_lower or "tel" in name_lower:
        return "+1-555-000-1234"
    if "name" in name_lower:
        return "Test User"
    if "url" in name_lower or "website" in name_lower:
        return "https://example.com"
    if "zip" in name_lower or "postal" in name_lower:
        return "90210"
    return "Test input value"
