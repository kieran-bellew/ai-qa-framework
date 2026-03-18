"""Run post-auth actions on an authenticated page.

Shared between the crawler (before crawling) and the executor (before each test).
"""

from __future__ import annotations

import logging

from playwright.async_api import Page

logger = logging.getLogger(__name__)


async def run_post_auth_actions(page: Page, actions: list) -> None:
    """Execute post-auth actions to navigate through entry flows.

    Supports: click, click_text, click_label, fill, select, wait.
    """
    from src.auth.smart_auth import _type_into_field

    for i, action in enumerate(actions):
        act_type = action.action_type
        selector = action.selector
        value = action.value
        desc = action.description or f"step {i + 1}"
        logger.info("Post-auth action [%d]: %s %s — %s", i + 1, act_type, selector, desc)

        try:
            if act_type == "wait":
                ms = int(value) if value else 2000
                await page.wait_for_timeout(ms)
            elif act_type == "click":
                await page.wait_for_selector(selector, state="visible", timeout=10000)
                await page.click(selector, timeout=5000)
            elif act_type == "click_text":
                found = await page.evaluate("""(text) => {
                    const candidates = document.querySelectorAll(
                        'mat-option, [role="option"], [role="menuitem"], ' +
                        'button, a, li, [role="listbox"] > *, ' +
                        '.mat-mdc-option, .mdc-list-item'
                    );
                    for (const el of candidates) {
                        const elText = (el.textContent || '').trim();
                        if (elText === text || elText.startsWith(text)) {
                            if (el.id) return '#' + CSS.escape(el.id);
                            return null;
                        }
                    }
                    return null;
                }""", value)
                if found:
                    await page.click(found, timeout=5000)
                else:
                    await page.locator(f"text='{value}'").first.click(timeout=5000)
            elif act_type == "click_label":
                # Find and click an interactive element by its form field label.
                # Uses page.evaluate to locate, then clicks via JS to avoid
                # CDK overlay and dynamic ID issues.
                clicked = await page.evaluate("""(label) => {
                    const fields = document.querySelectorAll(
                        'mat-form-field, .form-group, .form-field'
                    );
                    for (const ff of fields) {
                        const lbl = ff.querySelector(
                            'mat-label, label, .mat-label, .form-label'
                        );
                        if (lbl && lbl.textContent.trim() === label) {
                            const target = ff.querySelector(
                                'mat-select, select, input, textarea'
                            );
                            if (!target) continue;

                            // For mat-select: click the trigger directly via JS
                            if (target.tagName.toLowerCase() === 'mat-select') {
                                const trigger = target.querySelector(
                                    '[role="combobox"], .mat-mdc-select-trigger, .mat-select-trigger'
                                );
                                if (trigger) { trigger.click(); return 'clicked_trigger'; }
                                target.click(); return 'clicked_select';
                            }

                            // For regular inputs: focus and click
                            target.focus();
                            target.click();
                            return 'clicked_input';
                        }
                    }
                    return null;
                }""", value)
                if not clicked:
                    logger.warning("Post-auth: no form field found with label '%s'", value)
            elif act_type == "fill":
                await page.wait_for_selector(selector, state="visible", timeout=10000)
                await _type_into_field(page, selector, value)
            elif act_type == "select":
                await page.wait_for_selector(selector, state="visible", timeout=10000)
                await page.select_option(selector, value)
            else:
                logger.warning("Unknown post-auth action type: %s", act_type)
                continue

            # Wait for SPA transition after each action
            from src.utils.smart_wait import wait_for_stable
            await wait_for_stable(page, timeout_ms=5000)

        except Exception as e:
            logger.error("Post-auth action failed [%d] %s: %s", i + 1, desc, e)
            break

    logger.info("Post-auth actions complete, now at %s", page.url)
