"""Smart wait strategies — replaces hardcoded waits with intelligent detection."""

from __future__ import annotations

import logging

from playwright.async_api import Page

logger = logging.getLogger(__name__)


async def wait_for_stable(
    page: Page,
    timeout_ms: int = 10000,
    poll_ms: int = 300,
) -> None:
    """Wait for a page to reach a stable state using multiple signals.

    Strategy order:
    1. Angular testability API (zone.js stability)
    2. Network idle (no pending requests for 500ms)
    3. DOM mutation stability (no changes for 500ms)
    4. Hard timeout fallback
    """
    # Strategy 1: Angular testability (fastest, most reliable for Angular apps)
    try:
        is_angular = await page.evaluate("() => !!window.getAllAngularTestabilities")
        if is_angular:
            stable = await page.evaluate("""(timeout) => {
                return new Promise((resolve) => {
                    const testabilities = window.getAllAngularTestabilities();
                    if (!testabilities.length) { resolve(false); return; }
                    testabilities[0].whenStable(() => resolve(true));
                    setTimeout(() => resolve(false), timeout);
                });
            }""", min(timeout_ms, 5000))
            if stable:
                # Small buffer for Angular rendering after zone stabilizes
                await page.wait_for_timeout(100)
                return
    except Exception:
        pass

    # Strategy 2: Network idle
    try:
        await page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 5000))
        return
    except Exception:
        pass

    # Strategy 3: DOM mutation stability
    try:
        await page.evaluate("""(pollMs) => {
            return new Promise((resolve) => {
                let lastMutationTime = Date.now();
                const observer = new MutationObserver(() => {
                    lastMutationTime = Date.now();
                });
                observer.observe(document.body, {
                    childList: true, subtree: true, attributes: true
                });
                const check = () => {
                    if (Date.now() - lastMutationTime >= pollMs) {
                        observer.disconnect();
                        resolve(true);
                    } else {
                        setTimeout(check, 100);
                    }
                };
                setTimeout(check, pollMs);
                // Hard limit
                setTimeout(() => { observer.disconnect(); resolve(false); }, 5000);
            });
        }""", poll_ms)
    except Exception:
        # Strategy 4: Fallback
        await page.wait_for_timeout(min(1000, timeout_ms))


async def wait_after_action(
    page: Page,
    action_type: str = "",
    timeout_ms: int = 5000,
) -> None:
    """Wait after an action with action-type-appropriate strategy.

    Navigate/spa_navigate: full stability wait.
    Click: short stability wait (might trigger transitions).
    Fill/select: minimal wait (just change detection).
    """
    match action_type:
        case "navigate" | "spa_navigate":
            await wait_for_stable(page, timeout_ms=timeout_ms)
        case "click":
            await wait_for_stable(page, timeout_ms=min(timeout_ms, 3000), poll_ms=200)
        case "fill" | "select":
            await page.wait_for_timeout(100)
        case _:
            await page.wait_for_timeout(200)
