"""Dismiss Angular CDK overlay backdrops that block pointer events."""

from __future__ import annotations

from playwright.async_api import Page


async def dismiss_cdk_overlays(page: Page) -> None:
    """Dismiss open Angular CDK overlays/menus/dialogs that block clicks."""
    try:
        dismissed = await page.evaluate("""() => {
            let count = 0;
            document.querySelectorAll('.cdk-overlay-backdrop').forEach(el => {
                el.click();
                count++;
            });
            return count;
        }""")
        if dismissed:
            await page.wait_for_timeout(300)
    except Exception:
        pass
    try:
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(200)
    except Exception:
        pass
