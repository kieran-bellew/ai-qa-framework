"""Web storage restore — shared between crawler and executor."""

from __future__ import annotations

import logging

from playwright.async_api import Page

logger = logging.getLogger(__name__)


async def restore_web_storage(page: Page, storage: dict) -> None:
    """Restore localStorage/sessionStorage and reload for SPA token pickup."""
    try:
        await page.evaluate("""(storage) => {
            for (const [k, v] of storage.localStorage) localStorage.setItem(k, v);
            for (const [k, v] of storage.sessionStorage) sessionStorage.setItem(k, v);
        }""", storage)
        await page.reload(wait_until="domcontentloaded", timeout=15000)
        try:
            await page.wait_for_load_state("networkidle", timeout=10000)
        except Exception:
            await page.wait_for_timeout(2000)
    except Exception as e:
        logger.debug("Failed to restore web storage: %s", e)
