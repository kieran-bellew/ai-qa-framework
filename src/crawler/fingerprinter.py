"""State fingerprinting for SPA UI states."""

from __future__ import annotations

import hashlib
import json

from playwright.async_api import Page


async def fingerprint_state(page: Page) -> str:
    """Collect visible UI signals and return a SHA-256 hash (first 16 chars).

    Signals collected:
    - Visible heading text (h1-h3), sorted
    - Active/selected elements
    - Visible form field names+types, sorted
    - Tab/panel structure
    - Modal/dialog presence
    - URL path + hash (excluding query params)
    """
    data = await page.evaluate("""() => {
        const trunc = (s, n) => (s || '').trim().substring(0, n);

        // Headings
        const headings = Array.from(document.querySelectorAll('h1, h2, h3'))
            .filter(el => el.offsetParent !== null)
            .map(el => trunc(el.textContent, 50))
            .filter(Boolean)
            .sort();

        // Active/selected elements
        const active = Array.from(document.querySelectorAll(
            '[aria-selected="true"], .active, .selected, [aria-current]'
        ))
            .map(el => trunc(el.textContent, 50))
            .filter(Boolean)
            .sort();

        // Visible form fields
        const fields = Array.from(document.querySelectorAll(
            'input, textarea, select'
        ))
            .filter(el => el.offsetParent !== null)
            .map(el => {
                const name = el.name || el.id || el.getAttribute('aria-label') || '';
                return trunc(name, 50) + ':' + (el.type || el.tagName.toLowerCase());
            })
            .filter(Boolean)
            .sort();

        // Tab/panel structure
        const tabs = Array.from(document.querySelectorAll('[role="tablist"] [role="tab"]'))
            .map(el => trunc(el.textContent, 50))
            .filter(Boolean)
            .sort();

        // Modal/dialog presence
        const modals = Array.from(document.querySelectorAll(
            '[role="dialog"], .modal.show, dialog[open]'
        ))
            .map(el => trunc(el.getAttribute('aria-label') || el.querySelector('h1,h2,h3')?.textContent || 'dialog', 50))
            .sort();

        // URL path + hash (no query params)
        const loc = window.location;
        const urlKey = loc.pathname + (loc.hash || '');

        return {
            headings,
            active,
            fields,
            tabs,
            modals,
            url: urlKey,
        };
    }""")

    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def state_id_from_fingerprint(url: str, fingerprint: str) -> str:
    """Generate a state ID from normalized URL and fingerprint.

    Returns MD5 hash (first 12 chars), matching the existing page_id length.
    """
    from src.url_utils import normalize_url

    normalized = normalize_url(url)
    combined = f"{normalized}|{fingerprint}"
    return hashlib.md5(combined.encode()).hexdigest()[:12]
