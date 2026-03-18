"""Interaction-based SPA crawler — discovers UI states by clicking elements."""

from __future__ import annotations

import logging
import re
from collections import deque
from pathlib import Path
from typing import Any

from playwright.async_api import BrowserContext, Page

from src.models.config import FrameworkConfig
from src.models.site_model import PageModel

from .element_extractor import extract_elements
from .fingerprinter import fingerprint_state, state_id_from_fingerprint
from .form_analyzer import analyze_forms

logger = logging.getLogger(__name__)

# Selectors commonly associated with destructive actions
_DESTRUCTIVE_SELECTORS = (
    "[class*='delete']", "[class*='remove']", "[class*='destroy']",
    "[data-action='delete']", "[data-action='remove']",
)


class InteractionCrawler:
    """Explores SPA states by interacting with UI elements (BFS)."""

    def __init__(
        self,
        config: FrameworkConfig,
        output_dir: Path,
        ai_client: Any = None,
        auth_storage: dict | None = None,
    ):
        self.config = config
        self.interaction_config = config.crawl.interaction
        self.output_dir = output_dir
        self.baselines_dir = output_dir / "baselines"
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        self._ai_client = ai_client
        self._auth_storage = auth_storage

    async def explore_states(
        self,
        context: BrowserContext,
        seed_pages: list[PageModel],
        existing_page: Page | None = None,
    ) -> tuple[list[PageModel], dict[str, list[dict]]]:
        """Explore UI states starting from seed pages via BFS interaction.

        Args:
            context: The browser context to use.
            seed_pages: Pages discovered by the URL crawler.
            existing_page: An existing page with auth state (avoids losing
                SPA in-memory tokens by reusing the crawl page).

        Returns:
            Tuple of (new_page_models, state_graph).
            state_graph maps state_id -> [{target_state_id, action}].
        """
        max_states = self.interaction_config.max_states
        max_depth = self.interaction_config.max_depth
        # Use word boundaries so "login" matches "Login" and "Log in"
        # but not "Login Page Header" substring false positives in containers
        skip_patterns = [
            re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE)
            for p in self.interaction_config.skip_text_patterns
        ]

        visited_fingerprints: dict[str, str] = {}  # fingerprint -> state_id
        state_graph: dict[str, list[dict]] = {}
        new_pages: list[PageModel] = []

        # Queue entries: (url, state_id, depth, action_chain)
        # action_chain is the list of actions to replay to reach this state
        queue: deque[tuple[str, str, int, list[dict]]] = deque()

        # Reuse the authenticated page if available to preserve SPA auth state
        page = existing_page or await context.new_page()
        owns_page = existing_page is None

        try:
            # Seed: fingerprint each seed page, register it
            for i, seed in enumerate(seed_pages):
                if len(visited_fingerprints) >= max_states:
                    break

                # For the first seed with an existing page, DON'T navigate —
                # the page is already authenticated and at the right state.
                # Navigating would re-bootstrap the SPA and kill in-memory auth.
                if i == 0 and existing_page is not None:
                    logger.info("Using existing authenticated page at %s", page.url)
                else:
                    try:
                        await page.goto(seed.url, wait_until="domcontentloaded", timeout=15000)
                        if self._auth_storage:
                            await self._restore_web_storage(page, self._auth_storage)
                        else:
                            await self._wait_for_stable(page)
                    except Exception as e:
                        logger.debug("Failed to load seed page %s: %s", seed.url, e)
                        continue

                current_url = page.url
                fp = await fingerprint_state(page)
                sid = state_id_from_fingerprint(current_url, fp)
                visited_fingerprints[fp] = sid
                state_graph[sid] = []
                queue.append((current_url, sid, 0, []))
                logger.info("Seed state: %s (fingerprint=%s) at %s", sid, fp[:8], current_url)

            # BFS
            while queue and len(visited_fingerprints) < max_states:
                current_url, current_sid, depth, action_chain = queue.popleft()

                if depth >= max_depth:
                    continue

                # Navigate to current state by replaying the action chain
                if not await self._navigate_to_state(page, current_url, action_chain):
                    logger.debug("Failed to reach state %s, skipping", current_sid)
                    continue

                # Find interactive elements
                clickables = await self._find_clickables(page)
                logger.info(
                    "State %s at depth %d: %d clickable elements",
                    current_sid, depth, len(clickables),
                )
                for c in clickables:
                    logger.debug("  Clickable: [%s] '%s'", c.get("selector", "?"), c.get("text", "")[:60])

                for clickable in clickables:
                    if len(visited_fingerprints) >= max_states:
                        break

                    text = clickable.get("text", "")
                    selector = clickable.get("selector", "")

                    # Skip unsafe elements
                    if self._is_unsafe(text, selector, skip_patterns):
                        logger.debug("Skipping unsafe: [%s] '%s'", selector, text[:40])
                        continue

                    # Fingerprint before click to detect no-ops
                    pre_fp = await fingerprint_state(page)

                    try:
                        # Dismiss any open CDK overlays/menus before clicking
                        await self._dismiss_overlays(page)

                        # Click the element
                        logger.debug("Clicking: [%s] '%s'", selector, text[:60])
                        await page.click(selector, timeout=5000)
                        await self._wait_for_stable(page)

                        # Fingerprint the resulting state
                        fp = await fingerprint_state(page)
                        action_info = {
                            "action_type": "click",
                            "selector": selector,
                            "description": text[:80] if text else selector,
                        }

                        # Skip if the click didn't change anything
                        if fp == pre_fp:
                            logger.debug("No state change for [%s]", selector)
                            continue

                        if fp in visited_fingerprints:
                            # Known state — just record transition edge
                            target_sid = visited_fingerprints[fp]
                            if current_sid in state_graph:
                                state_graph[current_sid].append({
                                    "target_state_id": target_sid,
                                    "action": action_info,
                                })
                            logger.debug("Known state %s reached via '%s'", target_sid, text[:40])
                        else:
                            # New state — register it
                            new_url = page.url
                            target_sid = state_id_from_fingerprint(new_url, fp)
                            visited_fingerprints[fp] = target_sid

                            # Record transition
                            state_graph.setdefault(current_sid, []).append({
                                "target_state_id": target_sid,
                                "action": action_info,
                            })
                            state_graph[target_sid] = []

                            # Check if a transient overlay (dropdown/menu) opened.
                            # If so, explore its options in-place before restoring,
                            # since they won't exist in the DOM after dismiss.
                            overlay_options = await self._find_overlay_options(page)
                            if overlay_options:
                                logger.info(
                                    "Overlay opened with %d options via '%s'",
                                    len(overlay_options), text[:40],
                                )
                                await self._explore_overlay_options(
                                    page, overlay_options, current_sid, current_url,
                                    action_chain, action_info, depth,
                                    visited_fingerprints, state_graph, new_pages,
                                    queue, skip_patterns, max_states,
                                )
                            else:
                                # Regular new state — extract and enqueue
                                elements = await extract_elements(page)
                                forms = await analyze_forms(page)

                                screenshot_path = ""
                                try:
                                    screenshot_path = str(
                                        self.baselines_dir / f"{target_sid}_screenshot.png"
                                    )
                                    await page.screenshot(path=screenshot_path, full_page=True)
                                except Exception:
                                    screenshot_path = ""

                                new_page = PageModel(
                                    page_id=target_sid,
                                    url=new_url,
                                    page_type="interactive",
                                    title=await page.title() or "",
                                    elements=elements,
                                    forms=forms,
                                    screenshot_path=screenshot_path,
                                    fingerprint=fp,
                                    parent_page_id=current_sid,
                                    trigger_action=action_info,
                                )
                                new_pages.append(new_page)

                                new_chain = action_chain + [action_info]
                                queue.append((new_url, target_sid, depth + 1, new_chain))
                                logger.info(
                                    "New state: %s (fp=%s) via '%s'",
                                    target_sid, fp[:8], text[:40] if text else selector,
                                )

                    except Exception as e:
                        logger.debug("Click failed [%s]: %s", selector, e)

                    # Restore state by replaying the action chain (not page.goto which kills SPA)
                    if not await self._navigate_to_state(page, current_url, action_chain):
                        logger.debug("Failed to restore state %s, moving to next state", current_sid)
                        break

                # --- Additional interaction strategies ---

                # Scroll discovery: scroll to bottom to trigger lazy loading
                if self.interaction_config.enable_scroll_discovery:
                    if len(visited_fingerprints) < max_states:
                        await self._scroll_discover(
                            page, current_sid, current_url, action_chain, depth,
                            visited_fingerprints, state_graph, new_pages, queue,
                            max_states,
                        )

                # Grid interaction: sort columns, paginate, expand rows
                if self.interaction_config.enable_grid_interaction:
                    if len(visited_fingerprints) < max_states:
                        await self._grid_discover(
                            page, current_sid, current_url, action_chain, depth,
                            visited_fingerprints, state_graph, new_pages, queue,
                            skip_patterns, max_states,
                        )

        finally:
            if owns_page:
                await page.close()

        return new_pages, state_graph

    async def _navigate_to_state(
        self, page: Page, url: str, action_chain: list[dict]
    ) -> bool:
        """Navigate to a state by loading the URL then replaying the action chain.

        For SPAs, page.goto() re-bootstraps the app at the root route. We
        restore auth storage so the SPA picks up the token, then replay each
        click in the chain to reach the target state.
        """
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            # Restore auth storage after full navigation
            if self._auth_storage:
                await self._restore_web_storage(page, self._auth_storage)
            else:
                await self._wait_for_stable(page)
        except Exception as e:
            logger.debug("Failed to navigate to %s: %s", url, e)
            return False

        # Replay action chain to reach the target state
        for action in action_chain:
            selector = action.get("selector", "")
            if not selector:
                continue
            try:
                await page.wait_for_selector(selector, state="visible", timeout=5000)
                await page.click(selector, timeout=5000)
                await self._wait_for_stable(page)
            except Exception as e:
                logger.debug(
                    "Action chain replay failed at '%s': %s", selector, e
                )
                return False

        return True

    async def _scroll_discover(
        self, page, current_sid, current_url, action_chain, depth,
        visited_fingerprints, state_graph, new_pages, queue, max_states,
    ) -> None:
        """Scroll to bottom to trigger lazy-loaded content."""
        try:
            pre_fp = await fingerprint_state(page)
            for scroll_round in range(3):
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1500)

                fp = await fingerprint_state(page)
                if fp != pre_fp and fp not in visited_fingerprints:
                    new_url = page.url
                    target_sid = state_id_from_fingerprint(new_url, fp)
                    visited_fingerprints[fp] = target_sid

                    elements = await extract_elements(page)
                    forms = await analyze_forms(page)

                    action_info = {
                        "action_type": "scroll",
                        "selector": "",
                        "description": f"Scroll to bottom (round {scroll_round + 1})",
                    }
                    new_pages.append(PageModel(
                        page_id=target_sid, url=new_url,
                        page_type="interactive",
                        title=await page.title() or "",
                        elements=elements, forms=forms,
                        fingerprint=fp, parent_page_id=current_sid,
                        trigger_action=action_info,
                    ))
                    state_graph.setdefault(current_sid, []).append({
                        "target_state_id": target_sid, "action": action_info,
                    })
                    state_graph[target_sid] = []
                    logger.info("New state from scroll: %s", target_sid)
                    break
                elif fp == pre_fp:
                    break  # No new content loaded
                pre_fp = fp
        except Exception as e:
            logger.debug("Scroll discovery failed: %s", e)
        finally:
            await page.evaluate("window.scrollTo(0, 0)")

    async def _grid_discover(
        self, page, current_sid, current_url, action_chain, depth,
        visited_fingerprints, state_graph, new_pages, queue,
        skip_patterns, max_states,
    ) -> None:
        """Interact with data grids: sort columns, paginate."""
        grid_budget = self.interaction_config.grid_states_per_page
        grid_found = 0

        try:
            grid_actions = await page.evaluate("""() => {
                const results = [];

                // Sort headers
                document.querySelectorAll(
                    '[mat-sort-header], th[sortable], [role="columnheader"][aria-sort], ' +
                    '.ag-header-cell-sortable, .p-sortable-column'
                ).forEach(el => {
                    if (el.offsetParent === null) return;
                    let sel = '';
                    if (el.id) sel = '#' + CSS.escape(el.id);
                    else if (el.getAttribute('mat-sort-header'))
                        sel = '[mat-sort-header="' + el.getAttribute('mat-sort-header') + '"]';
                    if (sel) results.push({
                        selector: sel,
                        text: (el.textContent || '').trim().substring(0, 40),
                        type: 'sort',
                    });
                });

                // Pagination buttons
                document.querySelectorAll(
                    '.mat-mdc-paginator-navigation-next, .mat-paginator-navigation-next, ' +
                    '.p-paginator-next, [aria-label="Next page"], ' +
                    'button.next-page, [class*="paginator"] button:last-of-type'
                ).forEach(el => {
                    if (el.offsetParent === null || el.disabled) return;
                    let sel = '';
                    if (el.id) sel = '#' + CSS.escape(el.id);
                    else if (el.getAttribute('aria-label'))
                        sel = '[aria-label="' + el.getAttribute('aria-label') + '"]';
                    else if (el.className && typeof el.className === 'string') {
                        const cls = el.className.trim().split(/\\s+/)[0];
                        if (cls) sel = 'button.' + CSS.escape(cls);
                    }
                    if (sel) results.push({selector: sel, text: 'Next page', type: 'paginate'});
                });

                return results.slice(0, 10);
            }""")

            if not grid_actions:
                return

            logger.debug("Found %d grid interactions", len(grid_actions))

            for action in grid_actions:
                if grid_found >= grid_budget or len(visited_fingerprints) >= max_states:
                    break

                selector = action.get("selector", "")
                text = action.get("text", "")

                if self._is_unsafe(text, selector, skip_patterns):
                    continue

                pre_fp = await fingerprint_state(page)
                try:
                    await self._dismiss_overlays(page)
                    await page.click(selector, timeout=3000)
                    await self._wait_for_stable(page)

                    fp = await fingerprint_state(page)
                    if fp != pre_fp and fp not in visited_fingerprints:
                        new_url = page.url
                        target_sid = state_id_from_fingerprint(new_url, fp)
                        visited_fingerprints[fp] = target_sid

                        elements = await extract_elements(page)
                        forms = await analyze_forms(page)
                        action_info = {
                            "action_type": "click",
                            "selector": selector,
                            "description": f"Grid: {action['type']} {text}",
                        }
                        new_pages.append(PageModel(
                            page_id=target_sid, url=new_url,
                            page_type="interactive",
                            title=await page.title() or "",
                            elements=elements, forms=forms,
                            fingerprint=fp, parent_page_id=current_sid,
                            trigger_action=action_info,
                        ))
                        state_graph.setdefault(current_sid, []).append({
                            "target_state_id": target_sid, "action": action_info,
                        })
                        state_graph[target_sid] = []
                        grid_found += 1
                        logger.info("New grid state: %s via '%s %s'",
                                   target_sid, action["type"], text[:30])
                except Exception as e:
                    logger.debug("Grid interaction failed [%s]: %s", selector, e)

                # Restore state
                if not await self._navigate_to_state(page, current_url, action_chain):
                    break

        except Exception as e:
            logger.debug("Grid discovery failed: %s", e)

    async def _find_clickables(self, page: Page) -> list[dict]:
        """Find visible interactive elements on the page.

        Covers standard HTML, ARIA roles, Angular Material, PrimeNG,
        and common SPA navigation patterns.
        """
        return await page.evaluate("""() => {
            const results = [];
            const seen = new Set();

            // Broad selector covering standard, ARIA, Angular Material, PrimeNG,
            // and common SPA navigation elements
            const candidates = document.querySelectorAll(
                // Standard HTML
                'button, [role="tab"], [role="menuitem"], [role="button"], ' +
                'a[href]:not([href^="http"]):not([href^="mailto"]):not([href^="javascript"]), ' +
                '[class*="nav-link"], [data-toggle], [data-bs-toggle], ' +
                'li[role="presentation"] > a, .accordion-header, details > summary, ' +

                // Angular router links
                '[routerLink], [routerlink], [ng-click], [\\\\(click\\\\)], ' +

                // Angular Material (legacy mat-* and MDC mdc-*)
                'mat-tab, [mat-tab-link], mat-list-item, mat-nav-list a, ' +
                'mat-sidenav a, [mat-button], [mat-raised-button], [mat-flat-button], ' +
                '[mat-stroked-button], [mat-icon-button], [mat-fab], [mat-mini-fab], ' +
                'mat-expansion-panel-header, mat-menu-item, [mat-menu-item], ' +
                'mat-card[routerLink], mat-card[routerlink], ' +
                'mat-tree-node, [matTreeNodeToggle], ' +

                // MDC-based Angular Material (v15+ uses mdc-* classes)
                '.mdc-tab, .mdc-button, .mdc-icon-button, .mdc-fab, ' +
                '.mdc-list-item, .mdc-card__action, .mdc-card[routerLink], ' +
                '.mdc-evolution-chip, .mdc-switch, ' +
                '.mat-mdc-tab, .mat-mdc-button, .mat-mdc-icon-button, ' +
                '.mat-mdc-list-item, .mat-mdc-menu-item, .mat-mdc-fab, ' +
                '.mat-mdc-card, .mat-mdc-chip, ' +

                // PrimeNG
                'p-tabPanel, p-menuitem, .p-menuitem, p-button, .p-button, ' +
                '.p-tabview-nav li, .p-panelmenu-header, .p-tree-toggler, ' +
                '.p-accordion-header, ' +

                // Kendo UI
                '.k-tabstrip-items .k-item, .k-menu-item, .k-panelbar-item, ' +

                // Common SPA patterns
                '[class*="sidebar"] a, [class*="sidebar"] button, ' +
                '[class*="menu-item"], [class*="menuitem"], ' +
                'nav a, nav button, .nav a, .nav button, ' +
                '[class*="tree-node"], [class*="treenode"]'
            );

            for (const el of candidates) {
                // Must be visible — check offsetParent and bounding rect
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden') continue;
                if (el.offsetParent === null && style.position !== 'fixed' && !el.closest('details')) continue;
                const rect = el.getBoundingClientRect();
                if (rect.width === 0 || rect.height === 0) continue;
                // Must be in viewport (not scrolled out of view by a huge margin)
                if (rect.top > window.innerHeight * 2 || rect.bottom < -window.innerHeight) continue;

                // Build a unique, stable selector (prefer id > data-testid > aria-label > class)
                let sel = '';
                if (el.id) {
                    sel = '#' + CSS.escape(el.id);
                } else if (el.getAttribute('data-testid')) {
                    sel = '[data-testid="' + el.getAttribute('data-testid') + '"]';
                } else if (el.getAttribute('aria-label')) {
                    sel = '[aria-label="' + el.getAttribute('aria-label') + '"]';
                } else if (el.getAttribute('routerLink') || el.getAttribute('routerlink')) {
                    const rl = el.getAttribute('routerLink') || el.getAttribute('routerlink');
                    sel = '[routerLink="' + rl + '"]';
                    // Fall back to case-insensitive variant
                    if (!document.querySelector(sel)) {
                        sel = '[routerlink="' + rl + '"]';
                    }
                } else if (el.className && typeof el.className === 'string') {
                    const classes = el.className.trim().split(/\\s+/).filter(c => c.length > 0);
                    // Pick the most specific class (longest, not utility-prefix)
                    const specific = classes.filter(c => !c.startsWith('ng-') && !c.startsWith('cdk-') && c.length > 2);
                    const cls = (specific[0] || classes[0] || '');
                    if (cls) sel = el.tagName.toLowerCase() + '.' + CSS.escape(cls);
                }

                if (!sel || seen.has(sel)) continue;
                seen.add(sel);

                // Get meaningful text (skip massive text blocks from containers)
                let text = '';
                const directText = Array.from(el.childNodes)
                    .filter(n => n.nodeType === 3)
                    .map(n => n.textContent.trim())
                    .join(' ');
                if (directText) {
                    text = directText.substring(0, 100);
                } else {
                    text = (el.textContent || '').trim().substring(0, 100);
                }

                results.push({
                    selector: sel,
                    text: text,
                    tag: el.tagName.toLowerCase(),
                });
            }

            return results.slice(0, 50);
        }""")

    def _is_unsafe(
        self,
        text: str,
        selector: str,
        skip_patterns: list[re.Pattern],
    ) -> bool:
        """Check if an element is likely destructive or should be skipped."""
        text_lower = text.lower()
        for pat in skip_patterns:
            if pat.search(text_lower):
                return True

        selector_lower = selector.lower()
        for sel in _DESTRUCTIVE_SELECTORS:
            # Simple substring check on the selector
            key = sel.strip("[]").split("=")[0].replace("*", "").strip("'\"")
            if key in selector_lower:
                return True

        return False

    @staticmethod
    async def _find_overlay_options(page: Page) -> list[dict]:
        """Check if a transient overlay (dropdown/menu/popover) is open and return its options.

        Transient overlays only exist while open — their options can't be
        clicked after state restore. Returns clickable items inside CDK
        overlays, mat-select panels, and similar containers.
        """
        try:
            return await page.evaluate("""() => {
                const results = [];
                const seen = new Set();

                // Angular CDK overlay panes (mat-select, mat-menu, mat-autocomplete)
                const overlayPanes = document.querySelectorAll('.cdk-overlay-pane');
                for (const pane of overlayPanes) {
                    if (pane.offsetParent === null) continue;
                    const rect = pane.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) continue;

                    // Find clickable options inside the overlay
                    const options = pane.querySelectorAll(
                        'mat-option, [role="option"], [role="menuitem"], ' +
                        '.mat-mdc-option, .mdc-list-item, .mat-menu-item, ' +
                        '.p-menuitem, .p-listbox-item, li[mat-menu-item]'
                    );

                    for (const opt of options) {
                        const optRect = opt.getBoundingClientRect();
                        if (optRect.width === 0 || optRect.height === 0) continue;

                        let sel = '';
                        if (opt.id) {
                            sel = '#' + CSS.escape(opt.id);
                        } else if (opt.getAttribute('data-testid')) {
                            sel = '[data-testid="' + opt.getAttribute('data-testid') + '"]';
                        }

                        if (!sel || seen.has(sel)) continue;
                        seen.add(sel);

                        const text = (opt.textContent || '').trim().substring(0, 100);
                        results.push({ selector: sel, text, tag: opt.tagName.toLowerCase() });
                    }
                }

                return results;
            }""")
        except Exception:
            return []

    async def _explore_overlay_options(
        self,
        page: Page,
        options: list[dict],
        parent_sid: str,
        parent_url: str,
        parent_chain: list[dict],
        trigger_action: dict,
        depth: int,
        visited_fingerprints: dict[str, str],
        state_graph: dict[str, list[dict]],
        new_pages: list[PageModel],
        queue: deque,
        skip_patterns: list[re.Pattern],
        max_states: int,
    ) -> None:
        """Explore options inside a transient overlay (dropdown/menu).

        For each option: click it, fingerprint the result, register the state,
        then restore to parent and re-open the overlay to try the next option.
        """
        trigger_sid = state_id_from_fingerprint(
            parent_url, list(visited_fingerprints.keys())[-1]
        )
        # The action chain to reach the overlay-open state
        overlay_chain = parent_chain + [trigger_action]

        for opt in options:
            if len(visited_fingerprints) >= max_states:
                break

            opt_text = opt.get("text", "")
            opt_selector = opt.get("selector", "")

            if self._is_unsafe(opt_text, opt_selector, skip_patterns):
                logger.debug("Skipping unsafe overlay option: '%s'", opt_text[:40])
                continue

            try:
                # Click the option
                logger.debug("Clicking overlay option: [%s] '%s'", opt_selector, opt_text[:40])
                await page.click(opt_selector, timeout=3000)
                await self._wait_for_stable(page)

                fp = await fingerprint_state(page)
                opt_action = {
                    "action_type": "click",
                    "selector": opt_selector,
                    "description": opt_text[:80] if opt_text else opt_selector,
                }

                if fp not in visited_fingerprints:
                    new_url = page.url
                    target_sid = state_id_from_fingerprint(new_url, fp)
                    visited_fingerprints[fp] = target_sid

                    elements = await extract_elements(page)
                    forms = await analyze_forms(page)

                    screenshot_path = ""
                    try:
                        screenshot_path = str(
                            self.baselines_dir / f"{target_sid}_screenshot.png"
                        )
                        await page.screenshot(path=screenshot_path, full_page=True)
                    except Exception:
                        screenshot_path = ""

                    # The trigger_action for this page is a compound action:
                    # open overlay + click option
                    compound_action = {
                        "action_type": "click",
                        "selector": opt_selector,
                        "description": f"{trigger_action.get('description', '')[:30]} > {opt_text[:30]}",
                    }

                    new_page = PageModel(
                        page_id=target_sid,
                        url=new_url,
                        page_type="interactive",
                        title=await page.title() or "",
                        elements=elements,
                        forms=forms,
                        screenshot_path=screenshot_path,
                        fingerprint=fp,
                        parent_page_id=trigger_sid,
                        trigger_action=compound_action,
                    )
                    new_pages.append(new_page)

                    state_graph.setdefault(trigger_sid, []).append({
                        "target_state_id": target_sid,
                        "action": opt_action,
                    })
                    state_graph[target_sid] = []

                    # Enqueue: chain = parent chain + overlay trigger + option click
                    new_chain = overlay_chain + [opt_action]
                    queue.append((new_url, target_sid, depth + 1, new_chain))
                    logger.info(
                        "New state from overlay: %s (fp=%s) via '%s'",
                        target_sid, fp[:8], opt_text[:40],
                    )
                else:
                    target_sid = visited_fingerprints[fp]
                    state_graph.setdefault(trigger_sid, []).append({
                        "target_state_id": target_sid,
                        "action": opt_action,
                    })

            except Exception as e:
                logger.debug("Overlay option click failed [%s]: %s", opt_selector, e)

            # Restore to parent state and re-open the overlay for the next option
            if not await self._navigate_to_state(page, parent_url, parent_chain):
                logger.debug("Failed to restore parent state for overlay re-open")
                break

            # Re-open the overlay by replaying the trigger action
            try:
                trigger_sel = trigger_action.get("selector", "")
                if trigger_sel:
                    await page.wait_for_selector(trigger_sel, state="visible", timeout=5000)
                    await page.click(trigger_sel, timeout=5000)
                    await self._wait_for_stable(page)
            except Exception as e:
                logger.debug("Failed to re-open overlay: %s", e)
                break

    @staticmethod
    async def _dismiss_overlays(page: Page) -> None:
        """Dismiss open Angular CDK overlays/menus/dialogs that block clicks."""
        try:
            dismissed = await page.evaluate("""() => {
                let count = 0;
                // Click CDK overlay backdrops to close menus/dialogs
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
        # Also try Escape key as a universal dismiss
        try:
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(200)
        except Exception:
            pass

    @staticmethod
    async def _restore_web_storage(page: Page, storage: dict) -> None:
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

    async def _wait_for_stable(self, page: Page) -> None:
        """Wait for the page to stabilize after navigation or interaction.

        Angular apps often have route transitions, lazy module loading, and
        animations that need more time than a simple networkidle check.
        """
        # First wait for Angular to finish rendering (if Angular is present)
        try:
            await page.evaluate("""() => {
                return new Promise((resolve) => {
                    // If Angular testability API is available, wait for it
                    if (window.getAllAngularTestabilities) {
                        const testabilities = window.getAllAngularTestabilities();
                        if (testabilities.length > 0) {
                            testabilities[0].whenStable(() => resolve(true));
                            // Safety timeout — don't wait forever
                            setTimeout(() => resolve(false), 5000);
                            return;
                        }
                    }
                    // If ng.probe exists (older Angular), wait a bit for digest
                    if (window.ng) {
                        setTimeout(() => resolve(true), 500);
                        return;
                    }
                    // Not Angular, resolve immediately
                    resolve(true);
                });
            }""")
        except Exception:
            pass

        # Then wait for network idle
        try:
            await page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            await page.wait_for_timeout(2000)
