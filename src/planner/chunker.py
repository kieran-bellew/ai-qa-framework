"""Page chunker — groups pages for parallel AI planning calls."""

from __future__ import annotations

import math
from urllib.parse import urlparse

from src.models.site_model import PageModel, SiteModel


def chunk_site_model(
    site_model: SiteModel,
    max_pages_per_chunk: int = 15,
) -> list[list[PageModel]]:
    """Group pages into chunks for separate AI planning calls.

    Uses a hybrid strategy:
    1. Group interactive states with their root page (parent chain)
    2. Group remaining pages by URL path prefix
    3. Split oversized groups

    Returns a list of page groups (each group is a list of PageModel).
    """
    pages = site_model.pages
    if len(pages) <= max_pages_per_chunk:
        return [pages]

    # Build parent lookup
    page_by_id = {p.page_id: p for p in pages}

    # Step 1: Find root page for each interactive state
    root_map: dict[str, str] = {}  # page_id -> root_page_id
    for p in pages:
        root_id = _find_root(p, page_by_id)
        root_map[p.page_id] = root_id

    # Step 2: Group by root page, then by URL prefix for non-interactive pages
    groups: dict[str, list[PageModel]] = {}
    for p in pages:
        root_id = root_map[p.page_id]
        root_page = page_by_id.get(root_id)

        if root_page and root_page.fingerprint:
            # Interactive state — group with its root
            key = f"state:{root_id}"
        else:
            # Regular page — group by URL prefix
            key = f"url:{_url_prefix(p.url)}"

        groups.setdefault(key, []).append(p)

    # Step 3: Split oversized groups, merge tiny groups
    chunks: list[list[PageModel]] = []
    small_buffer: list[PageModel] = []

    for pages_in_group in groups.values():
        if len(pages_in_group) > max_pages_per_chunk:
            # Split into sub-chunks
            for i in range(0, len(pages_in_group), max_pages_per_chunk):
                chunks.append(pages_in_group[i:i + max_pages_per_chunk])
        elif len(pages_in_group) <= 3:
            # Too small — buffer for merging
            small_buffer.extend(pages_in_group)
            if len(small_buffer) >= max_pages_per_chunk:
                chunks.append(small_buffer[:max_pages_per_chunk])
                small_buffer = small_buffer[max_pages_per_chunk:]
        else:
            chunks.append(pages_in_group)

    if small_buffer:
        chunks.append(small_buffer)

    return chunks


def allocate_test_budget(
    chunks: list[list[PageModel]],
    total_budget: int,
) -> list[int]:
    """Allocate test budget proportionally across chunks."""
    total_pages = sum(len(c) for c in chunks)
    if total_pages == 0:
        return [0] * len(chunks)

    budgets = []
    for chunk in chunks:
        budget = max(2, math.ceil(total_budget * len(chunk) / total_pages))
        budgets.append(budget)

    # Cap total to not exceed budget
    while sum(budgets) > total_budget and any(b > 2 for b in budgets):
        # Reduce the largest chunk's budget
        max_idx = max(range(len(budgets)), key=lambda i: budgets[i])
        budgets[max_idx] -= 1

    return budgets


def _find_root(page: PageModel, page_by_id: dict[str, PageModel]) -> str:
    """Walk parent chain to find the root page."""
    visited = set()
    current = page
    while current and current.page_id not in visited:
        visited.add(current.page_id)
        if not current.parent_page_id or current.parent_page_id not in page_by_id:
            return current.page_id
        current = page_by_id[current.parent_page_id]
    return page.page_id


def _url_prefix(url: str) -> str:
    """Extract first path segment as group key."""
    path = urlparse(url).path.strip("/")
    parts = path.split("/")
    return parts[0] if parts and parts[0] else "root"
