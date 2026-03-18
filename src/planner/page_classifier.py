"""Pre-planning page classification — AI-driven analysis of what each page IS."""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.ai.client import AIClient
from src.models.site_model import PageModel

logger = logging.getLogger(__name__)

_CLASSIFICATION_PROMPT = """You are a QA page classifier. Given page data (elements, forms, URL), classify each page and suggest test strategies.

Return ONLY valid JSON (no markdown fences):
{
  "pages": [
    {
      "page_id": "string",
      "classification": "data_grid | form_page | dashboard | settings | wizard | list_detail | search | navigation | static | error",
      "test_strategies": ["list of applicable strategies"],
      "description": "one-sentence description of what this page does"
    }
  ]
}

Available test strategies:
- "crud_operations" — Create, Read, Update, Delete operations
- "form_validation" — Test required fields, invalid inputs, boundary values
- "form_submission" — Fill and submit the form with valid data
- "pagination" — Test next/previous page navigation in data grids
- "sort_columns" — Test column sorting in data grids
- "filter_search" — Test search/filter functionality
- "row_interaction" — Test clicking rows to view details
- "tab_navigation" — Test switching between tabs
- "navigation_links" — Test sidebar/menu navigation links
- "modal_dialogs" — Test opening and interacting with modals
- "file_upload" — Test file upload functionality
- "date_range" — Test date picker/range selection
- "export_data" — Test data export functionality
- "responsive_layout" — Test responsive behavior across viewports
"""


def classify_pages(
    pages: list[PageModel],
    ai_client: AIClient,
    max_tokens: int = 4000,
    batch_size: int = 6,
) -> None:
    """Classify pages in-place using AI analysis.

    Batches pages and makes parallel AI calls. Updates each PageModel's
    classification, test_strategies, and page_description fields.
    """
    if not pages:
        return

    # Split into batches
    batches = [pages[i:i + batch_size] for i in range(0, len(pages), batch_size)]
    logger.info("Classifying %d pages in %d batches", len(pages), len(batches))

    def _classify_batch(batch: list[PageModel]) -> dict:
        batch_data = []
        for p in batch:
            page_info = {
                "page_id": p.page_id,
                "url": p.url,
                "title": p.title,
                "page_type": p.page_type,
                "element_count": len(p.elements),
                "interactive_elements": sum(1 for e in p.elements if e.is_interactive),
                "form_count": len(p.forms),
                "key_elements": [
                    {"type": e.element_type, "text": e.text_content[:40]}
                    for e in p.elements[:10]
                    if e.is_interactive
                ],
                "forms": [
                    {
                        "fields": [f.name for f in form.fields[:8]],
                        "form_pattern": form.form_pattern,
                    }
                    for form in p.forms[:3]
                ],
            }
            batch_data.append(page_info)

        import json
        user_msg = f"Classify these pages:\n\n```json\n{json.dumps(batch_data, indent=2)}\n```"

        try:
            result = ai_client.complete_json(
                system_prompt=_CLASSIFICATION_PROMPT,
                user_message=user_msg,
                max_tokens=max_tokens,
            )
            return {p["page_id"]: p for p in result.get("pages", [])}
        except Exception as e:
            logger.debug("Page classification batch failed: %s", e)
            return {}

    # Run batches in parallel
    page_lookup = {p.page_id: p for p in pages}
    max_workers = min(len(batches), 4)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_classify_batch, b): i for i, b in enumerate(batches)}
        for future in as_completed(futures):
            try:
                classifications = future.result()
                for pid, data in classifications.items():
                    page = page_lookup.get(pid)
                    if page:
                        page.classification = data.get("classification", "")
                        page.test_strategies = data.get("test_strategies", [])
                        page.page_description = data.get("description", "")
            except Exception as e:
                logger.debug("Classification future failed: %s", e)

    classified = sum(1 for p in pages if p.classification)
    logger.info("Classified %d/%d pages", classified, len(pages))
