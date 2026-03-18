"""Test plan caching — skip AI planning when the site model hasn't changed."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from src.models.site_model import SiteModel
from src.models.test_plan import TestPlan

logger = logging.getLogger(__name__)


def compute_site_hash(site_model: SiteModel) -> str:
    """Compute a stable hash of the site model content.

    Includes page URLs, element counts, form structures, and state graph.
    Excludes volatile fields (timestamps, screenshot paths, DOM snapshots).
    """
    hashable = {
        "base_url": site_model.base_url,
        "pages": [
            {
                "url": p.url,
                "page_type": p.page_type,
                "element_count": len(p.elements),
                "interactive_count": sum(1 for e in p.elements if e.is_interactive),
                "form_count": len(p.forms),
                "form_fields": [
                    [f.name for f in form.fields]
                    for form in p.forms
                ],
                "fingerprint": p.fingerprint,
            }
            for p in sorted(site_model.pages, key=lambda p: p.url)
        ],
        "state_graph_edges": sum(len(v) for v in site_model.state_graph.values()),
        "api_endpoint_count": len(site_model.api_endpoints),
    }
    content = json.dumps(hashable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def load_cached_plan(cache_path: Path, site_hash: str) -> TestPlan | None:
    """Load a cached plan if the site hash matches."""
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text())
        if data.get("_site_hash") != site_hash:
            logger.info("Plan cache miss: site model changed (hash %s != %s)",
                       data.get("_site_hash", "?")[:8], site_hash[:8])
            return None
        logger.info("Plan cache hit: reusing %d test cases (hash=%s)",
                    len(data.get("test_cases", [])), site_hash[:8])
        return TestPlan(**{k: v for k, v in data.items() if not k.startswith("_")})
    except Exception as e:
        logger.debug("Failed to load cached plan: %s", e)
        return None


def save_plan_with_hash(cache_path: Path, plan: TestPlan, site_hash: str) -> None:
    """Save a plan with its site model hash for future cache lookups."""
    data = plan.model_dump()
    data["_site_hash"] = site_hash
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data, indent=2, default=str))
    logger.debug("Saved plan cache with hash=%s", site_hash[:8])
