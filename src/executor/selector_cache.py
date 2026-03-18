"""Persistent selector mapping — learns from AI fallback successes across runs."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MAX_MAPPINGS = 500
_CONFIDENCE_DECAY = 0.05
_MIN_CONFIDENCE = 0.3
_STALE_DAYS = 30


class SelectorMapping(BaseModel):
    original_selector: str
    replacement_selector: str
    page_url_pattern: str = ""  # URL path prefix where this applies
    action_type: str = ""
    success_count: int = 1
    failure_count: int = 0
    last_success: str = ""
    last_failure: str = ""
    source: str = "ai_fallback"  # ai_fallback | smart_resolve
    confidence: float = 1.0


class SelectorCache(BaseModel):
    version: int = 1
    target_url: str = ""
    mappings: list[SelectorMapping] = Field(default_factory=list)
    updated_at: str = ""

    # ---- Lookup ----

    def find(
        self,
        original_selector: str,
        page_url: str = "",
    ) -> SelectorMapping | None:
        """Find the best cached replacement for a selector."""
        candidates = [
            m for m in self.mappings
            if m.original_selector == original_selector
            and m.confidence >= _MIN_CONFIDENCE
            and (not m.page_url_pattern or page_url.startswith(m.page_url_pattern))
        ]
        if not candidates:
            return None
        # Best = highest confidence * success_count
        candidates.sort(key=lambda m: m.confidence * m.success_count, reverse=True)
        return candidates[0]

    # ---- Record ----

    def record_success(
        self,
        original_selector: str,
        replacement_selector: str,
        page_url: str = "",
        action_type: str = "",
        source: str = "ai_fallback",
    ) -> None:
        """Record a successful selector replacement."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        url_pattern = self._url_pattern(page_url)

        existing = self._find_exact(original_selector, replacement_selector, url_pattern)
        if existing:
            existing.success_count += 1
            existing.last_success = now
            existing.confidence = min(1.0, existing.confidence + 0.1)
        else:
            self.mappings.append(SelectorMapping(
                original_selector=original_selector,
                replacement_selector=replacement_selector,
                page_url_pattern=url_pattern,
                action_type=action_type,
                success_count=1,
                last_success=now,
                source=source,
                confidence=1.0,
            ))
        self._enforce_limits()

    def record_failure(self, original_selector: str, page_url: str = "") -> None:
        """Record that a cached replacement failed."""
        url_pattern = self._url_pattern(page_url)
        for m in self.mappings:
            if m.original_selector == original_selector and (
                not m.page_url_pattern or m.page_url_pattern == url_pattern
            ):
                m.failure_count += 1
                m.last_failure = time.strftime("%Y-%m-%dT%H:%M:%SZ")
                m.confidence = max(0.0, m.confidence - 0.15)

    # ---- Lifecycle ----

    def decay(self) -> None:
        """Decay confidence on run start (sites change over time)."""
        for m in self.mappings:
            m.confidence = max(0.0, m.confidence - _CONFIDENCE_DECAY)
        # Evict stale/low-confidence entries
        self.mappings = [
            m for m in self.mappings
            if m.confidence >= _MIN_CONFIDENCE
        ]

    # ---- Persistence ----

    @classmethod
    def load(cls, path: Path, target_url: str = "") -> SelectorCache:
        """Load from disk or create empty."""
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return cls(**data)
            except Exception as e:
                logger.warning("Failed to load selector cache: %s", e)
        return cls(target_url=target_url)

    def save(self, path: Path) -> None:
        """Persist to disk."""
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2))
        logger.debug("Saved selector cache (%d mappings) to %s", len(self.mappings), path)

    # ---- Internal ----

    def _find_exact(
        self, original: str, replacement: str, url_pattern: str,
    ) -> SelectorMapping | None:
        for m in self.mappings:
            if (m.original_selector == original
                    and m.replacement_selector == replacement
                    and m.page_url_pattern == url_pattern):
                return m
        return None

    @staticmethod
    def _url_pattern(url: str) -> str:
        """Extract a stable URL path prefix (first 2 segments)."""
        if not url:
            return ""
        from urllib.parse import urlparse
        path = urlparse(url).path.rstrip("/")
        parts = path.split("/")
        # Keep first 2 non-empty segments
        segments = [p for p in parts if p][:2]
        return "/" + "/".join(segments) if segments else "/"

    def _enforce_limits(self) -> None:
        if len(self.mappings) > _MAX_MAPPINGS:
            self.mappings.sort(key=lambda m: m.confidence * m.success_count)
            self.mappings = self.mappings[-_MAX_MAPPINGS:]
