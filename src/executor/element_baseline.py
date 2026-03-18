"""Element baseline capture and visual matching for selector recovery.

Captures small screenshots of elements after successful interactions.
When a selector fails, uses template matching to find the element visually.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from playwright.async_api import Page

logger = logging.getLogger(__name__)

_BASELINES_DIR_NAME = "element_baselines"


class ElementBaselineStore:
    """Stores and retrieves element screenshot baselines."""

    def __init__(self, store_dir: Path):
        self.store_dir = store_dir / _BASELINES_DIR_NAME
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.store_dir / "index.json"
        self._index: dict[str, dict] = {}
        self._load_index()

    def _load_index(self) -> None:
        if self._index_path.exists():
            try:
                self._index = json.loads(self._index_path.read_text())
            except Exception:
                self._index = {}

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(self._index, indent=2))

    async def capture_element(
        self,
        page: Page,
        selector: str,
        action_type: str = "",
    ) -> None:
        """Capture a screenshot of an element for future visual matching."""
        try:
            el = await page.query_selector(selector)
            if not el:
                return

            box = await el.bounding_box()
            if not box or box["width"] < 5 or box["height"] < 5:
                return

            # Generate a stable key from the selector
            key = hashlib.md5(selector.encode()).hexdigest()[:12]
            img_path = self.store_dir / f"{key}.png"

            # Crop the element from a full screenshot
            await el.screenshot(path=str(img_path))

            self._index[selector] = {
                "key": key,
                "path": str(img_path),
                "bounding_box": box,
                "action_type": action_type,
                "page_url": page.url,
            }
            self._save_index()
            logger.debug("Captured element baseline for '%s'", selector[:60])

        except Exception as e:
            logger.debug("Element baseline capture failed for '%s': %s", selector[:40], e)

    async def find_element_visually(
        self,
        page: Page,
        original_selector: str,
    ) -> str | None:
        """Try to find an element by matching its baseline screenshot.

        Returns Playwright coordinates selector (e.g., ">> nth=0") or None.
        Uses bounding box position matching as a lightweight visual strategy.
        """
        baseline = self._index.get(original_selector)
        if not baseline or not Path(baseline["path"]).exists():
            return None

        original_box = baseline["bounding_box"]

        try:
            from PIL import Image
            import io

            # Load baseline image
            baseline_img = Image.open(baseline["path"])
            bw, bh = baseline_img.size

            # Take a fresh full-page screenshot
            screenshot_bytes = await page.screenshot(full_page=False)
            current_img = Image.open(io.BytesIO(screenshot_bytes))

            # Simple approach: scan the current screenshot for a region that
            # matches the baseline image using pixel comparison.
            # Use a coarse search centered on the original bounding box position.
            best_match = None
            best_score = 0.0

            ox, oy = int(original_box["x"]), int(original_box["y"])

            # Search in a region around the original position
            search_radius = 150
            step = 4  # Coarse step for speed

            for dy in range(-search_radius, search_radius + 1, step):
                for dx in range(-search_radius, search_radius + 1, step):
                    x, y = ox + dx, oy + dy
                    if x < 0 or y < 0:
                        continue
                    if x + bw > current_img.width or y + bh > current_img.height:
                        continue

                    # Crop the candidate region
                    candidate = current_img.crop((x, y, x + bw, y + bh))

                    # Compare pixels (fast: sample every 4th pixel)
                    score = _compare_images_fast(baseline_img, candidate)
                    if score > best_score:
                        best_score = score
                        best_match = (x, y)

            if best_match and best_score > 0.7:
                # Found a visual match — click at the center of the matched region
                cx = best_match[0] + bw // 2
                cy = best_match[1] + bh // 2
                logger.info(
                    "Visual match for '%s': score=%.2f at (%d,%d)",
                    original_selector[:40], best_score, cx, cy,
                )
                # Return as a Playwright locator using page.mouse.click coordinates
                # The caller should use page.mouse.click(cx, cy) instead of selector
                return f"__visual_match:{cx},{cy}"

        except ImportError:
            logger.debug("Pillow not available for visual matching")
        except Exception as e:
            logger.debug("Visual matching failed: %s", e)

        return None


def _compare_images_fast(img1, img2, sample_step: int = 4) -> float:
    """Fast pixel comparison between two PIL images. Returns 0.0-1.0 similarity."""
    if img1.size != img2.size:
        return 0.0

    p1 = img1.load()
    p2 = img2.load()
    w, h = img1.size

    matches = 0
    total = 0

    for y in range(0, h, sample_step):
        for x in range(0, w, sample_step):
            total += 1
            px1 = p1[x, y]
            px2 = p2[x, y]
            # Allow some tolerance per channel
            if all(abs(a - b) < 30 for a, b in zip(px1[:3], px2[:3])):
                matches += 1

    return matches / total if total > 0 else 0.0
