"""Git context provider — clones a repo at a specific commit and extracts context."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)

GIT_TIMEOUT = 60  # seconds per git operation

# Default total budget for all git context injected into the prompt.
DEFAULT_MAX_CONTEXT_CHARS = 8000


class GitContext(BaseModel):
    """Git context configuration supplied by the user."""

    repo: Optional[str] = None
    branch: Optional[str] = None
    commit: Optional[str] = None


class GitContextProvider:
    """Clones a git repo at a specific point and extracts context for LLM prompts.

    Applies a total character budget across all extracted sections to prevent
    bloating the planning prompt.  Budget is allocated in priority order:
      1. Recent log  (cheap, high signal — what's been changing)
      2. Commit diff stat  (compact summary of current change)
      3. README  (app overview)
      4. Changed-file tree  (files touched by the commit)
      5. Full diff  (remaining budget)
      6. Repo-wide tree  (remaining budget)
    """

    def __init__(self, git_context: GitContext, max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS):
        self.git_context = git_context
        self.max_context_chars = max_context_chars
        self._work_dir: Path | None = None

    def extract(self) -> dict[str, str]:
        """Clone the repo (if needed), checkout the commit, and extract context.

        Returns a dict with keys: repo, branch, commit, readme, structure,
        recent_log, commit_diff. All values are strings (possibly empty).
        """
        result: dict[str, str] = {
            "repo": self.git_context.repo or "",
            "branch": self.git_context.branch or "",
            "commit": self.git_context.commit or "",
            "readme": "",
            "structure": "",
            "recent_log": "",
            "commit_diff": "",
        }

        if not self.git_context.repo:
            return result

        try:
            self._work_dir = Path(tempfile.mkdtemp(prefix="qa-git-"))
            self._clone()
            self._checkout()

            result = self._extract_within_budget(result)
        except Exception as e:
            logger.warning("Failed to extract git context: %s", e)
        finally:
            self._cleanup()

        total = sum(len(v) for v in result.values())
        logger.debug("Git context total: %d chars (budget: %d)", total, self.max_context_chars)
        return result

    def _extract_within_budget(self, result: dict[str, str]) -> dict[str, str]:
        """Extract context sections in priority order, respecting the total budget."""
        budget = self.max_context_chars

        # 1. Recent log — compact, high signal (cap: 15% of budget)
        log_cap = min(budget // 6, 1500)
        recent_log = self._get_recent_log(max_chars=log_cap)
        result["recent_log"] = recent_log
        budget -= len(recent_log)

        # 2. Diff stat — compact summary of what changed (cap: 10% of budget)
        stat_cap = min(budget // 5, 1500)
        diff_stat = self._get_diff_stat(max_chars=stat_cap)

        # 3. README — app overview (cap: 30% of budget)
        readme_cap = min(budget // 3, 3000)
        readme = self._read_readme(max_chars=readme_cap)
        result["readme"] = readme
        budget -= len(readme)

        # 4. Changed-file tree — files touched by the commit (always small)
        changed_tree = self._get_changed_files_tree()
        result["structure"] = changed_tree
        budget -= len(changed_tree)

        # 5. Full diff — fill remaining budget (stat + patch)
        if budget > 0:
            full_diff = self._get_full_diff(max_chars=budget)
            # Combine stat + full diff, preferring stat header
            if diff_stat and full_diff:
                combined = f"{diff_stat}\n---\n{full_diff}"[:budget]
            elif diff_stat:
                combined = diff_stat[:budget]
            else:
                combined = full_diff[:budget]
            result["commit_diff"] = combined
            budget -= len(result["commit_diff"])

        # 6. Repo-wide tree — only if significant budget remains (>500 chars)
        if budget > 500 and not result["structure"]:
            repo_tree = self._get_repo_tree(max_chars=budget)
            result["structure"] = repo_tree
            budget -= len(repo_tree)

        return result

    # ------------------------------------------------------------------
    # Git helpers
    # ------------------------------------------------------------------

    def _run_git(self, *args: str, cwd: Path | None = None) -> str:
        """Run a git command and return stdout."""
        cmd = ["git", *args]
        proc = subprocess.run(
            cmd,
            cwd=cwd or self._work_dir,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git {args[0]} failed: {proc.stderr.strip()}")
        return proc.stdout

    def _clone(self) -> None:
        """Shallow-clone the repo."""
        args = ["clone", "--depth", "50"]
        if self.git_context.branch:
            args += ["--branch", self.git_context.branch]
        args += [self.git_context.repo, str(self._work_dir / "repo")]
        self._run_git(*args, cwd=self._work_dir)
        self._work_dir = self._work_dir / "repo"

    def _checkout(self) -> None:
        """Checkout specific commit if provided."""
        if self.git_context.commit:
            self._run_git("checkout", self.git_context.commit)

    # ------------------------------------------------------------------
    # Context extraction methods
    # ------------------------------------------------------------------

    def _read_readme(self, max_chars: int) -> str:
        """Read README.md (or similar) from the repo root."""
        for name in ("README.md", "README.rst", "README.txt", "README"):
            readme_path = self._work_dir / name
            if readme_path.exists():
                try:
                    text = readme_path.read_text(encoding="utf-8", errors="replace")
                    if len(text) > max_chars:
                        # Truncate at last complete line within budget
                        text = text[:max_chars].rsplit("\n", 1)[0] + "\n[...truncated]"
                    return text
                except Exception:
                    pass
        return ""

    def _get_recent_log(self, max_chars: int) -> str:
        """Get recent git log (one-line format)."""
        try:
            log = self._run_git("log", "--oneline", "-20", "--no-decorate").strip()
            return log[:max_chars]
        except Exception:
            return ""

    def _get_diff_stat(self, max_chars: int) -> str:
        """Get --stat summary for the current commit."""
        try:
            stat = self._run_git("diff", "HEAD~1..HEAD", "--stat").strip()
            return stat[:max_chars]
        except Exception:
            return ""

    def _get_changed_files_tree(self) -> str:
        """Get list of files changed in the current commit — always compact."""
        try:
            files = self._run_git("diff", "HEAD~1..HEAD", "--name-only").strip()
            if not files:
                return ""
            lines = files.split("\n")
            header = f"Files changed in commit ({len(lines)} files):\n"
            return header + "\n".join(f"  {f}" for f in lines[:50])
        except Exception:
            return ""

    def _get_full_diff(self, max_chars: int) -> str:
        """Get the full patch diff, truncated to budget."""
        try:
            diff = self._run_git("diff", "HEAD~1..HEAD")
            if len(diff) > max_chars:
                diff = diff[:max_chars].rsplit("\n", 1)[0] + "\n[...diff truncated]"
            return diff
        except Exception:
            return ""

    def _get_repo_tree(self, max_chars: int) -> str:
        """Get a tree-like directory listing of the repo, capped by char budget."""
        lines: list[str] = []
        skip = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build", ".mypy_cache"}
        self._walk_tree(self._work_dir, "", 0, lines, skip, max_depth=3)
        tree = "\n".join(lines)
        if len(tree) > max_chars:
            tree = tree[:max_chars].rsplit("\n", 1)[0] + "\n[...truncated]"
        return tree

    def _walk_tree(self, path: Path, prefix: str, depth: int, lines: list[str],
                   skip: set[str], max_depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return
        entries = [e for e in entries if e.name not in skip]
        for entry in entries:
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                self._walk_tree(entry, prefix + "  ", depth + 1, lines, skip, max_depth)
            else:
                lines.append(f"{prefix}{entry.name}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Remove the temporary clone directory."""
        if self._work_dir is None:
            return
        cleanup_dir = self._work_dir
        if cleanup_dir.name == "repo" and cleanup_dir.parent.name.startswith("qa-git-"):
            cleanup_dir = cleanup_dir.parent
        try:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
        except Exception as e:
            logger.debug("Failed to clean up git work dir: %s", e)
