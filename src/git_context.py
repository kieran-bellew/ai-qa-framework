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
MAX_README_CHARS = 3000
MAX_DIFF_CHARS = 5000
MAX_LOG_ENTRIES = 20
MAX_TREE_DEPTH = 3


class GitContext(BaseModel):
    """Git context configuration supplied by the user."""

    repo: Optional[str] = None
    branch: Optional[str] = None
    commit: Optional[str] = None


class GitContextProvider:
    """Clones a git repo at a specific point and extracts context for LLM prompts."""

    def __init__(self, git_context: GitContext):
        self.git_context = git_context
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

            result["readme"] = self._read_readme()
            result["structure"] = self._get_structure()
            result["recent_log"] = self._get_recent_log()
            result["commit_diff"] = self._get_commit_diff()
        except Exception as e:
            logger.warning("Failed to extract git context: %s", e)
        finally:
            self._cleanup()

        return result

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
        # After clone, work inside the repo dir
        self._work_dir = self._work_dir / "repo"

    def _checkout(self) -> None:
        """Checkout specific commit if provided."""
        if self.git_context.commit:
            self._run_git("checkout", self.git_context.commit)

    def _read_readme(self) -> str:
        """Read README.md (or similar) from the repo root."""
        for name in ("README.md", "README.rst", "README.txt", "README"):
            readme_path = self._work_dir / name
            if readme_path.exists():
                try:
                    text = readme_path.read_text(encoding="utf-8", errors="replace")
                    return text[:MAX_README_CHARS]
                except Exception:
                    pass
        return ""

    def _get_structure(self) -> str:
        """Get a tree-like directory listing of the repo."""
        lines: list[str] = []
        self._walk_tree(self._work_dir, "", 0, lines)
        return "\n".join(lines[:200])  # cap at 200 lines

    def _walk_tree(self, path: Path, prefix: str, depth: int, lines: list[str]) -> None:
        if depth > MAX_TREE_DEPTH:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return

        # Skip hidden dirs and common noise
        skip = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build"}
        entries = [e for e in entries if e.name not in skip]

        for entry in entries:
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                self._walk_tree(entry, prefix + "  ", depth + 1, lines)
            else:
                lines.append(f"{prefix}{entry.name}")

    def _get_recent_log(self) -> str:
        """Get recent git log."""
        try:
            return self._run_git(
                "log", f"--oneline", f"-{MAX_LOG_ENTRIES}", "--no-decorate"
            ).strip()
        except Exception:
            return ""

    def _get_commit_diff(self) -> str:
        """Get the diff for the current commit (or HEAD)."""
        try:
            diff = self._run_git("diff", "HEAD~1..HEAD", "--stat")
            full_diff = self._run_git("diff", "HEAD~1..HEAD")
            combined = f"{diff}\n---\n{full_diff}"
            return combined[:MAX_DIFF_CHARS]
        except Exception:
            return ""

    def _cleanup(self) -> None:
        """Remove the temporary clone directory."""
        if self._work_dir is None:
            return
        # Walk up to find the temp root (we may have descended into /repo)
        cleanup_dir = self._work_dir
        if cleanup_dir.name == "repo" and cleanup_dir.parent.name.startswith("qa-git-"):
            cleanup_dir = cleanup_dir.parent
        try:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
        except Exception as e:
            logger.debug("Failed to clean up git work dir: %s", e)
