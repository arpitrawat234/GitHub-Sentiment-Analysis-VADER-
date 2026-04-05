"""
github_fetcher.py
─────────────────
Fetches Issues, Pull Request comments, and Commit messages
from the GitHub REST API for a given owner/repo.
"""

import requests
import pandas as pd
from datetime import datetime


class GitHubFetcher:
    BASE = "https://api.github.com"

    def __init__(self, token: str = None):
        self.headers = {"Accept": "application/vnd.github+json"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"

    # ── internal ──────────────────────────────────────────────
    def _get(self, url: str, params: dict = None):
        resp = requests.get(url, headers=self.headers, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _paginate(self, url: str, params: dict = None, max_pages: int = 5):
        params = params or {}
        params["per_page"] = 100
        results = []
        for page in range(1, max_pages + 1):
            params["page"] = page
            data = self._get(url, params)
            if not data:
                break
            results.extend(data)
            if len(data) < 100:
                break
        return results

    # ── public ────────────────────────────────────────────────
    def fetch_issues(self, owner: str, repo: str, max_pages: int = 5) -> pd.DataFrame:
        """Fetch open + closed issues (excludes PRs)."""
        url  = f"{self.BASE}/repos/{owner}/{repo}/issues"
        raw  = self._paginate(url, {"state": "all"}, max_pages)
        rows = []
        for item in raw:
            if "pull_request" in item:      # skip PRs — they appear in issues endpoint
                continue
            rows.append({
                "id":         item["number"],
                "type":       "issue",
                "author":     (item.get("user") or {}).get("login", ""),
                "text":       (item.get("title") or "") + " " + (item.get("body") or ""),
                "created_at": item.get("created_at", ""),
                "state":      item.get("state", ""),
            })
        return pd.DataFrame(rows)

    def fetch_pr_comments(self, owner: str, repo: str, max_pages: int = 5) -> pd.DataFrame:
        """Fetch pull-request review comments."""
        url  = f"{self.BASE}/repos/{owner}/{repo}/pulls/comments"
        raw  = self._paginate(url, {"state": "all"}, max_pages)
        rows = []
        for item in raw:
            rows.append({
                "id":         item.get("id", ""),
                "type":       "pr_comment",
                "author":     (item.get("user") or {}).get("login", ""),
                "text":       item.get("body") or "",
                "created_at": item.get("created_at", ""),
                "state":      "n/a",
            })
        return pd.DataFrame(rows)

    def fetch_issue_comments(self, owner: str, repo: str, max_pages: int = 5) -> pd.DataFrame:
        """Fetch comments on issues."""
        url  = f"{self.BASE}/repos/{owner}/{repo}/issues/comments"
        raw  = self._paginate(url, {}, max_pages)
        rows = []
        for item in raw:
            rows.append({
                "id":         item.get("id", ""),
                "type":       "issue_comment",
                "author":     (item.get("user") or {}).get("login", ""),
                "text":       item.get("body") or "",
                "created_at": item.get("created_at", ""),
                "state":      "n/a",
            })
        return pd.DataFrame(rows)

    def fetch_commits(self, owner: str, repo: str, max_pages: int = 5) -> pd.DataFrame:
        """Fetch commit messages."""
        url  = f"{self.BASE}/repos/{owner}/{repo}/commits"
        raw  = self._paginate(url, {}, max_pages)
        rows = []
        for item in raw:
            commit = item.get("commit", {})
            rows.append({
                "id":         item.get("sha", "")[:7],
                "type":       "commit",
                "author":     (commit.get("author") or {}).get("name", ""),
                "text":       commit.get("message") or "",
                "created_at": (commit.get("author") or {}).get("date", ""),
                "state":      "n/a",
            })
        return pd.DataFrame(rows)

    def fetch_all(self, owner: str, repo: str,
                  include_issues: bool = True,
                  include_pr_comments: bool = True,
                  include_issue_comments: bool = True,
                  include_commits: bool = True,
                  max_pages: int = 5) -> pd.DataFrame:
        """Fetch all selected data sources and combine."""
        frames = []
        if include_issues:
            frames.append(self.fetch_issues(owner, repo, max_pages))
        if include_pr_comments:
            frames.append(self.fetch_pr_comments(owner, repo, max_pages))
        if include_issue_comments:
            frames.append(self.fetch_issue_comments(owner, repo, max_pages))
        if include_commits:
            frames.append(self.fetch_commits(owner, repo, max_pages))

        if not frames:
            return pd.DataFrame()

        df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
        df["date"]       = df["created_at"].dt.date
        df["month"]      = df["created_at"].dt.to_period("M").astype(str)
        return df
