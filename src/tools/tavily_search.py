"""Tavily search tool for supplementary academic discovery and web-grounded research papers."""

from __future__ import annotations

import asyncio
import os
from typing import List, Optional

ACADEMIC_DOMAINS = [
    "arxiv.org",
    "scholar.google.com",
    "semanticscholar.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ieee.org",
    "acm.org",
    "nature.com",
    "sciencedirect.com",
    "springer.com",
    "biorxiv.org",
]


async def search_tavily(query: str, max_results: int = 5) -> List[dict]:
    """Search Tavily for academic preprints, papers, and authoritative research articles.

    Parameters
    ----------
    query:
        Research topic or query string.
    max_results:
        Maximum number of academic results to return.

    Returns
    -------
    List[dict]:
        List of paper metadata dictionaries matching the ResearchPaper schema.
    """
    clean_query = query.strip()
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not clean_query or not api_key:
        return []

    try:
        from tavily import TavilyClient

        def _do_search():
            client = TavilyClient(api_key=api_key)
            return client.search(
                query=f"{clean_query} research paper",
                search_depth="advanced",
                max_results=max_results,
                include_domains=ACADEMIC_DOMAINS,
            )

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, _do_search)

        results = response.get("results", [])
        papers: List[dict] = []

        for item in results:
            title = (item.get("title") or "").strip()
            url = item.get("url")
            content = item.get("content") or item.get("snippet") or ""

            if not title:
                continue

            # Try to infer year or authors if present
            papers.append({
                "title": title,
                "authors": ["Academic Publication"],
                "year": None,
                "abstract": content[:1200] if content else None,
                "doi": None,
                "url": url,
                "source": "tavily",
                "paper_id": url,
            })

        return papers
    except Exception as exc:
        # Non-fatal error; print and return empty list
        print(f"  [Tavily] Search note: {exc}")
        return []
