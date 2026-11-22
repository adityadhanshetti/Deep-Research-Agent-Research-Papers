"""Semantic Scholar API search tool for discovering peer-reviewed academic papers."""

from __future__ import annotations

import asyncio
from typing import List, Optional
import httpx

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "paperId,title,authors,year,abstract,externalIds,url,venue"


async def search_semantic_scholar(query: str, max_results: int = 10) -> List[dict]:
    """Search Semantic Scholar for papers matching the given query.

    Parameters
    ----------
    query:
        Research keywords or topic concepts.
    max_results:
        Number of papers to retrieve (default: 10, max: 100).

    Returns
    -------
    List[dict]:
        List of paper metadata dictionaries matching the ResearchPaper schema.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    params = {
        "query": clean_query,
        "limit": min(max_results, 20),
        "fields": FIELDS,
    }

    papers: List[dict] = []
    retries = 3
    backoff = 1.0

    for attempt in range(retries):
        try:
            await asyncio.sleep(0.5)
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(SEMANTIC_SCHOLAR_URL, params=params)

                if response.status_code == 429:
                    # Rate limited: back off
                    await asyncio.sleep(backoff)
                    backoff *= 2.0
                    continue

                if response.status_code != 200:
                    return []

                data = response.json()
                items = data.get("data", [])
                for item in items:
                    title = (item.get("title") or "").strip()
                    if not title:
                        continue

                    raw_authors = item.get("authors") or []
                    authors = [a.get("name", "").strip() for a in raw_authors if a.get("name")]

                    year = item.get("year")
                    abstract = item.get("abstract")

                    ext_ids = item.get("externalIds") or {}
                    doi = ext_ids.get("DOI")
                    paper_id = item.get("paperId")

                    url = item.get("url") or (f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else None)

                    papers.append({
                        "title": title,
                        "authors": authors if authors else ["Unknown Author"],
                        "year": year,
                        "abstract": abstract,
                        "doi": doi,
                        "url": url,
                        "source": "semantic_scholar",
                        "paper_id": paper_id,
                    })
                break
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  [Semantic Scholar] Error querying '{query}': {exc}")
            await asyncio.sleep(backoff)
            backoff *= 2.0

    return papers
