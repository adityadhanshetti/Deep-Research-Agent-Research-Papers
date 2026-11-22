"""OpenAlex API search tool for comprehensive global scientific literature."""

from __future__ import annotations

import asyncio
from typing import List, Optional
import httpx

OPENALEX_API_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted_index: Optional[dict]) -> Optional[str]:
    """Reconstruct plain-text abstract from OpenAlex abstract_inverted_index."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return None
    try:
        word_positions: List[tuple[int, str]] = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)
    except Exception:
        return None


async def search_openalex(query: str, max_results: int = 10) -> List[dict]:
    """Search OpenAlex for academic works matching the query.

    Parameters
    ----------
    query:
        Academic keywords or question concepts.
    max_results:
        Number of works to retrieve (default: 10).

    Returns
    -------
    List[dict]:
        List of paper metadata dictionaries matching the ResearchPaper schema.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    params = {
        "search": clean_query,
        "per_page": min(max_results, 25),
        "mailto": "research-agent@example.com",  # OpenAlex polite pool
    }

    papers: List[dict] = []
    try:
        await asyncio.sleep(0.3)
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(OPENALEX_API_URL, params=params)
            if response.status_code != 200:
                return []
            data = response.json()
            results = data.get("results", [])

            for item in results:
                title = (item.get("display_name") or item.get("title") or "").strip()
                if not title:
                    continue

                # Authorships
                authors: List[str] = []
                for auth in item.get("authorships", []):
                    author_obj = auth.get("author", {})
                    name = author_obj.get("display_name")
                    if name:
                        authors.append(name.strip())

                year = item.get("publication_year")

                # Abstract
                abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))

                # DOI
                raw_doi = item.get("doi")
                doi = raw_doi.replace("https://doi.org/", "").strip() if raw_doi else None

                # URL
                primary_loc = item.get("primary_location") or {}
                url = primary_loc.get("landing_page_url") or raw_doi or item.get("id")

                papers.append({
                    "title": title,
                    "authors": authors if authors else ["Unknown Author"],
                    "year": year,
                    "abstract": abstract,
                    "doi": doi,
                    "url": url,
                    "source": "openalex",
                    "paper_id": item.get("id"),
                })
    except Exception as exc:
        print(f"  [OpenAlex] Error querying '{query}': {exc}")
        return []

    return papers
