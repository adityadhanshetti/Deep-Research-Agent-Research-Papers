"""arXiv API search tool for discovering academic research papers."""

from __future__ import annotations

import asyncio
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Optional
import httpx

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


async def search_arxiv(query: str, max_results: int = 10) -> List[dict]:
    """Search the arXiv API for preprints matching the query.

    Parameters
    ----------
    query:
        Keywords or research concepts to search for.
    max_results:
        Maximum number of results to fetch (default: 10).

    Returns
    -------
    List[dict]:
        List of paper metadata dictionaries matching the ResearchPaper schema.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    encoded_query = urllib.parse.quote(f"all:{clean_query}")
    url = f"{ARXIV_API_URL}?search_query={encoded_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"

    try:
        await asyncio.sleep(0.5)  # Rate limiting courtesy
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(url)
            if response.status_code != 200:
                return []
            xml_text = response.text
    except Exception as exc:
        print(f"  [arXiv] Error querying '{query}': {exc}")
        return []

    papers: List[dict] = []
    try:
        root = ET.fromstring(xml_text)
        entries = root.findall("atom:entry", ATOM_NS)

        for entry in entries:
            title_elem = entry.find("atom:title", ATOM_NS)
            title = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else ""
            if not title:
                continue

            # Authors
            authors: List[str] = []
            for author_elem in entry.findall("atom:author", ATOM_NS):
                name_elem = author_elem.find("atom:name", ATOM_NS)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())

            # Published date -> year
            year: Optional[int] = None
            pub_elem = entry.find("atom:published", ATOM_NS)
            if pub_elem is not None and pub_elem.text:
                try:
                    year = int(pub_elem.text[:4])
                except (ValueError, TypeError):
                    pass

            # Summary / Abstract
            summary_elem = entry.find("atom:summary", ATOM_NS)
            abstract = " ".join(summary_elem.text.split()) if summary_elem is not None and summary_elem.text else None

            # arXiv ID and URLs
            id_elem = entry.find("atom:id", ATOM_NS)
            arxiv_id = id_elem.text.strip() if id_elem is not None and id_elem.text else ""
            paper_id = arxiv_id.split("/abs/")[-1] if "/abs/" in arxiv_id else arxiv_id

            url = arxiv_id
            pdf_link = entry.find("atom:link[@title='pdf']", ATOM_NS)
            if pdf_link is not None:
                url = pdf_link.attrib.get("href", url)

            # DOI if present
            doi: Optional[str] = None
            doi_elem = entry.find("arxiv:doi", ATOM_NS)
            if doi_elem is not None and doi_elem.text:
                doi = doi_elem.text.strip()

            papers.append({
                "title": title,
                "authors": authors if authors else ["Unknown"],
                "year": year,
                "abstract": abstract,
                "doi": doi,
                "url": url if url else f"https://arxiv.org/abs/{paper_id}",
                "source": "arxiv",
                "paper_id": paper_id,
            })
    except Exception as exc:
        print(f"  [arXiv] Error parsing results for '{query}': {exc}")
        return []

    return papers
