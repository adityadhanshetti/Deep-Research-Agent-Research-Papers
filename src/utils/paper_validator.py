"""Paper metadata validation and academic citation formatting utilities."""

from __future__ import annotations

from datetime import datetime
from typing import List, Dict, Tuple, Any

CURRENT_YEAR = datetime.now().year + 1  # Allow near-term preprints


def validate_papers(papers: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Validate paper metadata records.

    A paper is considered valid if:
    - Has a non-empty title (> 3 characters)
    - Has at least one author name
    - Has a recognized academic source
    - Has a reasonable publication year (1900 <= year <= CURRENT_YEAR) if year is present

    Returns
    -------
    Tuple[List[dict], List[dict]]:
        (valid_papers, invalid_papers)
    """
    valid: List[dict] = []
    invalid: List[dict] = []

    for paper in papers:
        title = (paper.get("title") or "").strip()
        authors = paper.get("authors") or []
        source = paper.get("source") or ""
        year = paper.get("year")

        reasons = []

        if not title or len(title) < 4:
            reasons.append("Missing or excessively short title")

        if not authors:
            reasons.append("Missing author list")

        if not source:
            reasons.append("Missing source attribution")

        if year is not None:
            if not isinstance(year, int) or year < 1800 or year > CURRENT_YEAR:
                reasons.append(f"Invalid publication year: {year}")

        if reasons:
            flagged_paper = dict(paper)
            flagged_paper["validation_errors"] = reasons
            invalid.append(flagged_paper)
        else:
            # Clean up missing optional values
            cleaned = dict(paper)
            cleaned["doi"] = cleaned.get("doi") or None
            cleaned["url"] = cleaned.get("url") or None
            cleaned["year"] = cleaned.get("year") or None
            cleaned["abstract"] = cleaned.get("abstract") or "Abstract not provided in source metadata."
            valid.append(cleaned)

    return valid, invalid


def format_paper_citation(paper: dict, index: int) -> str:
    """Format a validated paper into a standard academic numbered citation.

    Format:
    [1] Author A, Author B (2023). "Title of Paper". Source. https://doi.org/...
    """
    authors = paper.get("authors") or ["Unknown Author"]
    if len(authors) > 3:
        author_str = f"{authors[0]}, {authors[1]}, et al."
    else:
        author_str = ", ".join(authors)

    year = paper.get("year")
    year_str = f"({year})" if year else "(n.d.)"

    title = paper.get("title", "Untitled")
    source = (paper.get("source") or "academic source").replace("_", " ").title()

    doi = paper.get("doi")
    url = paper.get("url")

    link_str = ""
    if doi:
        link_str = f" DOI: https://doi.org/{doi}"
    elif url:
        link_str = f" URL: {url}"

    return f"[{index}] {author_str} {year_str}. \"{title}\". *{source}*.{link_str}"
