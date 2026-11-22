"""Paper deduplication utilities using DOI, paper ID, and fuzzy title similarity."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Dict, Any


def _normalize_title(title: str) -> str:
    """Normalize paper title for accurate similarity comparison."""
    if not title:
        return ""
    text = title.lower()
    # Remove punctuation and extra whitespace
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _count_populated_fields(paper: dict) -> int:
    """Count non-empty, non-None metadata fields in a paper dictionary."""
    count = 0
    for key, val in paper.items():
        if val is not None and val != "" and val != []:
            count += 1
    return count


def _merge_papers(existing: dict, incoming: dict) -> dict:
    """Merge two matching paper records, retaining the richest metadata."""
    merged = dict(existing)
    for key, val in incoming.items():
        existing_val = merged.get(key)
        # If existing is missing or less specific, take incoming
        if existing_val is None or existing_val == "" or existing_val == []:
            merged[key] = val
        elif key == "abstract" and isinstance(val, str) and isinstance(existing_val, str):
            if len(val) > len(existing_val):
                merged[key] = val
        elif key == "authors" and isinstance(val, list) and isinstance(existing_val, list):
            if len(val) > len(existing_val) and "Unknown" not in val[0]:
                merged[key] = val
    return merged


def deduplicate_papers(papers: List[dict]) -> List[dict]:
    """Deduplicate a list of paper records.

    Matches by:
    1. Exact normalized DOI
    2. Exact source + paper_id
    3. High title similarity (> 0.85)

    Parameters
    ----------
    papers:
        List of raw paper dictionaries.

    Returns
    -------
    List[dict]:
        Unique papers with merged, best-available metadata.
    """
    if not papers:
        return []

    unique_papers: List[dict] = []
    doi_map: Dict[str, int] = {}
    id_map: Dict[str, int] = {}

    for paper in papers:
        title = (paper.get("title") or "").strip()
        if not title:
            continue

        norm_title = _normalize_title(title)
        doi = (paper.get("doi") or "").strip().lower()
        source = paper.get("source") or ""
        paper_id = (paper.get("paper_id") or "").strip()
        matched_idx = None

        # 1. Match by DOI
        if doi and doi in doi_map:
            matched_idx = doi_map[doi]

        # 2. Match by source + paper_id
        if matched_idx is None and paper_id:
            key = f"{source}:{paper_id}"
            if key in id_map:
                matched_idx = id_map[key]

        # 3. Match by Title Similarity
        if matched_idx is None and norm_title:
            for idx, existing in enumerate(unique_papers):
                existing_norm = _normalize_title(existing.get("title", ""))
                if not existing_norm:
                    continue
                # Exact normalized title
                if norm_title == existing_norm:
                    matched_idx = idx
                    break
                # Fuzzy ratio
                ratio = SequenceMatcher(None, norm_title, existing_norm).ratio()
                if ratio >= 0.85:
                    matched_idx = idx
                    break

        if matched_idx is not None:
            # Merge with existing
            unique_papers[matched_idx] = _merge_papers(unique_papers[matched_idx], paper)
        else:
            new_idx = len(unique_papers)
            unique_papers.append(dict(paper))
            if doi:
                doi_map[doi] = new_idx
            if paper_id:
                id_map[f"{source}:{paper_id}"] = new_idx

    removed = len(papers) - len(unique_papers)
    if removed > 0:
        print(f"  [Deduplication] Removed {removed} duplicate paper(s). Total unique: {len(unique_papers)}")

    return unique_papers
