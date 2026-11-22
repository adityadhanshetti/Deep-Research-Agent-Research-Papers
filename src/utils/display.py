"""Terminal display formatting helpers for the Deep Research Agent UI."""

from __future__ import annotations

from typing import List, Dict, Any


def print_header(title: str, emoji: str = "📌"):
    """Print a major phase banner."""
    print(f"\n{'='*75}")
    print(f"  {emoji}  {title.upper()}")
    print(f"{'='*75}")


def print_subheader(title: str, emoji: str = "▸"):
    """Print a section subheader."""
    print(f"\n{'-'*55}")
    print(f"  {emoji} {title}")
    print(f"{'-'*55}")


def print_research_brief(brief: dict):
    """Pretty-print the structured research brief."""
    print_subheader("RESEARCH BRIEF", "📋")
    print(f"  Refined Topic       : {brief.get('refined_topic')}")
    print(f"  Scope Boundaries    : {brief.get('scope_boundaries')}")
    print(f"  Year Range          : {brief.get('year_range') or 'Any recent'}")
    print(f"  Methodology Focus   : {brief.get('methodology_focus') or 'Comprehensive'}")

    key_aspects = brief.get("key_aspects") or []
    if key_aspects:
        print("  Key Aspects:")
        for aspect in key_aspects:
            print(f"    • {aspect}")

    search_queries = brief.get("search_queries") or []
    if search_queries:
        print("  Initial Search Queries:")
        for q in search_queries:
            print(f"    🔍 {q}")


def print_delegation_plan(plan: dict):
    """Pretty-print the supervisor's delegation plan."""
    num_agents = plan.get("num_agents", 0)
    rationale = plan.get("rationale", "")
    print_subheader(f"DELEGATION PLAN: {num_agents} Parallel Research Sub-Agent(s)", "🧭")
    print(f"  Rationale: {rationale}\n")

    assignments = plan.get("assignments") or []
    for a in assignments:
        agent_id = a.get("subtopic_id")
        title = a.get("title")
        desc = a.get("description")
        queries = a.get("search_queries") or []
        focus = a.get("focus_areas") or []

        print(f"  [Agent #{agent_id}] {title}")
        print(f"    Description : {desc}")
        print(f"    Focus Areas : {', '.join(focus)}")
        print(f"    Queries     : {', '.join(queries[:3])}")
        print()


def print_paper_summary(papers: List[dict]):
    """Print an aggregate breakdown of discovered papers."""
    total = len(papers)
    counts: Dict[str, int] = {}
    for p in papers:
        src = p.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1

    src_breakdown = ", ".join(f"{k}: {v}" for k, v in counts.items())
    print(f"  📚 Total papers collected: {total} ({src_breakdown})")


def print_supervisor_decision(evaluation: dict):
    """Print supervisor's evaluation and next-step decision."""
    round_no = evaluation.get("round_number", 1)
    decision = evaluation.get("decision", "sufficient")
    gap_analysis = evaluation.get("gap_analysis") or {}

    print_subheader(f"SUPERVISOR EVALUATION (Round {round_no})", "⚖️")
    print(f"  Total Unique Papers : {evaluation.get('total_papers', 0)}")
    print(f"  Decision            : {'✅ RESEARCH SUFFICIENT' if decision == 'sufficient' else '🔄 ADDITIONAL RESEARCH REQUIRED'}")

    covered = gap_analysis.get("covered_areas") or []
    if covered:
        print("  Covered Dimensions  :")
        for c in covered:
            print(f"    ✔ {c}")

    missing = gap_analysis.get("missing_areas") or []
    if missing:
        print("  Identified Gaps     :")
        for m in missing:
            print(f"    ⚠️ {m}")

    reasoning = gap_analysis.get("reasoning", "")
    if reasoning:
        print(f"  Reasoning           : {reasoning}")


def print_progress(agent_id: int, message: str):
    """Print sub-agent progress update."""
    print(f"  🤖 [Sub-Agent #{agent_id}] {message}")
