"""LangGraph workflow definition for the Deep Research Agent."""

from __future__ import annotations

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from src.state import ResearchState
from src.phases.scope import (
    check_clarity,
    ask_clarification,
    generate_brief,
    route_clarification,
)
from src.phases.research import research_phase, route_research
from src.phases.write import write_report


def build_research_graph():
    """Build and compile the end-to-end 3-phase Deep Research Agent graph."""
    workflow = StateGraph(ResearchState)

    # 1. Register Nodes
    workflow.add_node("check_clarity", check_clarity)
    workflow.add_node("ask_clarification", ask_clarification)
    workflow.add_node("generate_brief", generate_brief)
    workflow.add_node("research_phase", research_phase)
    workflow.add_node("write_report", write_report)

    # 2. Add Edges
    workflow.add_edge(START, "check_clarity")

    workflow.add_conditional_edges(
        "check_clarity",
        route_clarification,
        {
            "ask_clarification": "ask_clarification",
            "generate_brief": "generate_brief",
        },
    )

    workflow.add_edge("ask_clarification", "generate_brief")
    workflow.add_edge("generate_brief", "research_phase")

    workflow.add_conditional_edges(
        "research_phase",
        route_research,
        {
            "research": "research_phase",
            "write": "write_report",
        },
    )

    workflow.add_edge("write_report", END)

    return workflow.compile()


# Pre-compiled research graph
research_graph = build_research_graph()


def run_research(topic: str, clarification_answers: Optional[str] = None) -> dict:
    """Execute the Deep Research Agent on a research topic.

    Parameters
    ----------
    topic:
        The scientific topic or research question.
    clarification_answers:
        Optional pre-populated answers to clarification questions (useful for automated evals).

    Returns
    -------
    dict:
        Final state of the research run containing brief, findings, papers, and report.
    """
    initial_state: ResearchState = {
        "topic": topic,
        "clarification_answers": clarification_answers,
        "needs_clarification": False,
        "clarification_questions": [],
        "research_brief": None,
        "delegation_plan": None,
        "all_findings": [],
        "all_papers": [],
        "supervisor_evaluations": [],
        "current_round": 1,
        "research_complete": False,
        "final_report": None,
        "phase": "scope",
        "messages": [],
    }

    final_state = research_graph.invoke(initial_state)
    return final_state
