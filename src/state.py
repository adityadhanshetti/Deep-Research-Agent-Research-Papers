"""LangGraph state definition for the Deep Research Agent.

Uses ``TypedDict`` with ``Annotated`` reducers so that list-valued channels
accumulate across graph nodes rather than being overwritten.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional, TypedDict

from langgraph.graph import add_messages


class ResearchState(TypedDict):
    """Top-level state threaded through every node of the research graph."""

    # ------------------------------------------------------------------
    # User input
    # ------------------------------------------------------------------
    topic: str
    """The user's original research topic or question."""

    clarification_answers: Optional[str]
    """User's answers to any clarification questions, if applicable."""

    # ------------------------------------------------------------------
    # Phase 1 – Scoping outputs
    # ------------------------------------------------------------------
    needs_clarification: bool
    """Whether the scoping phase determined clarification is needed."""

    clarification_questions: List[str]
    """List of clarification questions generated during scoping."""

    research_brief: Optional[dict]
    """Serialized ``ResearchBrief`` produced after scoping is finalized."""

    # ------------------------------------------------------------------
    # Phase 2 – Research outputs
    # ------------------------------------------------------------------
    delegation_plan: Optional[dict]
    """Serialized ``DelegationPlan`` describing how research is split across sub-agents."""

    all_findings: Annotated[list, operator.add]
    """Accumulated ``SubAgentFindings`` dicts from every research round."""

    all_papers: Annotated[list, operator.add]
    """Accumulated ``ResearchPaper`` dicts from every research round."""

    supervisor_evaluations: Annotated[list, operator.add]
    """Accumulated ``SupervisorEvaluation`` dicts, one per completed round."""

    current_round: int
    """Current research round number (1-indexed)."""

    research_complete: bool
    """Flag indicating whether the supervisor has deemed research sufficient."""

    # ------------------------------------------------------------------
    # Phase 3 – Report outputs
    # ------------------------------------------------------------------
    final_report: Optional[str]
    """The fully synthesized research report in Markdown format."""

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------
    phase: str
    """Current phase of the pipeline: ``'scope'``, ``'research'``, ``'write'``, or ``'done'``."""

    messages: Annotated[list, add_messages]
    """LangGraph message list with the ``add_messages`` reducer for chat history."""
