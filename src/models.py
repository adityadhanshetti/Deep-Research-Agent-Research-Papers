"""Pydantic v2 data models for structured outputs used throughout the Deep Research Agent."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Paper representation
# ---------------------------------------------------------------------------

class ResearchPaper(BaseModel):
    """A single research paper discovered during the search phase."""

    title: str = Field(
        description="Title of the research paper."
    )
    authors: List[str] = Field(
        description="List of author names for the paper."
    )
    year: Optional[int] = Field(
        default=None,
        description="Publication year of the paper, if known."
    )
    abstract: Optional[str] = Field(
        default=None,
        description="Abstract or summary text of the paper."
    )
    doi: Optional[str] = Field(
        default=None,
        description="Digital Object Identifier (DOI) of the paper."
    )
    url: Optional[str] = Field(
        default=None,
        description="URL where the paper can be accessed."
    )
    source: str = Field(
        description=(
            "Source database that provided this paper. "
            "One of: 'arxiv', 'semantic_scholar', 'openalex', 'tavily'."
        )
    )
    paper_id: Optional[str] = Field(
        default=None,
        description="Source-specific unique identifier for the paper."
    )


# ---------------------------------------------------------------------------
# Phase 1 – Scoping
# ---------------------------------------------------------------------------

class ClarificationResult(BaseModel):
    """Result of evaluating whether the user's topic needs clarification."""

    needs_clarification: bool = Field(
        description="Whether the topic requires clarification before research can begin."
    )
    questions: List[str] = Field(
        default_factory=list,
        max_length=4,
        description="Up to four clarifying questions to ask the user (empty if none needed)."
    )
    reasoning: str = Field(
        description="Explanation of why clarification is or is not needed."
    )


class ResearchBrief(BaseModel):
    """Structured brief that guides the research phase after scoping."""

    original_topic: str = Field(
        description="The user's original research topic as provided."
    )
    refined_topic: str = Field(
        description="A refined, more precise formulation of the research topic."
    )
    key_aspects: List[str] = Field(
        description="Key aspects or dimensions of the topic to investigate."
    )
    search_queries: List[str] = Field(
        description="Initial search queries to use when discovering papers."
    )
    year_range: Optional[str] = Field(
        default=None,
        description="Preferred publication year range for filtering results (e.g. '2020-2024')."
    )
    methodology_focus: Optional[str] = Field(
        default=None,
        description="Specific methodology or approach to focus on, if any."
    )
    scope_boundaries: str = Field(
        description="Clear boundaries defining what is in-scope and out-of-scope."
    )


# ---------------------------------------------------------------------------
# Phase 2 – Research / Delegation
# ---------------------------------------------------------------------------

class SubtopicAssignment(BaseModel):
    """Assignment descriptor for a single sub-agent research task."""

    subtopic_id: int = Field(
        description="Unique integer identifier for this subtopic."
    )
    title: str = Field(
        description="Short, descriptive title of the subtopic."
    )
    description: str = Field(
        description="Detailed description of what the sub-agent should investigate."
    )
    search_queries: List[str] = Field(
        description="Search queries the sub-agent should execute."
    )
    focus_areas: List[str] = Field(
        description="Specific focus areas or angles within the subtopic."
    )


class DelegationPlan(BaseModel):
    """Plan produced by the supervisor to partition research across sub-agents."""

    num_agents: int = Field(
        description="Number of sub-agents to dispatch."
    )
    rationale: str = Field(
        description="Explanation of how and why the topic was divided this way."
    )
    assignments: List[SubtopicAssignment] = Field(
        description="Ordered list of subtopic assignments, one per sub-agent."
    )


class SubAgentFindings(BaseModel):
    """Collected findings returned by a single sub-agent after researching its subtopic."""

    subtopic_id: int = Field(
        description="The subtopic identifier this finding set corresponds to."
    )
    subtopic_title: str = Field(
        description="Human-readable title of the subtopic."
    )
    papers: List[ResearchPaper] = Field(
        description="Papers discovered and deemed relevant by the sub-agent."
    )
    synthesis: str = Field(
        description="Organized narrative synthesizing the key findings from the papers."
    )
    key_themes: List[str] = Field(
        description="Major themes or patterns identified across the papers."
    )
    limitations_noted: List[str] = Field(
        description="Limitations, gaps, or caveats noted during the research."
    )


class GapAnalysis(BaseModel):
    """Analysis of coverage gaps after one or more research rounds."""

    covered_areas: List[str] = Field(
        description="Topic areas that have been adequately covered by the research."
    )
    missing_areas: List[str] = Field(
        description="Topic areas that are missing or insufficiently covered."
    )
    unanswered_questions: List[str] = Field(
        description="Important questions that remain unanswered."
    )
    needs_more_research: bool = Field(
        description="Whether additional research rounds are required."
    )
    reasoning: str = Field(
        description="Explanation supporting the gap analysis conclusions."
    )
    additional_queries: List[str] = Field(
        default_factory=list,
        description="Suggested additional search queries to fill identified gaps."
    )


class SupervisorEvaluation(BaseModel):
    """Supervisor's evaluation of a completed research round."""

    round_number: int = Field(
        description="Sequential round number of this evaluation (1-indexed)."
    )
    total_papers: int = Field(
        description="Total number of unique papers collected across all sub-agents so far."
    )
    gap_analysis: GapAnalysis = Field(
        description="Detailed gap analysis for the current round."
    )
    decision: str = Field(
        description=(
            "Supervisor's decision after the evaluation. "
            "One of: 'sufficient' (research is complete) or 'needs_more' (another round is needed)."
        )
    )
