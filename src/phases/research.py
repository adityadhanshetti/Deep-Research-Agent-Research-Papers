"""Phase 2: Research — Multi-agent supervisor, parallel sub-agents, findings cleanup, and gap analysis."""

from __future__ import annotations

import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import get_llm, get_fast_llm, settings
from src.models import (
    DelegationPlan,
    SubAgentFindings,
    SupervisorEvaluation,
    GapAnalysis,
    ResearchPaper,
    SubtopicAssignment,
)
from src.state import ResearchState
from src.tools.arxiv_search import search_arxiv
from src.tools.semantic_scholar import search_semantic_scholar
from src.tools.openalex_search import search_openalex
from src.tools.tavily_search import search_tavily
from src.utils.deduplication import deduplicate_papers
from src.utils.structured_helper import invoke_structured, ainvoke_structured
from src.utils.display import (
    print_header,
    print_subheader,
    print_delegation_plan,
    print_paper_summary,
    print_supervisor_decision,
    print_progress,
)


def supervisor_plan(brief: dict, round_number: int = 1, gaps: Optional[List[str]] = None) -> DelegationPlan:
    """Analyze the research brief and dynamically partition the work across sub-agents."""
    llm = get_llm(temperature=0.1)

    if round_number == 1 or not gaps:
        system_prompt = (
            "You are the Research Supervisor coordinating a team of specialized AI research sub-agents.\n"
            "Analyze the provided research brief and dynamically determine the optimal number of sub-agents required:\n"
            "- For focused, straightforward topics: use 1 or 2 sub-agents.\n"
            "- For multifaceted, broad, or complex topics: divide into 2 to 3 independent, non-overlapping subtopics.\n"
            "Do NOT hardcode the number of agents. Tailor the count directly to the complexity of the topic.\n"
            "For each sub-agent, assign specific focus areas and 2 tailored search queries for academic paper search."
        )
        content = f"Research Brief:\n{brief}"
    else:
        system_prompt = (
            "You are the Research Supervisor conducting a targeted follow-up research round.\n"
            "Previous research left specific gaps. Formulate 1 to 2 targeted sub-agent assignments "
            "designed specifically to investigate and resolve these identified missing areas."
        )
        content = f"Original Brief:\n{brief}\n\nIdentified Gaps To Target:\n" + "\n".join(f"- {g}" for g in gaps)

    plan: DelegationPlan = invoke_structured(
        llm,
        DelegationPlan,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ]
    )
    return plan


async def _fetch_papers_for_query(query: str) -> List[dict]:
    """Execute parallel queries across all academic databases for a single search phrase."""
    results = await asyncio.gather(
        search_arxiv(query, max_results=settings.max_papers_per_search),
        search_semantic_scholar(query, max_results=settings.max_papers_per_search),
        search_openalex(query, max_results=settings.max_papers_per_search),
        search_tavily(query, max_results=3),
        return_exceptions=True
    )

    batch_papers: List[dict] = []
    for res in results:
        if isinstance(res, list):
            batch_papers.extend(res)
        elif isinstance(res, Exception):
            pass

    return batch_papers


async def run_single_subagent(
    assignment: SubtopicAssignment,
    brief_context: str,
    agent_id: int,
    semaphore: asyncio.Semaphore
) -> SubAgentFindings:
    """Execute an isolated research sub-agent focusing exclusively on its assigned subtopic."""
    print_progress(agent_id, f"Initiated research on '{assignment.title}'")

    raw_papers: List[dict] = []
    for query in assignment.search_queries:
        print_progress(agent_id, f"Querying academic APIs for: '{query}'")
        found = await _fetch_papers_for_query(query)
        raw_papers.extend(found)

    # Deduplicate local sub-agent papers
    subagent_papers = deduplicate_papers(raw_papers)
    print_progress(agent_id, f"Discovered {len(subagent_papers)} relevant academic paper(s)")

    # Prepare concise evidence summary for sub-agent synthesis (token-optimized)
    paper_evidence_snippets = []
    for idx, p in enumerate(subagent_papers[:6], 1):
        p_title = p.get("title", "Untitled")
        p_year = p.get("year", "n.d.")
        p_auth = ", ".join(p.get("authors", ["Unknown"])[:2])
        p_abs = (p.get("abstract") or "No abstract available.")[:180]
        paper_evidence_snippets.append(
            f"Paper [{idx}]: \"{p_title}\" by {p_auth} ({p_year})\nSummary: {p_abs}\n"
        )

    evidence_text = "\n".join(paper_evidence_snippets) if paper_evidence_snippets else "No papers retrieved."

    # Sub-agent synthesizes findings in isolated context under concurrency semaphore
    async with semaphore:
        llm = get_fast_llm(temperature=0.1)
        subagent_prompt = (
            f"You are an isolated specialist research sub-agent assigned to: {assignment.title}.\n"
            f"Subtopic Objective: {assignment.description}\n"
            f"Focus Areas: {', '.join(assignment.focus_areas)}\n\n"
            f"Context from Brief:\n{brief_context}\n\n"
            f"Retrieved Academic Papers:\n{evidence_text}\n\n"
            "Instructions:\n"
            "1. Synthesize the core findings, experimental results, and methodologies from these papers.\n"
            "2. Directly cite papers using the paper numbers (e.g. [1], [2]).\n"
            "3. Highlight key technical themes and concrete limitations.\n"
            "4. DO NOT invent citations or unsupported claims."
        )

        findings: SubAgentFindings = await ainvoke_structured(
            llm,
            SubAgentFindings,
            [
                SystemMessage(content="You are a rigorous academic researcher."),
                HumanMessage(content=subagent_prompt)
            ]
        )

    # Convert raw papers to ResearchPaper models
    valid_models: List[ResearchPaper] = []
    for p in subagent_papers:
        try:
            valid_models.append(ResearchPaper(**p))
        except Exception:
            pass

    findings.papers = valid_models
    findings.subtopic_id = agent_id
    findings.subtopic_title = assignment.title

    return findings


async def cleanup_findings(findings: SubAgentFindings, semaphore: asyncio.Semaphore) -> SubAgentFindings:
    """Clean, structure, and eliminate noise from sub-agent findings using a separate LLM call."""
    fast_llm = get_fast_llm(temperature=0.0)

    cleanup_prompt = (
        f"You are a meticulous scientific editor.\n"
        f"Review and clean the following research subtopic findings:\n"
        f"Subtopic: {findings.subtopic_title}\n"
        f"Current Synthesis:\n{findings.synthesis}\n\n"
        "Tasks:\n"
        "1. Remove any conversational filler or raw search artifacts.\n"
        "2. Ensure citations [1], [2] are preserved and claims are concise and evidence-grounded.\n"
        "Return the polished academic synthesis directly."
    )

    async with semaphore:
        for attempt in range(3):
            try:
                response = await fast_llm.ainvoke([HumanMessage(content=cleanup_prompt)])
                polished = response.content.strip()
                if polished and len(polished) > 50:
                    findings.synthesis = polished
                break
            except Exception as exc:
                if "429" in str(exc):
                    await asyncio.sleep(5.0)
                    continue
                break

    return findings


def supervisor_evaluate(
    brief: dict,
    all_findings: List[dict],
    all_papers: List[dict],
    current_round: int
) -> SupervisorEvaluation:
    """Perform gap analysis and determine if an additional research round is required."""
    llm = get_llm(temperature=0.0)

    findings_summary = []
    for f in all_findings:
        findings_summary.append(
            f"Subtopic [{f.get('subtopic_id')}]: {f.get('subtopic_title')}\n"
            f"Themes: {', '.join(f.get('key_themes', []))}\n"
            f"Synthesis Excerpt: {f.get('synthesis', '')[:160]}...\n"
        )

    eval_prompt = (
        f"Research Brief:\n{brief}\n\n"
        f"Current Round: {current_round} (Maximum additional rounds allowed: {settings.max_additional_rounds})\n"
        f"Total Unique Papers Collected: {len(all_papers)}\n\n"
        f"Synthesized Findings Coverage:\n" + "\n".join(findings_summary) + "\n\n"
        "Evaluate whether the collected research is sufficient to write a comprehensive literature review:\n"
        "1. Identify covered areas vs. missing areas.\n"
        "2. If significant research gaps exist and current_round <= max_additional_rounds, decision='needs_more'.\n"
        "3. If the brief is adequately covered, or if maximum rounds reached, decision='sufficient'."
    )

    evaluation: SupervisorEvaluation = invoke_structured(
        llm,
        SupervisorEvaluation,
        [
            SystemMessage(content="You are an exacting academic research supervisor."),
            HumanMessage(content=eval_prompt)
        ]
    )

    # Enforce programmatic bound on additional rounds
    if current_round >= (1 + settings.max_additional_rounds):
        evaluation.decision = "sufficient"
        evaluation.gap_analysis.needs_more_research = False

    return evaluation


def research_phase(state: ResearchState) -> dict:
    """LangGraph node orchestrating Phase 2: multi-agent research and supervisor iterations."""
    brief = state.get("research_brief") or {}
    current_round = state.get("current_round", 1)
    accumulated_findings = list(state.get("all_findings") or [])
    accumulated_papers = list(state.get("all_papers") or [])
    evaluations = list(state.get("supervisor_evaluations") or [])

    print_header(f"Phase 2: Research (Round {current_round})", "🔬")

    # Determine gaps if round > 1
    gaps = None
    if current_round > 1 and evaluations:
        last_eval = evaluations[-1]
        gaps = last_eval.get("gap_analysis", {}).get("missing_areas", [])

    # Supervisor develops dynamic delegation plan
    plan = supervisor_plan(brief, round_number=current_round, gaps=gaps)
    plan_dict = plan.model_dump()
    print_delegation_plan(plan_dict)

    # Concurrency semaphore to stay safely within Groq token-per-minute limits
    semaphore = asyncio.Semaphore(2)

    # Run sub-agents in parallel using asyncio
    async def _execute_parallel_agents():
        tasks = [
            run_single_subagent(
                assignment=assignment,
                brief_context=brief.get("refined_topic", ""),
                agent_id=assignment.subtopic_id,
                semaphore=semaphore,
            )
            for assignment in plan.assignments
        ]
        raw_results: List[SubAgentFindings] = await asyncio.gather(*tasks)

        # Cleanup step for each sub-agent
        cleaned_tasks = [cleanup_findings(f, semaphore) for f in raw_results]
        cleaned_results: List[SubAgentFindings] = await asyncio.gather(*cleaned_tasks)
        return cleaned_results

    try:
        findings_objects = asyncio.run(_execute_parallel_agents())
    except RuntimeError:
        # If an event loop is already active
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        findings_objects = loop.run_until_complete(_execute_parallel_agents())

    # Accumulate papers & findings
    new_findings_dicts = [f.model_dump() for f in findings_objects]
    new_papers: List[dict] = []
    for f in findings_objects:
        for p in f.papers:
            new_papers.append(p.model_dump())

    all_round_papers = deduplicate_papers(accumulated_papers + new_papers)
    all_round_findings = accumulated_findings + new_findings_dicts

    print_paper_summary(all_round_papers)

    # Supervisor Evaluation
    evaluation = supervisor_evaluate(
        brief=brief,
        all_findings=all_round_findings,
        all_papers=all_round_papers,
        current_round=current_round
    )
    eval_dict = evaluation.model_dump()
    print_supervisor_decision(eval_dict)

    research_complete = (evaluation.decision == "sufficient")

    return {
        "delegation_plan": plan_dict,
        "all_findings": new_findings_dicts,
        "all_papers": new_papers,
        "supervisor_evaluations": [eval_dict],
        "current_round": current_round + 1 if not research_complete else current_round,
        "research_complete": research_complete,
        "phase": "write" if research_complete else "research",
    }


def route_research(state: ResearchState) -> str:
    """Conditional edge router: continue research round or advance to report writing."""
    if state.get("research_complete", False):
        return "write"
    current_round = state.get("current_round", 1)
    if current_round > (1 + settings.max_additional_rounds):
        return "write"
    return "research"
