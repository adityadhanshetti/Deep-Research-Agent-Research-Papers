"""Phase 1: Scope — Topic clarification check and structured research brief generation."""

from __future__ import annotations

import sys
import json
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import get_llm
from src.models import ClarificationResult, ResearchBrief
from src.state import ResearchState
from src.utils.display import print_header, print_research_brief, print_subheader
from src.utils.structured_helper import invoke_structured


def check_clarity(state: ResearchState) -> dict:
    """Analyze the user's research topic to determine if clarification is required."""
    topic = state.get("topic", "").strip()
    print_header("Phase 1: Scope & Clarification", "🎯")
    print(f"  User Query: \"{topic}\"")

    llm = get_llm(temperature=0.0)

    system_prompt = (
        "You are an expert academic research advisor. "
        "Assess whether the user's research topic is sufficiently focused for a deep academic literature review.\n\n"
        "Guidelines:\n"
        "1. If the topic is specific and well-defined (e.g. mentions concrete algorithms, architectures, specific problems, or methods), "
        "set needs_clarification=False and questions=[].\n"
        "2. If the topic is broad, vague, or ambiguous (e.g. 'AI in healthcare', 'blockchain'), "
        "set needs_clarification=True and formulate 2 to 4 concise clarification questions regarding scope, "
        "timeframe/year range, target methodology, or specific clinical/domain focus.\n"
        "3. Provide clear reasoning for your decision."
    )

    result: ClarificationResult = invoke_structured(
        llm,
        ClarificationResult,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Topic to evaluate: {topic}")
        ]
    )

    if result.needs_clarification:
        print("\n  🔍 Status: Clarification REQUIRED.")
        print(f"  Reasoning: {result.reasoning}")
        print("  Clarification Questions:")
        for idx, q in enumerate(result.questions, 1):
            print(f"    {idx}. {q}")
    else:
        print("\n  ⚡ Status: Topic is sufficiently clear. Clarification SKIPPED.")
        print(f"  Reasoning: {result.reasoning}")

    return {
        "needs_clarification": result.needs_clarification,
        "clarification_questions": result.questions,
        "phase": "scope",
    }


def ask_clarification(state: ResearchState) -> dict:
    """Collect clarification answers from the user or automated state input."""
    existing_answers = state.get("clarification_answers")
    questions = state.get("clarification_questions") or []

    if existing_answers:
        # Pre-provided answers (e.g., from evaluation harness)
        print_subheader("Clarification Answers (Provided)", "💬")
        print(f"  {existing_answers}")
        return {"clarification_answers": existing_answers}

    print_subheader("Please provide clarification", "❓")
    answers = []
    for idx, q in enumerate(questions, 1):
        try:
            if not sys.stdin.isatty():
                ans = "Recent peer-reviewed advancements, formal algorithmic formulations, and benchmarks from 2022 to 2025."
                print(f"\n  Q{idx}: {q}\n  Automated Answer: {ans}")
            else:
                ans = input(f"\n  Q{idx}: {q}\n  Your Answer: ").strip()
            answers.append(f"Q: {q}\nA: {ans}")
        except Exception:
            answers.append(f"Q: {q}\nA: General recent advancements and state-of-the-art methodology.")

    collected = "\n".join(answers)
    return {"clarification_answers": collected}


def generate_brief(state: ResearchState) -> dict:
    """Generate a structured, authoritative research brief based on topic and clarification."""
    topic = state.get("topic", "")
    clarification_answers = state.get("clarification_answers")

    llm = get_llm(temperature=0.1)

    prompt = f"Original User Topic: {topic}\n"
    if clarification_answers:
        prompt += f"\nUser Clarification Answers:\n{clarification_answers}\n"

    system_prompt = (
        "You are the Lead Scientific Director for an autonomous literature review system.\n"
        "Formulate a precise, comprehensive research brief based on the topic and any clarification.\n"
        "Generate 4-6 specific search queries optimized for academic databases like arXiv, Semantic Scholar, and OpenAlex.\n"
        "Define clear boundaries for what is in-scope and out-of-scope to strictly guide sub-agents."
    )

    brief: ResearchBrief = invoke_structured(
        llm,
        ResearchBrief,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
    )

    brief_dict = brief.model_dump()
    print_research_brief(brief_dict)

    return {
        "research_brief": brief_dict,
        "phase": "research",
    }


def route_clarification(state: ResearchState) -> str:
    """Conditional edge router: determine whether clarification interaction is needed."""
    if state.get("needs_clarification") and not state.get("clarification_answers"):
        return "ask_clarification"
    return "generate_brief"
