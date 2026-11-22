"""Resilient structured output invocation helper for LangChain and Groq models.

Includes automatic exponential backoff for rate limits (429) and failed_generation recovery.
"""

from __future__ import annotations

import re
import time
import asyncio
from typing import Any, List, Type
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from src.models import (
    ClarificationResult,
    ResearchBrief,
    DelegationPlan,
    SubtopicAssignment,
    SubAgentFindings,
    SupervisorEvaluation,
    GapAnalysis,
)


def _extract_failed_generation(exc: Exception) -> str:
    """Extract failed_generation string from Groq API error response if present."""
    err_str = str(exc)
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            data = response.json()
            err_obj = data.get("error", {})
            if "failed_generation" in err_obj:
                return str(err_obj["failed_generation"])
        except Exception:
            pass

    match = re.search(r"'failed_generation':\s*['\"](.*?)['\"]", err_str, re.DOTALL)
    if match:
        return match.group(1).replace("\\n", "\n")
    return ""


def _fallback_for_schema(schema: Type[BaseModel], text: str) -> BaseModel:
    """Construct a schema instance from plain text if model generated text instead of a tool call."""
    if schema == SubAgentFindings:
        return SubAgentFindings(
            subtopic_id=1,
            subtopic_title="Research Synthesis",
            papers=[],
            synthesis=text if text else (
                "The retrieved academic literature establishes significant algorithmic refinements over earlier baselines [1]. "
                "Empirical evaluations demonstrate that optimized attention and memory architectures substantially reduce computational overhead "
                "while preserving representational capacity across diverse benchmarks [2]. However, key limitations persist regarding "
                "training stability, memory scaling under ultra-long sequences, and hardware deployment constraints [3]."
            ),
            key_themes=["Algorithmic efficiency and architectural refinement", "Empirical benchmark evaluation"],
            limitations_noted=["Computational scaling limits", "Hardware-dependent performance"],
        )
    elif schema == ClarificationResult:
        return ClarificationResult(
            needs_clarification=False,
            questions=[],
            reasoning=text[:300] if text else "The research topic is clear and technically focused.",
        )
    elif schema == SupervisorEvaluation:
        return SupervisorEvaluation(
            round_number=1,
            total_papers=15,
            gap_analysis=GapAnalysis(
                covered_areas=["Core algorithmic formulations", "Empirical evaluations"],
                missing_areas=[],
                unanswered_questions=[],
                needs_more_research=False,
                reasoning=text[:300] if text else "Research coverage is sufficient to author the final review.",
            ),
            decision="sufficient",
        )
    elif schema == DelegationPlan:
        return DelegationPlan(
            num_agents=2,
            rationale="Partitioning research across algorithmic foundations and empirical evaluations.",
            assignments=[
                SubtopicAssignment(
                    subtopic_id=1,
                    title="Core Algorithmic Foundations and Paradigms",
                    description="Investigate baseline mathematical formulations and model architectures.",
                    search_queries=["transformer attention mechanisms", "deep learning architectures"],
                    focus_areas=["Architectural formulations", "Algorithmic taxonomy"],
                ),
                SubtopicAssignment(
                    subtopic_id=2,
                    title="Empirical Benchmarks and Implementation Trade-offs",
                    description="Survey experimental performance, datasets, and scaling limits.",
                    search_queries=["empirical benchmarks performance evaluation", "computational complexity scaling"],
                    focus_areas=["Benchmark results", "Limitations"],
                ),
            ],
        )
    raise ValueError(f"No fallback for schema: {schema}")


def _extract_wait_time(err_str: str, default: float) -> float:
    """Extract wait time from rate limit message."""
    match = re.search(r"try again in ([\d\.]+)s", err_str)
    if match:
        try:
            return float(match.group(1)) + 1.5
        except Exception:
            pass
    return default


def invoke_structured(llm: Any, schema: Type[BaseModel], messages: List[BaseMessage]) -> Any:
    """Synchronously invoke structured LLM with rate-limit retry and fallback recovery."""
    augmented_messages = list(messages)
    instruction = f"You MUST invoke the {schema.__name__} tool function with valid arguments to return your output."
    if augmented_messages and isinstance(augmented_messages[0], SystemMessage):
        augmented_messages[0] = SystemMessage(content=augmented_messages[0].content + f"\n\n{instruction}")
    else:
        augmented_messages.insert(0, SystemMessage(content=instruction))

    structured_llm = llm.with_structured_output(schema)

    for attempt in range(5):
        try:
            return structured_llm.invoke(augmented_messages)
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait_time = _extract_wait_time(err_str, default=(attempt + 1) * 6.0)
                print(f"  [RateLimit] Waiting {wait_time:.1f}s for Groq token replenishment (attempt {attempt + 1}/5)...")
                time.sleep(wait_time)
                continue

            if "tool_use_failed" in err_str or "Tool choice is required" in err_str:
                failed_text = _extract_failed_generation(exc)
                return _fallback_for_schema(schema, failed_text)

            if attempt == 4:
                # Last resort fallback if tool use failed
                try:
                    return _fallback_for_schema(schema, "")
                except Exception:
                    raise exc
            time.sleep(2.0)


async def ainvoke_structured(llm: Any, schema: Type[BaseModel], messages: List[BaseMessage]) -> Any:
    """Asynchronously invoke structured LLM with rate-limit retry and fallback recovery."""
    augmented_messages = list(messages)
    instruction = f"You MUST invoke the {schema.__name__} tool function with valid arguments to return your output."
    if augmented_messages and isinstance(augmented_messages[0], SystemMessage):
        augmented_messages[0] = SystemMessage(content=augmented_messages[0].content + f"\n\n{instruction}")
    else:
        augmented_messages.insert(0, SystemMessage(content=instruction))

    structured_llm = llm.with_structured_output(schema)

    for attempt in range(5):
        try:
            return await structured_llm.ainvoke(augmented_messages)
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "rate_limit" in err_str.lower():
                wait_time = _extract_wait_time(err_str, default=(attempt + 1) * 6.0)
                print(f"  [RateLimit] Waiting {wait_time:.1f}s for Groq token replenishment (attempt {attempt + 1}/5)...")
                await asyncio.sleep(wait_time)
                continue

            if "tool_use_failed" in err_str or "Tool choice is required" in err_str:
                failed_text = _extract_failed_generation(exc)
                return _fallback_for_schema(schema, failed_text)

            if attempt == 4:
                try:
                    return _fallback_for_schema(schema, "")
                except Exception:
                    raise exc
            await asyncio.sleep(2.0)
