"""Evaluation harness for the Deep Research Agent.

Evaluates the system across five diverse research topics:
1. Clear & straightforward topic 1 (Transformer architectures)
2. Clear & straightforward topic 2 (Reinforcement learning in robotics)
3. Vague topic requiring clarification (AI in healthcare)
4. Highly niche topic (Neuromorphic memristor architectures)
5. Topic triggering additional research round (Quantum optimization on NISQ)

Generates the required evaluation summary table.
"""

from __future__ import annotations

import os
import sys
import re
from datetime import datetime
from typing import List, Dict, Any
from src.graph import run_research
from src.utils.display import print_header, print_subheader

EVAL_TOPICS = [
    {
        "type": "Clear & Straightforward (Topic 1)",
        "topic": "Transformer architectures and attention mechanisms for long-context natural language processing",
        "clarification_answers": None,
    },
    {
        "type": "Clear & Straightforward (Topic 2)",
        "topic": "Deep reinforcement learning methods for robotic dexterous in-hand manipulation",
        "clarification_answers": None,
    },
    {
        "type": "Vague (Requires Clarification)",
        "topic": "AI in healthcare",
        "clarification_answers": (
            "Focus area: Deep learning diagnostic models in oncology and histopathology imaging.\n"
            "Year range: 2022 to 2025.\n"
            "Methodology: Multi-modal vision-language models and self-supervised learning for tumor classification."
        ),
    },
    {
        "type": "Highly Niche",
        "topic": "Spike-timing-dependent plasticity (STDP) learning rules in memristor crossbar arrays for neuromorphic computing",
        "clarification_answers": None,
    },
    {
        "type": "Complex / Multi-Round Trigger",
        "topic": "Quantum approximate optimization algorithm (QAOA) and variational quantum eigensolvers for combinatorial graph optimization on noisy intermediate-scale quantum (NISQ) processors",
        "clarification_answers": None,
    },
]


def verify_citation_coverage(report_text: str) -> bool:
    """Verify that claims in the report contain citations in [1], [2], [3] format."""
    citations = re.findall(r"\[\d+\]", report_text)
    return len(citations) >= 5


def verify_references_structure(report_text: str) -> bool:
    """Verify that the references section is populated with numbered references."""
    if "References" not in report_text and "REFERENCES" not in report_text:
        return False
    return bool(re.search(r"\[\d+\]\s+[A-Za-z]+", report_text))


def run_evaluation():
    """Run evaluation on all five benchmark topics."""
    print_header("DEEP RESEARCH AGENT BENCHMARK EVALUATION", "🧪")
    print(f"  Evaluating {len(EVAL_TOPICS)} distinct topics across all architectural dimensions...")

    results: List[Dict[str, Any]] = []

    for idx, test_case in enumerate(EVAL_TOPICS, 1):
        topic_type = test_case["type"]
        topic = test_case["topic"]
        pre_answers = test_case["clarification_answers"]

        print_header(f"Evaluation {idx}/5: {topic_type}", "🔬")
        print(f"  Query: {topic}")

        try:
            state = run_research(topic, clarification_answers=pre_answers)

            # Analyze run results
            plan = state.get("delegation_plan") or {}
            num_agents = plan.get("num_agents", 1)

            evals = state.get("supervisor_evaluations") or []
            research_rounds = len(evals)

            papers = state.get("all_papers") or []
            papers_count = len(papers)

            report = state.get("final_report", "")
            citations_valid = verify_citation_coverage(report)
            refs_valid = verify_references_structure(report)

            results.append({
                "topic": topic[:45] + "..." if len(topic) > 45 else topic,
                "type": topic_type,
                "subagents": num_agents,
                "rounds": research_rounds,
                "papers": papers_count,
                "cited": "Yes" if citations_valid else "Partial",
                "verified": "Yes" if refs_valid else "No",
            })

        except Exception as exc:
            print(f"  ❌ Error evaluating '{topic}': {exc}")
            import traceback
            traceback.print_exc()
            results.append({
                "topic": topic[:45] + "...",
                "type": topic_type,
                "subagents": "Error",
                "rounds": "Error",
                "papers": 0,
                "cited": "No",
                "verified": "No",
            })

    # Print final evaluation summary table
    print_header("EVALUATION SUMMARY REPORT", "📊")
    print(f"{'| Topic':<50} | {'Sub-agents':<12} | {'Rounds':<8} | {'Papers':<8} | {'Cited':<8} | {'Verified':<10} |")
    print(f"|{'-'*50}|{'-'*14}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*12}|")

    for r in results:
        print(
            f"| {r['topic']:<48} "
            f"| {str(r['subagents']):<12} "
            f"| {str(r['rounds']):<8} "
            f"| {str(r['papers']):<8} "
            f"| {r['cited']:<8} "
            f"| {r['verified']:<10} |"
        )
    print("\n")


if __name__ == "__main__":
    run_evaluation()
