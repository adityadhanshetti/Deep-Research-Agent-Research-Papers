"""Mock LLM implementation for testing and dry-run evaluation without API keys.

Allows full verification of the LangGraph workflow, multi-agent parallel execution,
academic API retrieval (arXiv, OpenAlex), deduplication, gap analysis, and citation formatting.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from src.models import (
    ClarificationResult,
    ResearchBrief,
    DelegationPlan,
    SubtopicAssignment,
    SubAgentFindings,
    SupervisorEvaluation,
    GapAnalysis,
    ResearchPaper,
)


class MockStructuredModel:
    """Mock model that returns valid Pydantic models matching the prompt context."""

    def __init__(self, schema: Type[BaseModel]):
        self.schema = schema

    def invoke(self, messages: List[BaseMessage]) -> Any:
        # Extract user / human message specifically
        human_parts = []
        for m in messages:
            if isinstance(m, HumanMessage) or getattr(m, "type", "") == "human":
                human_parts.append(str(m.content))
        human_content = "\n".join(human_parts) if human_parts else str(messages[-1].content)

        # 1. ClarificationResult
        if self.schema == ClarificationResult:
            topic_lower = human_content.lower()
            # Only trigger clarification for vague queries like "ai in healthcare"
            is_vague = "ai in healthcare" in topic_lower or (
                "topic to evaluate:" in topic_lower and len(topic_lower.split()) < 7 and "ai" in topic_lower
            )

            if is_vague:
                return ClarificationResult(
                    needs_clarification=True,
                    questions=[
                        "What specific clinical or medical domain should be prioritized (e.g., oncology, pathology)?",
                        "What is the preferred publication year window (e.g., 2022-2025)?",
                        "Which methodologies are of interest (e.g., multimodal foundation models, self-supervised learning)?",
                    ],
                    reasoning="The query 'AI in healthcare' is broad and requires domain and methodological focus.",
                )
            return ClarificationResult(
                needs_clarification=False,
                questions=[],
                reasoning="The research topic is technically specific, well-bounded, and suitable for deep literature review.",
            )

        # 2. ResearchBrief
        elif self.schema == ResearchBrief:
            topic_match = re.search(r"Original User Topic:\s*(.+)", human_content)
            topic = topic_match.group(1).strip() if topic_match else "Advanced Research Topic"

            clean_words = [w for w in re.sub(r"[^\w\s]", "", topic).split() if len(w) > 3]
            q1 = " ".join(clean_words[:2]) + " architectures" if len(clean_words) >= 2 else topic
            q2 = " ".join(clean_words[-2:]) + " benchmark" if len(clean_words) >= 2 else topic

            return ResearchBrief(
                original_topic=topic,
                refined_topic=f"Systematic Review of {topic}",
                key_aspects=[
                    "Foundational theoretical principles and algorithmic formulations",
                    "Empirical benchmarks, standard datasets, and comparative performance",
                    "Architectural scalability, computational complexity, and hardware constraints",
                    "Current limitations, unaddressed challenges, and future research frontiers",
                ],
                search_queries=[q1, q2, f"{topic[:25]} algorithms", f"{topic[:25]} survey"],
                year_range="2020-2025",
                methodology_focus="Empirical validation and comparative algorithmic analysis",
                scope_boundaries="Focus on peer-reviewed and preprint technical literature; exclude popular media.",
            )

        # 3. DelegationPlan
        elif self.schema == DelegationPlan:
            content_lower = human_content.lower()
            is_simple = "simple" in content_lower
            is_quantum = "quantum" in content_lower or "nisq" in content_lower

            if is_simple:
                assignments = [
                    SubtopicAssignment(
                        subtopic_id=1,
                        title="Core Architectures and Algorithmic Foundations",
                        description="Investigate the fundamental principles, baseline models, and training objectives.",
                        search_queries=["transformer attention mechanisms", "neural architecture optimization"],
                        focus_areas=["Architectural taxonomy", "Attention formulation"],
                    )
                ]
            elif is_quantum:
                assignments = [
                    SubtopicAssignment(
                        subtopic_id=1,
                        title="Quantum Approximate Optimization Algorithm (QAOA) on Graph Problems",
                        description="Investigate QAOA parameter optimization and ansatz design on MaxCut and graph partitioning.",
                        search_queries=["QAOA graph optimization NISQ", "quantum approximate optimization MaxCut"],
                        focus_areas=["Ansatz depth", "Parameter optimization heuristics"],
                    ),
                    SubtopicAssignment(
                        subtopic_id=2,
                        title="Variational Quantum Eigensolver (VQE) and Hybrid Classical-Quantum Schemes",
                        description="Examine VQE formulations for combinatorial and electronic structure problems.",
                        search_queries=["variational quantum eigensolver combinatorial", "VQE hybrid classical quantum"],
                        focus_areas=["Error mitigation", "Barren plateaus"],
                    ),
                    SubtopicAssignment(
                        subtopic_id=3,
                        title="NISQ Hardware Noise, Error Mitigation, and Scalability Limits",
                        description="Analyze decoherence, two-qubit gate fidelities, and zero-noise extrapolation on NISQ chips.",
                        search_queries=["NISQ noise error mitigation quantum", "quantum processor scalability benchmarks"],
                        focus_areas=["Decoherence impact", "Extrapolation methods"],
                    ),
                ]
            else:
                assignments = [
                    SubtopicAssignment(
                        subtopic_id=1,
                        title="Foundational Algorithmic Formulations and Baselines",
                        description="Examine baseline mathematical formulations and core models.",
                        search_queries=["deep learning algorithmic foundations", "computational models survey"],
                        focus_areas=["Core formulations", "Mathematical taxonomy"],
                    ),
                    SubtopicAssignment(
                        subtopic_id=2,
                        title="State-of-the-Art Implementations & Empirical Results",
                        description="Survey state-of-the-art implementations, benchmarks, and experimental setups.",
                        search_queries=["state of the art empirical benchmarks", "experimental performance evaluation"],
                        focus_areas=["Benchmark datasets", "Ablation studies"],
                    ),
                    SubtopicAssignment(
                        subtopic_id=3,
                        title="Hardware Scalability, Limitations, and Open Frontiers",
                        description="Analyze memory complexity, scalability barriers, and open research challenges.",
                        search_queries=["computational complexity scalability limits", "open research challenges survey"],
                        focus_areas=["Computational bottlenecks", "Future directions"],
                    ),
                ]

            return DelegationPlan(
                num_agents=len(assignments),
                rationale=f"Divided the research into {len(assignments)} independent investigation tracks to ensure thorough academic coverage.",
                assignments=assignments,
            )

        # 4. SubAgentFindings
        elif self.schema == SubAgentFindings:
            return SubAgentFindings(
                subtopic_id=1,
                subtopic_title="Research Findings",
                papers=[],
                synthesis=(
                    "The retrieved papers establish significant improvements over earlier baselines [1]. "
                    "Empirical evaluations demonstrate that modern algorithmic designs substantially reduce computational overhead "
                    "while preserving model fidelity across diverse benchmarks [2]. However, significant limitations remain regarding "
                    "sample efficiency and robustness under severe noise conditions [3]."
                ),
                key_themes=[
                    "Algorithmic efficiency and architectural refinement",
                    "Empirical benchmark gains across standard datasets",
                    "Hardware constraints and parameter sensitivity",
                ],
                limitations_noted=[
                    "High computational cost during training phases",
                    "Susceptibility to noise in non-ideal environments",
                    "Lack of standardized evaluation protocol across disparate labs",
                ],
            )

        # 5. SupervisorEvaluation
        elif self.schema == SupervisorEvaluation:
            content_lower = human_content.lower()
            is_quantum = "quantum" in content_lower or "nisq" in content_lower
            current_round_match = re.search(r"Current Round:\s*(\d+)", human_content)
            current_round = int(current_round_match.group(1)) if current_round_match else 1

            if is_quantum and current_round == 1:
                return SupervisorEvaluation(
                    round_number=current_round,
                    total_papers=16,
                    gap_analysis=GapAnalysis(
                        covered_areas=["QAOA theoretical formulation", "VQE baselines"],
                        missing_areas=[
                            "Barren plateau mitigation in deep variational circuits",
                            "Experimental NISQ processor fidelity comparisons on hardware > 50 qubits",
                        ],
                        unanswered_questions=[
                            "What are the precise scaling limits of zero-noise extrapolation on superconducting qubits?",
                        ],
                        needs_more_research=True,
                        reasoning="Critical gaps remain regarding hardware noise scaling and barren plateaus on physical NISQ processors.",
                        additional_queries=["barren plateaus mitigation variational quantum", "NISQ hardware benchmarks superconducting qubits"],
                    ),
                    decision="needs_more",
                )

            return SupervisorEvaluation(
                round_number=current_round,
                total_papers=22,
                gap_analysis=GapAnalysis(
                    covered_areas=[
                        "Core mathematical formulations and baseline taxonomies",
                        "Empirical benchmark evaluations across leading datasets",
                        "Hardware and computational complexity profiles",
                    ],
                    missing_areas=[],
                    unanswered_questions=[],
                    needs_more_research=False,
                    reasoning="All core research dimensions specified in the brief have been thoroughly investigated and supported by verified literature.",
                ),
                decision="sufficient",
            )

        raise ValueError(f"Unknown schema: {self.schema}")

    async def ainvoke(self, messages: List[BaseMessage]) -> Any:
        return self.invoke(messages)


class MockChatModel:
    """Mock Chat Model that produces realistic academic responses and supports structured outputs."""

    def with_structured_output(self, schema: Type[BaseModel]):
        return MockStructuredModel(schema)

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        content = "\n".join(str(m.content) for m in messages)

        if "OFFICIAL CANONICAL REFERENCE LIST" in content or "REQUIRED SECTIONS IN ORDER" in content:
            report = (
                "# Deep Systematic Literature Review: Algorithmic Advances and Empirical Frontiers\n\n"
                "## 1. Executive Summary\n"
                "Recent advancements in modern scientific computing have revolutionized foundational paradigms [1]. "
                "This comprehensive literature review synthesizes key developments across theoretical principles, "
                "algorithmic improvements, and empirical benchmarks [2], highlighting critical trade-offs between "
                "computational complexity and model expressivity [3].\n\n"
                "## 2. Research Scope and Methodology\n"
                "This review examined peer-reviewed literature and authoritative preprints indexed across arXiv, "
                "Semantic Scholar, and OpenAlex. The scoping criteria prioritized primary sources presenting reproducible "
                "empirical findings and formal algorithmic designs published between 2020 and 2025.\n\n"
                "## 3. Introduction and Background\n"
                "Theoretical foundations were originally established to overcome scalability bottlenecks in classical architectures [1]. "
                "Subsequent refinements have introduced specialized mechanisms to stabilize optimization landscapes and improve "
                "sample efficiency [2], [4].\n\n"
                "## 4. Detailed Literature Review\n"
                "### 4.1 Foundational Architectures and Algorithmic Paradigms\n"
                "Early framework implementations focused on core computational primitives [1]. "
                "Contemporary investigations demonstrate that modular attention and factorization strategies yield "
                "substantial speedups while maintaining representational capacity [2], [5].\n\n"
                "### 4.2 Empirical Benchmarks and Comparative Evaluations\n"
                "Cross-study comparisons confirm that optimized formulations consistently outperform conventional baselines "
                "across standard task suites [3], [6]. Nevertheless, performance degrades predictably when input distribution "
                "shifts occur [4].\n\n"
                "## 5. Comparative Analysis\n"
                "A comparative assessment reveals distinct methodological divergences across existing paradigms. "
                "Approaches based on sparse representations exhibit superior memory scaling but introduce optimization instability [2], "
                "whereas dense parameterizations guarantee convergence at the expense of quadratic computational overhead [5].\n\n"
                "## 6. Comparison Table\n"
                "| Paper | Authors | Year | Methodology | Key Contribution | Limitations |\n"
                "|---|---|---|---|---|---|\n"
                "| Paper 1 [1] | Primary Authors | 2023 | Empirical Benchmark | Core Algorithmic Framework | High training latency |\n"
                "| Paper 2 [2] | Specialized Team | 2024 | Optimization Protocol | Reduced Computational Complexity | Parameter sensitivity |\n"
                "| Paper 3 [3] | Research Consortium | 2024 | Comparative Study | Multi-Domain Validation | Data distribution shift |\n\n"
                "## 7. Research Gaps\n"
                "Despite substantial empirical gains, critical gaps persist regarding:\n"
                "1. Theoretical guarantees under non-convex and noisy optimization settings [2].\n"
                "2. Standardized evaluation protocols capable of isolating hardware efficiency from software heuristics [4].\n"
                "3. Long-tail generalization across out-of-distribution real-world datasets [5].\n\n"
                "## 8. Future Research Directions\n"
                "Future research should focus on hybrid classical-quantum paradigms, hardware-software co-design, "
                "and self-supervised pretraining objectives tailored to resource-constrained environments [1], [6].\n\n"
                "## 9. Conclusion\n"
                "This literature review has synthesized the current state-of-the-art across key technical dimensions. "
                "Continued progress will depend on addressing foundational bottlenecks in scalability, robustness, and interpretability.\n\n"
                "## 10. References\n"
                "[1] Vaswani, A., et al. (2023). \"Foundational Architectures for Scalable Processing\". *Arxiv*. DOI: https://doi.org/10.1234/567\n"
                "[2] Smith, J., & Doe, A. (2024). \"Efficient Algorithmic Optimization Strategies\". *Semantic Scholar*. URL: https://example.com/paper2\n"
                "[3] Chen, L., et al. (2024). \"Cross-Domain Empirical Benchmarks\". *Openalex*. DOI: https://doi.org/10.5678/901\n"
                "[4] Patel, R., et al. (2023). \"Robustness Under Distributional Shift\". *Arxiv*. URL: https://example.com/paper4\n"
                "[5] Williams, K., & Taylor, M. (2024). \"Sparse vs Dense Parameterizations\". *Openalex*. DOI: https://doi.org/10.3456/789\n"
                "[6] Zhang, Y., et al. (2025). \"Future Frontiers in Scalable Learning\". *Semantic Scholar*. URL: https://example.com/paper6\n"
            )
            return AIMessage(content=report)

        return AIMessage(content="Simulated LLM synthesis output.")
