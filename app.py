"""Streamlit UI for the Deep Research Agent.

Interactive web application for autonomous academic literature review generation:
Phase 1 (Scope) -> Phase 2 (Multi-Agent Research) -> Phase 3 (Synthesis & Literature Review).
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
from datetime import datetime
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.config import settings
from src.state import ResearchState
from src.phases.scope import check_clarity, generate_brief
from src.phases.research import research_phase
from src.phases.write import write_report
from src.utils.deduplication import deduplicate_papers
from src.utils.paper_validator import validate_papers

# Streamlit Page Setup
st.set_page_config(
    page_title="Deep Research Agent — Academic Papers",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "state" not in st.session_state:
    st.session_state.state = {
        "topic": "",
        "clarification_answers": "",
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
        "phase": "init",
    }


def reset_session():
    """Reset the research session to start fresh."""
    st.session_state.state = {
        "topic": "",
        "clarification_answers": "",
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
        "phase": "init",
    }
    st.rerun()


# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    st.markdown("Configure inference providers and research depth.")

    provider = st.selectbox(
        "LLM Provider",
        options=["Groq (Ultra-Fast)", "OpenAI", "Google Gemini"],
        index=0,
    )

    if "Groq" in provider:
        settings.llm_provider = "groq"
        model_name = st.selectbox(
            "Primary Model",
            options=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
            index=0,
        )
        settings.llm_model = model_name
    elif "OpenAI" in provider:
        settings.llm_provider = "openai"
        model_name = st.selectbox(
            "Primary Model",
            options=["gpt-4o", "gpt-4o-mini", "o3-mini"],
            index=0,
        )
        settings.llm_model = model_name
    else:
        settings.llm_provider = "google"
        model_name = st.selectbox(
            "Primary Model",
            options=["gemini-2.0-flash", "gemini-1.5-pro"],
            index=0,
        )
        settings.llm_model = model_name

    st.divider()

    st.subheader("Research Controls")
    max_rounds = st.slider(
        "Max Additional Rounds",
        min_value=0,
        max_value=2,
        value=settings.max_additional_rounds,
        help="Maximum gap-filling iteration rounds the supervisor can trigger.",
    )
    settings.max_additional_rounds = max_rounds

    papers_per_query = st.slider(
        "Max Papers Per Search",
        min_value=3,
        max_value=12,
        value=settings.max_papers_per_search,
        help="Number of papers to pull from each academic API (arXiv, OpenAlex, Semantic Scholar).",
    )
    settings.max_papers_per_search = papers_per_query

    st.divider()

    st.subheader("API Status")
    groq_active = bool(settings.groq_api_key)
    tavily_active = bool(settings.tavily_api_key)

    st.markdown(f"**Groq API**: {'🟢 Active' if groq_active else '🔴 Missing'}")
    st.markdown(f"**Tavily Search**: {'🟢 Active' if tavily_active else '⚪ Inactive (Optional)'}")
    st.markdown("**arXiv API**: 🟢 Active (Public)")
    st.markdown("**OpenAlex API**: 🟢 Active (Public)")
    st.markdown("**Semantic Scholar**: 🟢 Active (Public)")

    st.divider()
    if st.button("🔄 Reset / New Topic", use_container_width=True):
        reset_session()


# -----------------------------------------------------------------------------
# Main Content Area
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">🔬 Deep Research Agent for Academic Papers</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated, evidence-based scientific literature reviews powered by multi-agent research loops.</div>', unsafe_allow_html=True)

# Example topic selector
EXAMPLE_TOPICS = [
    "Transformer attention mechanisms for long context NLP",
    "Deep reinforcement learning for robotic dexterous in-hand manipulation",
    "AI in healthcare",
    "Spike-timing-dependent plasticity in memristor crossbars for neuromorphic computing",
    "Quantum approximate optimization algorithm (QAOA) on NISQ processors",
]

col1, col2 = st.columns([3, 1])
with col1:
    topic_input = st.text_input(
        "Enter Research Topic or Question:",
        value=st.session_state.state["topic"],
        placeholder="e.g. Scalable state space models vs transformers for long-sequence modeling...",
    )
with col2:
    selected_example = st.selectbox("Or choose an example:", options=[""] + EXAMPLE_TOPICS)
    if selected_example and selected_example != topic_input:
        st.session_state.state["topic"] = selected_example
        st.rerun()

if topic_input != st.session_state.state["topic"]:
    st.session_state.state["topic"] = topic_input


# -----------------------------------------------------------------------------
# PHASE 1: SCOPE & CLARIFICATION
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("Phase 1: Scope & Clarification")

if not st.session_state.state["research_brief"]:
    col_scope1, col_scope2 = st.columns([1, 4])
    with col_scope1:
        scope_btn = st.button("🎯 Analyze & Scope Topic", type="primary", use_container_width=True)

    if scope_btn and st.session_state.state["topic"]:
        with st.spinner("Analyzing topic clarity and academic boundaries..."):
            temp_state: ResearchState = {
                "topic": st.session_state.state["topic"],
                "clarification_answers": None,
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
            clarity_res = check_clarity(temp_state)
            st.session_state.state["needs_clarification"] = clarity_res["needs_clarification"]
            st.session_state.state["clarification_questions"] = clarity_res["clarification_questions"]

            if not clarity_res["needs_clarification"]:
                # Clear topic: generate brief immediately
                brief_res = generate_brief(temp_state)
                st.session_state.state["research_brief"] = brief_res["research_brief"]
                st.session_state.state["phase"] = "brief_ready"
                st.rerun()
            else:
                st.session_state.state["phase"] = "clarification_needed"
                st.rerun()

    # If clarification questions were generated
    if st.session_state.state["phase"] == "clarification_needed":
        st.warning("⚠️ **Clarification Required:** The topic is broad or ambiguous. Please provide focus below:")
        clarif_answers = []
        with st.form("clarification_form"):
            for idx, q in enumerate(st.session_state.state["clarification_questions"], 1):
                ans = st.text_input(f"Q{idx}: {q}", key=f"q_{idx}")
                clarif_answers.append(f"Q: {q}\nA: {ans}")

            submit_clarif = st.form_submit_button("Submit Clarification & Generate Brief")
            if submit_clarif:
                combined_ans = "\n".join(clarif_answers)
                st.session_state.state["clarification_answers"] = combined_ans

                with st.spinner("Formulating customized research brief..."):
                    temp_state: ResearchState = {
                        "topic": st.session_state.state["topic"],
                        "clarification_answers": combined_ans,
                        "needs_clarification": True,
                        "clarification_questions": st.session_state.state["clarification_questions"],
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
                    brief_res = generate_brief(temp_state)
                    st.session_state.state["research_brief"] = brief_res["research_brief"]
                    st.session_state.state["phase"] = "brief_ready"
                    st.rerun()

# Display Research Brief if ready
if st.session_state.state["research_brief"]:
    brief = st.session_state.state["research_brief"]
    with st.expander("📋 **Active Research Brief** (Click to collapse)", expanded=True):
        st.markdown(f"**Refined Topic:** {brief.get('refined_topic')}")
        st.markdown(f"**Scope Boundaries:** {brief.get('scope_boundaries')}")
        st.markdown(f"**Target Year Range:** {brief.get('year_range') or 'Any recent'}")
        st.markdown(f"**Methodology Focus:** {brief.get('methodology_focus') or 'Empirical & Theoretical'}")

        c_asp, c_qry = st.columns(2)
        with c_asp:
            st.markdown("**Key Aspects to Investigate:**")
            for asp in brief.get("key_aspects", []):
                st.markdown(f"- {asp}")
        with c_qry:
            st.markdown("**Academic API Search Queries:**")
            for qry in brief.get("search_queries", []):
                st.markdown(f"- `🔍 {qry}`")


# -----------------------------------------------------------------------------
# PHASE 2 & 3: EXECUTION & PROGRESS
# -----------------------------------------------------------------------------
if st.session_state.state["research_brief"] and not st.session_state.state["final_report"]:
    st.markdown("---")
    st.subheader("Phase 2: Multi-Agent Deep Research")

    start_research = st.button("🚀 Execute Autonomous Research Pipeline", type="primary", use_container_width=True)

    if start_research:
        progress_bar = st.progress(0, text="Initializing Research Supervisor...")
        status_container = st.empty()

        # Step 1: Run Research Phase (Multi-agent supervisor + iterations)
        status_container.info("Supervisor analyzing brief and dispatching parallel sub-agents across arXiv, Semantic Scholar, OpenAlex, Tavily...")
        progress_bar.progress(20, text="Running Sub-Agents across Academic APIs...")

        res_state: ResearchState = {
            "topic": st.session_state.state["topic"],
            "clarification_answers": st.session_state.state.get("clarification_answers"),
            "needs_clarification": st.session_state.state.get("needs_clarification", False),
            "clarification_questions": st.session_state.state.get("clarification_questions", []),
            "research_brief": st.session_state.state["research_brief"],
            "delegation_plan": None,
            "all_findings": [],
            "all_papers": [],
            "supervisor_evaluations": [],
            "current_round": 1,
            "research_complete": False,
            "final_report": None,
            "phase": "research",
            "messages": [],
        }

        # Multi-round supervisor loop
        round_count = 1
        max_loop = 1 + settings.max_additional_rounds

        while round_count <= max_loop:
            progress_bar.progress(20 + (round_count * 20), text=f"Executing Research Round {round_count}/{max_loop}...")
            res_update = research_phase(res_state)

            res_state["delegation_plan"] = res_update["delegation_plan"]
            res_state["all_findings"] = res_state.get("all_findings", []) + res_update["all_findings"]
            res_state["all_papers"] = res_state.get("all_papers", []) + res_update["all_papers"]
            res_state["supervisor_evaluations"] = res_state.get("supervisor_evaluations", []) + res_update["supervisor_evaluations"]
            res_state["current_round"] = res_update["current_round"]
            res_state["research_complete"] = res_update["research_complete"]

            if res_update["research_complete"]:
                break
            round_count += 1

        st.session_state.state["delegation_plan"] = res_state["delegation_plan"]
        st.session_state.state["all_findings"] = res_state["all_findings"]
        st.session_state.state["all_papers"] = res_state["all_papers"]
        st.session_state.state["supervisor_evaluations"] = res_state["supervisor_evaluations"]

        # Step 2: Run Phase 3 (Write single LLM call)
        progress_bar.progress(85, text="Phase 3: Synthesizing Final Literature Review via Single LLM Call...")
        status_container.info("Drafting comprehensive literature review, comparison table, and verified citations...")

        write_update = write_report(res_state)
        st.session_state.state["final_report"] = write_update["final_report"]
        st.session_state.state["phase"] = "completed"

        progress_bar.progress(100, text="Deep Research Complete!")
        status_container.success("Literature Review successfully generated and verified!")
        st.rerun()


# -----------------------------------------------------------------------------
# RESULTS DISPLAY (TABS)
# -----------------------------------------------------------------------------
if st.session_state.state["final_report"]:
    st.markdown("---")
    st.subheader("Research Findings & Final Literature Review")

    # Metrics Bar
    all_papers = st.session_state.state["all_papers"]
    evals = st.session_state.state["supervisor_evaluations"]
    unique_papers = deduplicate_papers(all_papers)
    valid_papers, _ = validate_papers(unique_papers)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Papers Discovered", len(unique_papers))
    with m2:
        st.metric("Verified References", len(valid_papers))
    with m3:
        st.metric("Research Rounds", len(evals))
    with m4:
        st.metric("Report Word Count", len(st.session_state.state["final_report"].split()))

    # Tabs for detailed inspection
    tab_report, tab_plan, tab_papers, tab_eval = st.tabs([
        "📄 Literature Review",
        "🧭 Delegation Plan",
        "📚 Discovered Papers",
        "⚖️ Supervisor Gap Analysis"
    ])

    with tab_report:
        report_text = st.session_state.state["final_report"]

        # Download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Download Literature Review (.md)",
            data=report_text,
            file_name=f"literature_review_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        st.markdown(report_text)

    with tab_plan:
        plan = st.session_state.state.get("delegation_plan") or {}
        st.markdown(f"**Supervisor Rationale:** {plan.get('rationale', 'N/A')}")
        st.markdown(f"**Parallel Sub-Agents Deployed:** {plan.get('num_agents', 0)}")

        for a in plan.get("assignments", []):
            with st.expander(f"🤖 Sub-Agent #{a.get('subtopic_id')}: {a.get('title')}", expanded=True):
                st.markdown(f"**Objective:** {a.get('description')}")
                st.markdown(f"**Focus Areas:** {', '.join(a.get('focus_areas', []))}")
                st.markdown(f"**Search Queries:** `{', '.join(a.get('search_queries', []))}`")

    with tab_papers:
        if unique_papers:
            table_data = []
            for p in unique_papers[:50]:
                table_data.append({
                    "Title": p.get("title", ""),
                    "Authors": ", ".join(p.get("authors", []))[:40],
                    "Year": p.get("year") or "N/A",
                    "Source": p.get("source", "").title(),
                    "DOI / URL": p.get("doi") or p.get("url") or "N/A",
                })
            st.dataframe(table_data, use_container_width=True)
            if len(unique_papers) > 50:
                st.caption(f"Showing top 50 of {len(unique_papers)} total papers.")
        else:
            st.info("No papers recorded.")

    with tab_eval:
        for idx, ev in enumerate(evals, 1):
            st.markdown(f"### Round {ev.get('round_number', idx)} Evaluation")
            gap = ev.get("gap_analysis", {})
            st.markdown(f"**Decision:** `{'✅ SUFFICIENT' if ev.get('decision') == 'sufficient' else '🔄 NEEDS MORE RESEARCH'}`")
            st.markdown(f"**Supervisor Reasoning:** {gap.get('reasoning', '')}")

            col_cov, col_gap = st.columns(2)
            with col_cov:
                st.markdown("**Covered Dimensions:**")
                for c in gap.get("covered_areas", []):
                    st.markdown(f"- ✔ {c}")
            with col_gap:
                st.markdown("**Identified Gaps:**")
                for m in gap.get("missing_areas", []):
                    st.markdown(f"- ⚠️ {m}")
            st.divider()
