"""Interactive CLI entry point for the Deep Research Agent."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from src.graph import run_research
from src.utils.display import print_header, print_subheader

WELCOME_BANNER = """
===========================================================================
  🔬 DEEP RESEARCH AGENT FOR ACADEMIC RESEARCH PAPERS
  Inspired by Open Deep Research (LangChain) & Gemini/OpenAI Deep Research
===========================================================================
  Architecture:
    Phase 1: Scope    -> Clarity analysis, clarification, research brief
    Phase 2: Research -> Dynamic supervisor, parallel sub-agents, arXiv /
                         Semantic Scholar / OpenAlex / Tavily search, cleanup,
                         and gap analysis (max 2 additional rounds)
    Phase 3: Write    -> Single LLM call comprehensive literature review
===========================================================================
"""


def main():
    """Main interactive loop."""
    print(WELCOME_BANNER)

    # Ensure output directory exists
    output_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    while True:
        try:
            print("\n" + "-"*75)
            topic = input("\n👉 Enter your research topic (or 'exit' / 'quit' to stop): ").strip()
            print("-"*75)

            if not topic:
                continue

            if topic.lower() in ("exit", "quit", "q"):
                print("\n👋 Exiting Deep Research Agent. Happy researching!\n")
                break

            print(f"\n[Initiating Deep Research on: '{topic}']")
            start_time = datetime.now()

            # Execute the research graph
            final_state = run_research(topic)

            elapsed = (datetime.now() - start_time).total_seconds()
            report = final_state.get("final_report", "No report generated.")

            # Save report to markdown artifact file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.md"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report)

            print_header("RESEARCH COMPLETED", "🏆")
            print(f"  Total time elapsed: {elapsed:.1f} seconds")
            print(f"  Report saved to: {filepath}")

            brief = final_state.get("research_brief") or {}
            all_papers = final_state.get("all_papers") or []
            evals = final_state.get("supervisor_evaluations") or []

            print("\n  Summary Statistics:")
            print(f"    • Total Papers Discovered : {len(all_papers)}")
            print(f"    • Research Rounds Run     : {len(evals)}")
            print(f"    • Report Word Count       : {len(report.split())} words")

            # Display final report excerpt
            print("\n" + "="*75)
            print("  FINAL LITERATURE REVIEW EXCERPT:")
            print("="*75)
            print(report[:2000] + ("\n\n... [Full report saved to file] ..." if len(report) > 2000 else ""))
            print("="*75)

        except KeyboardInterrupt:
            print("\n\nResearch cancelled by user.")
            break
        except Exception as exc:
            print(f"\n❌ Error during research execution: {exc}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
