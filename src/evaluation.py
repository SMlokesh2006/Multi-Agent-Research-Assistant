"""LangSmith evaluation suite for the Multi-Agent Research Assistant.

Evaluates research reports on four criteria using Gemini Flash as judge:
- **Relevance**: Does the report address the original query?
- **Accuracy**: Are claims properly sourced and factual?
- **Completeness**: Does the report cover the topic sufficiently?
- **Coherence**: Is the report well-structured and clearly written?

Only runs when ``LANGSMITH_API_KEY`` is set in the environment.

Usage:
    python -m src.evaluation
    python -m src.evaluation --queries "query1" "query2"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


# ── Evaluation criteria ──────────────────────────────────────


@dataclass
class EvaluationCriterion:
    """A single evaluation criterion for LLM-as-judge scoring."""

    name: str
    description: str
    prompt_template: str
    weight: float = 1.0


CRITERIA: list[EvaluationCriterion] = [
    EvaluationCriterion(
        name="relevance",
        description="How well the report addresses the original research query",
        prompt_template=(
            "You are evaluating a research report for RELEVANCE.\n\n"
            "Research Query: {query}\n\n"
            "Report:\n{report}\n\n"
            "Score the report on relevance from 1-10 where:\n"
            "1 = Completely irrelevant, does not address the query at all\n"
            "5 = Partially relevant, addresses some aspects but misses key points\n"
            "10 = Highly relevant, directly and comprehensively addresses the query\n\n"
            "Respond with ONLY a JSON object: "
            '{{"score": <1-10>, "reasoning": "<brief explanation>"}}'
        ),
        weight=1.5,
    ),
    EvaluationCriterion(
        name="accuracy",
        description="Whether claims are properly sourced and appear factual",
        prompt_template=(
            "You are evaluating a research report for ACCURACY.\n\n"
            "Research Query: {query}\n\n"
            "Report:\n{report}\n\n"
            "Score the report on accuracy from 1-10 where:\n"
            "1 = Contains major factual errors or unsourced claims\n"
            "5 = Mostly accurate with some unverified statements\n"
            "10 = All claims are well-sourced and factually sound\n\n"
            "Respond with ONLY a JSON object: "
            '{{"score": <1-10>, "reasoning": "<brief explanation>"}}'
        ),
        weight=1.5,
    ),
    EvaluationCriterion(
        name="completeness",
        description="Whether the report sufficiently covers the topic",
        prompt_template=(
            "You are evaluating a research report for COMPLETENESS.\n\n"
            "Research Query: {query}\n\n"
            "Report:\n{report}\n\n"
            "Score the report on completeness from 1-10 where:\n"
            "1 = Extremely shallow, covers almost nothing\n"
            "5 = Covers main points but lacks depth or misses subtopics\n"
            "10 = Comprehensive coverage with good depth across all aspects\n\n"
            "Respond with ONLY a JSON object: "
            '{{"score": <1-10>, "reasoning": "<brief explanation>"}}'
        ),
        weight=1.0,
    ),
    EvaluationCriterion(
        name="coherence",
        description="Whether the report is well-structured and clearly written",
        prompt_template=(
            "You are evaluating a research report for COHERENCE.\n\n"
            "Research Query: {query}\n\n"
            "Report:\n{report}\n\n"
            "Score the report on coherence from 1-10 where:\n"
            "1 = Disorganized, hard to follow, poor writing\n"
            "5 = Reasonably organized but with structural issues\n"
            "10 = Excellent structure, clear writing, logical flow\n\n"
            "Respond with ONLY a JSON object: "
            '{{"score": <1-10>, "reasoning": "<brief explanation>"}}'
        ),
        weight=1.0,
    ),
]


# ── Evaluation results ───────────────────────────────────────


@dataclass
class CriterionScore:
    """Score for a single evaluation criterion."""

    criterion: str
    score: float
    reasoning: str
    weight: float = 1.0

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


@dataclass
class EvaluationResult:
    """Complete evaluation result for a single query."""

    query: str
    report: str
    scores: list[CriterionScore] = field(default_factory=list)
    overall_score: float = 0.0
    execution_time_seconds: float = 0.0
    error: str = ""

    def compute_overall(self) -> None:
        """Calculate the weighted average score."""
        if not self.scores:
            self.overall_score = 0.0
            return

        total_weight = sum(s.weight for s in self.scores)
        weighted_sum = sum(s.weighted_score for s in self.scores)
        self.overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ── LLM-as-judge evaluator ───────────────────────────────────


async def _evaluate_criterion(
    query: str,
    report: str,
    criterion: EvaluationCriterion,
) -> CriterionScore:
    """Evaluate a report against a single criterion using Gemini Flash.

    Args:
        query: The original research query.
        report: The generated research report.
        criterion: The evaluation criterion to judge.

    Returns:
        A ``CriterionScore`` with the LLM's judgment.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    from src.utils.rate_limiter import get_rate_limiter

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini.model,
        temperature=0.1,  # Low temp for consistent judging
        google_api_key=settings.google_api_key,
    )

    prompt = criterion.prompt_template.format(query=query, report=report)
    rate_limiter = get_rate_limiter()

    try:
        response = await rate_limiter.execute_with_retry(
            llm.ainvoke, prompt
        )
        content = response.content.strip()

        # Parse JSON response — handle markdown fences
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(content)
        score = float(parsed.get("score", 5))
        reasoning = parsed.get("reasoning", "No reasoning provided")

        return CriterionScore(
            criterion=criterion.name,
            score=min(max(score, 1.0), 10.0),  # Clamp to 1-10
            reasoning=reasoning,
            weight=criterion.weight,
        )

    except json.JSONDecodeError:
        logger.warning("Failed to parse judge response for %s: %s", criterion.name, content[:200])
        return CriterionScore(
            criterion=criterion.name,
            score=5.0,
            reasoning=f"Parse error — raw response: {content[:200]}",
            weight=criterion.weight,
        )
    except Exception as e:
        logger.exception("Evaluation failed for criterion %s", criterion.name)
        return CriterionScore(
            criterion=criterion.name,
            score=0.0,
            reasoning=f"Evaluation error: {e}",
            weight=criterion.weight,
        )


async def evaluate_report(query: str, report: str) -> EvaluationResult:
    """Evaluate a research report against all criteria.

    Args:
        query: The original research query.
        report: The generated report text.

    Returns:
        An ``EvaluationResult`` with scores for each criterion.
    """
    start_time = time.time()
    result = EvaluationResult(query=query, report=report)

    if not report.strip():
        result.error = "Empty report — nothing to evaluate"
        return result

    # Evaluate each criterion sequentially (rate-limit-friendly)
    for criterion in CRITERIA:
        score = await _evaluate_criterion(query, report, criterion)
        result.scores.append(score)
        logger.info(
            "Criterion %s: %.1f/10 — %s",
            score.criterion,
            score.score,
            score.reasoning[:80],
        )

    result.compute_overall()
    result.execution_time_seconds = round(time.time() - start_time, 2)

    return result


# ── Batch evaluation runner ──────────────────────────────────

DEFAULT_TEST_QUERIES: list[str] = [
    "What are the latest breakthroughs in quantum computing in 2025?",
    "Compare renewable energy adoption rates across G7 countries.",
    "What is the current state of AI regulation worldwide?",
    "Explain the impact of microplastics on marine ecosystems.",
    "What are the most promising treatments for Alzheimer's disease?",
]


async def run_evaluation(
    queries: list[str] | None = None,
    max_iterations: int = 3,
) -> list[EvaluationResult]:
    """Run the full evaluation pipeline on a set of test queries.

    For each query:
    1. Run the research graph to generate a report.
    2. Evaluate the report with LLM-as-judge on all criteria.
    3. Collect and summarize results.

    Args:
        queries: Test queries to evaluate. Defaults to ``DEFAULT_TEST_QUERIES``.
        max_iterations: Max research iterations per query.

    Returns:
        A list of ``EvaluationResult`` objects.
    """
    from src.graph import run_research

    queries = queries or DEFAULT_TEST_QUERIES
    results: list[EvaluationResult] = []

    for i, query in enumerate(queries, 1):
        logger.info("=== Evaluation %d/%d: %s ===", i, len(queries), query[:60])

        try:
            # Generate report
            final_state = await run_research(
                query=query,
                max_iterations=max_iterations,
                thread_id=f"eval-{i}-{int(time.time())}",
            )
            report = final_state.get("report", "")

            # Evaluate report
            eval_result = await evaluate_report(query, report)
            results.append(eval_result)

            logger.info(
                "Query %d overall score: %.1f/10 (%.1fs)",
                i,
                eval_result.overall_score,
                eval_result.execution_time_seconds,
            )

        except Exception as e:
            logger.exception("Evaluation failed for query %d", i)
            results.append(
                EvaluationResult(query=query, report="", error=str(e))
            )

    return results


def print_evaluation_summary(results: list[EvaluationResult]) -> None:
    """Print a formatted summary of evaluation results.

    Args:
        results: List of evaluation results to summarize.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Per-query results table
    table = Table(title="📊 Evaluation Results", show_lines=True)
    table.add_column("Query", style="cyan", max_width=40)
    table.add_column("Relevance", justify="center")
    table.add_column("Accuracy", justify="center")
    table.add_column("Complete.", justify="center")
    table.add_column("Coherence", justify="center")
    table.add_column("Overall", justify="center", style="bold")
    table.add_column("Time", justify="right", style="dim")

    for r in results:
        if r.error:
            table.add_row(
                r.query[:40], "ERR", "ERR", "ERR", "ERR",
                "[red]Error[/red]", "-",
            )
            continue

        scores_by_name = {s.criterion: s.score for s in r.scores}

        def _color_score(score: float) -> str:
            if score >= 8:
                return f"[green]{score:.1f}[/green]"
            elif score >= 5:
                return f"[yellow]{score:.1f}[/yellow]"
            return f"[red]{score:.1f}[/red]"

        table.add_row(
            r.query[:40],
            _color_score(scores_by_name.get("relevance", 0)),
            _color_score(scores_by_name.get("accuracy", 0)),
            _color_score(scores_by_name.get("completeness", 0)),
            _color_score(scores_by_name.get("coherence", 0)),
            _color_score(r.overall_score),
            f"{r.execution_time_seconds:.0f}s",
        )

    console.print(table)

    # Aggregate stats
    valid = [r for r in results if not r.error]
    if valid:
        avg_overall = sum(r.overall_score for r in valid) / len(valid)
        console.print(
            f"\n[bold]Average overall score: {avg_overall:.1f}/10[/bold] "
            f"({len(valid)} queries evaluated, {len(results) - len(valid)} errors)"
        )


# ── Entrypoint ───────────────────────────────────────────────


def main() -> None:
    """Run evaluations from the command line."""
    if not os.getenv("LANGSMITH_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
        print("⚠ Neither LANGSMITH_API_KEY nor GOOGLE_API_KEY is set.")
        print("  Set at least GOOGLE_API_KEY to run evaluations.")
        return

    parser = argparse.ArgumentParser(description="Run research evaluation suite")
    parser.add_argument(
        "--queries", nargs="+", help="Custom test queries (default: built-in set)"
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=3, help="Max iterations per query"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    if settings.langsmith_enabled:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "research-assistant-eval")
        logger.info("LangSmith tracing enabled for evaluation")

    results = asyncio.run(
        run_evaluation(queries=args.queries, max_iterations=args.iterations)
    )
    print_evaluation_summary(results)

    # Optionally save results to file
    output_path = "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
