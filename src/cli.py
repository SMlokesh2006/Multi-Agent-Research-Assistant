"""Interactive CLI for the Multi-Agent Research Assistant.

Run with: ``python -m src.cli`` or ``python -m src.cli "your query"``

Features:
- Rich terminal output with live status updates
- Real-time streaming of graph node updates
- HITL interrupt handling via terminal prompts
- Final report rendered as rich Markdown
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from src.graph import create_graph
from src.persistence import get_checkpointer, get_session_manager
from src.state import create_initial_state

logger = logging.getLogger(__name__)
console = Console()


# ── Status display helpers ───────────────────────────────────

_STATUS_ICONS: dict[str, str] = {
    "initialized": "🔧",
    "planning": "📋",
    "searching": "🔍",
    "reading_content": "📖",
    "analyzing": "🧪",
    "writing": "✍️",
    "awaiting_human_review": "👀",
    "human_feedback_received": "💬",
    "complete": "✅",
    "error": "❌",
}


def _status_panel(status: str, detail: str = "") -> Panel:
    """Create a styled status panel for the current pipeline stage."""
    icon = _STATUS_ICONS.get(status, "⏳")
    title = f"{icon} {status.replace('_', ' ').title()}"
    content = detail if detail else f"Status: {status}"
    return Panel(content, title=title, border_style="blue", width=80)


def _display_search_results(results: list[dict]) -> None:
    """Print search results as a rich table."""
    if not results:
        return

    table = Table(title="🔍 Search Results", show_lines=True, width=80)
    table.add_column("Title", style="cyan", max_width=35)
    table.add_column("URL", style="dim", max_width=40)

    for r in results[-5:]:  # Show last 5 results
        title = r.get("title", "Untitled")[:35]
        url = r.get("url", "")[:40]
        table.add_row(title, url)

    console.print(table)


def _display_analysis(analysis: dict) -> None:
    """Print analysis summary."""
    if not analysis:
        return

    findings = analysis.get("key_findings", [])
    console.print(f"\n[bold green]📊 Analysis: {len(findings)} findings[/bold green]")

    for i, f in enumerate(findings[:5], 1):
        claim = f.get("claim", "")[:80]
        confidence = f.get("confidence", 0.0)
        console.print(f"  {i}. {claim} [dim](confidence: {confidence:.0%})[/dim]")

    gaps = analysis.get("knowledge_gaps", [])
    if gaps:
        console.print(f"\n[yellow]⚠ Knowledge gaps: {len(gaps)}[/yellow]")
        for gap in gaps[:3]:
            console.print(f"  • {gap[:60]}")

    conflicts = analysis.get("conflicts", [])
    if conflicts:
        console.print(f"\n[red]⚡ Conflicts: {len(conflicts)}[/red]")
        for conflict in conflicts[:3]:
            console.print(f"  • {conflict[:60]}")


# ── Main research runner ─────────────────────────────────────


async def run_cli(query: str, max_iterations: int = 3) -> None:
    """Run the full research pipeline with rich terminal output.

    Streams graph updates in real-time and handles HITL interrupts
    by prompting the user in the terminal.

    Args:
        query: The research question to investigate.
        max_iterations: Maximum supervisor loop iterations.
    """
    thread_id = str(uuid.uuid4())

    console.print(
        Panel(
            f"[bold]{query}[/bold]\n\n"
            f"[dim]Thread: {thread_id}[/dim]\n"
            f"[dim]Max iterations: {max_iterations}[/dim]",
            title="🔬 Multi-Agent Research Assistant",
            border_style="bright_blue",
            width=80,
        )
    )

    # Initialize persistence
    checkpointer = await get_checkpointer()
    session_mgr = get_session_manager()
    await session_mgr.save_session(thread_id=thread_id, query=query, status="started")

    # Build graph
    graph = create_graph(checkpointer=checkpointer)
    initial_state = create_initial_state(query, max_iterations=max_iterations)
    config = {"configurable": {"thread_id": thread_id}}

    # Stream with HITL loop
    current_input: dict[str, Any] | Any = initial_state
    final_report = ""

    while True:
        interrupted = False

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Researching...", total=None)

            try:
                async for event in graph.astream(
                    current_input, config=config, stream_mode="updates"
                ):
                    for node_name, node_output in event.items():
                        status = node_output.get("status", "")
                        progress.update(task, description=f"[cyan]{node_name}[/cyan] — {status}")

                        # Display node-specific output
                        if node_name == "web_searcher" and "search_results" in node_output:
                            _display_search_results(node_output["search_results"])

                        elif node_name == "analyst" and "analysis" in node_output:
                            _display_analysis(node_output["analysis"])

                        elif node_name == "writer" and "report" in node_output:
                            final_report = node_output["report"]

                        elif node_name == "supervisor":
                            console.print(_status_panel(status))

                        # Show errors inline
                        if "errors" in node_output and node_output["errors"]:
                            for err in node_output["errors"]:
                                console.print(f"[red]⚠ Error: {err}[/red]")

            except Exception as e:
                console.print(f"[bold red]Pipeline error: {e}[/bold red]")
                break

        # Check if the graph is done
        state = await graph.aget_state(config)
        if not state.next:
            break
        
        current_input = None

        # If not interrupted, we're done
        if not interrupted:
            break

    # ── Display final report ─────────────────────────────────
    if final_report:
        console.print("\n")
        console.print(
            Panel(
                Markdown(final_report),
                title="📝 Research Report",
                border_style="green",
                width=100,
                padding=(1, 2),
            )
        )

        # Save session
        await session_mgr.save_session(
            thread_id=thread_id,
            query=query,
            status="complete",
            report=final_report,
        )
        console.print(f"\n[dim]Session saved: {thread_id}[/dim]")
    else:
        console.print("[yellow]No report generated.[/yellow]")

    # Show rate limiter stats
    try:
        from src.utils.rate_limiter import get_rate_limiter

        stats = get_rate_limiter().stats
        console.print(
            f"[dim]API calls: {stats['total_requests']} | "
            f"Waits: {stats['total_waits']} | "
            f"Wait time: {stats['total_wait_time_seconds']}s[/dim]"
        )
    except Exception:
        pass


# ── Entrypoint ───────────────────────────────────────────────


def main() -> None:
    """CLI entrypoint with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        '  python -m src.cli "What are the latest advances in quantum computing?"\n'
        "  python -m src.cli --iterations 5\n"
        "  python -m src.cli  # interactive prompt",
    )
    parser.add_argument("query", nargs="?", help="Research query (or enter interactively)")
    parser.add_argument(
        "--iterations", "-n", type=int, default=3, help="Max research iterations (default: 3)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # Get query interactively if not provided
    query = args.query
    if not query:
        console.print(
            Panel(
                "[bold]Welcome to the Multi-Agent Research Assistant![/bold]\n\n"
                "Enter a research question and I'll search, analyze, and write a report.",
                border_style="bright_blue",
                width=80,
            )
        )
        query = Prompt.ask("\n[bold]Research query[/bold]")

    if not query or not query.strip():
        console.print("[red]No query provided. Exiting.[/red]")
        sys.exit(1)

    asyncio.run(run_cli(query.strip(), max_iterations=args.iterations))


if __name__ == "__main__":
    main()
