"""Fiscal CLI — entry point."""
import logging
import sys

import typer

app = typer.Typer(
    name="fiscal",
    help="LLM-powered financial report pipeline.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def _callback() -> None:
    """LLM-powered financial report pipeline."""


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-40s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address"),
    port: int = typer.Option(8080, "--port", "-p", help="Port to listen on"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start the web UI — click a company button to generate and download its PDF report."""
    from fiscal.server import run_server

    _setup_logging(verbose)

    from rich.console import Console
    console = Console()
    console.print(
        f"[green]✓[/green] Fiscal server starting at "
        f"[bold cyan]http://{host}:{port}[/bold cyan]"
    )
    console.print("[dim]Press Ctrl+C to stop.[/dim]")
    run_server(host=host, port=port)
