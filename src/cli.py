"""
Slooze Data Engineering Challenge - CLI Application
====================================================

Typer-based command-line interface for the complete pipeline.
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR, DATA_DIR, LOGS_DIR

app = typer.Typer(
    name="slooze",
    help="🚀 Slooze Data Engineering Pipeline - B2B Marketplace Intelligence",
    add_completion=False
)

console = Console()


@app.command()
def collect(
    mock: bool = typer.Option(False, "--mock", "-m", help="Use mock data generation"),
    categories: str = typer.Option(None, "--categories", "-c", help="Comma-separated categories")
):
    """
    🕷️ Collect data from B2B marketplaces.
    
    This command scrapes product data from IndiaMART and similar marketplaces.
    Use --mock flag to generate realistic test data instead of live scraping.
    """
    console.print(Panel.fit(
        "[bold blue]Data Collection Module[/bold blue]\n"
        "Gathering B2B marketplace intelligence...",
        border_style="blue"
    ))
    
    from collector import AdvancedDataCollector
    
    collector = AdvancedDataCollector()
    count = collector.run(use_mock=mock)
    
    if count > 0:
        rprint(f"\n[green]✅ Successfully collected {count} product records![/green]")
    else:
        rprint("\n[red]❌ Collection failed. Check logs for details.[/red]")


@app.command()
def analyze():
    """
    📊 Run Exploratory Data Analysis.
    
    Generates comprehensive analysis including:
    - Statistical summaries
    - Interactive visualizations
    - NLP text analysis
    - Product clustering
    - HTML report
    """
    console.print(Panel.fit(
        "[bold magenta]EDA & Analytics Module[/bold magenta]\n"
        "Generating insights and visualizations...",
        border_style="magenta"
    ))
    
    from eda_analysis import AdvancedEDAEngine
    
    engine = AdvancedEDAEngine()
    engine.run()


@app.command()
def pipeline(
    mock: bool = typer.Option(False, "--mock", "-m", help="Use mock data")
):
    """
    🔄 Run the complete ETL + EDA pipeline.
    
    Executes data collection followed by analysis in one command.
    """
    console.print(Panel.fit(
        "[bold cyan]Full Pipeline Execution[/bold cyan]\n"
        "Running: Collection -> Validation -> Analysis -> Reporting",
        border_style="cyan"
    ))
    
    # Step 1: Collect
    console.print("\n[bold]Step 1/2: Data Collection[/bold]")
    from collector import AdvancedDataCollector
    collector = AdvancedDataCollector()
    count = collector.run(use_mock=mock)
    
    if count == 0:
        rprint("[red]Pipeline aborted: No data collected.[/red]")
        raise typer.Exit(1)
    
    # Step 2: Analyze
    console.print("\n[bold]Step 2/2: Data Analysis[/bold]")
    from eda_analysis import AdvancedEDAEngine
    engine = AdvancedEDAEngine()
    engine.run()
    
    console.print(Panel.fit(
        "[bold green]Pipeline Complete![/bold green]\n\n"
        f"Data: {DATA_DIR}\n"
        f"Reports: {OUTPUT_DIR / 'reports'}\n"
        f"Charts: {OUTPUT_DIR / 'charts'}",
        border_style="green"
    ))


@app.command()
def status():
    """
    📋 Show current data statistics.
    """
    from database import db_manager
    
    console.print(Panel.fit(
        "[bold yellow]Pipeline Status[/bold yellow]",
        border_style="yellow"
    ))
    
    try:
        products = db_manager.get_all_products()
        stats = db_manager.get_category_stats()
        
        console.print(f"\n[bold]Database Records:[/bold] {len(products)}")
        
        if stats:
            console.print("\n[bold]Category Breakdown:[/bold]")
            for stat in stats:
                console.print(f"  • {stat['category']}: {stat['count']} products (Avg: ₹{stat['avg_price']:,.0f})")
        else:
            console.print("[dim]No data collected yet. Run 'collect' first.[/dim]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def clean():
    """
    🗑️ Clear all collected data and outputs.
    """
    from database import db_manager
    import shutil
    
    confirm = typer.confirm("This will delete all data. Are you sure?")
    
    if confirm:
        # Clear database
        db_manager.clear_all_products()
        
        # Clear output directories
        for directory in [OUTPUT_DIR / "charts", OUTPUT_DIR / "reports"]:
            if directory.exists():
                shutil.rmtree(directory)
                directory.mkdir(parents=True, exist_ok=True)
        
        console.print("[green]✅ All data cleared successfully.[/green]")
    else:
        console.print("[dim]Operation cancelled.[/dim]")


@app.callback()
def main():
    """
    🚀 Slooze Data Engineering Challenge
    
    A production-grade data collection and analysis pipeline
    for B2B marketplace intelligence.
    """
    pass


if __name__ == "__main__":
    app()
