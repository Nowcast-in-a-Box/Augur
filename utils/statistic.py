import numpy as np
from typing import Dict, Tuple, List, Any
from torchmetrics import MetricCollection
from rich.table import Table
from rich.console import Console

def print_result(
    model_name: str, 
    metric_collection: Dict[str, MetricCollection], 
    lead_times: Tuple[int, ...],
    metric_order: List[str] = None
) -> Table:
    """
    Computes and formats the results into a Rich Table.
    
    Args:
        model_name: Name of the model.
        metric_collection: Dict mapping "lead_X" to a computed MetricCollection or the collection itself.
        lead_times: Tuple of lead time values.
        metric_order: Optional list of metric names to define column order.
    """
    # Compute results if not already computed
    results = {}
    available_metrics = set()
    for lead, collection in metric_collection.items():
        computed = collection.compute()
        results[lead] = {k: v.item() if hasattr(v, "item") else v for k, v in computed.items()}
        available_metrics.update(results[lead].keys())

    # Determine column order
    if metric_order:
        headers = [m for m in metric_order if m in available_metrics]
    else:
        headers = sorted(list(available_metrics))

    # Create table
    table = Table(
        title=f"Evaluation Results: [bold]{model_name}[/bold]", 
        show_header=True, 
        header_style="bold magenta",
        box=None,
        border_style="bright_black"
    )

    # Add columns
    table.add_column("Lead Time", style="cyan", justify="center", header_style="bold cyan")
    for header in headers:
        table.add_column(header.upper(), justify="center", style="green", header_style="bold green")

    # Storage for mean calculation
    all_values = {h: [] for h in headers}

    # Fill data rows
    for lt in lead_times:
        lead_key = f"lead_{lt}"
        if lead_key not in results:
            continue
            
        row = [str(lt)]
        for h in headers:
            val = results[lead_key].get(h, float("nan"))
            all_values[h].append(val)
            row.append(f"{val:.6f}")
        table.add_row(*row)

    table.add_section()

    # Mean value row
    mean_row = ["[bold]Mean[/bold]"]
    for h in headers:
        vals = [v for v in all_values[h] if not np.isnan(v)] if "np" in globals() else all_values[h]
        # Using simple sum/len for mean
        valid_vals = [v for v in vals if v == v] # filter out nans
        mean_val = sum(valid_vals) / len(valid_vals) if valid_vals else 0.0
        mean_row.append(f"[bold]{mean_val:.6f}[/bold]")

    table.add_row(*mean_row, style="bold yellow")

    return table