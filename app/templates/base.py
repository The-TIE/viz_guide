"""Base utilities for visualization templates.

Imports branding from token_labs and provides common formatting functions.
"""

import sys
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go

# Add token_labs to path for importing visualization module
# This avoids installing the full package with its heavy ML dependencies
_token_labs_path = Path("/home/quantfiction/repositories/token_labs_python")
if str(_token_labs_path) not in sys.path:
    sys.path.insert(0, str(_token_labs_path))

from token_labs.visualization.plotly import colorway, get_template_tie, get_row_col

# Re-export for convenience
__all__ = [
    "colorway",
    "get_template_tie",
    "get_row_col",
    "format_with_B",
    "add_source_annotation",
    "add_updated_annotation",
    "configure_hover",
]


def format_with_B(value: float, prefix: str = "$", decimals: int = 1) -> str:
    """Format large numbers with B/M/k suffixes.

    Args:
        value: Number to format
        prefix: Prefix string (e.g., "$" for currency)
        decimals: Number of decimal places

    Returns:
        Formatted string like "$1.2B" or "$45.3M"
    """
    abs_val = abs(value)
    sign = "-" if value < 0 else ""

    if abs_val >= 1e12:
        return f"{sign}{prefix}{abs_val/1e12:.{decimals}f}T"
    elif abs_val >= 1e9:
        return f"{sign}{prefix}{abs_val/1e9:.{decimals}f}B"
    elif abs_val >= 1e6:
        return f"{sign}{prefix}{abs_val/1e6:.{decimals}f}M"
    elif abs_val >= 1e3:
        return f"{sign}{prefix}{abs_val/1e3:.{decimals}f}k"
    else:
        return f"{sign}{prefix}{abs_val:.{decimals}f}"


def add_source_annotation(fig: go.Figure, source: str) -> go.Figure:
    """Add source annotation to bottom left of figure.

    Args:
        fig: Plotly figure to annotate
        source: Source text (e.g., "CoinGecko", "Glassnode")

    Returns:
        Modified figure with source annotation
    """
    fig.add_annotation(
        text=f"Source: {source}",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.12,
        showarrow=False,
        font=dict(size=10, color="#9ca3af"),
        xanchor="left",
    )
    return fig


def add_updated_annotation(fig: go.Figure, timestamp: datetime = None) -> go.Figure:
    """Add 'Updated' timestamp annotation to bottom right of figure.

    Args:
        fig: Plotly figure to annotate
        timestamp: Datetime to display (defaults to now)

    Returns:
        Modified figure with updated annotation
    """
    if timestamp is None:
        timestamp = datetime.now()

    fig.add_annotation(
        text=f"Updated: {timestamp.strftime('%b %d, %Y')}",
        xref="paper",
        yref="paper",
        x=1,
        y=-0.12,
        showarrow=False,
        font=dict(size=10, color="#9ca3af"),
        xanchor="right",
    )
    return fig


def configure_hover(fig: go.Figure, mode: str = "x unified") -> go.Figure:
    """Configure hover mode and styling.

    Args:
        fig: Plotly figure to configure
        mode: Hover mode ('x', 'y', 'closest', 'x unified', 'y unified')

    Returns:
        Modified figure with hover configuration
    """
    fig.update_layout(
        hovermode=mode,
        hoverlabel=dict(
            bgcolor="#1f2937",
            font_size=12,
            font_color="#d3d4d6",
        ),
    )
    return fig


def get_y_tickformat(y_format: str) -> str:
    """Get Plotly tickformat string based on format type.

    Args:
        y_format: Format type ('number', 'currency', 'percent')

    Returns:
        Plotly tickformat string
    """
    formats = {
        "number": ",.0f",
        "currency": "$,.0f",
        "percent": ".1%",
    }
    return formats.get(y_format, ",.0f")
