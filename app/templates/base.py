"""Base utilities for visualization templates.

Imports branding from token_labs and provides common formatting functions.
"""

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import plotly.graph_objects as go

# Add token_labs to path for importing visualization module
# This avoids installing the full package with its heavy ML dependencies
_token_labs_path = Path("/home/quantfiction/repositories/token_labs_python")
if str(_token_labs_path) not in sys.path:
    sys.path.insert(0, str(_token_labs_path))

from token_labs.visualization.plotly import colorway, get_template_tie, get_row_col

# =============================================================================
# Shared Style Configuration
# =============================================================================

# Font settings
FONT_FAMILY = "Trebuchet MS, sans-serif"
FONT_SIZE = 12
FONT_SIZE_TICK = 11
FONT_SIZE_TITLE = 18
FONT_SIZE_SUBTITLE = 14

# Title positioning (uses container reference for consistent scaling)
TITLE_X = 0.04
TITLE_Y = 0.96
TITLE_YREF = "container"

# Legend styling
LEGEND_Y = 1.02
LEGEND_X = 0  # Align with y-axis (left edge of plot area)
LEGEND_YREF = "paper"
LEGEND_BGCOLOR = "rgba(51, 65, 85, 0.7)"
LEGEND_BORDERCOLOR = "rgba(255,255,255,0.15)"
LEGEND_BORDERWIDTH = 1

# Axis styling
AXIS_LINECOLOR = "rgba(255,255,255,0.3)"
AXIS_LINEWIDTH = 1

# Margins
MARGIN_TOP = 90
MARGIN_BOTTOM = 60
MARGIN_LEFT = 60
MARGIN_RIGHT = 30
MARGIN_LEFT_HBAR = 120  # Extra left margin for horizontal bar labels


def get_base_font() -> dict:
    """Get base font configuration."""
    return dict(family=FONT_FAMILY, size=FONT_SIZE)


def get_title_config(title_text: str) -> dict:
    """Get title configuration with consistent positioning."""
    return dict(
        text=title_text,
        x=TITLE_X,
        xanchor="left",
        y=TITLE_Y,
        yanchor="top",
        yref=TITLE_YREF,
        font=dict(size=FONT_SIZE_TITLE),
    )


def get_axis_config(tickformat: str = None, extra: dict = None) -> dict:
    """Get axis configuration with consistent styling.

    Args:
        tickformat: Optional tick format string
        extra: Optional extra settings to merge

    Returns:
        Axis configuration dict
    """
    config = dict(
        showgrid=False,
        showline=True,
        linecolor=AXIS_LINECOLOR,
        linewidth=AXIS_LINEWIDTH,
        ticks="",
        tickfont=dict(size=FONT_SIZE_TICK),
        automargin=True,
    )
    if tickformat:
        config["tickformat"] = tickformat
    if extra:
        config.update(extra)
    return config


def get_legend_config(extra: dict = None) -> dict:
    """Get legend configuration with consistent styling.

    Args:
        extra: Optional extra settings to merge

    Returns:
        Legend configuration dict
    """
    config = dict(
        orientation="h",
        yanchor="bottom",
        y=LEGEND_Y,
        xanchor="left",
        x=LEGEND_X,
        yref=LEGEND_YREF,
        font=dict(size=FONT_SIZE),
        bgcolor=LEGEND_BGCOLOR,
        bordercolor=LEGEND_BORDERCOLOR,
        borderwidth=LEGEND_BORDERWIDTH,
    )
    if extra:
        config.update(extra)
    return config


def get_margin_config(left: int = None, extra: dict = None) -> dict:
    """Get margin configuration.

    Args:
        left: Optional left margin override (for horizontal bars)
        extra: Optional extra settings to merge

    Returns:
        Margin configuration dict
    """
    config = dict(
        l=left if left is not None else MARGIN_LEFT,
        r=MARGIN_RIGHT,
        t=MARGIN_TOP,
        b=MARGIN_BOTTOM,
        autoexpand=True,
    )
    if extra:
        config.update(extra)
    return config


def build_subtitle(title: str, subtitle: str = None) -> str:
    """Build title text with optional subtitle.

    Args:
        title: Main title text
        subtitle: Optional subtitle text

    Returns:
        Formatted title string with HTML subtitle if provided
    """
    if subtitle:
        return f"{title}<br><span style='font-size:{FONT_SIZE_SUBTITLE}px;color:#9ca3af'>{subtitle}</span>"
    return title


# Re-export for convenience
__all__ = [
    # From token_labs
    "colorway",
    "get_template_tie",
    "get_row_col",
    # Annotations
    "add_source_annotation",
    "add_updated_annotation",
    "configure_hover",
    # Style config
    "FONT_FAMILY",
    "FONT_SIZE",
    "FONT_SIZE_TICK",
    "FONT_SIZE_TITLE",
    "MARGIN_LEFT_HBAR",
    # Layout helpers
    "get_base_font",
    "get_title_config",
    "get_axis_config",
    "get_legend_config",
    "get_margin_config",
    "build_subtitle",
    # Value formatting
    "format_with_B",  # Legacy, prefer format_bar_label
    "get_y_tickformat",  # Legacy, prefer get_tick_config
    "get_nice_ticks",
    "format_tick_label",
    "get_tick_config",
    "format_hover_value",
    "format_bar_label",
    "get_bar_labels",
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


# =============================================================================
# Value Formatting System
# =============================================================================
# Three contexts for formatting values:
# 1. Tick labels: Abbreviated, no decimals unless small values, nice intervals
# 2. Hover values: Abbreviated, 3 significant figures
# 3. Bar labels: Same as hover


def _get_magnitude_suffix(abs_val: float) -> tuple[float, str]:
    """Get the divisor and suffix for a value's magnitude.

    Returns:
        Tuple of (divisor, suffix) e.g. (1e9, "B")
    """
    if abs_val >= 1e12:
        return 1e12, "T"
    elif abs_val >= 1e9:
        return 1e9, "B"
    elif abs_val >= 1e6:
        return 1e6, "M"
    elif abs_val >= 1e3:
        return 1e3, "k"
    else:
        return 1, ""


def _round_to_sig_figs(value: float, sig_figs: int = 3) -> float:
    """Round a value to a specified number of significant figures."""
    if value == 0:
        return 0
    return round(value, sig_figs - int(math.floor(math.log10(abs(value)))) - 1)


def _get_nice_interval(data_range: float, target_ticks: int = 5) -> float:
    """Calculate a 'nice' tick interval using standard intervals.

    Uses intervals like 1, 2, 2.5, 5, 10 (and their multiples/divisions).

    Args:
        data_range: The range of data (max - min)
        target_ticks: Desired number of ticks

    Returns:
        A nice interval value
    """
    if data_range <= 0:
        return 1

    # Rough interval
    rough_interval = data_range / target_ticks

    # Get the magnitude
    magnitude = 10 ** math.floor(math.log10(rough_interval))

    # Normalize to 1-10 range
    normalized = rough_interval / magnitude

    # Choose nice interval: 1, 2, 2.5, 5, or 10
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 2.5:
        nice = 2.5
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10

    return nice * magnitude


def get_nice_ticks(
    data_min: float, data_max: float, target_ticks: int = 5
) -> list[float]:
    """Generate nice tick values for an axis.

    Args:
        data_min: Minimum data value
        data_max: Maximum data value
        target_ticks: Desired number of ticks

    Returns:
        List of tick values at nice intervals
    """
    data_range = data_max - data_min
    if data_range <= 0:
        return [data_min]

    interval = _get_nice_interval(data_range, target_ticks)

    # Find the first tick (round down to nearest interval)
    first_tick = math.floor(data_min / interval) * interval

    # Find the last tick (round up to nearest interval to ensure coverage)
    last_tick = math.ceil(data_max / interval) * interval

    # Generate ticks
    ticks = []
    tick = first_tick
    while tick <= last_tick + interval * 0.01:  # Small tolerance for floating point
        ticks.append(tick)
        tick += interval

    return ticks


def format_tick_label(
    value: float,
    prefix: str = "",
    suffix: str = "",
    max_value: float = None,
) -> str:
    """Format a value for tick label display.

    Rules:
    - Use B/M/k abbreviation for large numbers
    - No decimals unless max_value < 3 (then use 1 decimal)
    - Include prefix ($) and suffix (%)

    Args:
        value: The value to format
        prefix: Prefix string (e.g., "$")
        suffix: Suffix string (e.g., "%")
        max_value: Maximum value in the dataset (for decimal decision)

    Returns:
        Formatted tick label string
    """
    if value == 0:
        return f"{prefix}0{suffix}"

    abs_val = abs(value)
    sign = "-" if value < 0 else ""

    # Determine if we need decimals (only for small max values)
    use_decimals = max_value is not None and abs(max_value) < 3
    decimals = 1 if use_decimals else 0

    # Get magnitude
    divisor, mag_suffix = _get_magnitude_suffix(abs_val)
    scaled = abs_val / divisor

    # Format the number
    if decimals > 0:
        num_str = f"{scaled:.{decimals}f}"
    else:
        num_str = f"{scaled:.0f}"

    return f"{sign}{prefix}{num_str}{mag_suffix}{suffix}"


def get_tick_config(
    data_values: Sequence[float],
    value_format: str = "number",
    target_ticks: int = 5,
    include_zero: bool = False,
) -> dict:
    """Get tickvals and ticktext for custom axis formatting.

    Args:
        data_values: Sequence of data values to base ticks on
        value_format: Format type ('number', 'currency', 'percent')
        target_ticks: Desired number of ticks
        include_zero: If True, ensure 0 is included (for bar charts)

    Returns:
        Dict with tickvals and ticktext for axis config
    """
    # Handle empty or single value
    if len(data_values) == 0:
        return {}

    data_min = min(data_values)
    data_max = max(data_values)

    # For bar charts, always start from 0
    if include_zero:
        data_min = min(0, data_min)

    # Get nice tick values
    tick_vals = get_nice_ticks(data_min, data_max, target_ticks)

    # Determine prefix/suffix based on format
    prefix = "$" if value_format == "currency" else ""
    suffix = "%" if value_format == "percent" else ""

    # For percent, values are typically 0-1, multiply for display
    if value_format == "percent":
        tick_text = [
            format_tick_label(v * 100, prefix="", suffix="%", max_value=data_max * 100)
            for v in tick_vals
        ]
    else:
        tick_text = [
            format_tick_label(v, prefix=prefix, suffix=suffix, max_value=data_max)
            for v in tick_vals
        ]

    return {"tickvals": tick_vals, "ticktext": tick_text}


def format_hover_value(
    value: float,
    value_format: str = "number",
    sig_figs: int = 3,
) -> str:
    """Format a value for hover display.

    Rules:
    - Use B/M/k abbreviation
    - Round to 3 significant figures
    - Include prefix ($) and suffix (%)

    Args:
        value: The value to format
        value_format: Format type ('number', 'currency', 'percent')
        sig_figs: Number of significant figures

    Returns:
        Formatted hover string
    """
    if value == 0:
        prefix = "$" if value_format == "currency" else ""
        suffix = "%" if value_format == "percent" else ""
        return f"{prefix}0{suffix}"

    # For percent, multiply by 100 for display
    display_val = value * 100 if value_format == "percent" else value

    abs_val = abs(display_val)
    sign = "-" if display_val < 0 else ""

    # Get magnitude
    divisor, mag_suffix = _get_magnitude_suffix(abs_val)
    scaled = abs_val / divisor

    # Round to significant figures
    scaled = _round_to_sig_figs(scaled, sig_figs)

    # Format - use appropriate decimal places based on magnitude
    if scaled >= 100:
        num_str = f"{scaled:.0f}"
    elif scaled >= 10:
        num_str = f"{scaled:.1f}"
    else:
        num_str = f"{scaled:.2f}"

    # Remove trailing zeros after decimal
    if "." in num_str:
        num_str = num_str.rstrip("0").rstrip(".")

    prefix = "$" if value_format == "currency" else ""
    suffix = "%" if value_format == "percent" else ""

    return f"{sign}{prefix}{num_str}{mag_suffix}{suffix}"


def format_bar_label(
    value: float,
    value_format: str = "number",
) -> str:
    """Format a value for bar label display.

    Same rules as hover: abbreviated, 3 significant figures.

    Args:
        value: The value to format
        value_format: Format type ('number', 'currency', 'percent')

    Returns:
        Formatted bar label string
    """
    return format_hover_value(value, value_format, sig_figs=3)


def get_bar_labels(
    values: Sequence[float],
    value_format: str = "number",
) -> list[str]:
    """Generate formatted labels for a series of bar values.

    Args:
        values: Sequence of values to format
        value_format: Format type ('number', 'currency', 'percent')

    Returns:
        List of formatted label strings
    """
    return [format_bar_label(v, value_format) for v in values]


# Legacy function - kept for backwards compatibility
def get_y_tickformat(y_format: str) -> str:
    """Get Plotly tickformat string based on format type.

    DEPRECATED: Use get_tick_config() for proper B/M/k formatting.
    This function is kept for backwards compatibility but returns
    basic d3 format strings that don't support abbreviations.

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
