"""Standardized visualization templates for production-ready Plotly charts.

Import template functions directly:
    from app.templates import multi_line_chart, horizontal_bar_chart

All templates:
- Apply dark theme and branding from token_labs
- Handle common formatting (billions, hover, annotations)
- Return configured go.Figure ready for display
"""

from .base import (
    format_with_B,
    add_source_annotation,
    add_updated_annotation,
    configure_hover,
)

from .line import (
    line_chart,
    multi_line_chart,
    small_multiples_chart,
)

from .bar import (
    bar_chart,
    horizontal_bar_chart,
    stacked_bar_chart,
    grouped_bar_chart,
)

from .detector import detect_chart_type

__all__ = [
    # Utilities
    "format_with_B",
    "add_source_annotation",
    "add_updated_annotation",
    "configure_hover",
    # Line charts
    "line_chart",
    "multi_line_chart",
    "small_multiples_chart",
    # Bar charts
    "bar_chart",
    "horizontal_bar_chart",
    "stacked_bar_chart",
    "grouped_bar_chart",
    # Detection
    "detect_chart_type",
]
