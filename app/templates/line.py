"""Line chart templates for time series visualizations."""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base import (
    colorway,
    get_template_tie,
    get_row_col,
    add_source_annotation,
    add_updated_annotation,
    configure_hover,
    get_base_font,
    get_title_config,
    get_axis_config,
    get_legend_config,
    get_margin_config,
    build_subtitle,
    # New formatting functions
    get_tick_config,
    format_hover_value,
)


def line_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    subtitle: Optional[str] = None,
    y_format: str = "number",
    source: str = "The Tie",
    watermark: str = "tie",
) -> go.Figure:
    """Create a single line chart for time series trends.

    Args:
        df: DataFrame with data
        x_column: Column name for x-axis (typically date)
        y_column: Column name for y-axis values
        title: Chart title
        subtitle: Optional subtitle below title
        y_format: Value format ('number', 'currency', 'percent')
        source: Source attribution (default: "The Tie")
        watermark: Watermark type ('tie', 'labs', 'qf', 'none')

    Returns:
        Configured Plotly figure
    """
    fig = go.Figure()

    # Pre-format hover values
    hover_labels = [format_hover_value(v, y_format) for v in df[y_column]]

    # Add line trace (date shown once at top in unified hover mode)
    fig.add_trace(
        go.Scatter(
            x=df[x_column],
            y=df[y_column],
            mode="lines",
            name=y_column,
            line=dict(width=2, color=colorway[0]),
            customdata=hover_labels,
            hovertemplate="%{customdata}<extra></extra>",
        )
    )

    # Apply template - always use token_labs styling, watermark controls only the logo
    template = get_template_tie(watermark if watermark != "none" else "tie")
    if watermark == "none":
        template.layout.images = []  # Remove watermark but keep all styling

    # Get formatted tick configuration
    y_tick_config = get_tick_config(df[y_column], y_format)

    fig.update_layout(
        template=template,
        font=get_base_font(),
        title=get_title_config(build_subtitle(title, subtitle)),
        xaxis=get_axis_config(tickformat="%b %Y"),
        yaxis=get_axis_config(extra=y_tick_config),
        margin=get_margin_config(),
        showlegend=False,
    )

    # Configure hover
    configure_hover(fig, mode="x unified")

    # Add annotations
    add_source_annotation(fig, source)
    add_updated_annotation(fig)

    return fig


def multi_line_chart(
    df: pd.DataFrame,
    x_column: str,
    y_columns: list[str],
    title: str,
    subtitle: Optional[str] = None,
    normalize: bool | str = False,
    colors: Optional[list[str]] = None,
    y_format: str = "number",
    source: str = "The Tie",
    watermark: str = "tie",
) -> go.Figure:
    """Create a multi-line chart for comparing 2-5 series.

    Args:
        df: DataFrame with data
        x_column: Column name for x-axis (typically date)
        y_columns: List of column names for y-axis series (2-5 recommended)
        title: Chart title
        subtitle: Optional subtitle below title
        normalize: Normalization mode:
            - False: No normalization (raw values)
            - True or 'indexed': Index to 100 at start
            - 'returns': Percentage returns from 0% at start
        colors: Optional list of hex colors for each series (e.g., ['#F7931A', '#627EEA'])
        y_format: Value format ('number', 'currency', 'percent')
        source: Source attribution (default: "The Tie")
        watermark: Watermark type ('tie', 'labs', 'qf', 'none')

    Returns:
        Configured Plotly figure
    """
    fig = go.Figure()

    # Prepare data - normalize if requested
    plot_df = df.copy()
    normalize_mode = None
    if normalize:
        normalize_mode = 'returns' if normalize == 'returns' else 'indexed'
        for col in y_columns:
            first_val = plot_df[col].iloc[0]
            if first_val != 0:
                if normalize_mode == 'returns':
                    # Percentage returns from 0%
                    plot_df[col] = ((plot_df[col] / first_val) - 1) * 100
                else:
                    # Index to 100 at start
                    plot_df[col] = (plot_df[col] / first_val) * 100

    # Use custom colors if provided, otherwise fall back to default colorway
    color_list = colors if colors else colorway

    # Collect all values for tick calculation (non-normalized case)
    all_values = []
    for col in y_columns:
        all_values.extend(plot_df[col].tolist())

    # Add traces for each series
    for i, col in enumerate(y_columns):
        color = color_list[i % len(color_list)]

        # Format hover values based on mode
        if normalize_mode == 'returns':
            hover_labels = [f"{v:+.1f}%" for v in plot_df[col]]
        elif normalize_mode == 'indexed':
            hover_labels = [f"{v:,.0f}" for v in plot_df[col]]
        else:
            hover_labels = [format_hover_value(v, y_format) for v in plot_df[col]]

        # Date shown once at top in unified hover mode
        fig.add_trace(
            go.Scatter(
                x=plot_df[x_column],
                y=plot_df[col],
                mode="lines",
                name=col,
                line=dict(width=2, color=color),
                customdata=hover_labels,
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

    # Apply template - always use token_labs styling, watermark controls only the logo
    template = get_template_tie(watermark if watermark != "none" else "tie")
    if watermark == "none":
        template.layout.images = []  # Remove watermark but keep all styling

    # Y-axis configuration based on normalization mode
    if normalize_mode == 'returns':
        y_title = "Cumulative Return (%)"
        yaxis_extra = {
            "title": dict(text=y_title, font=dict(size=12)),
            "tickformat": "+.0f",
            "ticksuffix": "%",
        }
    elif normalize_mode == 'indexed':
        y_title = "Indexed Value (Start = 100)"
        yaxis_extra = {
            "title": dict(text=y_title, font=dict(size=12)),
            "tickformat": ",.0f",
        }
    else:
        # Use B/M/k formatting for raw values
        yaxis_extra = get_tick_config(all_values, y_format)

    fig.update_layout(
        template=template,
        font=get_base_font(),
        title=get_title_config(build_subtitle(title, subtitle)),
        xaxis=get_axis_config(tickformat="%b %Y"),
        yaxis=get_axis_config(extra=yaxis_extra),
        legend=get_legend_config(),
        margin=get_margin_config(),
    )

    # Add reference line at 0% for returns mode
    if normalize_mode == 'returns':
        fig.add_hline(
            y=0,
            line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'),
        )

    # Configure unified hover
    configure_hover(fig, mode="x unified")

    # Add annotations
    add_source_annotation(fig, source)
    add_updated_annotation(fig)

    return fig


def small_multiples_chart(
    df: pd.DataFrame,
    x_column: str,
    y_columns: list[str],
    title: str,
    cols: int = 2,
    y_format: str = "number",
    source: str = "The Tie",
    watermark: str = "tie",
) -> go.Figure:
    """Create a small multiples chart (subplot grid) for 6+ series.

    Args:
        df: DataFrame with data
        x_column: Column name for x-axis (typically date)
        y_columns: List of column names for y-axis series
        title: Main chart title
        cols: Number of columns in subplot grid (default 2)
        y_format: Value format ('number', 'currency', 'percent')
        source: Source attribution (default: "The Tie")
        watermark: Watermark type ('tie', 'labs', 'qf', 'none')

    Returns:
        Configured Plotly figure with subplot grid
    """
    n_series = len(y_columns)
    rows = (n_series + cols - 1) // cols

    # Collect all values for tick calculation
    all_values = []
    for col in y_columns:
        all_values.extend(df[col].tolist())

    # Create subplots
    fig = make_subplots(
        rows=rows,
        cols=cols,
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=y_columns,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    # Add traces to each subplot
    for i, col in enumerate(y_columns):
        row, col_idx = get_row_col(i, cols)
        hover_labels = [format_hover_value(v, y_format) for v in df[col]]

        fig.add_trace(
            go.Scatter(
                x=df[x_column],
                y=df[col],
                mode="lines",
                name=col,
                line=dict(width=2, color=colorway[0]),
                showlegend=False,
                customdata=hover_labels,
                hovertemplate=f"<b>{col}</b><br>%{{x|%b %d}}: %{{customdata}}<extra></extra>",
            ),
            row=row,
            col=col_idx,
        )

    # Apply template - always use token_labs styling, watermark controls only the logo
    template = get_template_tie(watermark if watermark != "none" else "tie")
    if watermark == "none":
        template.layout.images = []  # Remove watermark but keep all styling

    fig.update_layout(
        template=template,
        font=get_base_font(),
        title=get_title_config(title),
        height=250 * rows,
        showlegend=False,
        margin=get_margin_config(),
    )

    # Get formatted tick configuration for y-axes
    y_tick_config = get_tick_config(all_values, y_format)

    # Update all x and y axes using shared config
    x_config = get_axis_config(tickformat="%b")
    y_config = get_axis_config(extra=y_tick_config)
    fig.update_xaxes(**x_config)
    fig.update_yaxes(**y_config)

    # Add annotations
    add_source_annotation(fig, source)
    add_updated_annotation(fig)

    return fig
