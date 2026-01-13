"""Line chart templates for time series visualizations."""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base import (
    colorway,
    get_template_tie,
    get_row_col,
    format_with_B,
    add_source_annotation,
    add_updated_annotation,
    configure_hover,
    get_y_tickformat,
)


def line_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    subtitle: Optional[str] = None,
    y_format: str = "number",
    source: Optional[str] = None,
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
        source: Optional source attribution
        watermark: Watermark type ('tie', 'labs', 'qf', 'none')

    Returns:
        Configured Plotly figure
    """
    fig = go.Figure()

    # Add line trace
    fig.add_trace(
        go.Scatter(
            x=df[x_column],
            y=df[y_column],
            mode="lines",
            name=y_column,
            line=dict(width=2, color=colorway[0]),
            hovertemplate="%{x|%b %d, %Y}<br>%{y:,.2f}<extra></extra>",
        )
    )

    # Build title with optional subtitle
    title_text = title
    if subtitle:
        title_text = f"{title}<br><span style='font-size:14px;color:#9ca3af'>{subtitle}</span>"

    # Apply template - always use token_labs styling, watermark controls only the logo
    template = get_template_tie(watermark if watermark != "none" else "tie")
    if watermark == "none":
        template.layout.images = []  # Remove watermark but keep all styling

    fig.update_layout(
        template=template,
        title=dict(text=title_text, x=0, xanchor="left"),
        xaxis=dict(
            showgrid=False,
            tickformat="%b %Y",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            tickformat=get_y_tickformat(y_format),
        ),
        margin=dict(l=60, r=30, t=80, b=80),
        showlegend=False,
    )

    # Configure hover
    configure_hover(fig, mode="x unified")

    # Add annotations
    if source:
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
    source: Optional[str] = None,
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
        source: Optional source attribution
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

    # Determine hover format based on normalization
    if normalize_mode == 'returns':
        hover_format = "%{y:+.1f}%"
    elif normalize_mode == 'indexed':
        hover_format = "%{y:,.0f}"
    else:
        hover_format = "%{y:,.2f}"

    # Use custom colors if provided, otherwise fall back to default colorway
    color_list = colors if colors else colorway

    # Add traces for each series
    for i, col in enumerate(y_columns):
        color = color_list[i % len(color_list)]

        fig.add_trace(
            go.Scatter(
                x=plot_df[x_column],
                y=plot_df[col],
                mode="lines",
                name=col,
                line=dict(width=2, color=color),
                hovertemplate=f"<b>{col}</b>: {hover_format}<extra></extra>",
            )
        )

    # Build title with optional subtitle
    title_text = title
    if subtitle:
        title_text = f"{title}<br><span style='font-size:14px;color:#9ca3af'>{subtitle}</span>"

    # Apply template - always use token_labs styling, watermark controls only the logo
    template = get_template_tie(watermark if watermark != "none" else "tie")
    if watermark == "none":
        template.layout.images = []  # Remove watermark but keep all styling

    # Y-axis configuration based on normalization mode
    if normalize_mode == 'returns':
        y_title = "Cumulative Return (%)"
        y_tickformat = "+.0f"
        y_ticksuffix = "%"
    elif normalize_mode == 'indexed':
        y_title = "Indexed Value (Start = 100)"
        y_tickformat = ",.0f"
        y_ticksuffix = None
    else:
        y_title = None
        y_tickformat = get_y_tickformat(y_format)
        y_ticksuffix = None

    fig.update_layout(
        template=template,
        title=dict(text=title_text, x=0, xanchor="left"),
        xaxis=dict(
            showgrid=False,
            tickformat="%b %Y",
        ),
        yaxis=dict(
            title=y_title,
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            tickformat=y_tickformat,
            ticksuffix=y_ticksuffix,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=60, r=30, t=100, b=80),
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
    if source:
        add_source_annotation(fig, source)
    add_updated_annotation(fig)

    return fig


def small_multiples_chart(
    df: pd.DataFrame,
    x_column: str,
    y_columns: list[str],
    title: str,
    cols: int = 2,
    source: Optional[str] = None,
    watermark: str = "tie",
) -> go.Figure:
    """Create a small multiples chart (subplot grid) for 6+ series.

    Args:
        df: DataFrame with data
        x_column: Column name for x-axis (typically date)
        y_columns: List of column names for y-axis series
        title: Main chart title
        cols: Number of columns in subplot grid (default 2)
        source: Optional source attribution
        watermark: Watermark type ('tie', 'labs', 'qf', 'none')

    Returns:
        Configured Plotly figure with subplot grid
    """
    n_series = len(y_columns)
    rows = (n_series + cols - 1) // cols

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

        fig.add_trace(
            go.Scatter(
                x=df[x_column],
                y=df[col],
                mode="lines",
                name=col,
                line=dict(width=2, color=colorway[0]),
                showlegend=False,
                hovertemplate=f"<b>{col}</b><br>%{{x|%b %d}}: %{{y:,.2f}}<extra></extra>",
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
        title=dict(text=title, x=0, xanchor="left"),
        height=250 * rows,
        showlegend=False,
        margin=dict(l=60, r=30, t=80, b=80),
    )

    # Update all x and y axes
    fig.update_xaxes(showgrid=False, tickformat="%b")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")

    # Add annotations
    if source:
        add_source_annotation(fig, source)
    add_updated_annotation(fig)

    return fig
