"""Bar chart templates for categorical and ranking visualizations."""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from .base import (
    colorway,
    get_template_tie,
    format_with_B,
    add_source_annotation,
    add_updated_annotation,
    configure_hover,
    get_y_tickformat,
)


def bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    subtitle: Optional[str] = None,
    sort: bool = False,
    y_format: str = "number",
    source: Optional[str] = None,
    watermark: str = "tie",
) -> go.Figure:
    """Create a vertical bar chart.

    Args:
        df: DataFrame with data
        x_column: Column name for x-axis categories
        y_column: Column name for y-axis values
        title: Chart title
        subtitle: Optional subtitle below title
        sort: If True, sort bars by value (descending)
        y_format: Value format ('number', 'currency', 'percent')
        source: Optional source attribution
        watermark: Watermark type ('tie', 'labs', 'qf', 'none')

    Returns:
        Configured Plotly figure
    """
    # Sort if requested
    plot_df = df.copy()
    if sort:
        plot_df = plot_df.sort_values(y_column, ascending=False)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=plot_df[x_column],
            y=plot_df[y_column],
            marker=dict(color=colorway[0]),
            hovertemplate="%{x}: %{y:,.2f}<extra></extra>",
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
        xaxis=dict(showgrid=False),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            tickformat=get_y_tickformat(y_format),
        ),
        bargap=0.2,
        margin=dict(l=60, r=30, t=80, b=80),
        showlegend=False,
    )

    # Configure hover
    configure_hover(fig, mode="closest")

    # Add annotations
    if source:
        add_source_annotation(fig, source)
    add_updated_annotation(fig)

    return fig


def horizontal_bar_chart(
    df: pd.DataFrame,
    category_column: str,
    value_column: str,
    title: str,
    subtitle: Optional[str] = None,
    sort: bool = True,
    value_format: str = "number",
    source: Optional[str] = None,
    watermark: str = "tie",
) -> go.Figure:
    """Create a horizontal bar chart for rankings and long labels.

    Args:
        df: DataFrame with data
        category_column: Column name for categories (y-axis)
        value_column: Column name for values (x-axis)
        title: Chart title
        subtitle: Optional subtitle below title
        sort: If True (default), sort bars by value (largest at top)
        value_format: Value format ('number', 'currency', 'percent')
        source: Optional source attribution
        watermark: Watermark type ('tie', 'labs', 'qf', 'none')

    Returns:
        Configured Plotly figure
    """
    # Sort for ranking (ascending=True puts largest at top for horizontal)
    plot_df = df.copy()
    if sort:
        plot_df = plot_df.sort_values(value_column, ascending=True)

    fig = go.Figure()

    # Determine if we need text inside or outside bars
    max_val = plot_df[value_column].max()

    fig.add_trace(
        go.Bar(
            y=plot_df[category_column],
            x=plot_df[value_column],
            orientation="h",
            marker=dict(color=colorway[0]),
            text=plot_df[value_column].apply(lambda x: format_with_B(x, prefix="", decimals=1)),
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color="white"),
            hovertemplate="%{y}: %{x:,.2f}<extra></extra>",
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
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            tickformat=get_y_tickformat(value_format),
        ),
        yaxis=dict(
            showgrid=False,
            categoryorder="total ascending" if sort else None,
        ),
        bargap=0.2,
        margin=dict(l=120, r=30, t=80, b=80),  # Extra left margin for labels
        showlegend=False,
    )

    # Configure hover
    configure_hover(fig, mode="closest")

    # Add annotations
    if source:
        add_source_annotation(fig, source)
    add_updated_annotation(fig)

    return fig


def stacked_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    stack_columns: list[str],
    title: str,
    subtitle: Optional[str] = None,
    horizontal: bool = False,
    source: Optional[str] = None,
    watermark: str = "tie",
) -> go.Figure:
    """Create a stacked bar chart for composition.

    Args:
        df: DataFrame with data
        x_column: Column name for x-axis categories
        stack_columns: List of column names to stack
        title: Chart title
        subtitle: Optional subtitle below title
        horizontal: If True, create horizontal stacked bars
        source: Optional source attribution
        watermark: Watermark type ('tie', 'labs', 'qf', 'none')

    Returns:
        Configured Plotly figure
    """
    fig = go.Figure()

    # Add trace for each stack segment
    for i, col in enumerate(stack_columns):
        color = colorway[i % len(colorway)]

        if horizontal:
            fig.add_trace(
                go.Bar(
                    y=df[x_column],
                    x=df[col],
                    name=col,
                    orientation="h",
                    marker=dict(color=color),
                    hovertemplate=f"<b>{col}</b>: %{{x:,.2f}}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=df[x_column],
                    y=df[col],
                    name=col,
                    marker=dict(color=color),
                    hovertemplate=f"<b>{col}</b>: %{{y:,.2f}}<extra></extra>",
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
        barmode="stack",
        xaxis=dict(showgrid=False if not horizontal else True, gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(showgrid=True if not horizontal else False, gridcolor="rgba(255,255,255,0.1)"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=60 if not horizontal else 120, r=30, t=100, b=80),
    )

    # Configure hover
    configure_hover(fig, mode="closest")

    # Add annotations
    if source:
        add_source_annotation(fig, source)
    add_updated_annotation(fig)

    return fig


def grouped_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    group_columns: list[str],
    title: str,
    subtitle: Optional[str] = None,
    source: Optional[str] = None,
    watermark: str = "tie",
) -> go.Figure:
    """Create a grouped bar chart for comparing 2-3 measures.

    Args:
        df: DataFrame with data
        x_column: Column name for x-axis categories
        group_columns: List of column names to group (2-3 recommended)
        title: Chart title
        subtitle: Optional subtitle below title
        source: Optional source attribution
        watermark: Watermark type ('tie', 'labs', 'qf', 'none')

    Returns:
        Configured Plotly figure
    """
    fig = go.Figure()

    # Add trace for each group
    for i, col in enumerate(group_columns):
        color = colorway[i % len(colorway)]

        fig.add_trace(
            go.Bar(
                x=df[x_column],
                y=df[col],
                name=col,
                marker=dict(color=color),
                hovertemplate=f"<b>{col}</b>: %{{y:,.2f}}<extra></extra>",
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
        barmode="group",
        xaxis=dict(showgrid=False),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
        ),
        bargap=0.15,
        bargroupgap=0.1,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=60, r=30, t=100, b=80),
    )

    # Configure hover
    configure_hover(fig, mode="closest")

    # Add annotations
    if source:
        add_source_annotation(fig, source)
    add_updated_annotation(fig)

    return fig
