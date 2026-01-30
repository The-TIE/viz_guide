"""Bar chart templates for categorical and ranking visualizations."""

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from .base import (
    colorway,
    get_template_tie,
    add_source_annotation,
    add_updated_annotation,
    configure_hover,
    get_base_font,
    get_title_config,
    get_axis_config,
    get_legend_config,
    get_margin_config,
    build_subtitle,
    MARGIN_LEFT_HBAR,
    # New formatting functions
    get_tick_config,
    get_bar_labels,
    format_hover_value,
)


def bar_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    subtitle: Optional[str] = None,
    sort: bool = False,
    y_format: str = "number",
    source: str = "The Tie",
    watermark: str = "tie",
    **kwargs,
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
        source: Source attribution (default: "The Tie")
        watermark: Watermark type ('tie', 'labs', 'qf', 'none')

    Returns:
        Configured Plotly figure
    """
    # Sort if requested
    plot_df = df.copy()
    if sort:
        plot_df = plot_df.sort_values(y_column, ascending=False)

    # Pre-format hover values (always needed)
    hover_labels = [format_hover_value(v, y_format) for v in plot_df[y_column]]

    # Only add bar labels if ≤12 bars (per guide/07_text.md)
    n_bars = len(plot_df)
    if n_bars <= 12:
        bar_labels = get_bar_labels(plot_df[y_column], y_format)
        # Position inside by default, outside only for short bars
        max_val = plot_df[y_column].max()
        threshold = max_val * 0.15  # Bars < 15% of max get outside labels
        text_positions = [
            "inside" if v > threshold else "outside"
            for v in plot_df[y_column]
        ]
    else:
        bar_labels = None
        text_positions = None

    fig = go.Figure()

    trace_kwargs = dict(
        x=plot_df[x_column],
        y=plot_df[y_column],
        marker=dict(color=colorway[0]),
        customdata=hover_labels,
        hovertemplate="<b>%{x}</b>: %{customdata}<extra></extra>",
    )
    if bar_labels is not None:
        trace_kwargs.update(
            text=bar_labels,
            textposition=text_positions,
            textfont=dict(color="white"),
            insidetextanchor="end",
            cliponaxis=False,
        )

    fig.add_trace(go.Bar(**trace_kwargs))

    # Apply template - always use token_labs styling, watermark controls only the logo
    template = get_template_tie(watermark if watermark != "none" else "tie")
    if watermark == "none":
        template.layout.images = []  # Remove watermark but keep all styling

    # Get formatted tick configuration (include_zero for bar charts)
    y_tick_config = get_tick_config(plot_df[y_column], y_format, include_zero=True)

    fig.update_layout(
        template=template,
        font=get_base_font(),
        title=get_title_config(build_subtitle(title, subtitle)),
        xaxis=get_axis_config(),
        yaxis=get_axis_config(extra=y_tick_config),
        bargap=0.2,
        margin=get_margin_config(),
        showlegend=False,
    )

    # Configure hover
    configure_hover(fig, mode="closest")

    # Add annotations
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
    source: str = "The Tie",
    watermark: str = "tie",
    **kwargs,
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
        source: Source attribution (default: "The Tie")
        watermark: Watermark type ('tie', 'labs', 'qf', 'none')

    Returns:
        Configured Plotly figure
    """
    # Sort for ranking (ascending=True puts largest at top for horizontal)
    plot_df = df.copy()
    if sort:
        plot_df = plot_df.sort_values(value_column, ascending=True)

    # Pre-format labels and hover values
    bar_labels = get_bar_labels(plot_df[value_column], value_format)
    hover_labels = [format_hover_value(v, value_format) for v in plot_df[value_column]]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=plot_df[category_column],
            x=plot_df[value_column],
            orientation="h",
            marker=dict(color=colorway[0]),
            text=bar_labels,
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(color="white"),
            customdata=hover_labels,
            hovertemplate="<b>%{y}</b>: %{customdata}<extra></extra>",
        )
    )

    # Apply template - always use token_labs styling, watermark controls only the logo
    template = get_template_tie(watermark if watermark != "none" else "tie")
    if watermark == "none":
        template.layout.images = []  # Remove watermark but keep all styling

    # Get formatted tick configuration for x-axis (values, include_zero for bar charts)
    x_tick_config = get_tick_config(plot_df[value_column], value_format, include_zero=True)

    # Y-axis with optional category ordering
    yaxis_extra = {"categoryorder": "total ascending"} if sort else {}

    fig.update_layout(
        template=template,
        font=get_base_font(),
        title=get_title_config(build_subtitle(title, subtitle)),
        xaxis=get_axis_config(extra=x_tick_config),
        yaxis=get_axis_config(extra=yaxis_extra),
        bargap=0.2,
        margin=get_margin_config(left=MARGIN_LEFT_HBAR),
        showlegend=False,
    )

    # Configure hover
    configure_hover(fig, mode="closest")

    # Add annotations
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
    value_format: str = "number",
    source: str = "The Tie",
    watermark: str = "tie",
    **kwargs,
) -> go.Figure:
    """Create a stacked bar chart for composition.

    Args:
        df: DataFrame with data
        x_column: Column name for x-axis categories
        stack_columns: List of column names to stack
        title: Chart title
        subtitle: Optional subtitle below title
        horizontal: If True, create horizontal stacked bars
        value_format: Value format ('number', 'currency', 'percent')
        source: Source attribution (default: "The Tie")
        watermark: Watermark type ('tie', 'labs', 'qf', 'none')

    Returns:
        Configured Plotly figure
    """
    fig = go.Figure()

    # Calculate stack totals for tick calculation (not individual values)
    stack_totals = df[stack_columns].sum(axis=1).tolist()

    # Add trace for each stack segment
    for i, col in enumerate(stack_columns):
        color = colorway[i % len(colorway)]
        hover_labels = [format_hover_value(v, value_format) for v in df[col]]

        if horizontal:
            fig.add_trace(
                go.Bar(
                    y=df[x_column],
                    x=df[col],
                    name=col,
                    orientation="h",
                    marker=dict(color=color),
                    customdata=hover_labels,
                    hovertemplate=f"<b>{col}</b>: %{{customdata}}<extra></extra>",
                )
            )
        else:
            fig.add_trace(
                go.Bar(
                    x=df[x_column],
                    y=df[col],
                    name=col,
                    marker=dict(color=color),
                    customdata=hover_labels,
                    hovertemplate=f"<b>{col}</b>: %{{customdata}}<extra></extra>",
                )
            )

    # Apply template - always use token_labs styling, watermark controls only the logo
    template = get_template_tie(watermark if watermark != "none" else "tie")
    if watermark == "none":
        template.layout.images = []  # Remove watermark but keep all styling

    # Get formatted tick configuration for value axis (use stack totals, include_zero)
    value_tick_config = get_tick_config(stack_totals, value_format, include_zero=True)

    if horizontal:
        xaxis_config = get_axis_config(extra=value_tick_config)
        yaxis_config = get_axis_config()
    else:
        xaxis_config = get_axis_config()
        yaxis_config = get_axis_config(extra=value_tick_config)

    fig.update_layout(
        template=template,
        font=get_base_font(),
        title=get_title_config(build_subtitle(title, subtitle)),
        barmode="stack",
        xaxis=xaxis_config,
        yaxis=yaxis_config,
        legend=get_legend_config(),
        margin=get_margin_config(left=MARGIN_LEFT_HBAR if horizontal else None),
    )

    # Configure hover
    configure_hover(fig, mode="closest")

    # Add annotations
    add_source_annotation(fig, source)
    add_updated_annotation(fig)

    return fig


def grouped_bar_chart(
    df: pd.DataFrame,
    x_column: str,
    group_columns: list[str],
    title: str,
    subtitle: Optional[str] = None,
    value_format: str = "number",
    source: str = "The Tie",
    watermark: str = "tie",
    **kwargs,
) -> go.Figure:
    """Create a grouped bar chart for comparing 2-3 measures.

    Args:
        df: DataFrame with data
        x_column: Column name for x-axis categories
        group_columns: List of column names to group (2-3 recommended)
        title: Chart title
        subtitle: Optional subtitle below title
        value_format: Value format ('number', 'currency', 'percent')
        source: Source attribution (default: "The Tie")
        watermark: Watermark type ('tie', 'labs', 'qf', 'none')

    Returns:
        Configured Plotly figure
    """
    fig = go.Figure()

    # Collect all values for tick calculation
    all_values = []
    for col in group_columns:
        all_values.extend(df[col].tolist())

    # Only add bar labels if total bars ≤12 (per guide/07_text.md)
    n_total_bars = len(df) * len(group_columns)
    show_labels = n_total_bars <= 12
    max_val = max(all_values) if all_values else 1
    threshold = max_val * 0.15  # Bars < 15% of max get outside labels

    # Add trace for each group
    for i, col in enumerate(group_columns):
        color = colorway[i % len(colorway)]
        hover_labels = [format_hover_value(v, value_format) for v in df[col]]

        trace_kwargs = dict(
            x=df[x_column],
            y=df[col],
            name=col,
            marker=dict(color=color),
            customdata=hover_labels,
            hovertemplate=f"<b>{col}</b>: %{{customdata}}<extra></extra>",
        )

        if show_labels:
            bar_labels = get_bar_labels(df[col], value_format)
            text_positions = [
                "inside" if v > threshold else "outside"
                for v in df[col]
            ]
            trace_kwargs.update(
                text=bar_labels,
                textposition=text_positions,
                textfont=dict(color="white"),
                insidetextanchor="end",
                cliponaxis=False,
            )

        fig.add_trace(go.Bar(**trace_kwargs))

    # Apply template - always use token_labs styling, watermark controls only the logo
    template = get_template_tie(watermark if watermark != "none" else "tie")
    if watermark == "none":
        template.layout.images = []  # Remove watermark but keep all styling

    # Get formatted tick configuration for y-axis (include_zero for bar charts)
    y_tick_config = get_tick_config(all_values, value_format, include_zero=True)

    fig.update_layout(
        template=template,
        font=get_base_font(),
        title=get_title_config(build_subtitle(title, subtitle)),
        barmode="group",
        xaxis=get_axis_config(),
        yaxis=get_axis_config(extra=y_tick_config),
        bargap=0.15,
        bargroupgap=0.1,
        legend=get_legend_config(),
        margin=get_margin_config(),
    )

    # Configure hover
    configure_hover(fig, mode="closest")

    # Add annotations
    add_source_annotation(fig, source)
    add_updated_annotation(fig)

    return fig
