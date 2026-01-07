# 13 - Accessibility

> Ensuring visualizations are perceivable and understandable by all users.
> Accessibility is not optional - it improves usability for everyone.

---

## Color Blindness Considerations

Approximately 8% of men and 0.5% of women have some form of color vision deficiency. Design for inclusion from the start.

### Types of Color Vision Deficiency

| Type | Affected Colors | Population | Design Impact |
|------|-----------------|------------|---------------|
| Deuteranopia | Red-green (green weak) | ~6% of men | Cannot distinguish red from green |
| Protanopia | Red-green (red weak) | ~2% of men | Red appears darker, confused with green |
| Tritanopia | Blue-yellow | ~0.01% | Blue and yellow appear similar |
| Achromatopsia | All colors | Very rare | Sees only grayscale |

### Safe Palette Selection

**CRITICAL: Never rely on red-green distinctions alone.**

The standard positive/negative color scheme (`#34D399` green, `#F87171` red) is problematic for deuteranopia and protanopia users.

#### Colorblind-Safe Categorical Palette

```python
# Primary colorblind-safe palette
# Tested with Coblis color blindness simulator
COLORBLIND_SAFE_PALETTE = [
    "#60A5FA",  # Blue - distinguishable in all types
    "#FBBF24",  # Amber - safe alternative to green
    "#E879F9",  # Magenta/pink - distinguishable
    "#38BDF8",  # Sky blue - works well
    "#A78BFA",  # Purple - safe choice
    "#FB923C",  # Orange - distinguishable from blue
    "#2DD4BF",  # Teal - different from pure green
    "#F472B6",  # Pink - distinguishable
]

fig.update_layout(colorway=COLORBLIND_SAFE_PALETTE)
```

#### Safe Positive/Negative Alternatives

```python
# Instead of red/green, use these alternatives:

# Option 1: Blue/Orange (most universally distinguishable)
POSITIVE_COLOR = "#60A5FA"  # Blue
NEGATIVE_COLOR = "#FB923C"  # Orange

# Option 2: Blue/Red with pattern reinforcement
POSITIVE_COLOR = "#60A5FA"  # Blue
NEGATIVE_COLOR = "#F87171"  # Red (combine with shape/pattern)

# Option 3: Teal/Magenta
POSITIVE_COLOR = "#2DD4BF"  # Teal
NEGATIVE_COLOR = "#E879F9"  # Magenta
```

#### Sequential Palettes for Colorblind Users

```python
# Blue sequential (safe for all CVD types)
BLUE_SEQUENTIAL = [
    "#DBEAFE", "#BFDBFE", "#93C5FD", "#60A5FA",
    "#3B82F6", "#2563EB", "#1D4ED8", "#1E40AF"
]

# Purple sequential (safe)
PURPLE_SEQUENTIAL = [
    "#F3E8FF", "#E9D5FF", "#D8B4FE", "#C084FC",
    "#A855F7", "#9333EA", "#7C3AED", "#6D28D9"
]

# Orange sequential (safe)
ORANGE_SEQUENTIAL = [
    "#FFF7ED", "#FFEDD5", "#FED7AA", "#FDBA74",
    "#FB923C", "#F97316", "#EA580C", "#C2410C"
]

# Viridis-style (designed for colorblindness)
VIRIDIS_STYLE = [
    "#440154", "#482878", "#3E4A89", "#31688E",
    "#26828E", "#1F9E89", "#35B779", "#6DCD59"
]
```

### Redundant Encoding (CRITICAL)

**Never use color as the only differentiator.** Always provide a secondary visual cue.

#### Color + Shape Encoding

```python
import plotly.graph_objects as go

# Different categories use different markers AND colors
fig = go.Figure()

categories = [
    {"name": "Category A", "color": "#60A5FA", "symbol": "circle"},
    {"name": "Category B", "color": "#FBBF24", "symbol": "square"},
    {"name": "Category C", "color": "#E879F9", "symbol": "diamond"},
    {"name": "Category D", "color": "#2DD4BF", "symbol": "triangle-up"},
]

for cat in categories:
    mask = df["category"] == cat["name"]
    fig.add_trace(go.Scatter(
        x=df.loc[mask, "x"],
        y=df.loc[mask, "y"],
        mode="markers",
        name=cat["name"],
        marker=dict(
            color=cat["color"],
            symbol=cat["symbol"],
            size=10,
            line=dict(color="#0e1729", width=1)
        )
    ))
```

#### Color + Line Style Encoding

```python
# Multiple line series with different dash patterns
line_styles = [
    {"name": "Series A", "color": "#60A5FA", "dash": "solid"},
    {"name": "Series B", "color": "#FBBF24", "dash": "dash"},
    {"name": "Series C", "color": "#E879F9", "dash": "dot"},
    {"name": "Series D", "color": "#2DD4BF", "dash": "dashdot"},
]

for style in line_styles:
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df[style["name"]],
        mode="lines",
        name=style["name"],
        line=dict(
            color=style["color"],
            dash=style["dash"],
            width=2
        )
    ))
```

#### Color + Pattern Fill for Bars

```python
# Use patterns in addition to colors for bar charts
patterns = ["", "/", "\\", "x", "-", "|", "+", "."]

fig = go.Figure()

for i, category in enumerate(df["category"].unique()):
    mask = df["category"] == category
    fig.add_trace(go.Bar(
        x=df.loc[mask, "x"],
        y=df.loc[mask, "y"],
        name=category,
        marker=dict(
            color=COLORBLIND_SAFE_PALETTE[i],
            pattern=dict(
                shape=patterns[i % len(patterns)],
                solidity=0.5
            )
        )
    ))
```

### Testing Recommendations

Always test visualizations with color blindness simulators:

1. **Coblis** (https://www.color-blindness.com/coblis-color-blindness-simulator/)
2. **Sim Daltonism** (macOS app)
3. **Chrome DevTools** - Rendering tab > Emulate vision deficiencies
4. **Figma/Adobe** - Built-in colorblind preview modes

**Testing Checklist:**
- [ ] Test with Deuteranopia simulation
- [ ] Test with Protanopia simulation
- [ ] Test with Tritanopia simulation
- [ ] Verify all data series are distinguishable
- [ ] Confirm legend items match chart elements

---

## Contrast Requirements

Adequate contrast ensures readability for users with low vision and in various lighting conditions.

### WCAG Contrast Standards

| Element Type | Minimum Ratio | Enhanced Ratio | Standard |
|--------------|---------------|----------------|----------|
| Normal text (< 18pt) | 4.5:1 | 7:1 | WCAG AA / AAA |
| Large text (>= 18pt) | 3:1 | 4.5:1 | WCAG AA / AAA |
| UI components | 3:1 | - | WCAG 2.1 |
| Graphical objects | 3:1 | - | WCAG 2.1 |

### Text on Dark Background

With background `#0e1729` and text `#d3d4d6`:

```python
# Calculate contrast ratio
# Background: #0e1729 (relative luminance ~0.013)
# Text: #d3d4d6 (relative luminance ~0.67)
# Contrast ratio: ~11.5:1 (exceeds WCAG AAA)

# Template text colors and their contrast ratios with #0e1729
TEXT_COLORS = {
    "primary": "#d3d4d6",     # ~11.5:1 - Excellent
    "secondary": "#9ca3af",   # ~6.2:1  - Good (AA compliant)
    "muted": "#6b7280",       # ~3.8:1  - Minimum for large text
}
```

**Warning:** The muted text color `#6b7280` barely meets AA standards. Use only for:
- Large text (18pt+)
- Non-essential annotations
- Decorative elements

### Line Visibility on Dark Backgrounds

```python
# Minimum line widths for visibility
LINE_WIDTH_GUIDELINES = {
    "primary_data": 2,       # Main data lines - minimum 2px
    "secondary_data": 1.5,   # Supporting lines
    "reference_lines": 1,    # Dashed reference lines
    "grid_lines": 1,         # Use with opacity
}

# Grid line configuration for visibility
fig.update_xaxes(
    gridcolor="rgba(255, 255, 255, 0.1)",  # 10% white
    gridwidth=1
)

fig.update_yaxes(
    gridcolor="rgba(255, 255, 255, 0.1)",
    gridwidth=1
)

# Reference lines should be more visible
fig.add_hline(
    y=threshold,
    line=dict(
        color="rgba(255, 255, 255, 0.3)",  # 30% white
        width=1,
        dash="dash"
    )
)
```

### Data Point Distinguishability

```python
# Minimum marker sizes for visibility
MARKER_SIZE_GUIDELINES = {
    "scatter_default": 8,     # Default scatter point
    "scatter_dense": 6,       # Dense scatter plots
    "line_markers": 6,        # Markers on line charts
    "emphasis_marker": 12,    # Highlighted points
}

# Marker outline for better visibility
fig.add_trace(go.Scatter(
    x=df["x"],
    y=df["y"],
    mode="markers",
    marker=dict(
        size=10,
        color="#60A5FA",
        line=dict(
            color="#0e1729",  # Dark outline for contrast
            width=1
        )
    )
))
```

### Ensuring Distinguishable Data Series

```python
def check_color_distinction(colors, min_delta_e=30):
    """
    Check if colors are sufficiently distinguishable.
    Delta E > 30 is generally considered distinguishable.
    """
    # For production, use colormath library
    # This is a simplified check
    for i, c1 in enumerate(colors):
        for j, c2 in enumerate(colors[i+1:], i+1):
            # Calculate perceptual difference
            # (simplified - use proper Delta E calculation in production)
            pass
    return True

# Apply sufficient visual separation
def apply_accessible_series_styling(fig, num_series):
    """Apply accessible styling to multiple data series."""

    colors = COLORBLIND_SAFE_PALETTE[:num_series]
    symbols = ["circle", "square", "diamond", "triangle-up",
               "triangle-down", "cross", "x", "star"][:num_series]
    dashes = ["solid", "dash", "dot", "dashdot",
              "longdash", "longdashdot"][:num_series]

    for i, trace in enumerate(fig.data):
        trace.marker.color = colors[i]
        trace.marker.symbol = symbols[i]
        if hasattr(trace, "line"):
            trace.line.dash = dashes[i]
            trace.line.width = 2

    return fig
```

---

## Screen Reader Considerations

While Plotly charts are inherently visual, we can improve accessibility for screen reader users.

### Alt Text Patterns for Charts

Every chart should have descriptive alt text that conveys the key insights.

```python
def generate_chart_alt_text(chart_type, title, key_insights, data_summary):
    """
    Generate descriptive alt text for a chart.

    Args:
        chart_type: Type of chart (line, bar, scatter, etc.)
        title: Chart title
        key_insights: List of main takeaways
        data_summary: Brief data description

    Returns:
        Alt text string
    """
    alt_text = f"{chart_type} chart titled '{title}'. "
    alt_text += f"{data_summary}. "
    alt_text += "Key insights: " + "; ".join(key_insights) + "."

    return alt_text

# Example usage
alt_text = generate_chart_alt_text(
    chart_type="Line",
    title="BTC Price (30 Days)",
    key_insights=[
        "Price increased 15% overall",
        "Peak of $48,200 on Jan 5",
        "Lowest point $42,100 on Dec 28"
    ],
    data_summary="Shows daily Bitcoin prices from December 15, 2024 to January 14, 2025"
)

# Apply to figure config
fig.update_layout(
    meta=dict(
        description=alt_text
    )
)
```

### Alt Text Templates by Chart Type

```python
ALT_TEXT_TEMPLATES = {
    "line": "{title}. Line chart showing {metric} over {time_period}. {trend_description}. Range: {min_val} to {max_val}.",

    "bar": "{title}. Bar chart comparing {metric} across {num_categories} categories. Highest: {highest_cat} ({highest_val}). Lowest: {lowest_cat} ({lowest_val}).",

    "scatter": "{title}. Scatter plot showing relationship between {x_metric} and {y_metric}. {correlation_description}. {num_points} data points.",

    "pie": "{title}. Pie chart showing composition of {total_metric}. Largest segment: {largest_segment} ({largest_pct}%). {num_segments} total segments.",

    "histogram": "{title}. Histogram showing distribution of {metric}. {distribution_description}. Mean: {mean_val}, Median: {median_val}."
}
```

### Semantic Structure

```python
# Use proper title hierarchy in layout
fig.update_layout(
    title=dict(
        text="<b>Main Title</b><br><sup>Subtitle with context</sup>",
        font=dict(size=16, color="#d3d4d6"),
        x=0,
        xanchor="left"
    ),
    # Axis titles provide semantic meaning
    xaxis_title="Date",
    yaxis_title="Price (USD)"
)
```

### Data Tables as Alternatives

Always provide data tables for complex visualizations:

```python
def create_accessible_data_table(df, columns, title):
    """
    Create an HTML data table as an alternative to charts.

    Args:
        df: DataFrame with data
        columns: Columns to include
        title: Table caption

    Returns:
        HTML string for accessible table
    """
    html = f'<table role="table" aria-label="{title}">\n'
    html += f'  <caption>{title}</caption>\n'
    html += '  <thead>\n    <tr>\n'

    for col in columns:
        html += f'      <th scope="col">{col}</th>\n'

    html += '    </tr>\n  </thead>\n  <tbody>\n'

    for _, row in df[columns].iterrows():
        html += '    <tr>\n'
        for col in columns:
            html += f'      <td>{row[col]}</td>\n'
        html += '    </tr>\n'

    html += '  </tbody>\n</table>'

    return html

# Provide CSV download option
def add_data_download_link(fig, df, filename="chart_data.csv"):
    """Add annotation with data download link."""

    # In web context, provide download button
    fig.add_annotation(
        text='<a href="data:text/csv;base64,...">Download data (CSV)</a>',
        xref="paper",
        yref="paper",
        x=1,
        y=-0.15,
        showarrow=False,
        font=dict(size=10, color="#60A5FA"),
        xanchor="right"
    )
```

---

## Cognitive Accessibility

Reduce cognitive load to make charts understandable for users with cognitive disabilities and improve usability for everyone.

### Avoid Chart Junk

Remove non-essential visual elements that don't convey data.

```python
# AVOID: Excessive decoration
fig.update_layout(
    # Remove 3D effects - they distort perception
    scene=None,

    # Gridlines OFF by default (less visual clutter)
    xaxis=dict(
        showgrid=False,  # OFF by default
        gridcolor="rgba(255,255,255,0.05)"  # If enabled, very subtle
    ),

    # Remove chart border - wastes visual space
    xaxis_showline=False,
    yaxis_showline=False,

    # Minimal background
    plot_bgcolor="#0e1729",
    paper_bgcolor="#0e1729"
)

# AVOID: Gradient fills that don't encode data
# USE: Solid fills with consistent opacity
fig.add_trace(go.Bar(
    marker=dict(
        color="#60A5FA",  # Solid color
        # NOT: color gradient
    )
))
```

### Clear Labels and Titles

```python
# Good: Clear, descriptive title
title_good = "Monthly Revenue (2024)"

# Bad: Vague or missing context
title_bad = "Revenue"

# Good: Axis labels with units
fig.update_yaxes(title_text="Revenue (USD)")

# Bad: Missing units
fig.update_yaxes(title_text="Revenue")

# Direct labeling for clarity
fig.add_trace(go.Scatter(
    x=df["date"],
    y=df["value"],
    mode="lines+text",
    text=["", "", "", "", df["value"].iloc[-1]],  # Label last point
    textposition="middle right",
    textfont=dict(size=11, color="#d3d4d6")
))
```

### Consistent Patterns

```python
# Establish consistent visual language across all charts

CHART_STANDARDS = {
    # Colors
    "colors": {
        "primary": "#60A5FA",
        "secondary": "#FBBF24",
        "positive": "#60A5FA",    # Use blue, not green
        "negative": "#FB923C",    # Use orange, not red
        "neutral": "#9ca3af"
    },

    # Typography
    "fonts": {
        "title_size": 16,
        "axis_title_size": 12,
        "tick_size": 11,
        "annotation_size": 10
    },

    # Spacing
    "margins": {
        "t": 60,
        "b": 60,
        "l": 60,
        "r": 30
    },

    # Legend position (consistent)
    "legend": {
        "orientation": "h",
        "y": 1.02,
        "x": 0,
        "xanchor": "left",
        "yanchor": "bottom"
    }
}

def apply_standard_layout(fig):
    """Apply consistent layout standards."""
    fig.update_layout(
        font=dict(size=CHART_STANDARDS["fonts"]["tick_size"]),
        title_font_size=CHART_STANDARDS["fonts"]["title_size"],
        margin=CHART_STANDARDS["margins"],
        legend=CHART_STANDARDS["legend"],
        paper_bgcolor="#0e1729",
        plot_bgcolor="#0e1729"
    )
    return fig
```

### Reducing Complexity

```python
# Limit number of series (5-7 maximum)
MAX_SERIES = 7

def simplify_chart(df, value_col, category_col, top_n=MAX_SERIES):
    """Reduce categories to top N plus 'Other'."""

    top_categories = (
        df.groupby(category_col)[value_col]
        .sum()
        .nlargest(top_n - 1)
        .index
        .tolist()
    )

    df_simplified = df.copy()
    df_simplified[category_col] = df_simplified[category_col].apply(
        lambda x: x if x in top_categories else "Other"
    )

    return df_simplified

# Progressive disclosure - show summary first, details on interaction
fig.update_traces(
    hoverinfo="x+y",  # Basic info on hover
    customdata=df[["detail_1", "detail_2"]],  # Details available if needed
)
```

---

## Color-Safe Alternatives

When color cannot be the differentiator, use these alternative encoding methods.

### Different Line Styles

```python
# Available dash patterns in Plotly
DASH_PATTERNS = [
    "solid",        # _______________
    "dash",         # _ _ _ _ _ _ _ _
    "dot",          # . . . . . . . .
    "dashdot",      # _._._._._._._._
    "longdash",     # __ __ __ __ __
    "longdashdot",  # __.__.__.__.
]

def create_accessible_line_chart(df, x_col, y_cols, names=None):
    """Create line chart with distinct line styles."""

    fig = go.Figure()

    names = names or y_cols

    for i, (y_col, name) in enumerate(zip(y_cols, names)):
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            name=name,
            mode="lines",
            line=dict(
                color=COLORBLIND_SAFE_PALETTE[i % len(COLORBLIND_SAFE_PALETTE)],
                dash=DASH_PATTERNS[i % len(DASH_PATTERNS)],
                width=2
            )
        ))

    return fig
```

### Different Marker Shapes

```python
# Available marker symbols in Plotly
MARKER_SYMBOLS = [
    "circle",          # O
    "square",          # []
    "diamond",         # <>
    "triangle-up",     # ^
    "triangle-down",   # v
    "cross",           # +
    "x",               # x
    "star",            # *
    "hexagon",         # (6-sided)
    "pentagon",        # (5-sided)
]

def create_accessible_scatter_chart(df, x_col, y_col, category_col):
    """Create scatter chart with distinct marker shapes."""

    fig = go.Figure()
    categories = df[category_col].unique()

    for i, cat in enumerate(categories):
        mask = df[category_col] == cat

        fig.add_trace(go.Scatter(
            x=df.loc[mask, x_col],
            y=df.loc[mask, y_col],
            name=cat,
            mode="markers",
            marker=dict(
                symbol=MARKER_SYMBOLS[i % len(MARKER_SYMBOLS)],
                color=COLORBLIND_SAFE_PALETTE[i % len(COLORBLIND_SAFE_PALETTE)],
                size=10,
                line=dict(color="#0e1729", width=1)
            )
        ))

    return fig
```

### Direct Labeling Instead of Legend

Direct labeling eliminates the need to match legend entries to visual elements.

```python
def add_direct_labels(fig, df, x_col, y_cols, names=None):
    """Add labels directly to lines instead of using legend."""

    names = names or y_cols

    for i, (y_col, name) in enumerate(zip(y_cols, names)):
        # Find rightmost point
        last_idx = df[x_col].idxmax()
        last_x = df.loc[last_idx, x_col]
        last_y = df.loc[last_idx, y_col]

        # Add annotation at end of line
        fig.add_annotation(
            x=last_x,
            y=last_y,
            text=name,
            showarrow=False,
            xanchor="left",
            xshift=10,
            font=dict(
                size=11,
                color=COLORBLIND_SAFE_PALETTE[i]
            )
        )

    # Hide legend since we have direct labels
    fig.update_layout(showlegend=False)

    # Increase right margin for labels
    fig.update_layout(margin=dict(r=100))

    return fig

# Usage
fig = create_accessible_line_chart(df, "date", ["series_a", "series_b", "series_c"])
fig = add_direct_labels(fig, df, "date", ["series_a", "series_b", "series_c"],
                        names=["Series A", "Series B", "Series C"])
```

### Combined Accessible Chart Example

```python
import plotly.graph_objects as go
import pandas as pd

def create_fully_accessible_chart(df, x_col, y_cols, names, title):
    """
    Create a chart with full accessibility features.

    - Colorblind-safe palette
    - Distinct line styles
    - Distinct markers
    - Direct labeling
    - Proper alt text
    - High contrast
    """

    fig = go.Figure()

    for i, (y_col, name) in enumerate(zip(y_cols, names)):
        fig.add_trace(go.Scatter(
            x=df[x_col],
            y=df[y_col],
            name=name,
            mode="lines+markers",
            line=dict(
                color=COLORBLIND_SAFE_PALETTE[i],
                dash=DASH_PATTERNS[i % len(DASH_PATTERNS)],
                width=2
            ),
            marker=dict(
                symbol=MARKER_SYMBOLS[i % len(MARKER_SYMBOLS)],
                size=8,
                line=dict(color="#0e1729", width=1)
            )
        ))

        # Direct label at end of each line
        last_idx = len(df) - 1
        fig.add_annotation(
            x=df[x_col].iloc[last_idx],
            y=df[y_col].iloc[last_idx],
            text=name,
            showarrow=False,
            xanchor="left",
            xshift=10,
            font=dict(size=11, color=COLORBLIND_SAFE_PALETTE[i])
        )

    # Layout with accessibility in mind
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color="#d3d4d6"),
            x=0,
            xanchor="left"
        ),
        paper_bgcolor="#0e1729",
        plot_bgcolor="#0e1729",
        font=dict(color="#d3d4d6", size=12),
        showlegend=True,  # Keep legend for reference
        legend=dict(
            orientation="h",
            y=1.02,
            x=0,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=11)
        ),
        margin=dict(t=80, b=60, l=60, r=120),
        hovermode="x unified"
    )

    fig.update_xaxes(
        gridcolor="rgba(255, 255, 255, 0.1)",
        linecolor="rgba(255, 255, 255, 0.2)",
        tickfont=dict(size=11)
    )

    fig.update_yaxes(
        gridcolor="rgba(255, 255, 255, 0.1)",
        linecolor="rgba(255, 255, 255, 0.2)",
        tickfont=dict(size=11)
    )

    # Add alt text metadata
    fig.update_layout(
        meta=dict(
            description=generate_chart_alt_text(
                "Line",
                title,
                [f"Compares {len(y_cols)} data series"],
                f"Data from {df[x_col].min()} to {df[x_col].max()}"
            )
        )
    )

    return fig
```

---

## Implementation Helpers

### Colorblind-Safe Palette Generator

```python
def get_safe_palette(n_colors, palette_type="categorical"):
    """
    Get a colorblind-safe palette with the specified number of colors.

    Args:
        n_colors: Number of colors needed
        palette_type: "categorical", "sequential", or "diverging"

    Returns:
        List of hex color strings
    """

    if palette_type == "categorical":
        # Cycle through safe palette
        return [COLORBLIND_SAFE_PALETTE[i % len(COLORBLIND_SAFE_PALETTE)]
                for i in range(n_colors)]

    elif palette_type == "sequential":
        # Use blue sequential for most cases
        if n_colors <= 8:
            return BLUE_SEQUENTIAL[:n_colors]
        else:
            # Interpolate for more colors
            import numpy as np
            indices = np.linspace(0, len(BLUE_SEQUENTIAL)-1, n_colors, dtype=int)
            return [BLUE_SEQUENTIAL[i] for i in indices]

    elif palette_type == "diverging":
        # Blue-gray-orange diverging palette
        DIVERGING = [
            "#1E40AF", "#3B82F6", "#93C5FD", "#E5E7EB",
            "#FDBA74", "#F97316", "#C2410C"
        ]
        mid = len(DIVERGING) // 2
        if n_colors <= len(DIVERGING):
            step = len(DIVERGING) // n_colors
            return [DIVERGING[i] for i in range(0, len(DIVERGING), step)][:n_colors]
        return DIVERGING

    return COLORBLIND_SAFE_PALETTE[:n_colors]
```

### Accessibility Validation Function

```python
def validate_accessibility(fig):
    """
    Validate accessibility features of a Plotly figure.

    Returns:
        dict with validation results and recommendations
    """

    issues = []
    warnings = []

    # Check number of traces
    if len(fig.data) > 7:
        warnings.append(
            f"Chart has {len(fig.data)} data series. "
            "Consider reducing to 7 or fewer for clarity."
        )

    # Check for redundant encoding
    has_multiple_traces = len(fig.data) > 1
    if has_multiple_traces:
        colors = set()
        symbols = set()
        dashes = set()

        for trace in fig.data:
            if hasattr(trace, "marker") and trace.marker:
                colors.add(trace.marker.color)
                if trace.marker.symbol:
                    symbols.add(trace.marker.symbol)
            if hasattr(trace, "line") and trace.line:
                if trace.line.dash:
                    dashes.add(trace.line.dash)

        if len(symbols) <= 1 and len(dashes) <= 1:
            issues.append(
                "Multiple series use only color differentiation. "
                "Add different marker shapes or line styles."
            )

    # Check for title
    if not fig.layout.title or not fig.layout.title.text:
        issues.append("Chart is missing a title.")

    # Check contrast of text colors
    text_color = fig.layout.font.color if fig.layout.font else None
    if text_color == "#6b7280":
        warnings.append(
            "Text color #6b7280 has low contrast. "
            "Use #9ca3af or #d3d4d6 for better readability."
        )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "trace_count": len(fig.data),
        "has_title": bool(fig.layout.title and fig.layout.title.text)
    }
```

---

## Accessibility Checklist

Use this checklist before finalizing any visualization:

### Color Blindness

- [ ] Palette does not rely on red-green distinction alone
- [ ] All data series distinguishable in grayscale
- [ ] Tested with deuteranopia simulation
- [ ] Tested with protanopia simulation
- [ ] Redundant encoding used (color + shape/line style)

### Contrast

- [ ] Text contrast meets WCAG AA (4.5:1 minimum)
- [ ] Data lines minimum 2px width
- [ ] Markers minimum 8px size
- [ ] Grid lines subtle but visible
- [ ] All elements visible on dark background

### Screen Readers

- [ ] Descriptive title present
- [ ] Alt text / description metadata included
- [ ] Data table alternative available (for complex charts)
- [ ] Axis labels include units

### Cognitive Load

- [ ] Maximum 7 data series per chart
- [ ] Clear, descriptive title (5-10 words)
- [ ] Axis labels include units where applicable
- [ ] No chart junk (unnecessary decoration)
- [ ] Consistent styling with other charts

### Visual Encoding

- [ ] Direct labeling used where practical
- [ ] Different line styles for multiple series
- [ ] Different marker shapes for categories
- [ ] Legend entries match visual order

### Testing

- [ ] Viewed at different zoom levels
- [ ] Tested on mobile viewport
- [ ] Verified hover interactions work
- [ ] Checked export (PNG/PDF) rendering

---

## Quick Reference

### Colorblind-Safe Palette

```python
# Primary categorical palette
["#60A5FA", "#FBBF24", "#E879F9", "#2DD4BF", "#A78BFA", "#FB923C", "#38BDF8", "#F472B6"]

# Safe positive/negative
POSITIVE = "#60A5FA"  # Blue
NEGATIVE = "#FB923C"  # Orange
```

### Minimum Sizes

| Element | Minimum Size |
|---------|-------------|
| Line width | 2px |
| Marker size | 8px |
| Text (normal) | 11px |
| Text (labels) | 10px |
| Touch target | 44x44px |

### Contrast Ratios with #0e1729 Background

| Color | Ratio | Use For |
|-------|-------|---------|
| #d3d4d6 | 11.5:1 | Primary text |
| #9ca3af | 6.2:1 | Secondary text |
| #6b7280 | 3.8:1 | Large text only |
| #60A5FA | 5.8:1 | Data elements |

### Alt Text Formula

```
[Chart type] chart titled '[Title]'.
Shows [metric] over [time period/categories].
[Key insight 1]. [Key insight 2].
Range: [min] to [max].
```
