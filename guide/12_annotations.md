# Annotations Guide

This guide covers annotation requirements and best practices for AI agents generating Plotly visualizations. Proper annotations transform charts from simple data displays into informative, professional visualizations.

---

## Required Annotations

Every chart MUST include these mandatory annotations. These are non-negotiable for production visualizations.

Always include a source and last updated annotation at the bottom of every chart. Place it at y=-0.15 (below the chart) with left alignment: `fig.add_annotation(text='Source: [Data Source] | Updated: [Date]', xref='paper', yref='paper', x=0, y=-0.15, showarrow=False, font=dict(size=10, color='#9ca3af'), xanchor='left', yanchor='top')`. Adjust bottom margin to accommodate this annotation (e.g., `margin=dict(b=100)`).

Always include source and update timestamp annotations at the bottom of visualizations. Position at `y=-0.15` (or lower if needed) with `xref='paper', yref='paper', x=0, xanchor='left'`. Format as: 'Source: [Data Source] | Updated: [Date]'. Use muted color like `#9ca3af` and smaller font size (10px).

### 1. Data Source Attribution

Position: Bottom left of the chart, small text.

```python
fig.add_annotation(
    text="Source: Federal Reserve Economic Data (FRED)",
    xref="paper",
    yref="paper",
    x=0,
    y=-0.12,
    showarrow=False,
    font=dict(size=10, color="#9ca3af"),
    xanchor="left",
    yanchor="top"
)
```

### 2. Last Updated Timestamp

Indicates when the underlying data was last refreshed.

```python
from datetime import datetime

data_updated = "2024-01-15"  # From data source metadata

fig.add_annotation(
    text=f"Source: Company Database | Updated: {data_updated}",
    xref="paper",
    yref="paper",
    x=0,
    y=-0.12,
    showarrow=False,
    font=dict(size=10, color="#9ca3af"),
    xanchor="left",
    yanchor="top"
)
```

### 3. Generation Timestamp (Static Exports)

For PNG/PDF exports, include when the chart was generated.

```python
generated_at = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

fig.add_annotation(
    text=f"Generated: {generated_at}",
    xref="paper",
    yref="paper",
    x=1,
    y=-0.12,
    showarrow=False,
    font=dict(size=10, color="#6b7280"),
    xanchor="right",
    yanchor="top"
)
```

### Avoiding Annotation Overlaps

**Common problem:** Source attribution at `y=-0.12` overlaps with x-axis tick labels or other bottom annotations.

**Solution:** Adjust bottom margin and annotation y-position based on what's below the chart:

| X-axis Configuration | Attribution y | Margin b |
|---------------------|---------------|----------|
| Short tick labels | -0.10 | 60 |
| Long/rotated tick labels | -0.15 | 80 |
| X-axis title present | -0.18 | 100 |
| Multiple bottom annotations | -0.20 | 120 |

```python
# When x-axis has long labels or a title, push attribution lower
fig.update_layout(margin=dict(b=100))  # Increase bottom margin

fig.add_annotation(
    text="Source: ...",
    y=-0.18,  # Lower position to clear x-axis labels
    ...
)
```

**Check for overlaps:** If x-axis tick labels are rotated or there's an x-axis title, use `y=-0.15` or lower and increase `margin.b` accordingly.

### Combined Attribution Pattern

Standard format combining source and timestamp:

```python
def add_attribution(fig, source_name, data_updated, include_generated=False):
    """Add required attribution annotations to any chart."""

    attribution_text = f"Source: {source_name} | Updated: {data_updated}"

    fig.add_annotation(
        text=attribution_text,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.12,  # Adjust to -0.15 or -0.18 if x-axis has title or long labels
        showarrow=False,
        font=dict(size=10, color="#9ca3af"),
        xanchor="left",
        yanchor="top"
    )

    if include_generated:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        fig.add_annotation(
            text=f"Generated: {generated_at}",
            xref="paper",
            yref="paper",
            x=1,
            y=-0.12,
            showarrow=False,
            font=dict(size=10, color="#6b7280"),
            xanchor="right",
            yanchor="top"
        )

    return fig
```

---

## Reference Lines

Reference lines provide context by showing thresholds, averages, or important markers.

### Horizontal Reference Lines

Use for thresholds, averages, and historical comparisons.

#### Threshold Lines (Breakeven, Targets)

```python
# Breakeven line
fig.add_hline(
    y=0,
    line=dict(
        color="rgba(255,255,255,0.3)",
        width=1,
        dash="dash"
    ),
    annotation=dict(
        text="Breakeven",
        font=dict(size=11, color="#d3d4d6"),
        xanchor="right",
        yanchor="bottom"
    ),
    annotation_position="right"
)

# Target line
fig.add_hline(
    y=target_value,
    line=dict(
        color="rgba(34,197,94,0.5)",
        width=1,
        dash="dash"
    ),
    annotation=dict(
        text=f"Target: {target_value:,.0f}",
        font=dict(size=11, color="#22c55e"),
        xanchor="right"
    ),
    annotation_position="right"
)
```

#### Average Lines (Mean, Median)

```python
mean_value = df["value"].mean()
median_value = df["value"].median()

# Mean line
fig.add_hline(
    y=mean_value,
    line=dict(
        color="rgba(96,165,250,0.5)",
        width=1,
        dash="dot"
    ),
    annotation=dict(
        text=f"Mean: {mean_value:,.1f}",
        font=dict(size=10, color="#60a5fa"),
        xanchor="left"
    ),
    annotation_position="left"
)

# Median line
fig.add_hline(
    y=median_value,
    line=dict(
        color="rgba(251,191,36,0.5)",
        width=1,
        dash="dot"
    ),
    annotation=dict(
        text=f"Median: {median_value:,.1f}",
        font=dict(size=10, color="#fbbf24"),
        xanchor="left",
        yshift=15  # Offset to avoid overlap with mean
    ),
    annotation_position="left"
)
```

#### Historical Comparison Lines

```python
# Previous period average
fig.add_hline(
    y=previous_year_avg,
    line=dict(
        color="rgba(255,255,255,0.2)",
        width=1,
        dash="dash"
    ),
    annotation=dict(
        text=f"2023 Avg: {previous_year_avg:,.0f}",
        font=dict(size=10, color="#9ca3af")
    ),
    annotation_position="right"
)
```

### Vertical Reference Lines

Use for events, announcements, and date markers.

#### Event Markers

```python
# Product launch event
fig.add_vline(
    x="2024-03-15",
    line=dict(
        color="rgba(168,85,247,0.5)",
        width=1,
        dash="dash"
    ),
    annotation=dict(
        text="Product Launch",
        font=dict(size=10, color="#a855f7"),
        textangle=-90,
        yanchor="bottom"
    ),
    annotation_position="top"
)
```

#### Quarter/Year Markers

```python
# Add quarter boundaries
quarters = ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"]
quarter_labels = ["Q1", "Q2", "Q3", "Q4"]

for date, label in zip(quarters, quarter_labels):
    fig.add_vline(
        x=date,
        line=dict(
            color="rgba(255,255,255,0.1)",
            width=1,
            dash="dot"
        ),
        annotation=dict(
            text=label,
            font=dict(size=9, color="#6b7280"),
            yanchor="top",
            yshift=-5
        ),
        annotation_position="top"
    )
```

#### Multiple Events Pattern

```python
events = [
    {"date": "2024-02-01", "label": "Earnings Call", "color": "#60a5fa"},
    {"date": "2024-05-15", "label": "Product Launch", "color": "#a855f7"},
    {"date": "2024-08-20", "label": "Acquisition", "color": "#22c55e"},
]

for i, event in enumerate(events):
    fig.add_vline(
        x=event["date"],
        line=dict(
            color=f"rgba({int(event['color'][1:3], 16)},{int(event['color'][3:5], 16)},{int(event['color'][5:7], 16)},0.5)",
            width=1,
            dash="dash"
        )
    )

    fig.add_annotation(
        x=event["date"],
        y=1,
        yref="paper",
        text=event["label"],
        showarrow=False,
        font=dict(size=9, color=event["color"]),
        textangle=-45,
        xanchor="left",
        yanchor="bottom",
        yshift=5 + (i * 12)  # Stagger vertically to avoid overlap
    )
```

---

## Event Markers

### Point Annotations

Use arrows to highlight specific data points such as outliers or key moments.

#### Basic Point Annotation

```python
fig.add_annotation(
    x="2024-06-15",
    y=peak_value,
    text="All-time High",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=1,
    arrowcolor="#d3d4d6",
    font=dict(size=11, color="#d3d4d6"),
    ax=40,  # Arrow x offset
    ay=-30,  # Arrow y offset (negative = above)
    bordercolor="#374151",
    borderwidth=1,
    borderpad=4,
    bgcolor="rgba(31,41,55,0.9)"
)
```

#### Outlier Highlighting

```python
def annotate_outliers(fig, df, x_col, y_col, threshold_std=2):
    """Annotate data points that are statistical outliers."""

    mean = df[y_col].mean()
    std = df[y_col].std()

    outliers = df[abs(df[y_col] - mean) > threshold_std * std]

    for idx, row in outliers.iterrows():
        direction = "above" if row[y_col] > mean else "below"
        ay = -40 if direction == "above" else 40

        fig.add_annotation(
            x=row[x_col],
            y=row[y_col],
            text=f"Outlier: {row[y_col]:,.0f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#ef4444",
            font=dict(size=10, color="#ef4444"),
            ax=0,
            ay=ay,
            bgcolor="rgba(31,41,55,0.9)",
            bordercolor="#ef4444",
            borderwidth=1,
            borderpad=3
        )

    return fig
```

#### Key Moment Annotation

```python
# Highlighting a significant event at a data point
fig.add_annotation(
    x=significant_date,
    y=value_at_date,
    text="<b>Record Sales</b><br>$2.5M in single day",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=1.5,
    arrowcolor="#22c55e",
    font=dict(size=11, color="#d3d4d6"),
    align="left",
    ax=60,
    ay=-50,
    bordercolor="#22c55e",
    borderwidth=1,
    borderpad=6,
    bgcolor="rgba(31,41,55,0.95)"
)
```

### Range Annotations (Shaded Regions)

Use shaded regions to highlight time periods such as recessions, promotional periods, or events.

#### Basic Shaded Region

```python
# Highlight a recession period
fig.add_vrect(
    x0="2024-03-01",
    x1="2024-06-30",
    fillcolor="rgba(239,68,68,0.1)",
    line=dict(width=0),
    layer="below"
)

# Add label for the region
fig.add_annotation(
    x="2024-05-01",  # Center of region
    y=1,
    yref="paper",
    text="Recession",
    showarrow=False,
    font=dict(size=10, color="#ef4444"),
    yanchor="top",
    yshift=-10
)
```

#### Event Period with Styling

```python
def add_shaded_period(fig, start, end, label, color="#60a5fa", opacity=0.1):
    """Add a shaded region with label for a time period."""

    # Add shaded rectangle
    fig.add_vrect(
        x0=start,
        x1=end,
        fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},{opacity})",
        line=dict(
            color=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.3)",
            width=1,
            dash="dot"
        ),
        layer="below"
    )

    # Add centered label
    fig.add_annotation(
        x=pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2,
        y=1.02,
        yref="paper",
        text=label,
        showarrow=False,
        font=dict(size=10, color=color),
        yanchor="bottom"
    )

    return fig

# Usage
add_shaded_period(fig, "2024-01-01", "2024-03-31", "Beta Period", "#a855f7")
add_shaded_period(fig, "2024-07-01", "2024-07-31", "Summer Sale", "#22c55e")
```

#### Horizontal Shaded Bands

```python
# Target range band
fig.add_hrect(
    y0=target_low,
    y1=target_high,
    fillcolor="rgba(34,197,94,0.1)",
    line=dict(width=0),
    layer="below"
)

fig.add_annotation(
    x=0,
    xref="paper",
    y=(target_low + target_high) / 2,
    text="Target Range",
    showarrow=False,
    font=dict(size=10, color="#22c55e"),
    xanchor="left",
    xshift=10
)
```

### Text Placement Strategies

#### Avoiding Data Overlap

```python
def smart_annotation_position(fig, x, y, data_df, x_col, y_col):
    """Determine best annotation position to avoid data overlap."""

    # Get nearby data points
    nearby = data_df[
        (data_df[x_col] >= x - pd.Timedelta(days=7)) &
        (data_df[x_col] <= x + pd.Timedelta(days=7))
    ]

    # Determine if annotation should go above or below
    avg_nearby = nearby[y_col].mean()

    if y >= avg_nearby:
        # Data point is above average, place annotation above
        ay = -40
        yanchor = "bottom"
    else:
        # Data point is below average, place annotation below
        ay = 40
        yanchor = "top"

    return ay, yanchor
```

#### Consistent Positioning Pattern

```python
# For multiple annotations on a line chart, alternate above/below
annotations = [
    {"x": "2024-01-15", "y": 100, "text": "Event A"},
    {"x": "2024-03-20", "y": 120, "text": "Event B"},
    {"x": "2024-06-10", "y": 90, "text": "Event C"},
    {"x": "2024-09-05", "y": 150, "text": "Event D"},
]

for i, ann in enumerate(annotations):
    # Alternate above and below
    ay = -35 if i % 2 == 0 else 35

    fig.add_annotation(
        x=ann["x"],
        y=ann["y"],
        text=ann["text"],
        showarrow=True,
        arrowhead=2,
        arrowcolor="#d3d4d6",
        font=dict(size=10, color="#d3d4d6"),
        ax=0,
        ay=ay,
        bgcolor="rgba(31,41,55,0.9)",
        bordercolor="#374151",
        borderwidth=1,
        borderpad=4
    )
```

#### Using Arrows for Offset Text

```python
# When annotation must be far from data point
fig.add_annotation(
    x="2024-06-15",
    y=crowded_peak_value,
    text="Peak value occurred during<br>promotional campaign",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=1,
    arrowcolor="rgba(255,255,255,0.5)",
    font=dict(size=10, color="#d3d4d6"),
    ax=80,  # Large horizontal offset
    ay=-60,  # Large vertical offset
    bgcolor="rgba(31,41,55,0.95)",
    bordercolor="#374151",
    borderwidth=1,
    borderpad=6,
    align="left"
)
```

---

## Callouts and Notes

### When to Use Callouts

Use callouts for:
- Explaining anomalies in the data
- Methodology notes
- Data quality warnings
- Important caveats

### Positioning Outside Plot Area

```python
# Methodology note below the chart
fig.add_annotation(
    text="<i>Note: Values adjusted for inflation using CPI-U. Seasonal adjustment applied.</i>",
    xref="paper",
    yref="paper",
    x=0,
    y=-0.18,
    showarrow=False,
    font=dict(size=9, color="#6b7280"),
    xanchor="left",
    yanchor="top",
    align="left"
)
```

### Styling Patterns

```python
# Anomaly explanation callout
fig.add_annotation(
    text="<i>*Spike due to one-time accounting adjustment</i>",
    xref="paper",
    yref="paper",
    x=0.5,
    y=-0.15,
    showarrow=False,
    font=dict(size=9, color="#9ca3af", style="italic"),
    xanchor="center",
    bgcolor="rgba(31,41,55,0.8)",
    borderpad=4
)

# Data quality warning
fig.add_annotation(
    text="⚠ Preliminary data - subject to revision",
    xref="paper",
    yref="paper",
    x=1,
    y=1.08,
    showarrow=False,
    font=dict(size=10, color="#fbbf24"),
    xanchor="right",
    yanchor="bottom"
)
```

### Multi-line Notes

```python
methodology_note = """<i>Methodology Notes:
• Revenue figures are gross, pre-tax
• YoY comparison uses same-store basis
• Excludes discontinued operations</i>"""

fig.add_annotation(
    text=methodology_note,
    xref="paper",
    yref="paper",
    x=0,
    y=-0.22,
    showarrow=False,
    font=dict(size=9, color="#6b7280"),
    xanchor="left",
    yanchor="top",
    align="left"
)

# Adjust bottom margin to accommodate
fig.update_layout(margin=dict(b=120))
```

---

## Plotly Implementation Reference

### add_annotation() Parameters

```python
fig.add_annotation(
    # Position
    x=value,                    # X coordinate
    y=value,                    # Y coordinate
    xref="x" | "paper",         # Reference: axis or paper (0-1)
    yref="y" | "paper",         # Reference: axis or paper (0-1)
    xanchor="left"|"center"|"right",
    yanchor="top"|"middle"|"bottom",
    xshift=0,                   # Pixel offset
    yshift=0,                   # Pixel offset

    # Text
    text="Annotation text",
    font=dict(size=11, color="#d3d4d6", family="sans-serif"),
    align="left"|"center"|"right",
    textangle=0,                # Rotation in degrees

    # Arrow
    showarrow=True|False,
    arrowhead=0-8,              # Arrow style
    arrowsize=1,                # Arrow scale
    arrowwidth=1,               # Line width
    arrowcolor="#d3d4d6",
    ax=40,                      # Arrow end x offset
    ay=-40,                     # Arrow end y offset
    axref="pixel"|"x",
    ayref="pixel"|"y",

    # Box
    bgcolor="rgba(31,41,55,0.9)",
    bordercolor="#374151",
    borderwidth=1,
    borderpad=4,

    # Behavior
    clicktoshow=False,
    visible=True
)
```

### add_hline() and add_vline()

```python
# Horizontal line
fig.add_hline(
    y=value,
    line=dict(
        color="rgba(255,255,255,0.3)",
        width=1,
        dash="solid"|"dash"|"dot"|"dashdot"
    ),
    annotation=dict(
        text="Label",
        font=dict(size=10, color="#d3d4d6"),
        xanchor="left"|"right",
        yanchor="top"|"bottom",
        xshift=0,
        yshift=0
    ),
    annotation_position="left"|"right"|"top"|"bottom",
    row="all"|int,              # For subplots
    col="all"|int
)

# Vertical line
fig.add_vline(
    x=value,
    line=dict(
        color="rgba(255,255,255,0.3)",
        width=1,
        dash="dash"
    ),
    annotation=dict(
        text="Event",
        font=dict(size=10, color="#d3d4d6"),
        textangle=-90
    ),
    annotation_position="top"
)
```

### add_vrect() and add_hrect()

```python
# Vertical shaded region
fig.add_vrect(
    x0=start_value,
    x1=end_value,
    fillcolor="rgba(96,165,250,0.1)",
    line=dict(
        color="rgba(96,165,250,0.3)",
        width=1,
        dash="dot"
    ),
    layer="below"|"above",
    row="all"|int,
    col="all"|int
)

# Horizontal shaded region
fig.add_hrect(
    y0=lower_value,
    y1=upper_value,
    fillcolor="rgba(34,197,94,0.1)",
    line=dict(width=0),
    layer="below"
)
```

### Coordinate Reference Systems

```python
# xref and yref options:

# "x" / "y" - Use axis coordinates (default)
fig.add_annotation(x="2024-06-15", y=100, xref="x", yref="y")

# "paper" - Use paper coordinates (0-1 range)
fig.add_annotation(x=0.5, y=1.05, xref="paper", yref="paper")

# "x domain" / "y domain" - Use domain coordinates for subplots
fig.add_annotation(x=0.5, y=1, xref="x domain", yref="y domain")

# For secondary axes
fig.add_annotation(x=value, y=value, xref="x", yref="y2")
```

---

## Template Colors Reference

Use these colors consistently across all annotations:

```python
ANNOTATION_COLORS = {
    # Text colors
    "primary_text": "#d3d4d6",      # Main annotation text
    "secondary_text": "#9ca3af",    # Attribution, notes
    "muted_text": "#6b7280",        # Timestamps, fine print

    # Reference lines
    "reference_line": "rgba(255,255,255,0.3)",
    "threshold_line": "rgba(255,255,255,0.5)",

    # Shaded regions
    "highlight_region": "rgba(96,165,250,0.1)",   # Blue
    "warning_region": "rgba(239,68,68,0.1)",      # Red
    "success_region": "rgba(34,197,94,0.1)",      # Green
    "neutral_region": "rgba(255,255,255,0.05)",   # Gray

    # Backgrounds
    "annotation_bg": "rgba(31,41,55,0.9)",
    "annotation_border": "#374151",

    # Status colors (for text)
    "positive": "#22c55e",
    "negative": "#ef4444",
    "warning": "#fbbf24",
    "info": "#60a5fa",
    "accent": "#a855f7"
}
```

---

## Annotation Checklist

Use this checklist before finalizing any visualization:

### Required Elements
- [ ] Data source attribution present (bottom left)
- [ ] Last updated timestamp included
- [ ] Generation timestamp added (for static exports)
- [ ] Attribution format follows standard: "Source: [name] | Updated: [date]"

### Reference Lines
- [ ] Relevant thresholds marked (breakeven, targets)
- [ ] Averages shown if meaningful (mean, median)
- [ ] Reference lines use dashed styling
- [ ] Labels positioned at line ends
- [ ] Colors use template reference line opacity

### Event Markers
- [ ] Key events annotated with vertical lines
- [ ] Point annotations have arrows when offset
- [ ] Shaded regions use semi-transparent fill
- [ ] Event labels don't overlap each other

### Text Placement
- [ ] No annotations overlap data points
- [ ] Consistent positioning (above/below pattern)
- [ ] Arrows used when text is offset from point
- [ ] Text is legible against background

### Styling
- [ ] Font sizes appropriate (9-11px for annotations)
- [ ] Colors match template palette
- [ ] Backgrounds applied for readability
- [ ] Borders subtle but visible

### Notes and Callouts
- [ ] Anomalies explained with callouts
- [ ] Methodology notes included if needed
- [ ] Notes positioned outside plot area
- [ ] Smaller font and italic styling for notes

### Final Verification
- [ ] All annotations visible at default zoom
- [ ] Interactive tooltips don't conflict with annotations
- [ ] Margins adjusted to fit all annotations
- [ ] Export preview checked for static formats

## Mandatory Annotations

**CRITICAL: Source Attribution is Mandatory** - Every visualization MUST include a source annotation at the bottom of the chart. Use `fig.add_annotation(text='Source: [data source] | Updated: [date]', xref='paper', yref='paper', x=0, y=-0.12, showarrow=False, font=dict(size=10, color='#9ca3af'), xanchor='left', yanchor='top')`. This annotation should be added as the final step before returning the figure. Missing this annotation is a critical error that must be corrected.

## Required Metadata Annotations

Always include source and timestamp annotations at the bottom of the chart using `fig.add_annotation()` with `xref='paper', yref='paper', x=0, y=-0.15, xanchor='left'`. Format as 'Source: [data source] | Updated: [date]' with `font=dict(size=10, color='#9ca3af')`. Ensure adequate bottom margin (`margin=dict(b=100)`) to accommodate these annotations.

## Data Point Annotations

For cumulative comparisons, add an annotation at the last valid data point showing the percentage difference. Use `arrowhead=2, arrowsize=1, arrowwidth=2` with color matching the metric's performance (green for positive, red for negative). Position with `ax=60, ay=-40` for optimal readability and set `yref='y2'` when annotating secondary axis data.

## Source and Metadata Annotations

Always include source and update timestamp annotations. Position below the chart with `xref='paper', yref='paper', x=0, y=-0.15, xanchor='left'`. Use format: `'Source: [source name] | Updated: [date]'` with smaller font size (10px) and muted color (#9ca3af). Ensure bottom margin accommodates this (e.g., `margin=dict(b=100)`).

Add source and update date annotations at the bottom of charts: `fig.add_annotation(text='Source: <source> | Updated: <date>', xref='paper', yref='paper', x=0, y=-0.15, showarrow=False, font=dict(size=10, color='#9ca3af'), xanchor='left')`. Increase bottom margin to accommodate: `margin=dict(b=100)`. Use a muted color like '#9ca3af' for metadata text.

## Source Annotations

When adding source/timestamp annotations at the bottom, ensure adequate bottom margin to prevent overlap with axis labels. Use `margin=dict(b=100)` or higher and position annotations at `y=-0.15` to `-0.20` depending on axis label presence. Set `xanchor='left'` and `x=0` to align with the plot area.
