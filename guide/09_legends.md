# 09 - Legends

> Guidelines for when to include legends, where to place them, and how to style them.
> Legends should aid comprehension, not add clutter.

---

## When to Include Legends

### Required

| Condition | Legend Needed | Reason |
|-----------|---------------|--------|
| Multiple series without inline labels | Yes | Distinguish series |
| Color encoding represents categories | Yes | Decode colors |
| Different line styles (solid, dashed) | Yes | Explain styles |
| Dual-axis charts | Yes | Clarify which axis |

### Optional (Consider Omitting)

| Condition | Recommendation | Alternative |
|-----------|----------------|-------------|
| Single series | Omit legend | Use chart title |
| 2 series, clearly labeled in title | Omit legend | Title: "Price vs Volume" |
| Small multiples with subplot titles | Omit legend | Subplot titles serve as legend |
| Color is redundant with labels | Omit legend | Direct labeling |

### Never Include

| Condition | Why |
|-----------|-----|
| Legend repeats chart title | Redundant |
| Single trace with obvious meaning | Clutter |
| Legend items are cut off/unreadable | Unusable |

---

## Legend Placement

### Decision Tree

```
LEGEND PLACEMENT
├── Few items (≤5)
│   └── Horizontal, above chart
│
├── Many items (6-12)
│   └── Vertical, right side
│
├── Space constrained
│   ├── Inside chart (if whitespace available)
│   └── Below chart (last resort)
│
└── Small multiples
    └── Single shared legend, above all subplots
```

### Horizontal Legend (Default for ≤5 Items)

Best for most cases. Places legend above the chart, left-aligned.

```python
fig.update_layout(
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='left',
        x=0,
        font=dict(size=11, color='#d3d4d6'),
        bgcolor='rgba(0,0,0,0)',  # Transparent
        borderwidth=0
    )
)
```

**Important: Avoid Legend/Subtitle Overlap**

When using a subtitle (via `<br><sup>...</sup>` in the title), increase the top margin to prevent the legend from overlapping the subtitle:

```python
# With subtitle - increase top margin
fig.update_layout(
    title=dict(
        text='Main Title<br><sup>Subtitle text here</sup>',
        ...
    ),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.08,  # Higher y value to clear subtitle
        ...
    ),
    margin=dict(t=120)  # Increase from default 100 to accommodate legend + subtitle
)
```

| Title Type | Legend y | Margin t |
|------------|----------|----------|
| Title only | 1.02 | 80-100 |
| Title + subtitle | 1.08 | 120 |
| Title + subtitle + long legend | 1.12 | 140 |

### Vertical Legend (Right Side)

For many items that won't fit horizontally.

```python
fig.update_layout(
    legend=dict(
        orientation='v',
        yanchor='top',
        y=1,
        xanchor='left',
        x=1.02,
        font=dict(size=11, color='#d3d4d6'),
        bgcolor='rgba(0,0,0,0)',
        borderwidth=0
    ),
    margin=dict(r=150)  # Make room for legend
)
```

### Inside Chart (When Space Permits)

Only when there's clear whitespace that won't overlap data.

```python
fig.update_layout(
    legend=dict(
        orientation='v',
        yanchor='top',
        y=0.99,
        xanchor='right',
        x=0.99,
        font=dict(size=10, color='#d3d4d6'),
        bgcolor='rgba(14, 23, 41, 0.8)',  # Semi-transparent background
        bordercolor='rgba(255,255,255,0.1)',
        borderwidth=1
    )
)
```

### Below Chart (Last Resort)

Avoid if possible—takes space from chart. Use only when horizontal won't fit above.

```python
fig.update_layout(
    legend=dict(
        orientation='h',
        yanchor='top',
        y=-0.15,
        xanchor='center',
        x=0.5,
        font=dict(size=10, color='#d3d4d6')
    ),
    margin=dict(b=100)  # Make room below
)
```

---

## Legend Formatting

### Item Order

**Match visual order:** Legend items should appear in the same order as they appear in the chart (top-to-bottom for stacked, left-to-right for grouped).

```python
# Plotly orders by trace addition order
# Add traces in the order you want legend items to appear

# Or reverse programmatically:
fig.update_layout(legend=dict(traceorder='reversed'))

# Or group by category:
fig.update_layout(legend=dict(traceorder='grouped'))
```

### Grouping Related Items

For complex charts with multiple trace types:

```python
# Assign legend groups
fig.add_trace(go.Scatter(
    ...,
    legendgroup='prices',
    legendgrouptitle=dict(text='Prices', font=dict(size=12, color='#d3d4d6'))
))

fig.add_trace(go.Bar(
    ...,
    legendgroup='volume',
    legendgrouptitle=dict(text='Volume', font=dict(size=12, color='#d3d4d6'))
))
```

### Truncating Long Names

If legend items are too long:

```python
# Truncate in the trace name
def truncate_name(name, max_length=20):
    if len(name) > max_length:
        return name[:max_length-3] + '...'
    return name

fig.add_trace(go.Scatter(
    name=truncate_name(series_name),
    ...
))
```

---

## Interactive Legend Behavior

### Click to Toggle (Default)

By default, clicking a legend item toggles trace visibility. This is usually desirable.

### Click to Isolate

Double-click isolates a single trace. Useful for many-series charts.

### Disable Interaction

When legend should be for reference only:

```python
fig.update_layout(
    legend=dict(
        itemclick=False,
        itemdoubleclick=False
    )
)
```

---

## Alternatives to Legends

### Direct Labeling

For line charts, label the line directly at its endpoint:

```python
# Add text annotation at end of each line
for trace_name, trace_data in data_dict.items():
    last_x = trace_data['x'].iloc[-1]
    last_y = trace_data['y'].iloc[-1]

    fig.add_annotation(
        x=last_x,
        y=last_y,
        text=trace_name,
        xanchor='left',
        xshift=5,
        showarrow=False,
        font=dict(size=10, color='#d3d4d6')
    )

fig.update_layout(showlegend=False)
```

### Color-Coded Titles

For 2-3 series, encode in the title:

```python
fig.update_layout(
    title=dict(
        text='<span style="color:#60A5FA">Bitcoin</span> vs <span style="color:#F87171">Ethereum</span> Price',
        font=dict(size=16)
    ),
    showlegend=False
)
```

### Subplot Titles as Legend

For small multiples:

```python
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=['BTC', 'ETH', 'SOL', 'AVAX']  # These serve as legend
)

# Use consistent color for all subplots
for i, (name, data) in enumerate(datasets.items()):
    row = (i // 2) + 1
    col = (i % 2) + 1
    fig.add_trace(
        go.Scatter(x=data['date'], y=data['price'],
                   line=dict(color='#60A5FA'),
                   showlegend=False),
        row=row, col=col
    )
```

---

## Special Cases

### Dual-Axis Legend

Clearly indicate which axis each series uses:

```python
fig.add_trace(go.Bar(
    name='Volume (left axis)',
    ...
))

fig.add_trace(go.Scatter(
    name='Price (right axis)',
    yaxis='y2',
    ...
))
```

### Heatmap Color Scale

Heatmaps use a colorbar instead of a legend:

```python
fig.add_trace(go.Heatmap(
    z=values,
    colorbar=dict(
        title=dict(text='Value', side='right'),
        tickformat=',.2s',
        len=0.8,
        thickness=15,
        bgcolor='rgba(0,0,0,0)',
        tickfont=dict(color='#d3d4d6')
    )
))
```

### Categorical Color Scale

When color represents categories in scatter/heatmap:

```python
fig.add_trace(go.Scatter(
    marker=dict(
        color=df['category_numeric'],
        colorscale=[[0, '#60A5FA'], [0.5, '#34D399'], [1, '#F87171']],
        colorbar=dict(
            title='Category',
            tickmode='array',
            tickvals=[0, 0.5, 1],
            ticktext=['Low', 'Medium', 'High']
        )
    )
))
```

---

## Quick Reference

### Placement Decision Table

| # Items | Chart Type | Recommended Placement |
|---------|------------|----------------------|
| 1 | Any | No legend (use title) |
| 2-3 | Line/Bar | Horizontal above OR color-coded title |
| 4-5 | Line/Bar | Horizontal above |
| 6-10 | Line/Bar | Vertical right |
| 11+ | Any | Small multiples (no legend) |
| Any | Heatmap | Colorbar (not legend) |
| Any | Small multiples | Shared legend above OR subplot titles |

### Legend Styling Defaults

```python
LEGEND_DEFAULTS = dict(
    font=dict(size=11, color='#d3d4d6'),
    bgcolor='rgba(0,0,0,0)',  # Transparent
    borderwidth=0,
    itemsizing='constant',  # Consistent marker sizes
    tracegroupgap=10
)
```

---

## Legend Checklist

Before finalizing legend configuration:

- [ ] Legend is necessary (multiple series that need identification)
- [ ] Placement doesn't obscure data
- [ ] Item order matches visual order
- [ ] All items are readable (not truncated/cut off)
- [ ] Margin adjusted if legend is outside chart
- [ ] For single series: consider omitting legend entirely
- [ ] For 2-3 series: consider direct labeling or color-coded title
- [ ] Interactive behavior appropriate (toggle on/off is usually fine)

## Legend Positioning

For charts with many legend items or wide titles, position the legend to the right of the plot area to avoid overlap with the title. Use `legend=dict(orientation='v', xanchor='left', x=1.02, yanchor='top', y=1)` and increase right margin with `margin=dict(r=180)` to accommodate the legend outside the plot area.

For charts with titles, avoid placing legends in the top area where they may overlap. Position horizontal legends below the chart using `legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5)` and increase bottom margin accordingly with `margin=dict(b=100)` to accommodate the legend.
