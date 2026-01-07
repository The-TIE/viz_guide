# Hover & Tooltips Guide

This guide covers hover behavior configuration for Plotly visualizations. Proper hover design enhances data exploration without cluttering the visual display.

---

## Hover Mode Selection

The `hovermode` property controls how hover information is triggered and displayed. Select based on chart type and user intent.

For stacked bar charts and multi-series visualizations, use `hovermode='y unified'` (for horizontal bars) or `hovermode='x unified'` (for vertical bars) to show all series values simultaneously. This provides better context than hovering individual segments.

### Mode Options

#### 1. `x unified`
**Use when**: Comparing multiple series at the same x-value.

Best for:
- Time series with multiple traces
- Any chart where cross-series comparison at a single x-point matters
- Dashboard charts where users need to compare values across categories

```python
fig.update_layout(
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor='#1e293b',
        font=dict(color='#d3d4d6', size=12)
    )
)
```

#### 2. `closest`
**Use when**: Inspecting individual data points in detail.

Best for:
- Scatter plots
- Bubble charts
- Any sparse data visualization
- Charts where points don't align on a common axis

```python
fig.update_layout(
    hovermode='closest',
    hoverlabel=dict(
        bgcolor='#1e293b',
        font=dict(color='#d3d4d6', size=12)
    )
)
```

#### 3. `x`
**Use when**: Default time series behavior where each trace shows its own tooltip.

Best for:
- Simple time series (1-2 traces)
- When unified hover creates too much information
- Line charts with few overlapping points

```python
fig.update_layout(hovermode='x')
```

#### 4. `y`
**Use when**: Horizontal chart orientation.

Best for:
- Horizontal bar charts
- Horizontal box plots
- Any chart with categorical y-axis and numeric x-axis

```python
fig.update_layout(hovermode='y')
```

### Decision Tree

```
START
  │
  ├─► Is chart horizontal (bars, horizontal box)?
  │     YES → hovermode='y'
  │
  ├─► Is this a scatter/bubble plot with non-aligned points?
  │     YES → hovermode='closest'
  │
  ├─► Are you comparing multiple series at same x-values?
  │     YES → hovermode='x unified'
  │
  └─► Default time series or simple line chart
        → hovermode='x'
```

---

## Hover Template Design

Custom hover templates provide control over tooltip content and formatting.

### Information Hierarchy

Order information from most to least important:
1. **Primary value** - the metric the user is investigating
2. **Context** - date, category, or identifier
3. **Secondary values** - supporting metrics
4. **Metadata** - additional context if space permits

### Formatting Rules

Apply consistent formatting within hover templates:

| Data Type | Format | Example |
|-----------|--------|---------|
| Large numbers (≥10,000) | SI notation | `%{y:.3s}` → "1.23M" |
| Currency | Symbol + SI | `$%{y:.3s}` → "$1.23M" |
| Percentages | One decimal | `%{y:.1f}%` → "12.3%" |
| Dates | Context-appropriate | `%{x|%b %d, %Y}` → "Jan 15, 2024" |
| Small decimals | 2-3 decimals | `%{y:.2f}` → "0.12" |

### Common Template Patterns

#### Time Series: Date + Value

```python
fig.update_traces(
    hovertemplate=(
        '<b>%{y:$.3s}</b><br>'
        '%{x|%b %d, %Y}'
        '<extra></extra>'
    )
)
```

Output: **$1.23M** / Jan 15, 2024

#### Multi-Series: Date + All Values (with `x unified`)

For unified hover, set individual trace names and templates. The unified tooltip will display:
- **Date as header** at the top
- **Each series** with its name (bolded) and value

```python
fig.add_trace(go.Scatter(
    x=dates,
    y=revenue,
    name='Revenue',
    hovertemplate='<b>Revenue</b>: %{y:$.3s}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=dates,
    y=costs,
    name='Costs',
    hovertemplate='<b>Costs</b>: %{y:$.3s}<extra></extra>'
))

fig.update_layout(hovermode='x unified')
```

**Unified Hover Formatting Best Practices:**

1. **Bold the series name** in the template: `<b>SeriesName</b>: %{y}`
2. **Don't repeat contextual info** (like category/exchange) that's already in the trace name
3. **Use consistent value formatting** across all series in the tooltip
4. **Keep individual templates simple** - the date header is added automatically
5. **Don't include the date in individual trace templates** - it's shown once at the top

**Example output:**
```
Jan 15, 2024           ← Date header (automatic)
───────────────
Revenue: $1.23M        ← Bolded label + formatted value
Costs: $0.98M
```

#### Unified Hover Anti-Patterns (AVOID THESE)

**BAD: Repeating datetime on every line:**
```python
# WRONG - datetime appears on EVERY line, cluttering the tooltip
hovertemplate='%{x|%b %d}<br>Revenue: %{y:$.2s}<extra></extra>'
```

Output (bad):
```
Jan 15, 2024
───────────────
Jan 15 - Revenue: $1.2M    ← Date repeated!
Jan 15 - Costs: $0.9M      ← Date repeated again!
```

**BAD: Repeating the category/entity name on every line:**
```python
# WRONG when trace.name='Binance' - repeats exchange name
hovertemplate='<b>Binance</b>: %{x:.2f}B<extra></extra>'
```

Output (bad):
```
Binance                    ← From trace name
───────────────
Binance: $12.3B            ← Redundant!
```

**GOOD: Simple value-only template with bold label:**
```python
# CORRECT - let unified hover handle the structure
fig.add_trace(go.Scatter(
    name='Binance',  # Shows as header
    hovertemplate='<b>Spot</b>: %{x:.2f}B<extra></extra>'  # Category + value
))
```

Output (good):
```
Binance                    ← From trace name (once)
───────────────
Spot: $8.5B
Futures: $3.2B
Options: $0.6B
```

**Key rule:** In `x unified` mode, each trace's `name` appears automatically. Your `hovertemplate` should only contain the **value** (and optionally a label if showing multiple metrics per trace).

#### Scatter Plot: Both Axes + Category

```python
fig.update_traces(
    hovertemplate=(
        '<b>%{customdata[0]}</b><br>'
        'X: %{x:.2f}<br>'
        'Y: %{y:.2f}<br>'
        'Size: %{marker.size:,.0f}'
        '<extra></extra>'
    ),
    customdata=df[['category']].values
)
```

#### Bar Chart: Category + Value

```python
fig.update_traces(
    hovertemplate=(
        '<b>%{x}</b><br>'
        'Value: %{y:,.0f}'
        '<extra></extra>'
    )
)
```

#### Table Context: Multi-Field Display

```python
fig.update_traces(
    hovertemplate=(
        '<b>%{customdata[0]}</b><br>'
        'Region: %{customdata[1]}<br>'
        'Q1: %{customdata[2]:$,.0f}<br>'
        'Q2: %{customdata[3]:$,.0f}'
        '<extra></extra>'
    ),
    customdata=df[['product', 'region', 'q1', 'q2']].values
)
```

### The `<extra></extra>` Tag

Always include `<extra></extra>` to suppress the trace name box that appears by default:

```python
# Without <extra></extra> - shows trace name in secondary box
hovertemplate='Value: %{y}'

# With <extra></extra> - clean single tooltip
hovertemplate='Value: %{y}<extra></extra>'
```

---

## Spike Lines

Spike lines draw vertical or horizontal reference lines from the cursor to the axes, aiding alignment in dense charts.

### When to Enable

- Time series comparisons with multiple traces
- Charts where precise x-axis alignment matters
- Dense scatter plots where axis position is important

### Configuration

```python
fig.update_layout(
    hovermode='x unified',
    spikedistance=-1,  # Enable spikes for all points
    xaxis=dict(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='#475569',
        spikedash='dash'
    ),
    yaxis=dict(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='#475569',
        spikedash='dash'
    )
)
```

### Spike Line Properties

| Property | Recommended Value | Purpose |
|----------|-------------------|---------|
| `spikethickness` | 1 | Subtle, non-intrusive |
| `spikecolor` | `#475569` | Muted gray, visible on dark background |
| `spikedash` | `'dash'` | Distinguishes from data lines |
| `spikemode` | `'across'` | Extends full chart width/height |
| `spikesnap` | `'cursor'` | Snaps to cursor position |

---

## Hover Label Styling

Match hover labels to the visualization theme for visual consistency.

### Dark Theme Configuration

```python
fig.update_layout(
    hoverlabel=dict(
        bgcolor='#1e293b',
        font=dict(
            family='Inter, system-ui, sans-serif',
            size=12,
            color='#d3d4d6'
        ),
        bordercolor='#334155',
        namelength=-1  # Show full trace name
    )
)
```

### Style Properties

| Property | Value | Notes |
|----------|-------|-------|
| `bgcolor` | `#1e293b` | Slightly lighter than chart background (#0e1729) |
| `font.color` | `#d3d4d6` | Standard text color |
| `font.size` | 12 | Readable without dominating |
| `bordercolor` | `#334155` | Subtle border, matches theme |
| `namelength` | -1 | -1 shows full name; set positive int to truncate |

### Per-Trace Styling

Override global hover style for specific traces:

```python
fig.add_trace(go.Scatter(
    x=x_data,
    y=y_data,
    hoverlabel=dict(
        bgcolor='#7c3aed',  # Purple for emphasis
        font=dict(color='#ffffff')
    )
))
```

---

## Complete Example

Full hover configuration for a multi-series time series chart:

```python
import plotly.graph_objects as go

fig = go.Figure()

# Add traces with custom hover templates
fig.add_trace(go.Scatter(
    x=dates,
    y=revenue,
    name='Revenue',
    line=dict(color='#22d3ee'),
    hovertemplate='%{y:$.3s}<extra></extra>'
))

fig.add_trace(go.Scatter(
    x=dates,
    y=profit,
    name='Profit',
    line=dict(color='#a78bfa'),
    hovertemplate='%{y:$.3s}<extra></extra>'
))

# Configure layout with hover settings
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='#0e1729',
    plot_bgcolor='#0e1729',

    # Hover mode
    hovermode='x unified',

    # Hover label styling
    hoverlabel=dict(
        bgcolor='#1e293b',
        font=dict(
            family='Inter, system-ui, sans-serif',
            size=12,
            color='#d3d4d6'
        ),
        bordercolor='#334155'
    ),

    # Spike lines
    spikedistance=-1,
    xaxis=dict(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='#475569',
        spikedash='dash'
    )
)
```

---

## Decision Checklist

Before finalizing hover configuration, verify:

### Mode Selection
- [ ] Horizontal chart → `hovermode='y'`
- [ ] Scatter/bubble with sparse points → `hovermode='closest'`
- [ ] Multi-series time comparison → `hovermode='x unified'`
- [ ] Simple time series → `hovermode='x'`

### Template Design
- [ ] Most important value appears first
- [ ] Numbers use appropriate SI notation (≥10,000)
- [ ] Percentages show one decimal place
- [ ] Currency includes symbol
- [ ] Dates use context-appropriate format
- [ ] `<extra></extra>` included to hide trace name box

### Spike Lines
- [ ] Enabled for time series with multiple traces
- [ ] Style is subtle (dash, muted color, thin)
- [ ] `spikedistance=-1` set for consistent activation

### Label Styling
- [ ] Background color matches theme (`#1e293b`)
- [ ] Font color readable (`#d3d4d6`)
- [ ] Font size appropriate (12px)
- [ ] Border color subtle (`#334155`)

### Testing
- [ ] Hover activates at expected distance
- [ ] All traces show correct information
- [ ] No overlapping or truncated text
- [ ] Unified hover aligns correctly across traces

## Avoiding Redundancy

When using `hovermode='x unified'`, the x-axis value appears at the top of the hover box. Remove redundant x-axis information from individual trace `hovertemplate` strings to avoid duplication. For example, use `'<b>Today</b><br>Volume: %{y:.2f}B<extra></extra>'` instead of including the hour value.

## Hover Template Best Practices

Avoid redundant information in hover templates. When using `hovermode='x unified'`, the x-axis value (e.g., date) is automatically displayed at the top of the hover box, so don't include it again in the hovertemplate. Instead, include the series/category name using `<b>{category_name}</b>` and only the y-axis value: `hovertemplate='<b>Category Name</b><br>Value: %{y:.1f}<extra></extra>'`.

## Unified Hover Mode

When using subplots with `hovermode='x unified'`, include the category or series identifier in each trace's hovertemplate since the unified hover shows all traces at once. Use `hovertemplate=f'<b>{category}</b><br>Metric: %{{y}}<extra></extra>'` to help users distinguish between different series in the combined hover display.

When using `hovermode='y unified'` (or `'x unified'`), format values in `customdata` rather than relying on automatic formatting. Use a formatting function to ensure consistent significant figures or decimal places across all traces. Example: `customdata=[format_sig_figs(v, sig_figs=3) for v in values]` and `hovertemplate='<b>%{fullData.name}</b>: %{customdata}<extra></extra>'`

## Hover Date Formatting

Match hover date granularity to your data frequency. For daily data, ensure hover shows full dates (day-level precision), not just month-year. Use `hovertemplate` with appropriate date formatting or rely on default x-axis formatting when using `hovermode='x unified'`.

## Series Names in Hover

When using `hovermode='x unified'`, series names are automatically shown from the trace `name` parameter. If using custom `hovertemplate`, you must explicitly include series identification. Verify that trace names are properly set with `name=blockchain` or similar to ensure hover tooltips identify which series each value belongs to.

## Hover Template Format

Structure hover information with series name first, followed by formatted values: `hovertemplate='<b>%{fullData.name}</b>: %{customdata}<extra></extra>'`. This provides clear context before showing the value. Use `customdata` for pre-formatted values with specific precision requirements.

## Hover Label Formatting

For financial time series, format hover labels with bold series names and appropriate precision: `hovertemplate='<b>{series_name}:</b> ${value:.3g}B<extra></extra>'`. Use 3 significant figures (%.3g) for readability. Include units (B for billions) in the hover text to maintain context.

## Hover Formatting

Ensure hover text uses the same formatting as visible labels. If bar labels show '$3.62B' with 3 significant figures, hover text should match: use `customdata` with pre-formatted strings rather than relying on Plotly's default number formatting. Example: `customdata=list(zip(text_labels, other_values))` and reference in hovertemplate as `%{customdata[0]}`.

## Hover Templates for Stacked/Grouped Charts

For stacked or grouped bar charts, use `hovermode='y unified'` (horizontal) or `hovermode='x unified'` (vertical) to show all series values in a single hover tooltip. Format each series as '<b>{series name}:</b> {formatted value}' using `customdata` with a formatting function like `format_sig_figs(value, sig_figs=3)` to display values with k/M/B abbreviations. Use `hovertemplate='<b>' + series_name + ':</b> %{customdata}<extra></extra>'` to suppress the secondary box.

## Number Formatting in Hover

For financial data requiring specific significant figures, create a dedicated formatting function (e.g., `format_sig_figs()`) that calculates appropriate decimal places based on the magnitude of scaled values. For 3 sig figs: values ≥100 need 0 decimals, ≥10 need 1 decimal, <10 need 2 decimals after scaling to B/M/k units.
