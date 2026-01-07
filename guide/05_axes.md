# 05 - Axis Configuration

> Rules for scale types, ranges, ticks, labels, and axis arrangements in Plotly visualizations.
> Proper axis configuration ensures accurate data representation and optimal readability.

---

## Scale Types

### When to Use Each Scale

| Scale Type | Plotly Value | Use When | Example Data |
|------------|--------------|----------|--------------|
| Linear | `'linear'` | Equal differences matter equally | Revenue, counts, temperatures |
| Logarithmic | `'log'` | Multiplicative relationships, wide value ranges | Market caps, exponential growth, orders of magnitude |
| Date/Time | `'date'` | Temporal data | Time series, event sequences |
| Categorical | `'category'` | Discrete, unordered groups | Product names, exchange names, labels |

### Linear Scale (Default)

**When to use:**
- Most quantitative data
- When the difference between 10 and 20 should look the same as 90 and 100
- Data without extreme outliers or multiplicative patterns

```python
fig.update_yaxes(type='linear')  # Default, often not needed explicitly
```

### Logarithmic Scale

**When to use:**
- Data spans multiple orders of magnitude (e.g., 100 to 1,000,000)
- Percentage changes matter more than absolute changes
- Comparing growth rates across different scales
- Financial data where returns are multiplicative

**When NOT to use:**
- Data contains zero or negative values (log undefined)
- Audience unfamiliar with log scales
- When absolute differences matter more than ratios

```python
# Log scale for market cap comparison
fig.update_yaxes(
    type='log',
    tickformat='$,.2s',
    title_text='Market Cap (log scale)'
)

# Always indicate log scale in title or annotation
fig.add_annotation(
    text='Log scale',
    xref='paper', yref='paper',
    x=0.02, y=0.98,
    showarrow=False,
    font=dict(size=10, color='#d3d4d6')
)
```

**Decision rule for log scale:**

```python
def should_use_log_scale(values):
    """
    Determine if log scale is appropriate for data.

    Args:
        values: Array of numeric values

    Returns:
        Tuple of (should_use_log, reason)
    """
    import numpy as np

    values = np.array([v for v in values if v is not None and v > 0])

    if len(values) == 0:
        return False, 'No positive values'

    if np.any(values <= 0):
        return False, 'Contains zero or negative values'

    ratio = np.max(values) / np.min(values)

    if ratio > 1000:  # 3+ orders of magnitude
        return True, f'Range spans {ratio:.0f}x (>1000x threshold)'
    elif ratio > 100:
        return True, f'Range spans {ratio:.0f}x (consider log scale)'
    else:
        return False, f'Range spans {ratio:.0f}x (linear appropriate)'
```

### Date/Time Scale

**Automatic handling:**
Plotly automatically detects datetime columns and applies date scale. For explicit control:

```python
fig.update_xaxes(
    type='date',
    tickformat='%b %Y',  # Format ticks
    dtick='M1',  # One tick per month
    ticklabelmode='period'  # Label at period center
)
```

**See Section 07 (Text Formatting) for date format patterns by timeframe.**

### Categorical Scale

**When to use:**
- Discrete categories without inherent order
- Bar charts with text labels
- When you want to control category order explicitly

```python
# Explicit category order
fig.update_xaxes(
    type='category',
    categoryorder='array',
    categoryarray=['Top 10', 'Top 50', 'Top 100', 'All']  # Custom order
)

# Order by value (descending)
fig.update_xaxes(
    type='category',
    categoryorder='total descending'
)

# Alphabetical order
fig.update_xaxes(
    type='category',
    categoryorder='category ascending'
)
```

**Category ordering options:**

| Value | Description |
|-------|-------------|
| `'trace'` | Order as they appear in trace data (default) |
| `'category ascending'` | Alphabetical A-Z |
| `'category descending'` | Alphabetical Z-A |
| `'total ascending'` | By total value (smallest first) |
| `'total descending'` | By total value (largest first) |
| `'array'` | Custom order via `categoryarray` |

---

## Range Determination

### Auto vs Fixed Ranges

| Scenario | Range Type | Configuration |
|----------|------------|---------------|
| Single chart, unknown data | Auto | Default (no config needed) |
| Comparing multiple charts | Fixed | Same range for all |
| Data with outliers | Auto with constraints | `range` with calculated bounds |
| Bar charts with positive values | Include zero | `rangemode='tozero'` |
| Percentage data | Fixed 0-100 or 0-1 | `range=[0, 100]` or `range=[0, 1]` |

### When to Include Zero

**Always include zero for:**
- Bar charts (height represents magnitude)
- Stacked charts (area represents total)
- When absolute magnitude matters

**May exclude zero for:**
- Line charts showing variation
- Scatter plots showing correlation
- When the data range is far from zero

```python
# Force axis to include zero
fig.update_yaxes(rangemode='tozero')

# Force axis to include negative and positive
fig.update_yaxes(rangemode='nonnegative')

# Explicit range
fig.update_yaxes(range=[0, max_value * 1.1])  # 10% padding above max
```

### Range Padding

Add breathing room around data:

```python
def calculate_axis_range(values, include_zero=False, padding_pct=0.05):
    """
    Calculate axis range with appropriate padding.

    Args:
        values: Array of numeric values
        include_zero: Force range to include zero
        padding_pct: Padding as fraction of range (default 5%)

    Returns:
        Tuple of (min_val, max_val)
    """
    import numpy as np

    min_val = np.nanmin(values)
    max_val = np.nanmax(values)
    data_range = max_val - min_val

    # Add padding
    padding = data_range * padding_pct
    range_min = min_val - padding
    range_max = max_val + padding

    # Include zero if requested
    if include_zero:
        range_min = min(range_min, 0)
        range_max = max(range_max, 0)

    return range_min, range_max


# Usage
y_min, y_max = calculate_axis_range(df['value'], include_zero=True)
fig.update_yaxes(range=[y_min, y_max])
```

### Range Decision Table

| Chart Type | Include Zero? | Padding | Notes |
|------------|---------------|---------|-------|
| Vertical bar | Yes | 5-10% top | Always start from zero |
| Horizontal bar | Yes | 5-10% right | Always start from zero |
| Line chart | Depends | 5% both | Include if showing absolute values |
| Area chart | Yes | 5% top | Base must be zero |
| Scatter plot | Depends | 5% both | Include if origin is meaningful |
| Candlestick | No | 2-5% both | Focus on price range |

---

## Tick Configuration

For continuous axes, limit ticks to 4-6 maximum to prevent crowding. Calculate `dtick` based on the data range to achieve this target. Use `tickmode='array'` with explicit `tickvals` and `ticktext` for precise control. Example: For a range of 0-200B, use `dtick=50e9` to get 5 ticks (0, 50B, 100B, 150B, 200B).

### Controlling Tick Count

Too many ticks create clutter; too few reduce precision.

```python
# Target number of ticks (Plotly estimates)
fig.update_yaxes(nticks=6)  # Approximately 6 ticks

# Exact tick interval
fig.update_yaxes(dtick=1000)  # Tick every 1000 units

# First tick position
fig.update_yaxes(tick0=0, dtick=500)  # Start at 0, tick every 500
```

### Tick Count Guidelines

| Axis Length | Recommended Ticks | Notes |
|-------------|-------------------|-------|
| < 300px | 4-5 | Prevent overcrowding |
| 300-600px | 5-7 | Standard density |
| > 600px | 7-10 | More room for detail |

### Date Tick Intervals

| Timeframe | dtick Value | Example |
|-----------|-------------|---------|
| Hours | `3600000` | Hourly ticks (ms) |
| Days | `86400000` | Daily ticks (ms) |
| Weeks | `604800000` | Weekly ticks (ms) |
| Months | `'M1'` | Monthly ticks |
| Quarters | `'M3'` | Quarterly ticks |
| Years | `'M12'` | Yearly ticks |

```python
# Monthly ticks for multi-month data
fig.update_xaxes(
    dtick='M1',
    tickformat='%b %Y'
)

# Weekly ticks for daily data
fig.update_xaxes(
    dtick=604800000,  # 7 days in milliseconds
    tickformat='%b %d'
)
```

### Tick Format Patterns

Plotly uses d3-format syntax (see Section 07 for complete reference):

```python
# Numbers
fig.update_yaxes(tickformat=',.2s')      # 1.23M
fig.update_yaxes(tickformat='$,.2f')     # $1,234.56
fig.update_yaxes(tickformat='$,.2s')     # $1.23M
fig.update_yaxes(tickformat='.1%')       # 12.3%

# Dates
fig.update_xaxes(tickformat='%b %Y')     # Jan 2025
fig.update_xaxes(tickformat='%b %d')     # Jan 15
fig.update_xaxes(tickformat='%Y-%m-%d')  # 2025-01-15
```

### Tick Rotation

Rotate ticks when labels overlap:

```python
# Rotate x-axis ticks
fig.update_xaxes(
    tickangle=-45,  # Negative = rotate counter-clockwise
    tickfont=dict(size=10)
)

# Common angles
# 0: Horizontal (default)
# -45: Diagonal (good for moderate length labels)
# -90: Vertical (for long labels)
```

**When to rotate:**

| Condition | Recommended Angle |
|-----------|-------------------|
| Labels < 6 chars, few categories | 0 (horizontal) |
| Labels 6-12 chars, many categories | -45 (diagonal) |
| Labels > 12 chars | -90 (vertical) |
| Date labels in tight space | -45 (diagonal) |

### Custom Tick Values and Labels

```python
# Explicit tick positions
fig.update_yaxes(
    tickmode='array',
    tickvals=[0, 25, 50, 75, 100],
    ticktext=['0%', '25%', '50%', '75%', '100%']
)

# Useful for:
# - Non-linear breakpoints
# - Custom labels (e.g., "Low", "Medium", "High")
# - Specific percentiles or thresholds
```

---

## Axis Labels

### Title Placement and Formatting

```python
fig.update_yaxes(
    title=dict(
        text='Price (USD)',
        font=dict(size=12, color='#d3d4d6'),
        standoff=10  # Distance from axis
    )
)

# Alternative shorthand
fig.update_yaxes(
    title_text='Price (USD)',
    title_font=dict(size=12, color='#d3d4d6'),
    title_standoff=10
)
```

### When to Include Axis Titles

| Include Title | Omit Title |
|---------------|------------|
| Units not obvious from values | Date/time on x-axis (self-evident) |
| Multiple y-axes on same chart | Percentage when values show % |
| First chart in series (establishes context) | Subsequent charts in small multiples |
| Unfamiliar metrics | Well-known metrics (e.g., "USD") |

**Important: Time Series X-Axis Labels**

Never add a "Date" or "Time" title to the x-axis of time series charts. The date/time nature is obvious from:
- The tick labels (e.g., "Jan 2024", "Feb 15")
- The sequential nature of the data

Adding `title_text='Date'` wastes space and adds visual noise without providing information.

```python
# BAD: Redundant label
fig.update_xaxes(title_text='Date')

# GOOD: No title needed, dates are self-evident
fig.update_xaxes(tickformat='%b %Y')  # Just format the ticks
```

### Axis Title Best Practices

```python
# Good: concise with units
fig.update_yaxes(title_text='Volume (24h, USD)')
fig.update_yaxes(title_text='Price (USD)')
fig.update_yaxes(title_text='Return (%)')

# Avoid: redundant or verbose
# Bad: "Y-Axis: The Price in US Dollars"
# Bad: "Values" (too vague)
```

### Tick Label Styling

```python
fig.update_xaxes(
    tickfont=dict(
        size=11,
        color='#d3d4d6',
        family='Arial, sans-serif'
    ),
    tickcolor='rgba(255, 255, 255, 0.3)',  # Tick mark color
    ticklen=5,  # Tick mark length
    tickwidth=1  # Tick mark width
)
```

---

## Secondary/Dual Axes

### When Dual Axes Are Appropriate

**Appropriate uses:**
- Two related metrics with different scales (e.g., price and volume)
- Showing a metric and its derivative (e.g., value and % change)
- Overlay of different units on same time axis

**Avoid when:**
- Scales can mislead (unrelated metrics forced to align)
- More than two y-axes needed (use small multiples instead)
- Correlation between metrics is the focus (use scatter plot)

### Creating Dual Y-Axes

```python
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Primary y-axis trace (left)
fig.add_trace(
    go.Bar(
        x=dates,
        y=volumes,
        name='Volume',
        marker=dict(color='rgba(96, 165, 250, 0.5)'),
        hovertemplate='Volume: %{y:$,.2s}<extra></extra>'
    ),
    secondary_y=False
)

# Secondary y-axis trace (right)
fig.add_trace(
    go.Scatter(
        x=dates,
        y=prices,
        name='Price',
        mode='lines',
        line=dict(width=2, color='#F87171'),
        hovertemplate='Price: $%{y:,.2f}<extra></extra>'
    ),
    secondary_y=True
)

# Configure primary y-axis (left)
fig.update_yaxes(
    title_text='Volume',
    tickformat='$,.2s',
    secondary_y=False,
    gridcolor='rgba(255, 255, 255, 0.1)'
)

# Configure secondary y-axis (right)
fig.update_yaxes(
    title_text='Price (USD)',
    tickformat='$,.0f',
    secondary_y=True,
    showgrid=False,  # Only show grid for primary to avoid clutter
    side='right'
)
```

### Dual Axis Alignment Strategies

Align zero points when both metrics can be negative or when comparing ratios:

```python
def align_dual_axes(primary_range, secondary_range):
    """
    Calculate aligned ranges for dual y-axes.

    Aligns zero points proportionally.

    Args:
        primary_range: Tuple of (min, max) for primary axis
        secondary_range: Tuple of (min, max) for secondary axis

    Returns:
        Tuple of (primary_range, secondary_range) aligned
    """
    p_min, p_max = primary_range
    s_min, s_max = secondary_range

    # If both have same sign, no alignment needed
    if (p_min >= 0 and s_min >= 0) or (p_max <= 0 and s_max <= 0):
        return primary_range, secondary_range

    # Calculate proportion of negative range
    p_neg_ratio = abs(p_min) / (abs(p_min) + abs(p_max)) if p_min < 0 else 0
    s_neg_ratio = abs(s_min) / (abs(s_min) + abs(s_max)) if s_min < 0 else 0

    # Use larger negative ratio for both
    neg_ratio = max(p_neg_ratio, s_neg_ratio)

    # Recalculate ranges
    p_total = p_max - p_min
    s_total = s_max - s_min

    new_p_min = -neg_ratio * p_total / (1 - neg_ratio) if neg_ratio < 1 else p_min
    new_s_min = -neg_ratio * s_total / (1 - neg_ratio) if neg_ratio < 1 else s_min

    return (new_p_min, p_max), (new_s_min, s_max)
```

### Dual Axis Styling Best Practices

1. **Color-code axes to traces:**

```python
# Match axis title/ticks to trace color
fig.update_yaxes(
    title_text='Volume',
    title_font=dict(color='#60A5FA'),
    tickfont=dict(color='#60A5FA'),
    secondary_y=False
)

fig.update_yaxes(
    title_text='Price',
    title_font=dict(color='#F87171'),
    tickfont=dict(color='#F87171'),
    secondary_y=True
)
```

2. **Disable grid on secondary axis:**

```python
fig.update_yaxes(showgrid=False, secondary_y=True)
```

3. **Consider if dual axes are necessary:**

```python
def recommend_dual_vs_subplot(metric1_range, metric2_range):
    """
    Recommend dual axes vs separate subplots.

    Args:
        metric1_range: Tuple of (min, max) for first metric
        metric2_range: Tuple of (min, max) for second metric

    Returns:
        String recommendation with reasoning
    """
    ratio1 = metric1_range[1] / max(metric1_range[0], 0.001)
    ratio2 = metric2_range[1] / max(metric2_range[0], 0.001)

    # If scales differ by more than 100x, prefer separate subplots
    scale_ratio = max(ratio1, ratio2) / min(ratio1, ratio2)

    if scale_ratio > 100:
        return 'separate_subplots', f'Scale ratio {scale_ratio:.0f}x - use subplots for clarity'
    elif scale_ratio > 10:
        return 'dual_axes', f'Scale ratio {scale_ratio:.0f}x - dual axes acceptable'
    else:
        return 'normalize', f'Scale ratio {scale_ratio:.0f}x - consider normalizing to same axis'
```

---

## Shared Axes in Subplots

### When to Share Axes

| Share Axes When | Don't Share When |
|-----------------|------------------|
| Comparing same metric across categories | Different metrics with different scales |
| Small multiples of same visualization | Subplots show fundamentally different data |
| Direct visual comparison is the goal | Each subplot needs its own scale for clarity |
| Synchronized navigation is needed | Data ranges are vastly different |

### Shared X-Axes Configuration

```python
from plotly.subplots import make_subplots

# Time series small multiples with shared x-axis
fig = make_subplots(
    rows=2,
    cols=2,
    shared_xaxes=True,  # All subplots share x-axis
    shared_yaxes=False,  # Each has own y-scale (different magnitudes)
    subplot_titles=['BTC', 'ETH', 'SOL', 'AVAX'],
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# Add traces to each subplot
for i, (asset, color) in enumerate([
    ('BTC', '#F7931A'),
    ('ETH', '#627EEA'),
    ('SOL', '#00FFA3'),
    ('AVAX', '#E84142')
]):
    row = (i // 2) + 1
    col = (i % 2) + 1

    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df[asset],
            mode='lines',
            name=asset,
            line=dict(color=color),
            showlegend=False
        ),
        row=row,
        col=col
    )

# Only show x-axis labels on bottom row
for col in range(1, 3):
    fig.update_xaxes(showticklabels=False, row=1, col=col)
    fig.update_xaxes(showticklabels=True, tickformat='%b %Y', row=2, col=col)
```

### Shared Y-Axes Configuration

```python
# Compare distributions with same scale
fig = make_subplots(
    rows=1,
    cols=3,
    shared_xaxes=False,
    shared_yaxes=True,  # Same y-scale for comparison
    subplot_titles=['Group A', 'Group B', 'Group C'],
    horizontal_spacing=0.05
)

# Only leftmost subplot shows y-axis labels
fig.update_yaxes(showticklabels=True, row=1, col=1)
fig.update_yaxes(showticklabels=False, row=1, col=2)
fig.update_yaxes(showticklabels=False, row=1, col=3)
```

### Mixed Shared Axes

```python
# Rows share x-axis, columns share y-axis
fig = make_subplots(
    rows=2,
    cols=2,
    shared_xaxes='rows',    # Each row shares its x-axis
    shared_yaxes='columns', # Each column shares its y-axis
    vertical_spacing=0.1,
    horizontal_spacing=0.05
)
```

### Linking Axes Across Subplots

For synchronized zooming/panning on shared axes:

```python
# Create subplots
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

# Link x-axes explicitly (beyond shared_xaxes)
fig.update_xaxes(matches='x', row=1, col=1)
fig.update_xaxes(matches='x', row=2, col=1)

# All x-axes now zoom/pan together
```

### Axis Reference in Subplots

Understanding axis numbering in subplots:

```python
# In a 2x2 grid:
# Position (1,1): xaxis, yaxis
# Position (1,2): xaxis2, yaxis2
# Position (2,1): xaxis3, yaxis3
# Position (2,2): xaxis4, yaxis4

# Update specific subplot axis
fig.update_xaxes(tickformat='%b %Y', row=1, col=1)  # First subplot
fig.update_yaxes(tickformat='$,.2s', row=2, col=2)  # Fourth subplot

# Update all x-axes
fig.update_xaxes(gridcolor='rgba(255, 255, 255, 0.1)')  # Applies to all

# Update all y-axes
fig.update_yaxes(gridcolor='rgba(255, 255, 255, 0.1)')  # Applies to all
```

---

## Reversed Axes

### When to Reverse Axes

| Scenario | Axis to Reverse | Example |
|----------|-----------------|---------|
| Rankings (1 = best at top) | Y-axis | Leaderboard, top N lists |
| Depth charts | X-axis | Order book visualization |
| Inverted conventions | Depends | Some scientific conventions |
| Mirrored comparisons | One axis | Population pyramids |

### Implementing Reversed Axes

```python
# Reverse y-axis (high values at bottom)
fig.update_yaxes(autorange='reversed')

# Or with explicit range
fig.update_yaxes(range=[100, 0])  # 100 at bottom, 0 at top

# Reverse x-axis
fig.update_xaxes(autorange='reversed')
```

### Ranking Charts

```python
# Leaderboard with rank 1 at top
fig = go.Figure()

fig.add_trace(go.Bar(
    y=df['name'],
    x=df['score'],
    orientation='h',
    marker=dict(color='#60A5FA')
))

# Reverse so rank 1 appears at top
fig.update_yaxes(
    autorange='reversed',
    categoryorder='array',
    categoryarray=df.sort_values('rank')['name'].tolist()
)
```

### Depth Charts (Order Books)

```python
# Bid side (reversed x) and ask side
fig = go.Figure()

# Bids (buying) - price decreasing to the left
fig.add_trace(go.Scatter(
    x=bid_prices,
    y=bid_cumulative_volume,
    fill='tozeroy',
    name='Bids',
    line=dict(color='#34D399')
))

# Asks (selling) - price increasing to the right
fig.add_trace(go.Scatter(
    x=ask_prices,
    y=ask_cumulative_volume,
    fill='tozeroy',
    name='Asks',
    line=dict(color='#F87171')
))

# Center on current price
mid_price = (bid_prices[-1] + ask_prices[0]) / 2
spread = (ask_prices[-1] - bid_prices[0]) / 2

fig.update_xaxes(range=[mid_price - spread, mid_price + spread])
```

---

## Plotly Implementation

### Complete Axis Configuration Example

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def configure_time_series_axes(
    fig,
    date_range_days,
    y_format='$,.2s',
    include_zero=False,
    show_grid=True
):
    """
    Configure axes for time series chart.

    Args:
        fig: Plotly figure
        date_range_days: Number of days in the date range
        y_format: Tick format for y-axis
        include_zero: Whether y-axis should include zero
        show_grid: Whether to show grid lines
    """
    # X-axis (date) configuration
    if date_range_days <= 7:
        tick_format = '%b %d'
        dtick = 86400000  # 1 day in ms
    elif date_range_days <= 30:
        tick_format = '%b %d'
        dtick = 86400000 * 7  # 1 week in ms
    elif date_range_days <= 180:
        tick_format = '%b %Y'
        dtick = 'M1'
    else:
        tick_format = '%b %Y'
        dtick = 'M3'

    fig.update_xaxes(
        type='date',
        tickformat=tick_format,
        dtick=dtick,
        gridcolor='rgba(255, 255, 255, 0.1)' if show_grid else 'rgba(0,0,0,0)',
        linecolor='rgba(255, 255, 255, 0.2)',
        tickfont=dict(size=11, color='#d3d4d6'),
        showgrid=show_grid
    )

    # Y-axis configuration
    rangemode = 'tozero' if include_zero else 'normal'

    fig.update_yaxes(
        tickformat=y_format,
        rangemode=rangemode,
        gridcolor='rgba(255, 255, 255, 0.1)' if show_grid else 'rgba(0,0,0,0)',
        linecolor='rgba(255, 255, 255, 0.2)',
        tickfont=dict(size=11, color='#d3d4d6'),
        showgrid=show_grid
    )

    return fig


def configure_categorical_axes(
    fig,
    category_order='total descending',
    x_is_categorical=True,
    rotate_labels=0,
    value_format=',.2s'
):
    """
    Configure axes for categorical chart.

    Args:
        fig: Plotly figure
        category_order: How to order categories
        x_is_categorical: Whether x-axis is categorical (else y-axis)
        rotate_labels: Rotation angle for category labels
        value_format: Format for value axis
    """
    if x_is_categorical:
        fig.update_xaxes(
            type='category',
            categoryorder=category_order,
            tickangle=rotate_labels,
            gridcolor='rgba(0, 0, 0, 0)',  # No vertical grid for bar charts
            linecolor='rgba(255, 255, 255, 0.2)',
            tickfont=dict(size=11, color='#d3d4d6')
        )
        fig.update_yaxes(
            tickformat=value_format,
            rangemode='tozero',  # Bars should start at zero
            gridcolor='rgba(255, 255, 255, 0.1)',
            linecolor='rgba(255, 255, 255, 0.2)',
            tickfont=dict(size=11, color='#d3d4d6')
        )
    else:
        fig.update_yaxes(
            type='category',
            categoryorder=category_order,
            gridcolor='rgba(0, 0, 0, 0)',
            linecolor='rgba(255, 255, 255, 0.2)',
            tickfont=dict(size=11, color='#d3d4d6')
        )
        fig.update_xaxes(
            tickformat=value_format,
            rangemode='tozero',
            gridcolor='rgba(255, 255, 255, 0.1)',
            linecolor='rgba(255, 255, 255, 0.2)',
            tickfont=dict(size=11, color='#d3d4d6')
        )

    return fig
```

### Grid and Axis Line Styling

```python
# Standard axis styling
fig.update_xaxes(
    showline=True,
    linewidth=1,
    linecolor='rgba(255, 255, 255, 0.2)',
    showgrid=False,  # Gridlines OFF by default
    gridwidth=1,
    gridcolor='rgba(255, 255, 255, 0.1)',  # If enabled
    zeroline=False,  # Hide zero line (often redundant with grid)
    mirror=False,  # Don't mirror axis on opposite side
    ticks='outside',  # Tick marks outside plot area
    ticklen=5,
    tickcolor='rgba(255, 255, 255, 0.3)'
)

# Minimal axis styling (clean look)
fig.update_xaxes(
    showline=False,
    showgrid=False,  # Gridlines OFF by default
    gridwidth=1,
    gridcolor='rgba(255, 255, 255, 0.05)',  # If enabled
    zeroline=False,
    ticks=''  # No tick marks
)
```

### Axis Domain (Position in Figure)

```python
# Adjust axis position within figure
# Domain is [start, end] as fraction of figure (0-1)

# Default: full width
fig.update_xaxes(domain=[0, 1])

# Leave room for annotation on right
fig.update_xaxes(domain=[0, 0.85])

# Leave room for secondary chart on top
fig.update_yaxes(domain=[0, 0.7])
```

---

## Quick Reference

### Scale Type Decision

| Data Characteristic | Recommended Scale |
|---------------------|-------------------|
| Linear progression | `'linear'` |
| Orders of magnitude | `'log'` |
| Time/date values | `'date'` |
| Discrete categories | `'category'` |

### Common Axis Configurations

| Chart Type | X-Axis | Y-Axis |
|------------|--------|--------|
| Time series | `type='date'`, `tickformat='%b %Y'` | `tickformat='$,.2s'` |
| Vertical bar | `type='category'` | `rangemode='tozero'` |
| Horizontal bar | `rangemode='tozero'` | `type='category'` |
| Scatter | `tickformat=',.2s'` | `tickformat=',.2s'` |
| Log-scale | N/A | `type='log'` |

### Tick Format Quick Reference

| Format | Pattern | Example Output |
|--------|---------|----------------|
| SI notation | `,.2s` | 1.23M |
| Currency | `$,.2f` | $1,234.56 |
| Currency + SI | `$,.2s` | $1.23M |
| Percentage | `.1%` | 12.3% |
| Date (month) | `%b %Y` | Jan 2025 |
| Date (day) | `%b %d` | Jan 15 |

### Template Colors

| Element | Color |
|---------|-------|
| Background | #0e1729 |
| Text | #d3d4d6 |
| Grid (subtle) | rgba(255, 255, 255, 0.1) |
| Axis lines | rgba(255, 255, 255, 0.2) |
| Tick marks | rgba(255, 255, 255, 0.3) |

---

## Axis Configuration Checklist

Before finalizing axes, verify:

**Scale Type:**
- [ ] Linear scale for standard quantitative data
- [ ] Log scale only when data spans 3+ orders of magnitude
- [ ] Log scale indicated in title or annotation
- [ ] Date scale for temporal data
- [ ] Categorical scale with appropriate ordering

**Range:**
- [ ] Bar charts include zero on value axis
- [ ] Line charts have appropriate range (include zero if showing absolute values)
- [ ] Padding added to prevent data from touching axis bounds
- [ ] Consistent ranges across related/compared charts

**Ticks:**
- [ ] Tick count appropriate for axis length (5-7 typical)
- [ ] Tick format matches data type (SI notation, currency, percentage, date)
- [ ] Labels rotated if overlapping
- [ ] Date tick intervals appropriate for timeframe

**Labels:**
- [ ] Axis titles included when units are not obvious
- [ ] Axis titles omitted for self-evident axes (dates, percentages)
- [ ] Font size readable (11-12px for ticks)
- [ ] Color matches template (#d3d4d6)

**Dual/Secondary Axes:**
- [ ] Dual axes used only when necessary
- [ ] Secondary axis color-coded to match trace
- [ ] Grid shown only on primary axis
- [ ] Consider subplots if more than two y-axes needed

**Shared Axes (Subplots):**
- [ ] X-axes shared when comparing over same time range
- [ ] Y-axes shared only when comparing same metric/scale
- [ ] Only edge subplots show tick labels (reduce clutter)
- [ ] Axes linked for synchronized zooming when appropriate

**Reversed Axes:**
- [ ] Reversed only when convention demands (rankings, depth charts)
- [ ] Direction clearly indicated to avoid confusion

**Styling:**
- [ ] Grid color subtle: rgba(255, 255, 255, 0.1)
- [ ] Axis line color: rgba(255, 255, 255, 0.2)
- [ ] Zero line disabled (usually redundant with grid)
- [ ] Consistent styling across all axes in figure

## Axis Styling and Fonts

Use `title=dict(text='...', font=dict(...))` for axis titles, not `titlefont`. The correct structure is nested: `xaxis=dict(title=dict(text='Label', font=dict(size=14, color='#d3d4d6')))`. The deprecated `titlefont` parameter will cause an error in current Plotly versions.

## Date/Time Axes

For daily data, use `tickformat='%b %d'` to show month and day (e.g., 'Jan 15'). Avoid `tickformat='%b %Y'` which shows only month-year and can create confusing repeated labels like 'Jan 2024, Jan 2024' when multiple days from the same month are shown. Match tick granularity to your data frequency.

## Axis Tick Spacing

Use `nticks` parameter to control regular spacing on axes. For example, `yaxis=dict(nticks=6)` creates approximately 6 evenly-spaced ticks. This is preferred over letting Plotly auto-generate irregular intervals.

## Grid Lines

Grid lines are enabled by default in Plotly. For cleaner visualizations, explicitly disable them with `xaxis=dict(showgrid=False)` and `yaxis=dict(showgrid=False)` unless they aid readability.

Grid lines are enabled by default in Plotly but often reduce visual clarity, especially with dark themes. For cleaner visualizations, disable them with `xaxis=dict(showgrid=False)` and `yaxis=dict(showgrid=False)`. Consider keeping grid lines only when precise value reading is critical.

## Gridlines

By default, remove gridlines for cleaner visualizations unless they significantly aid readability. Set `xaxis=dict(showgrid=False)` and `yaxis=dict(showgrid=False)`. Gridlines are most useful for precise value reading in dense scatter plots or when comparing many data points.

Remove gridlines for cleaner visualizations when the data patterns are clear: set `xaxis=dict(showgrid=False)` and `yaxis=dict(showgrid=False)`. Gridlines can be removed when hover interactions provide precise values and the visual pattern is the primary focus.

For cleaner visualizations, especially with bar charts where values are labeled directly on bars, consider removing gridlines with `xaxis=dict(showgrid=False)` and `yaxis=dict(showgrid=False)`. Keep the zeroline visible for reference with `zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'`.

## Tick Spacing

Use Plotly's automatic tick spacing by default rather than custom tick intervals. Only override with `dtick` when specific intervals are required for domain-specific reasons (e.g., showing every quarter in financial data). Plotly's auto-generated ticks adapt well to different data ranges.

## Tick Spacing and Intervals

Use regularly-spaced tick intervals for better readability. Set `tickmode='linear'` with explicit `tick0=0` and `dtick=<interval>` values, or use `nticks=<number>` to specify the approximate number of ticks. Avoid irregular spacing that makes the axis harder to interpret. For financial data in billions, use round intervals like 20, 25, or 50.

For regular tick spacing, use `tickmode='array'` with `tickvals` generated at consistent intervals using `np.arange()`. Calculate an appropriate `dtick` based on the data range (e.g., for values in billions: use 25 for ranges >100, 10 for >50, 5 for >20, 2 otherwise). Example: `tickvals = list(np.arange(0, max_val + dtick, dtick))`

## Tick Formatting

For financial data, avoid decimal places on axis ticks unless precision is critical. Use `ticktext` with custom formatting functions to display whole numbers with k/M/B suffixes (e.g., '$25B' not '$25.0B'). Set `decimals=0` in formatting functions for cleaner axis labels.
