# 04 - Data Encoding

> Map data dimensions to visual properties systematically.
> This section provides encoding rules for AI-assisted chart generation.

---

## Core Principle

Data encoding is the process of mapping data values to visual properties. The effectiveness of a visualization depends on choosing the right encoding channels for each data dimension.

**Encoding Priority Rule**: Encode the most important data dimension using the most effective visual channel.

```
ENCODING EFFECTIVENESS (ranked by perceptual accuracy):
1. Position (x, y) - Most accurate
2. Length/height
3. Angle/slope
4. Area
5. Volume (rarely used, avoid)
6. Color saturation/lightness
7. Color hue - Good for categories, poor for quantities
8. Shape - Categories only, limited capacity
```

---

## Position Encoding

### Primary Encoding Channel

Position is the most effective encoding channel. Use it for your primary data dimensions.

| Axis | Common Use | Data Type |
|------|------------|-----------|
| **X-axis** | Time, categories, independent variable | Temporal, categorical, numeric |
| **Y-axis** | Measured value, dependent variable | Numeric |
| **Z-axis** | Avoid in 2D charts | N/A |

### X-Y Mapping Conventions

```
STANDARD MAPPINGS:
  Time series:
    X = datetime
    Y = measured value

  Categorical comparison:
    X = categories (vertical bars)
    Y = measured value
    OR
    Y = categories (horizontal bars - preferred for readability)
    X = measured value

  Relationship/scatter:
    X = independent variable
    Y = dependent variable
```

### Position Encoding Rules

| Rule | Description |
|------|-------------|
| **Consistency** | Same variable should occupy same axis across related charts |
| **Zero baseline** | Include zero for bar charts and area charts |
| **Time flows right** | Time on X-axis, left to right |
| **Vertical for comparison** | Put comparison categories on Y for horizontal bars |

**Key Configuration:**

```python
import plotly.graph_objects as go

# Time series: X = time, Y = value
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['date'],           # Position encoding: time
    y=df['price'],          # Position encoding: measured value
    mode='lines',
    line=dict(width=2, color='#60A5FA')
))

fig.update_layout(
    template=get_template_tie(),
    xaxis=dict(
        title='Date',
        type='date'
    ),
    yaxis=dict(
        title='Price ($)',
        rangemode='tozero'  # Include zero for fair comparison
    )
)
```

### Dual Position Encoding

Use secondary Y-axis only when:
1. Two metrics have genuinely different scales
2. Metrics are related (e.g., price and volume)
3. There are exactly 2 series

```python
fig.update_layout(
    yaxis=dict(
        title='Volume',
        side='left'
    ),
    yaxis2=dict(
        title='Price',
        side='right',
        overlaying='y',
        showgrid=False
    )
)
```

---

## Color Encoding

### Color Encoding Types

```
COLOR CHANNELS:
  HUE       → Category differentiation (red, blue, green)
  SATURATION → Magnitude/intensity (pale to vivid)
  LIGHTNESS  → Magnitude/intensity (dark to light)
```

### Hue for Categories

Use color hue to distinguish between categorical groups.

| Attribute | Value |
|-----------|-------|
| **Best for** | Distinguishing 2-8 categories |
| **Limit** | Maximum 8 distinct hues (cognitive limit) |
| **Requirement** | Colors must be distinguishable (colorblind-safe) |

**Standard Colorway:**

```python
colorway = ["#60A5FA", "#F87171", "#34D399", "#FBBF24", "#E879F9"]

# Index | Color  | Hex      | Typical Use
# ------|--------|----------|------------------
# 0     | Blue   | #60A5FA  | Primary series
# 1     | Red    | #F87171  | Secondary/negative
# 2     | Green  | #34D399  | Positive/tertiary
# 3     | Yellow | #FBBF24  | Highlight/warning
# 4     | Purple | #E879F9  | Additional series
```

**Key Configuration:**

```python
categories = df['category'].unique()
colorway = ["#60A5FA", "#F87171", "#34D399", "#FBBF24", "#E879F9"]

fig = go.Figure()
for i, cat in enumerate(categories[:5]):  # Max 5 categories
    mask = df['category'] == cat
    fig.add_trace(go.Scatter(
        x=df.loc[mask, 'date'],
        y=df.loc[mask, 'value'],
        mode='lines',
        name=cat,
        line=dict(width=2, color=colorway[i])
    ))
```

### Saturation/Lightness for Magnitude

Use saturation or lightness to encode continuous numeric values.

| Scale Type | Use When | Color Pattern |
|------------|----------|---------------|
| **Sequential** | Single-direction magnitude (0 to max) | Dark to light single hue |
| **Diverging** | Bidirectional from center (negative to positive) | Color A -> neutral -> Color B |

**Sequential Scale Example:**

```python
# For magnitude-only data (e.g., volume, count)
fig.add_trace(go.Heatmap(
    z=values,
    colorscale=[
        [0, '#0e1729'],      # Low: background (dark)
        [0.5, '#3B82F6'],    # Mid: medium blue
        [1, '#60A5FA']       # High: bright blue
    ],
    showscale=True,
    colorbar=dict(title='Volume')
))
```

**Diverging Scale Example:**

```python
# For +/- data (e.g., returns, correlation)
fig.add_trace(go.Heatmap(
    z=correlation_matrix,
    colorscale=[
        [0, '#F87171'],      # Negative: red
        [0.5, '#0e1729'],    # Zero: background (neutral)
        [1, '#34D399']       # Positive: green
    ],
    zmid=0,  # Center at zero
    zmin=-1,
    zmax=1
))
```

### When Color Adds Value

| Scenario | Color Adds Value |
|----------|------------------|
| Distinguishing 2-8 series | Yes |
| Highlighting specific data points | Yes |
| Encoding sign (positive/negative) | Yes |
| Showing magnitude on a surface/heatmap | Yes |
| Encoding a third dimension in scatter | Yes |

### When Color Adds Clutter

| Scenario | Why Color Fails |
|----------|-----------------|
| >8 categories | Too many colors to distinguish |
| Single series chart | Unnecessary visual element |
| Color duplicates position encoding | Redundant, not informative |
| Rainbow gradients for sequential data | Perceptually non-uniform |
| Decoration only | No data mapping = clutter |

**Decision Rule:**

```
IF color encodes meaningful data dimension → USE IT
IF color is purely decorative → SKIP IT
IF color duplicates another encoding without accessibility benefit → SKIP IT
```

### Semantic Color Conventions

Maintain consistent meaning across visualizations:

| Color | Semantic Meaning |
|-------|------------------|
| Green (#34D399) | Positive, increase, profit, long |
| Red (#F87171) | Negative, decrease, loss, short |
| Blue (#60A5FA) | Neutral primary, default |
| Yellow (#FBBF24) | Warning, highlight, attention |
| Gray (muted) | De-emphasized, background context |

---

## Size Encoding

### Area vs Radius

**Critical Rule**: Encode magnitude using **area**, not radius.

Human perception compares circle areas, not radii. Using radius creates misleading visual comparisons.

```
PERCEPTION ERROR:
  If Value B = 2x Value A

  Radius encoding (WRONG):
    Radius B = 2x Radius A
    Area B = 4x Area A  → Appears 4x larger, not 2x

  Area encoding (CORRECT):
    Area B = 2x Area A
    Radius B = sqrt(2) x Radius A ≈ 1.41x Radius A
    → Appears correctly as 2x
```

### Bubble Chart Configuration

| Attribute | Value |
|-----------|-------|
| **Use when** | Third numeric dimension in scatter plot |
| **Size mode** | Always use `sizemode='area'` |
| **Minimum size** | 4-6 pixels (must be visible) |
| **Maximum size** | 40-60 pixels (avoid overlap) |
| **Reference** | Calculate `sizeref` for consistent scaling |

**Key Configuration:**

```python
# Calculate sizeref for area-based scaling
max_size = df['magnitude'].max()
sizeref = 2. * max_size / (40.**2)  # 40 = max marker size in pixels

fig.add_trace(go.Scatter(
    x=df['x'],
    y=df['y'],
    mode='markers',
    marker=dict(
        size=df['magnitude'],
        sizemode='area',        # Critical: use area, not diameter
        sizeref=sizeref,
        sizemin=4,              # Minimum visible size
        color='#60A5FA',
        opacity=0.7,
        line=dict(width=1, color='#1e293b')
    ),
    text=df['label'],
    hovertemplate=(
        '%{text}<br>'
        'X: %{x:.2f}<br>'
        'Y: %{y:.2f}<br>'
        'Size: %{marker.size:,.0f}'
        '<extra></extra>'
    )
))
```

### When to Use Bubble Charts

| Scenario | Recommendation |
|----------|----------------|
| 3 numeric variables, want single view | Use bubble chart |
| Size values have large range (>100x) | Consider log transform or cap outliers |
| Many overlapping points | Reduce opacity, consider aggregation |
| Size is critical for analysis | Add size legend or annotations |
| Precise size comparison needed | Use bar chart instead |

### Size Legend

Always include a size reference for bubble charts:

```python
# Add size legend annotation
fig.add_annotation(
    x=0.98,
    y=0.02,
    xref='paper',
    yref='paper',
    text='Bubble size = Market Cap',
    showarrow=False,
    font=dict(size=10, color='#d3d4d6'),
    align='right'
)

# Or add reference bubbles
reference_sizes = [100, 500, 1000]
for i, size in enumerate(reference_sizes):
    fig.add_trace(go.Scatter(
        x=[0.95],
        y=[0.2 + i * 0.1],
        xref='paper',
        yref='paper',
        mode='markers+text',
        marker=dict(
            size=size,
            sizemode='area',
            sizeref=sizeref,
            color='rgba(96, 165, 250, 0.3)',
            line=dict(width=1, color='#60A5FA')
        ),
        text=[f'{size:,}'],
        textposition='middle right',
        showlegend=False,
        hoverinfo='skip'
    ))
```

---

## Shape Encoding

### Marker Symbols for Categories

Use shape to differentiate categories when color is already used or for accessibility.

| Attribute | Value |
|-----------|-------|
| **Best for** | Additional categorical dimension |
| **Limit** | Maximum 5-6 distinct shapes |
| **Common use** | Redundant encoding with color (accessibility) |

**Recommended Shapes:**

```python
# Easily distinguishable shapes (in order of preference)
shapes = [
    'circle',       # Default, most common
    'square',       # Clear distinction from circle
    'diamond',      # Rotated square, distinct silhouette
    'triangle-up',  # Directional, distinct
    'cross',        # Open center, distinct
    'x'             # Similar to cross but rotated
]
```

**Key Configuration:**

```python
categories = df['category'].unique()
colorway = ["#60A5FA", "#F87171", "#34D399", "#FBBF24", "#E879F9"]
shapes = ['circle', 'square', 'diamond', 'triangle-up', 'cross']

fig = go.Figure()
for i, cat in enumerate(categories[:5]):
    mask = df['category'] == cat
    fig.add_trace(go.Scatter(
        x=df.loc[mask, 'x'],
        y=df.loc[mask, 'y'],
        mode='markers',
        name=cat,
        marker=dict(
            size=10,
            color=colorway[i],
            symbol=shapes[i],
            line=dict(width=1, color='#1e293b')
        )
    ))

fig.update_layout(
    template=get_template_tie(),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02
    )
)
```

### When Shape is Useful

| Scenario | Shape Encoding |
|----------|----------------|
| Colorblind accessibility | Use shape + color redundantly |
| Scatter with overlapping categories | Shape helps distinguish when colors overlap |
| Print/grayscale output | Shape persists without color |
| Two categorical dimensions | Color for one, shape for other |

### Shape Encoding Rules

1. **Never use shape alone** - Always pair with color for visibility
2. **Limit to 5 shapes** - Beyond this, patterns become confusing
3. **Use filled shapes** - Open shapes are harder to see at small sizes
4. **Consistent size** - All shapes should have similar visual weight

```python
# Good: Filled shapes with consistent sizing
marker=dict(
    symbol='diamond',
    size=10,  # Consistent across all series
    color='#60A5FA',
    line=dict(width=1, color='#1e293b')
)

# Avoid: Open shapes at small sizes
marker=dict(
    symbol='circle-open',  # Hard to see when small
    size=6
)
```

---

## Line Encoding

### Line Style for Series Differentiation

Use line style (dash pattern) as a secondary encoding channel.

| Style | Plotly Value | Use For |
|-------|--------------|---------|
| Solid | `'solid'` | Primary data series |
| Dashed | `'dash'` | Reference lines, projections |
| Dotted | `'dot'` | Uncertainty, estimates |
| Dash-dot | `'dashdot'` | Additional differentiation |

**Key Configuration:**

```python
line_styles = ['solid', 'dash', 'dot', 'dashdot']

fig = go.Figure()

# Actual data (solid)
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['actual'],
    mode='lines',
    name='Actual',
    line=dict(width=2, color='#60A5FA', dash='solid')
))

# Forecast (dashed)
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['forecast'],
    mode='lines',
    name='Forecast',
    line=dict(width=2, color='#60A5FA', dash='dash')
))

# Confidence interval (dotted)
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['upper_bound'],
    mode='lines',
    name='95% CI',
    line=dict(width=1, color='#60A5FA', dash='dot'),
    showlegend=False
))
```

### Line Width for Emphasis

| Width | Use Case |
|-------|----------|
| 1px | Secondary/background series, confidence intervals |
| 2px | Standard primary series |
| 3px | Emphasized/highlighted series |
| 4px+ | Strong emphasis (use sparingly) |

**Emphasis Pattern:**

```python
# Emphasize one series among many
fig = go.Figure()

for i, series in enumerate(all_series):
    is_emphasized = (series == highlighted_series)

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[series],
        mode='lines',
        name=series,
        line=dict(
            width=3 if is_emphasized else 1.5,
            color='#60A5FA' if is_emphasized else 'rgba(96, 165, 250, 0.3)'
        )
    ))
```

### Line Style Decision Table

| Data Type | Style | Width |
|-----------|-------|-------|
| Historical actual | Solid | 2px |
| Forecast/projection | Dashed | 2px |
| Confidence bounds | Dotted | 1px |
| Reference/threshold | Dashed | 1px |
| Emphasized series | Solid | 3px |
| De-emphasized series | Solid | 1px (+ reduced opacity) |

---

## Opacity Encoding

### Reduce Overplotting

Use opacity when many data points overlap.

| Points | Opacity Range |
|--------|---------------|
| <100 | 0.8-1.0 |
| 100-1,000 | 0.5-0.8 |
| 1,000-10,000 | 0.2-0.5 |
| >10,000 | 0.1-0.3 (consider aggregation) |

**Key Configuration:**

```python
n_points = len(df)

# Calculate appropriate opacity
if n_points < 100:
    opacity = 0.8
elif n_points < 1000:
    opacity = 0.5
elif n_points < 10000:
    opacity = 0.3
else:
    opacity = 0.1

fig.add_trace(go.Scatter(
    x=df['x'],
    y=df['y'],
    mode='markers',
    marker=dict(
        size=6,
        color='#60A5FA',
        opacity=opacity
    )
))
```

### De-emphasize Secondary Series

Use opacity to create visual hierarchy - push less important data to background.

```python
# Primary series: full opacity
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['main_series'],
    mode='lines',
    name='Main',
    line=dict(width=2, color='#60A5FA')
    # No opacity specified = 1.0
))

# Context series: reduced opacity
for context_series in context_list:
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[context_series],
        mode='lines',
        name=context_series,
        line=dict(width=1, color='rgba(211, 212, 214, 0.3)')  # 30% opacity
    ))
```

### Opacity Encoding Patterns

| Pattern | Implementation |
|---------|----------------|
| **Focus + Context** | Primary at 1.0, secondary at 0.2-0.3 |
| **Dense scatter** | All points at 0.1-0.5 |
| **Layered fills** | Bottom layers more opaque, top more transparent |
| **Hover highlight** | Base at 0.5, increase to 1.0 on hover |

**Fill Opacity for Area Charts:**

```python
# Solid line, semi-transparent fill
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['value'],
    mode='lines',
    fill='tozeroy',
    line=dict(width=2, color='#60A5FA'),
    fillcolor='rgba(96, 165, 250, 0.3)'  # 30% opacity fill
))
```

---

## Encoding Priorities

### Preattentive Attributes

These visual properties are processed automatically by the brain (within ~250ms), before conscious attention:

| Attribute | Effectiveness | Best For |
|-----------|---------------|----------|
| Position | Highest | Primary quantitative data |
| Length | Very High | Bar charts, magnitude |
| Slope/Angle | High | Trends, changes |
| Area | Medium-High | Bubble charts (use carefully) |
| Color Hue | Medium | Categories (limited to ~8) |
| Color Intensity | Medium | Sequential magnitude |
| Shape | Low-Medium | Categories (limited to ~6) |
| Line Style | Low | Secondary differentiation |

### Encoding Priority Matrix

Match data importance to encoding effectiveness:

| Data Priority | Encoding Channel |
|---------------|------------------|
| **Primary dimension** | X or Y position |
| **Secondary dimension** | Remaining position axis |
| **Tertiary dimension** | Color hue OR size |
| **Quaternary dimension** | Remaining (shape, line style) |
| **Context/grouping** | Faceting (small multiples) |

### Recommended Encoding Combinations

| Variables | Recommended Encoding |
|-----------|---------------------|
| 1 numeric + time | X: time, Y: numeric |
| 1 numeric + 1 categorical | X/Y: categorical, Y/X: numeric, Color: categorical |
| 2 numeric | X: independent, Y: dependent |
| 2 numeric + 1 categorical | X, Y: numeric, Color: categorical |
| 3 numeric | X, Y: primary metrics, Size: third metric |
| 3 numeric + 1 categorical | X, Y: numeric, Size: numeric, Color: categorical |

**Key Configuration Example (4 dimensions):**

```python
# Encoding: X=gdp, Y=life_expectancy, Size=population, Color=continent
fig.add_trace(go.Scatter(
    x=df['gdp_per_capita'],     # Position: primary metric
    y=df['life_expectancy'],    # Position: secondary metric
    mode='markers',
    marker=dict(
        size=df['population'],   # Size: third metric
        sizemode='area',
        sizeref=2. * df['population'].max() / (40.**2),
        sizemin=4,
        color=df['continent'].map(color_map),  # Color: category
        opacity=0.7,
        line=dict(width=1, color='#1e293b')
    ),
    text=df['country'],
    hovertemplate=(
        '<b>%{text}</b><br>'
        'GDP/capita: $%{x:,.0f}<br>'
        'Life expectancy: %{y:.1f} years<br>'
        'Population: %{marker.size:,.0f}'
        '<extra></extra>'
    )
))
```

---

## Encoding Limits

### Maximum Categories per Channel

| Encoding Channel | Maximum Categories | Notes |
|------------------|-------------------|-------|
| Color hue | 8 | Beyond this, colors become indistinguishable |
| Shape | 6 | Limited by distinguishable marker types |
| Line style | 4 | Solid, dash, dot, dashdot |
| Size | 5-6 buckets | Continuous is fine, discrete categories limited |
| Position (categories) | 15-20 | Long lists need aggregation or scrolling |

### What to Do When Limits Are Exceeded

| Situation | Solution |
|-----------|----------|
| >8 color categories | Aggregate to "Other", use small multiples, or interactive filtering |
| >6 shape categories | Use color instead, or combine categories |
| Too many series | Small multiples, top-N selection, or interactive legend |
| >20 position categories | Show top N, aggregate rest to "Other" |

**Aggregation Example:**

```python
# Too many categories: aggregate to top 5 + Other
value_counts = df.groupby('category')['value'].sum().sort_values(ascending=False)
top_5 = value_counts.head(5).index.tolist()

df['category_grouped'] = df['category'].apply(
    lambda x: x if x in top_5 else 'Other'
)

# Ensure "Other" is last in color assignment
category_order = top_5 + ['Other']
colorway = ["#60A5FA", "#F87171", "#34D399", "#FBBF24", "#E879F9", "#6B7280"]
```

### Cognitive Load Guidelines

```
RULE OF THUMB:
  - 2-3 encodings: Easy to read, recommended for most charts
  - 4 encodings: Complex but manageable (e.g., bubble chart with color)
  - 5+ encodings: Difficult to interpret, avoid if possible

ENCODING COMBINATIONS TO AVOID:
  - Size + Line width (confusing)
  - Multiple sequential color scales (impossible to compare)
  - Shape alone without color (low visibility)
  - More than one double-encoding (e.g., color for two different meanings)
```

---

## Encoding Decision Table

### Quick Reference: Data Type to Encoding

| Data Type | Primary Encoding | Secondary Encoding |
|-----------|------------------|-------------------|
| **Continuous numeric** | Position (X or Y) | Color intensity, Size |
| **Discrete numeric** | Position | Color intensity |
| **Categorical (ordered)** | Position | Color intensity |
| **Categorical (nominal)** | Position, Color hue | Shape |
| **Temporal** | X position | - |
| **Boolean/binary** | Color hue | Shape |

### Quick Reference: Intent to Encoding

| Intent | Primary Encoding | Secondary Encoding |
|--------|------------------|-------------------|
| **Show trend** | Y position over X time | Color for series |
| **Compare categories** | Position (bar length) | Color for groups |
| **Show distribution** | X position (histogram) | Y for frequency |
| **Show relationship** | X and Y position | Size/Color for third variable |
| **Show composition** | Area (stacked) | Color for parts |
| **Highlight subset** | Color, Opacity | Size |

---

## Complete Encoding Example

This example demonstrates proper encoding for a multi-dimensional dataset:

```python
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Sample data: Companies by revenue, profit margin, employees, and sector
np.random.seed(42)
df = pd.DataFrame({
    'company': [f'Company {i}' for i in range(50)],
    'revenue': np.random.exponential(1000, 50) * 1e6,
    'profit_margin': np.random.uniform(-0.1, 0.4, 50),
    'employees': np.random.exponential(5000, 50),
    'sector': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Energy'], 50)
})

# Color mapping for sectors
sector_colors = {
    'Tech': '#60A5FA',
    'Finance': '#F87171',
    'Healthcare': '#34D399',
    'Energy': '#FBBF24'
}

# Calculate size reference for area encoding
max_employees = df['employees'].max()
sizeref = 2. * max_employees / (40.**2)

fig = go.Figure()

# Add trace for each sector (color encoding)
for sector in df['sector'].unique():
    sector_df = df[df['sector'] == sector]

    fig.add_trace(go.Scatter(
        x=sector_df['revenue'],           # Position: X = revenue
        y=sector_df['profit_margin'],     # Position: Y = profit margin
        mode='markers',
        name=sector,
        marker=dict(
            size=sector_df['employees'],   # Size: employees (area encoded)
            sizemode='area',
            sizeref=sizeref,
            sizemin=5,
            color=sector_colors[sector],   # Color: sector (categorical)
            opacity=0.7,
            line=dict(width=1, color='#1e293b')
        ),
        text=sector_df['company'],
        hovertemplate=(
            '<b>%{text}</b><br>'
            'Revenue: $%{x:,.0f}<br>'
            'Profit Margin: %{y:.1%}<br>'
            'Employees: %{marker.size:,.0f}'
            '<extra></extra>'
        )
    ))

fig.update_layout(
    template=get_template_tie(),
    title='Company Performance by Sector',
    xaxis=dict(
        title='Revenue ($)',
        type='log',           # Log scale for exponential distribution
        tickformat='$,.2s'
    ),
    yaxis=dict(
        title='Profit Margin',
        tickformat='.0%',
        zeroline=True,
        zerolinecolor='rgba(255,255,255,0.3)'
    ),
    legend=dict(
        title='Sector',
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='left',
        x=0
    )
)

# Add annotation explaining size encoding
fig.add_annotation(
    x=0.98,
    y=0.02,
    xref='paper',
    yref='paper',
    text='Bubble size = Employee count',
    showarrow=False,
    font=dict(size=10, color='#d3d4d6'),
    align='right'
)
```

**Encoding Summary for this example:**

| Data Dimension | Encoding | Channel Type |
|----------------|----------|--------------|
| Revenue | X position | Position (primary) |
| Profit margin | Y position | Position (primary) |
| Employees | Bubble size (area) | Size (secondary) |
| Sector | Color hue | Color (categorical) |

---

## Encoding Checklist

Before finalizing a visualization, verify:

### Position Encoding
- [ ] Primary metric uses position encoding (most effective channel)
- [ ] Time on X-axis flows left to right
- [ ] Bar charts include zero baseline
- [ ] Dual Y-axes used only when necessary (different scales, related metrics)

### Color Encoding
- [ ] Color encodes meaningful data (not decoration)
- [ ] Maximum 8 distinct hue categories
- [ ] Colorblind-safe palette used
- [ ] Sequential scale for magnitude, diverging for +/- values
- [ ] Consistent color meanings across related charts (green=positive, red=negative)

### Size Encoding
- [ ] Size uses area mode, not radius/diameter
- [ ] Size reference (sizeref) calculated for consistent scaling
- [ ] Minimum size ensures visibility (sizemin >= 4)
- [ ] Size legend or annotation included for bubble charts

### Shape Encoding
- [ ] Maximum 6 distinct shapes
- [ ] Shape paired with color for visibility
- [ ] Filled shapes used (not open) for small markers

### Line Encoding
- [ ] Solid lines for primary data
- [ ] Dashed lines for projections/forecasts
- [ ] Line width creates appropriate emphasis hierarchy
- [ ] Maximum 4 distinct line styles

### Opacity Encoding
- [ ] Opacity reduced for dense scatter plots
- [ ] Secondary/context series de-emphasized with lower opacity
- [ ] Fill colors have appropriate transparency (0.2-0.4)

### General
- [ ] No more than 4 encoding channels used simultaneously
- [ ] Most important data uses most effective encoding
- [ ] Encodings are distinguishable (not too similar)
- [ ] Legends/annotations explain non-obvious encodings

---

## Encoding Anti-Patterns

### Do NOT

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| Rainbow color scale for sequential data | Perceptually non-uniform | Use single-hue sequential scale |
| >8 color categories | Indistinguishable | Aggregate or use small multiples |
| Radius for bubble size | Misleading perception | Use area (sizemode='area') |
| Shape without color | Poor visibility | Always pair shape with color |
| Decorative color | Visual noise | Only encode data with color |
| 5+ simultaneous encodings | Cognitive overload | Simplify or split into multiple charts |
| Same color, different meaning | Confusion | Maintain semantic consistency |

### Common Mistakes

```python
# BAD: Using radius instead of area
marker=dict(
    size=df['value'],
    sizemode='diameter'  # Wrong! Misleading visual comparison
)

# GOOD: Using area
marker=dict(
    size=df['value'],
    sizemode='area',
    sizeref=2. * max(df['value']) / (40.**2)
)

# BAD: Too many colors
for i, cat in enumerate(df['category'].unique()):  # Could be 20+ categories
    fig.add_trace(go.Scatter(..., line=dict(color=colorway[i % len(colorway)])))

# GOOD: Aggregate to manageable number
top_categories = df.groupby('category')['value'].sum().nlargest(5).index
df['category_display'] = df['category'].where(
    df['category'].isin(top_categories), 'Other'
)

# BAD: Decorative gradient that doesn't encode data
marker=dict(
    color=list(range(len(df))),  # Just creates a rainbow effect
    colorscale='Rainbow'
)

# GOOD: Color encodes meaningful data
marker=dict(
    color=df['profit'],  # Actual data dimension
    colorscale=[[0, '#F87171'], [0.5, '#0e1729'], [1, '#34D399']],
    colorbar=dict(title='Profit ($M)')
)
```

---

## Template Reference

Standard colors for encoding:

```python
# Background and text
BACKGROUND = '#0e1729'
TEXT_COLOR = '#d3d4d6'

# Categorical colorway
COLORWAY = ["#60A5FA", "#F87171", "#34D399", "#FBBF24", "#E879F9"]

# Sequential scale (single hue)
SEQUENTIAL_SCALE = [
    [0, '#0e1729'],
    [0.5, '#3B82F6'],
    [1, '#60A5FA']
]

# Diverging scale (red-neutral-green)
DIVERGING_SCALE = [
    [0, '#F87171'],
    [0.5, '#0e1729'],
    [1, '#34D399']
]

# Opacity levels
OPACITY_PRIMARY = 1.0
OPACITY_SECONDARY = 0.5
OPACITY_BACKGROUND = 0.2
OPACITY_FILL = 0.3

# Line widths
LINE_WIDTH_EMPHASIS = 3
LINE_WIDTH_PRIMARY = 2
LINE_WIDTH_SECONDARY = 1

# Marker sizes
MARKER_SIZE_DEFAULT = 8
MARKER_SIZE_SMALL = 6
MARKER_SIZE_LARGE = 12
BUBBLE_SIZE_MIN = 4
BUBBLE_SIZE_MAX = 40
```

## Data Preparation for Stacked Charts

For stacked bar charts, ensure data arrays are properly aligned with category order before creating traces. Create aligned arrays by iterating through the category order and looking up values, filling with 0 for missing combinations. Example: `for category in category_order: value = data[data['cat']==category]['val'].values[0] if len(data[data['cat']==category]) > 0 else 0`
