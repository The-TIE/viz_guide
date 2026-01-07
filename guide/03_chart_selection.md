# 03 - Chart Type Selection

> Given data characteristics and visualization intent, select the optimal chart type.
> This section provides decision rules for AI-assisted chart generation.

---

## Decision Framework

### Step 1: Identify Primary Data Type

```
DATA_TYPE:
  - TIME_SERIES: Data indexed by datetime (prices, volumes, metrics over time)
  - CATEGORICAL: Data grouped by discrete categories (exchanges, coins, regions)
  - NUMERICAL: Continuous numeric values (correlations, scores, measurements)
  - HIERARCHICAL: Nested/tree-structured data (category > subcategory)
  - MATRIX: Two categorical dimensions with numeric values (exchange x expiry)
```

### Step 2: Identify Visualization Intent

```
INTENT:
  - TREND: Show change over time (single or multiple series)
  - COMPARISON: Compare values across categories
  - COMPOSITION: Show parts of a whole
  - DISTRIBUTION: Show spread/frequency of values
  - RELATIONSHIP: Show correlation between variables
  - RANKING: Order items by magnitude
```

### Step 3: Count Variables/Series

```
SERIES_COUNT:
  - SINGLE: One data series
  - FEW: 2-5 series (can overlay on single chart)
  - MANY: 6-12 series (consider small multiples)
  - EXCESSIVE: 12+ series (must aggregate or filter)
```

---

## Decision Tree

### Time Series Data

```
TIME_SERIES
├── TREND intent
│   ├── SINGLE series → Line Chart
│   ├── FEW series (2-5) → Multi-line Chart
│   └── MANY series (6+) → Small Multiples
│
├── COMPARISON intent (series vs series)
│   ├── Same scale → Multi-line Chart
│   ├── Different scales → Dual-axis Line Chart
│   └── Comparing returns → Normalize to percentage returns (see below)
│
├── COMPOSITION intent (parts of whole over time)
│   ├── Few categories (2-5) → Stacked Area Chart
│   └── Many categories → Stacked Bar Chart (aggregated)
│
├── TREND + MAGNITUDE (e.g., price + volume)
│   └── → Dual-axis: Line + Bar Combo
│
└── DISTRIBUTION over time
    └── → Small Multiples (histogram per period)
```

### Categorical Data

```
CATEGORICAL
├── COMPARISON intent
│   ├── Single measure
│   │   ├── Few categories (≤10) → Horizontal Bar Chart
│   │   └── Many categories (10+) → Horizontal Bar (top N + "Other")
│   │
│   └── Multiple measures
│       ├── 2-3 measures → Grouped Bar Chart
│       └── 4+ measures → Small Multiples (one bar chart per measure)
│
├── RANKING intent
│   └── → Horizontal Bar Chart (sorted descending)
│
└── COMPOSITION intent
    ├── Few categories (≤6), single period → Donut Chart (NOT pie)
    └── Many categories or multiple periods → Stacked Bar Chart
```

### Numerical Data

```
NUMERICAL
├── RELATIONSHIP intent (2 variables)
│   └── → Scatter Plot
│
├── RELATIONSHIP + third variable
│   ├── Third is numeric → Bubble Chart (size encoding)
│   └── Third is categorical → Scatter with Color Encoding
│
├── DISTRIBUTION intent
│   ├── Single variable → Histogram
│   └── Compare distributions → Box Plot or Violin Plot
│
└── CORRELATION matrix
    └── → Heatmap
```

### Hierarchical Data

```
HIERARCHICAL
├── Show structure + magnitude
│   ├── 2 levels → Treemap
│   └── 3+ levels → Sunburst
│
└── Show structure only
    └── → Tree Diagram (not common in financial dashboards)
```

### Matrix Data (Two Categorical Dimensions)

```
MATRIX
├── Dense values across grid
│   └── → Heatmap with color scale
│
└── Sparse values, need exact numbers
    └── → Data Table with Conditional Formatting
```

---

## Chart Type Reference

### Line Charts

#### Single Line Chart

| Attribute | Value |
|-----------|-------|
| **Use when** | Single time series, showing trend over time |
| **Do NOT use when** | Categorical data, comparing unrelated series |
| **Plotly trace** | `go.Scatter(mode='lines')` |
| **Data types** | x: datetime, y: numeric |
| **Max data points** | 10,000 (use WebGL beyond) |

**Key Configuration:**

```python
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['value'],
    mode='lines',
    name='Series Name',
    line=dict(
        width=2,
        color='#60A5FA'  # Primary blue from colorway
    ),
    hovertemplate='%{x|%b %d, %Y}<br>%{y:,.2f}<extra></extra>'
))

fig.update_layout(
    template=get_template_tie(),
    hovermode='x unified',
    xaxis=dict(
        type='date',
        tickformat='%b %Y'
    ),
    yaxis=dict(
        tickformat=',.2s'  # SI notation: 1.23M
    )
)
```

#### Multi-Line Chart

| Attribute | Value |
|-----------|-------|
| **Use when** | 2-5 related time series on same scale |
| **Do NOT use when** | Series have vastly different scales, >5 series |
| **Plotly trace** | Multiple `go.Scatter(mode='lines')` |
| **Legend required** | Yes |
| **Max series** | 5 (beyond: use small multiples) |

**Key Configuration:**

```python
colorway = ["#60A5FA", "#F87171", "#34D399", "#FBBF24", "#E879F9"]

fig = go.Figure()
for i, series_name in enumerate(series_list):
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[series_name],
        mode='lines',
        name=series_name,
        line=dict(width=2, color=colorway[i])
    ))

fig.update_layout(
    template=get_template_tie(),
    hovermode='x unified',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='left',
        x=0
    )
)
```

#### Comparing Returns Across Assets

When comparing multiple assets with different price levels (e.g., BTC at $45,000 vs SOL at $100), **never plot raw prices** on the same axis. The differences in magnitude make comparison meaningless.

**Normalization options:**

1. **Percentage returns** - Best for comparing relative performance
2. **Indexed to 100** - Start all series at 100, show cumulative performance

```python
# Option 1: Percentage returns (daily)
df['BTC_return'] = df['BTC_price'].pct_change() * 100
df['SOL_return'] = df['SOL_price'].pct_change() * 100

# Option 2: Index to 100 (cumulative performance)
df['BTC_indexed'] = (df['BTC_price'] / df['BTC_price'].iloc[0]) * 100
df['SOL_indexed'] = (df['SOL_price'] / df['SOL_price'].iloc[0]) * 100

# Plot indexed values
fig = go.Figure()
for asset, color in [('BTC_indexed', '#F7931A'), ('SOL_indexed', '#00FFA3')]:
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[asset],
        name=asset.replace('_indexed', ''),
        line=dict(color=color, width=2)
    ))

fig.update_layout(
    yaxis=dict(title_text='Indexed Value (Start = 100)'),
    hovermode='x unified'
)
```

**Decision rule:**
- Short timeframe (days): Use percentage returns
- Long timeframe (weeks/months): Use indexed to 100 for cleaner visualization

---

### Area Charts

#### Stacked Area Chart

| Attribute | Value |
|-----------|-------|
| **Use when** | Show composition over time, parts sum to total |
| **Do NOT use when** | Parts don't sum meaningfully, series overlap significantly |
| **Plotly trace** | `go.Scatter(fill='tonexty', stackgroup='one')` |
| **Max categories** | 6 (beyond: aggregate to "Other") |

**Key Configuration:**

```python
fig = go.Figure()

# Order matters: bottom to top
categories = ['Cat A', 'Cat B', 'Cat C']
colorway = ["#60A5FA", "#F87171", "#34D399"]

for i, cat in enumerate(categories):
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[cat],
        mode='lines',
        name=cat,
        stackgroup='one',  # Creates stacking
        line=dict(width=0.5, color=colorway[i]),
        fillcolor=colorway[i]
    ))

fig.update_layout(
    template=get_template_tie(),
    hovermode='x unified'
)
```

#### Filled Area Chart (Non-Stacked)

| Attribute | Value |
|-----------|-------|
| **Use when** | Single series, emphasize magnitude from zero |
| **Do NOT use when** | Y-axis doesn't start at zero, multiple overlapping series |
| **Plotly trace** | `go.Scatter(fill='tozeroy')` |

```python
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['value'],
    mode='lines',
    fill='tozeroy',
    line=dict(width=2, color='#60A5FA'),
    fillcolor='rgba(96, 165, 250, 0.3)'  # 30% opacity
))
```

---

### Bar Charts

#### Horizontal Bar Chart

| Attribute | Value |
|-----------|-------|
| **Use when** | Comparing categories, ranking, long category labels |
| **Do NOT use when** | Time series, showing trend |
| **Plotly trace** | `go.Bar(orientation='h')` |
| **Category order** | Sort by value (descending for rankings) |
| **Max categories** | 15 (beyond: show top N + "Other") |

**Bar Color Rules:**
- **Single metric comparison**: Use ONE color for all bars (e.g., all exchanges' volumes)
- **Multiple categories**: Use different colors only when bars represent different semantic groups
- **Do NOT** use rainbow colors just to make bars visually distinct - this implies false categorical differences

**Text Label Positioning:**
- `textposition='inside'`: Use when bars are long enough to fit labels (preferred for cleaner look)
- `textposition='outside'`: Use when bars are short or labels would be cramped inside
- Rule of thumb: If label width < 70% of bar width, place inside

**Key Configuration:**

```python
# Sort descending for ranking
df_sorted = df.sort_values('value', ascending=True)  # ascending=True for horizontal

fig = go.Figure()
fig.add_trace(go.Bar(
    y=df_sorted['category'],
    x=df_sorted['value'],
    orientation='h',
    marker=dict(color='#60A5FA'),  # Single color for same-metric comparison
    text=df_sorted['value'].apply(lambda x: f'{x:,.0f}'),
    textposition='inside',  # Cleaner look when bars are long enough
    insidetextanchor='end',  # Align text to right edge inside bar
    textfont=dict(color='white'),  # White text on colored bars
    hovertemplate='%{y}: %{x:,.2f}<extra></extra>'
))

fig.update_layout(
    template=get_template_tie(),
    yaxis=dict(categoryorder='total ascending'),  # Largest at top
    bargap=0.2,
    uniformtext=dict(minsize=10, mode='hide')  # Hide text if too small
)
```

#### Vertical Bar Chart

| Attribute | Value |
|-----------|-------|
| **Use when** | Time-based categories (months, quarters), few categories |
| **Do NOT use when** | Many categories (>10), long labels |
| **Plotly trace** | `go.Bar()` |

```python
fig.add_trace(go.Bar(
    x=df['period'],
    y=df['value'],
    marker=dict(color='#60A5FA'),
    hovertemplate='%{x}: %{y:,.2f}<extra></extra>'
))
```

#### Grouped Bar Chart

| Attribute | Value |
|-----------|-------|
| **Use when** | Comparing 2-3 measures across categories |
| **Do NOT use when** | >3 measures, many categories |
| **Plotly trace** | Multiple `go.Bar()` with `barmode='group'` |

**Key Configuration:**

```python
measures = ['Measure A', 'Measure B']
colorway = ["#60A5FA", "#F87171"]

fig = go.Figure()
for i, measure in enumerate(measures):
    fig.add_trace(go.Bar(
        x=df['category'],
        y=df[measure],
        name=measure,
        marker=dict(color=colorway[i])
    ))

fig.update_layout(
    template=get_template_tie(),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)
```

#### Stacked Bar Chart

| Attribute | Value |
|-----------|-------|
| **Use when** | Composition across categories, parts sum to whole |
| **Do NOT use when** | Comparing individual components is primary goal |
| **Plotly trace** | Multiple `go.Bar()` with `barmode='stack'` |

```python
fig.update_layout(
    barmode='stack',
    template=get_template_tie()
)
```

---

### Dual-Axis Charts

For comparing both discrete (hourly) and cumulative metrics, use dual-axis charts with `make_subplots(specs=[[{'secondary_y': True}]])`. Place bar charts on the primary y-axis for discrete values and line charts on the secondary y-axis for cumulative values. This allows direct comparison of both patterns simultaneously.

#### Line + Bar Combination

| Attribute | Value |
|-----------|-------|
| **Use when** | Two related metrics with different scales (price + volume, value + count) |
| **Do NOT use when** | Metrics can share a scale, >2 metrics |
| **Plotly trace** | `go.Bar()` + `go.Scatter()` with `yaxis='y2'` |
| **Axis alignment** | Align zeros if possible |

**Key Configuration:**

```python
fig = go.Figure()

# Bar on primary axis (typically volume/count)
fig.add_trace(go.Bar(
    x=df['date'],
    y=df['volume'],
    name='Volume',
    marker=dict(color='rgba(96, 165, 250, 0.5)'),  # Semi-transparent
    yaxis='y'
))

# Line on secondary axis (typically price/rate)
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['price'],
    name='Price',
    mode='lines',
    line=dict(width=2, color='#F87171'),
    yaxis='y2'
))

fig.update_layout(
    template=get_template_tie(),
    yaxis=dict(
        title='Volume',
        tickformat=',.2s',
        side='left'
    ),
    yaxis2=dict(
        title='Price',
        tickformat='$,.2f',
        side='right',
        overlaying='y',
        showgrid=False  # Avoid grid overlap
    ),
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='left',
        x=0
    ),
    hovermode='x unified'
)
```

#### Long/Short Liquidations Pattern

Common in derivatives dashboards: Green bars (up) for longs, red bars (down) for shorts, line overlay for price.

```python
# Long liquidations (positive, green)
fig.add_trace(go.Bar(
    x=df['date'],
    y=df['long_liquidations'],
    name='Long Liquidations',
    marker=dict(color='#34D399')  # Green
))

# Short liquidations (negative, red)
fig.add_trace(go.Bar(
    x=df['date'],
    y=-df['short_liquidations'],  # Negative to go downward
    name='Short Liquidations',
    marker=dict(color='#F87171')  # Red
))

fig.update_layout(barmode='relative')  # Allows positive/negative stacking
```

---

### Small Multiples / Subplots

#### When to Use Small Multiples

| Condition | Recommendation |
|-----------|----------------|
| 6+ series that would clutter single chart | Use small multiples |
| Same metric across categories | Use small multiples |
| Long/short breakdown by category | Use small multiples |
| Percentile series by category | Use small multiples |

| Attribute | Value |
|-----------|-------|
| **Use when** | Many series (6+), comparing patterns across categories |
| **Do NOT use when** | <5 series (use multi-line), need to compare exact values |
| **Plotly method** | `make_subplots()` |
| **Shared axes** | Yes, for comparability |

**Key Configuration:**

```python
from plotly.subplots import make_subplots

categories = ['Cat A', 'Cat B', 'Cat C', 'Cat D', 'Cat E']
num_cols = 2
num_rows = (len(categories) + num_cols - 1) // num_cols

fig = make_subplots(
    rows=num_rows,
    cols=num_cols,
    shared_xaxes=True,
    shared_yaxes=True,
    subplot_titles=categories,
    vertical_spacing=0.08,
    horizontal_spacing=0.05
)

for i, cat in enumerate(categories):
    row = (i // num_cols) + 1
    col = (i % num_cols) + 1

    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df[cat],
            mode='lines',
            name=cat,
            line=dict(color='#60A5FA'),
            showlegend=False
        ),
        row=row,
        col=col
    )

fig.update_layout(
    template=get_template_tie(),
    height=300 * num_rows,  # Scale height with rows
    showlegend=False
)
```

#### Helper Function for Row/Column Calculation

```python
def get_row_col(index, num_cols):
    """Calculate row and column position (1-based) for subplot grid."""
    row = (index // num_cols) + 1
    col = (index % num_cols) + 1
    return (row, col)
```

---

### Data Tables with Conditional Formatting

| Attribute | Value |
|-----------|-------|
| **Use when** | Exact values needed, matrix data, summary statistics |
| **Do NOT use when** | Trends are primary focus, large datasets |
| **Plotly trace** | `go.Table()` |

**Key Configuration:**

```python
def get_cell_colors(values, positive_color='#34D399', negative_color='#F87171', neutral_color='#0e1729'):
    """Return colors based on value sign."""
    colors = []
    for v in values:
        if v > 0:
            colors.append(f'rgba(52, 211, 153, 0.3)')  # Green, 30% opacity
        elif v < 0:
            colors.append(f'rgba(248, 113, 113, 0.3)')  # Red, 30% opacity
        else:
            colors.append(neutral_color)
    return colors

fig = go.Figure()
fig.add_trace(go.Table(
    header=dict(
        values=['Asset', 'Change 1D', 'Change 7D', 'Change 30D'],
        fill_color='#1e293b',
        font=dict(color='#d3d4d6', size=12),
        align='left',
        height=30
    ),
    cells=dict(
        values=[
            df['asset'],
            df['change_1d'].apply(lambda x: f'{x:+.2%}'),
            df['change_7d'].apply(lambda x: f'{x:+.2%}'),
            df['change_30d'].apply(lambda x: f'{x:+.2%}')
        ],
        fill_color=[
            '#0e1729',  # Asset column (no conditional formatting)
            get_cell_colors(df['change_1d']),
            get_cell_colors(df['change_7d']),
            get_cell_colors(df['change_30d'])
        ],
        font=dict(color='#d3d4d6', size=11),
        align='left',
        height=25
    )
))

fig.update_layout(
    template=get_template_tie(),
    margin=dict(l=0, r=0, t=30, b=0)
)
```

---

### Scatter Plots

| Attribute | Value |
|-----------|-------|
| **Use when** | Two numeric variables, exploring correlation/relationship |
| **Do NOT use when** | Time series (use line), categorical comparison |
| **Plotly trace** | `go.Scatter(mode='markers')` |
| **Max points** | 5,000 (use `go.Scattergl` beyond) |

**Key Configuration:**

```python
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['variable_x'],
    y=df['variable_y'],
    mode='markers',
    marker=dict(
        size=8,
        color='#60A5FA',
        opacity=0.7,
        line=dict(width=1, color='#1e293b')
    ),
    hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
))

fig.update_layout(
    template=get_template_tie(),
    xaxis=dict(title='Variable X'),
    yaxis=dict(title='Variable Y')
)
```

#### With Color Encoding (Third Variable - Categorical)

```python
categories = df['category'].unique()
colorway = ["#60A5FA", "#F87171", "#34D399", "#FBBF24"]

for i, cat in enumerate(categories):
    mask = df['category'] == cat
    fig.add_trace(go.Scatter(
        x=df.loc[mask, 'x'],
        y=df.loc[mask, 'y'],
        mode='markers',
        name=cat,
        marker=dict(size=8, color=colorway[i], opacity=0.7)
    ))
```

#### With Size Encoding (Third Variable - Numeric / Bubble Chart)

```python
fig.add_trace(go.Scatter(
    x=df['x'],
    y=df['y'],
    mode='markers',
    marker=dict(
        size=df['magnitude'],  # Numeric column for size
        sizemode='area',
        sizeref=2. * max(df['magnitude']) / (40.**2),
        sizemin=4,
        color='#60A5FA',
        opacity=0.7
    )
))
```

---

### Heatmaps

| Attribute | Value |
|-----------|-------|
| **Use when** | Matrix data, correlation matrices, dense grids |
| **Do NOT use when** | Sparse data, exact values critical |
| **Plotly trace** | `go.Heatmap()` |
| **Color scale** | Diverging for +/- values, sequential for magnitude only |

**Key Configuration:**

```python
# For correlation/diverging data (-1 to 1)
fig = go.Figure()
fig.add_trace(go.Heatmap(
    x=columns,
    y=rows,
    z=matrix_values,
    colorscale=[
        [0, '#F87171'],      # Negative: red
        [0.5, '#0e1729'],    # Zero: background
        [1, '#34D399']       # Positive: green
    ],
    zmid=0,
    text=matrix_values,
    texttemplate='%{text:.2f}',
    textfont=dict(size=10, color='#d3d4d6'),
    hovertemplate='%{x} vs %{y}: %{z:.3f}<extra></extra>'
))

fig.update_layout(
    template=get_template_tie(),
    xaxis=dict(side='top'),  # Labels on top
    yaxis=dict(autorange='reversed')  # First row at top
)
```

#### For Sequential Data (Magnitude Only)

```python
fig.add_trace(go.Heatmap(
    z=values,
    colorscale=[
        [0, '#0e1729'],      # Low: background
        [0.5, '#3B82F6'],    # Mid: blue
        [1, '#60A5FA']       # High: bright blue
    ],
    showscale=True,
    colorbar=dict(
        title='Value',
        tickformat=',.0f'
    )
))
```

#### Heatmap Cell Sizing

**Use constant-width cells** for better visual appearance. Variable-width cells (e.g., when x-axis represents non-uniform time intervals) can look unbalanced.

```python
# For uniform cell widths when x-values are non-uniform (like expiry dates):
# Option 1: Use categorical x-axis instead of numeric
fig.add_trace(go.Heatmap(
    x=['1W', '2W', '1M', '3M', '6M'],  # Categorical labels, not actual dates
    y=strikes,
    z=iv_matrix,
    ...
))

# Option 2: Use xgap/ygap for consistent spacing
fig.add_trace(go.Heatmap(
    x=x_values,
    y=y_values,
    z=values,
    xgap=2,  # Gap between cells in pixels
    ygap=2,
    ...
))
```

**When variable widths are acceptable:**
- When the width itself conveys information (e.g., time duration)
- When explicitly noted in the chart title/subtitle

---

## Quick Reference Matrix

### By Intent

| Intent | Primary Choice | Alternative |
|--------|---------------|-------------|
| **Trend** (single) | Line chart | Area chart |
| **Trend** (multiple) | Multi-line chart | Small multiples |
| **Comparison** (categories) | Horizontal bar | Grouped bar |
| **Comparison** (time series) | Multi-line | Dual-axis |
| **Composition** (static) | Donut chart | Stacked bar |
| **Composition** (over time) | Stacked area | Stacked bar |
| **Distribution** | Histogram | Box plot |
| **Relationship** | Scatter plot | Heatmap |
| **Ranking** | Horizontal bar (sorted) | - |

### By Data Characteristics

| Characteristic | Recommendation |
|----------------|----------------|
| >5 series, same metric | Small multiples |
| 2 series, different scales | Dual-axis chart |
| Long category labels | Horizontal bar (not vertical) |
| Exact values required | Data table |
| Dense matrix data | Heatmap |
| >10,000 points | Use WebGL (`scattergl`, etc.) |
| Positive/negative values | Use diverging colors |

### Series Count Guidelines

| Series Count | Chart Type | Notes |
|--------------|-----------|-------|
| 1 | Single line/bar | No legend needed |
| 2-5 | Multi-line or grouped | Legend required |
| 6-12 | Small multiples | One chart per series |
| 12+ | Aggregate or filter | Too many for any chart |

---

### Dot Plots

A cleaner alternative to bar charts, especially when comparing many categories.

| Attribute | Value |
|-----------|-------|
| **Use when** | Comparing values across many categories (10+), reducing visual clutter |
| **Do NOT use when** | Few categories (<5), time series data |
| **Plotly trace** | `go.Scatter(mode='markers')` with categorical y-axis |
| **Key advantage** | Less ink, easier to compare, works well with long labels |

**When to prefer dot plots over bar charts:**
- Many categories (10+) where bars become cluttered
- Need to show multiple measures per category
- Want cleaner, publication-ready aesthetics
- Horizontal orientation with long category labels

**Key Configuration:**

```python
fig = go.Figure()

# Sort by value for ranking
df_sorted = df.sort_values('value', ascending=True)

fig.add_trace(go.Scatter(
    x=df_sorted['value'],
    y=df_sorted['category'],
    mode='markers',
    marker=dict(
        size=10,
        color='#60A5FA',
        line=dict(width=1, color='#1e293b')
    ),
    hovertemplate='%{y}: %{x:,.2f}<extra></extra>'
))

fig.update_layout(
    template=get_template_tie(),
    yaxis=dict(categoryorder='total ascending'),
    xaxis=dict(tickformat=',.2s'),
    height=max(400, len(df) * 25)  # Scale height with categories
)
```

#### Connected Dot Plot (Comparing Two Values)

For comparing two measures per category (e.g., before/after, actual/target):

```python
fig = go.Figure()

for i, row in df.iterrows():
    # Connecting line
    fig.add_trace(go.Scatter(
        x=[row['value_1'], row['value_2']],
        y=[row['category'], row['category']],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.3)', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))

# First measure dots
fig.add_trace(go.Scatter(
    x=df['value_1'],
    y=df['category'],
    mode='markers',
    name='Before',
    marker=dict(size=10, color='#F87171')
))

# Second measure dots
fig.add_trace(go.Scatter(
    x=df['value_2'],
    y=df['category'],
    mode='markers',
    name='After',
    marker=dict(size=10, color='#34D399')
))
```

---

### Slope Charts

Show change between exactly two time points or conditions. More effective than grouped bars for showing direction and magnitude of change.

| Attribute | Value |
|-----------|-------|
| **Use when** | Comparing exactly 2 time points, showing direction of change |
| **Do NOT use when** | >2 time points (use line chart), single point in time |
| **Plotly trace** | `go.Scatter(mode='lines+markers')` |
| **Key advantage** | Clear direction indication, compact comparison |

**Key Configuration:**

```python
fig = go.Figure()

categories = df['category'].unique()
x_positions = [0, 1]  # Two time points
x_labels = ['2023', '2024']

for cat in categories:
    cat_data = df[df['category'] == cat]
    y_values = [cat_data['value_start'].iloc[0], cat_data['value_end'].iloc[0]]

    # Determine color based on direction
    color = '#34D399' if y_values[1] > y_values[0] else '#F87171'

    fig.add_trace(go.Scatter(
        x=x_positions,
        y=y_values,
        mode='lines+markers+text',
        name=cat,
        line=dict(width=2, color=color),
        marker=dict(size=8),
        text=[cat, f"{y_values[1]:,.0f}"],  # Label at start, value at end
        textposition=['middle left', 'middle right'],
        textfont=dict(size=10, color='#d3d4d6'),
        hovertemplate=f'{cat}<br>%{{x}}: %{{y:,.2f}}<extra></extra>'
    ))

fig.update_layout(
    template=get_template_tie(),
    xaxis=dict(
        tickmode='array',
        tickvals=x_positions,
        ticktext=x_labels,
        range=[-0.5, 1.5]  # Padding for labels
    ),
    showlegend=False,
    margin=dict(l=100, r=100)  # Room for text labels
)
```

---

## Anti-Patterns (What NOT to Do)

### Never Use Pie Charts

Use donut charts instead. Pie charts waste center space and are harder to read.

```python
# BAD: Pie chart
go.Pie(values=values, labels=labels)

# GOOD: Donut chart
go.Pie(values=values, labels=labels, hole=0.4)
```

### Never Use 3D Charts

**3D charts should be avoided unless absolutely necessary.** They obscure data, introduce perception errors, and rarely add information value over 2D alternatives.

Common cases where 3D is requested but 2D is better:

| 3D Request | Better 2D Alternative |
|------------|----------------------|
| Options IV surface | Heatmap (strikes × expiries) |
| 3D scatter plot | 2D scatter with color/size encoding |
| 3D bar chart | Grouped or stacked 2D bars |
| Surface plot | Contour plot or heatmap |

**Example: Implied Volatility Surface**

```python
# BAD: 3D surface is harder to read and has API quirks
go.Surface(x=strikes, y=expiries, z=iv_matrix)

# GOOD: 2D heatmap is cleaner and more readable
go.Heatmap(
    x=strikes,
    y=expiries,
    z=iv_matrix,
    colorscale=[[0, '#1E40AF'], [0.5, '#60A5FA'], [1, '#F87171']],
    hovertemplate='Strike: %{x}<br>Expiry: %{y}<br>IV: %{z:.1f}%<extra></extra>'
)
```

**Why avoid 3D:**
- Perspective distorts relative values
- Occluded data points are hidden
- Harder to read precise values
- 3D scene axis properties differ from 2D (causes errors)
- Rotation/interaction adds complexity without insight

### Avoid Dual Y-Axes When Possible

Only use when:
1. Two metrics are genuinely related (price + volume)
2. Scales are truly incompatible
3. There are exactly 2 series

```python
# Only acceptable dual-axis use cases:
# - Price + Volume
# - Value + Count
# - Rate + Level
```

### Don't Overload Small Multiples

Maximum 3x4 = 12 subplots. Beyond that, reconsider data selection.

### Avoid Stacked Area for Overlapping Trends

If series cross each other frequently, use multi-line instead.

---

## Template Integration

All charts should use the standard template:

```python
from token_labs.visualization.plotly import get_template_tie, colorway

fig.update_layout(
    template=get_template_tie(),
    # Additional layout options
)
```

### Standard Colors Reference

| Index | Color | Hex | Common Use |
|-------|-------|-----|------------|
| 0 | Blue | #60A5FA | Primary series |
| 1 | Red | #F87171 | Secondary / negative |
| 2 | Green | #34D399 | Positive / tertiary |
| 3 | Yellow | #FBBF24 | Highlight / warning |
| 4 | Purple | #E879F9 | Additional series |

### Background Colors

| Element | Color |
|---------|-------|
| Paper background | #0e1729 |
| Plot background | #0e1729 |
| Text color | #d3d4d6 |
| Grid (subtle) | rgba(255, 255, 255, 0.1) |

---

## Decision Checklist

Before finalizing chart type, verify:

- [ ] Chart type matches data type (time series -> line, categorical -> bar)
- [ ] Chart type matches intent (trend -> line, comparison -> bar)
- [ ] Series count is manageable (<= 5 for overlay, else small multiples)
- [ ] Labels will be readable (long labels -> horizontal bar)
- [ ] Color encoding is meaningful (not arbitrary)
- [ ] Dual axes are truly necessary (different scales, related metrics)
- [ ] No 3D or pie charts
- [ ] Template colors applied consistently

## Time-Based Patterns

For hourly or time-of-day patterns showing average values across multiple days, prefer bar charts over line charts. Bar charts better emphasize discrete hourly buckets and make it clearer that values represent aggregated averages rather than continuous time series. Use `go.Bar()` instead of `go.Scatter()` for this pattern.

## Time Series Visualization

For composition over time (showing how parts make up a whole), prefer stacked bar charts over line charts. Stacked bars clearly show both individual contributions and total volume at each time point. Use `barmode='stack'` in the layout configuration.

## Horizontal Stacked Bar Charts

For horizontal stacked bar charts comparing categories with multiple subcategories, use `orientation='h'` with `barmode='stack'`. Set `hovermode='y unified'` to show all stack components together. Sort the primary category axis in ascending order so the largest totals appear at the top of the chart (visual hierarchy principle).
