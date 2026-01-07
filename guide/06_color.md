# 06 - Color

> Rules and patterns for color selection in Plotly visualizations.
> Effective color usage improves readability, supports data interpretation, and ensures accessibility.

---

## Template Color Reference

### Base Colors

| Element | Color | Hex |
|---------|-------|-----|
| Background | Dark navy | #0e1729 |
| Text | Light gray | #d3d4d6 |
| Grid (subtle) | White 10% | rgba(255, 255, 255, 0.1) |
| Axis lines | White 20% | rgba(255, 255, 255, 0.2) |

### Standard Colorway

The default categorical palette for multi-series charts:

```python
COLORWAY = [
    "#60A5FA",  # Blue (primary)
    "#F87171",  # Red
    "#34D399",  # Green
    "#FBBF24",  # Yellow/Amber
    "#E879F9",  # Pink/Fuchsia
    "#818CF8",  # Indigo
    "#FB923C",  # Orange
    "#22D3EE",  # Cyan
    "#A3E635",  # Lime
    "#F472B6",  # Rose
]
```

### Semantic Colors

| Purpose | Color | Hex | Usage |
|---------|-------|-----|-------|
| Positive/Up | Green | #34D399 | Gains, increases, success |
| Negative/Down | Red | #F87171 | Losses, decreases, errors |
| Neutral | Gray | #9CA3AF | Zero change, baseline |
| Primary emphasis | Blue | #60A5FA | Highlighted series, main focus |
| Secondary | Purple | #818CF8 | Secondary focus, related data |

---

## Categorical Palettes

### When to Use

Use categorical (qualitative) palettes when:

- Differentiating discrete categories with no inherent order
- Showing multiple series in line, bar, or scatter charts
- Each category is equally important (no ranking)

**Examples:**
- Different cryptocurrencies (BTC, ETH, SOL)
- Exchange comparisons (Binance, Coinbase, Kraken)
- Product categories, regions, teams

### Decision Table: Category Count

| Categories | Recommendation | Notes |
|------------|----------------|-------|
| 1 | Use primary color (#60A5FA) | Single series needs no distinction |
| 2-3 | First 2-3 colors from colorway | High contrast, easily distinguishable |
| 4-6 | First 4-6 colors from colorway | Optimal range for categorical palettes |
| 7-10 | Full colorway | Maximum before confusion |
| 11+ | Consider grouping/filtering | Too many categories to distinguish |

### Implementation

```python
import plotly.graph_objects as go

# Standard colorway configuration
COLORWAY = [
    "#60A5FA", "#F87171", "#34D399", "#FBBF24", "#E879F9",
    "#818CF8", "#FB923C", "#22D3EE", "#A3E635", "#F472B6"
]

# Apply to figure
fig = go.Figure()
fig.update_layout(
    colorway=COLORWAY,
    paper_bgcolor='#0e1729',
    plot_bgcolor='#0e1729',
    font=dict(color='#d3d4d6')
)

# Colors are automatically assigned to traces in order
for i, name in enumerate(['BTC', 'ETH', 'SOL']):
    fig.add_trace(go.Scatter(
        x=dates,
        y=values[i],
        name=name,
        mode='lines'
        # Color automatically assigned from colorway
    ))
```

### Manual Color Assignment

When you need explicit control over colors:

```python
# Define color mapping for consistent colors across charts
ASSET_COLORS = {
    'BTC': '#60A5FA',   # Blue
    'ETH': '#818CF8',   # Indigo
    'SOL': '#34D399',   # Green
    'AVAX': '#F87171',  # Red
    'DOT': '#E879F9',   # Pink
}

# Apply explicit colors
for asset in assets:
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[asset],
        name=asset,
        mode='lines',
        line=dict(color=ASSET_COLORS.get(asset, '#60A5FA'))
    ))
```

### High-Contrast Subset for Small Category Counts

When you have only 2-4 categories, use maximally distinct colors:

```python
# 2 categories: blue and red (maximum contrast)
TWO_CATEGORY_COLORS = ['#60A5FA', '#F87171']

# 3 categories: blue, red, green
THREE_CATEGORY_COLORS = ['#60A5FA', '#F87171', '#34D399']

# 4 categories: blue, red, green, amber
FOUR_CATEGORY_COLORS = ['#60A5FA', '#F87171', '#34D399', '#FBBF24']
```

---

## Sequential Palettes

### When to Use

Use sequential (continuous) palettes when:

- Representing magnitude or intensity
- Values range from low to high
- Data has a natural ordering (more/less, high/low)

**Examples:**
- Volume (low to high)
- Intensity or concentration
- Frequency counts
- Heatmap values

### Recommended Sequential Scales

| Scale Name | Direction | Use Case |
|------------|-----------|----------|
| Blues | Light to dark | General magnitude (matches theme) |
| Viridis | Yellow to purple | Scientific data, colorblind-safe |
| Plasma | Yellow to purple | Alternative to Viridis |
| Inferno | Yellow to black | Heat/intensity data |
| Greys | Light to dark | Neutral, background data |

### Custom Sequential Scale (Theme-Matched)

```python
# Sequential scale matching the dark theme
# From transparent/dark to primary blue
SEQUENTIAL_BLUE = [
    [0.0, 'rgba(96, 165, 250, 0.1)'],   # Light/transparent
    [0.25, 'rgba(96, 165, 250, 0.3)'],
    [0.5, 'rgba(96, 165, 250, 0.5)'],
    [0.75, 'rgba(96, 165, 250, 0.75)'],
    [1.0, '#60A5FA']                     # Full blue
]

# Alternative: Multi-hue sequential
SEQUENTIAL_TEAL_BLUE = [
    [0.0, '#0e1729'],     # Background (near zero)
    [0.25, '#134e4a'],    # Dark teal
    [0.5, '#14b8a6'],     # Teal
    [0.75, '#38bdf8'],    # Sky blue
    [1.0, '#60A5FA']      # Primary blue
]
```

### Plotly Built-in Sequential Scales

```python
import plotly.express as px

# Recommended colorblind-safe sequential scales
RECOMMENDED_SEQUENTIAL = [
    'Viridis',    # Default, excellent for most uses
    'Plasma',     # Good contrast
    'Inferno',    # High contrast, avoid for accessibility
    'Blues',      # Single-hue, matches theme
    'Purples',    # Single-hue alternative
]

# Apply to heatmap
fig = px.imshow(
    data_matrix,
    color_continuous_scale='Viridis',
    labels=dict(color='Value')
)

# Or with Graph Objects
fig = go.Figure(data=go.Heatmap(
    z=data_matrix,
    colorscale='Viridis',
    colorbar=dict(title='Value')
))
```

### Sequential Color for Scatter/Bubble

```python
# Color points by a continuous variable
fig = go.Figure(data=go.Scatter(
    x=df['x'],
    y=df['y'],
    mode='markers',
    marker=dict(
        size=10,
        color=df['magnitude'],  # Continuous variable
        colorscale='Viridis',
        colorbar=dict(
            title='Magnitude',
            tickformat=',.2s'
        ),
        showscale=True
    )
))
```

---

## Diverging Palettes

### When to Use

Use diverging palettes when:

- Data has a meaningful midpoint (zero, average, threshold)
- Values can be positive or negative
- Showing deviation from a baseline
- Comparing above/below a reference

**Examples:**
- Price changes (positive/negative)
- Returns (gains/losses)
- Correlation matrices (-1 to +1)
- Deviation from average
- Above/below target performance

### Recommended Diverging Scales

| Scale | Negative | Neutral | Positive | Use Case |
|-------|----------|---------|----------|----------|
| RdBu | Red | White | Blue | General diverging |
| RdYlGn | Red | Yellow | Green | Performance metrics |
| Custom | #F87171 | #4B5563 | #34D399 | Theme-matched |

### Custom Diverging Scale (Theme-Matched)

```python
# Diverging scale: Red (negative) -> Gray (neutral) -> Green (positive)
DIVERGING_RG = [
    [0.0, '#F87171'],     # Red (negative)
    [0.5, '#4B5563'],     # Gray (neutral/zero)
    [1.0, '#34D399']      # Green (positive)
]

# With more granularity
DIVERGING_RG_DETAILED = [
    [0.0, '#F87171'],     # Strong negative (red)
    [0.25, '#FCA5A5'],    # Weak negative (light red)
    [0.5, '#4B5563'],     # Neutral (gray)
    [0.75, '#6EE7B7'],    # Weak positive (light green)
    [1.0, '#34D399']      # Strong positive (green)
]

# Blue-centered diverging (for non-financial data)
DIVERGING_BLUE = [
    [0.0, '#F87171'],     # Red (low)
    [0.5, '#0e1729'],     # Background/neutral
    [1.0, '#60A5FA']      # Blue (high)
]
```

### Setting Midpoint for Diverging Scales

```python
import plotly.graph_objects as go
import numpy as np

# Data with positive and negative values
values = np.array([-0.5, -0.2, 0.1, 0.3, -0.1, 0.4])

# Ensure colorscale is centered at zero
abs_max = max(abs(values.min()), abs(values.max()))

fig = go.Figure(data=go.Heatmap(
    z=values.reshape(2, 3),
    colorscale=DIVERGING_RG,
    zmid=0,  # Center at zero
    zmin=-abs_max,
    zmax=abs_max,
    colorbar=dict(title='Change')
))
```

### Correlation Matrix with Diverging Colors

```python
def create_correlation_heatmap(corr_matrix, labels):
    """
    Create a correlation matrix heatmap with diverging colors.

    Args:
        corr_matrix: 2D array of correlation values (-1 to 1)
        labels: List of row/column labels

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, '#F87171'],    # -1 (negative correlation)
            [0.5, '#4B5563'],    # 0 (no correlation)
            [1.0, '#34D399']     # +1 (positive correlation)
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title='Correlation',
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1.0', '-0.5', '0', '0.5', '1.0']
        ),
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        paper_bgcolor='#0e1729',
        plot_bgcolor='#0e1729',
        font=dict(color='#d3d4d6'),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')  # Diagonal from top-left
    )

    return fig
```

---

## Emphasis and De-emphasis

### Highlighting Specific Series

When one series needs attention while others provide context:

```python
def create_emphasis_chart(df, date_col, value_cols, highlight_col):
    """
    Create a chart with one emphasized series and others de-emphasized.

    Args:
        df: DataFrame with data
        date_col: Column name for x-axis
        value_cols: List of all columns to plot
        highlight_col: Column to emphasize

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    # De-emphasized series (plotted first, behind)
    for col in value_cols:
        if col != highlight_col:
            fig.add_trace(go.Scatter(
                x=df[date_col],
                y=df[col],
                name=col,
                mode='lines',
                line=dict(
                    color='rgba(156, 163, 175, 0.3)',  # Gray with low opacity
                    width=1
                ),
                hoverinfo='skip'  # Reduce hover clutter
            ))

    # Emphasized series (plotted last, on top)
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[highlight_col],
        name=highlight_col,
        mode='lines',
        line=dict(
            color='#60A5FA',  # Primary blue
            width=3           # Thicker line
        )
    ))

    fig.update_layout(
        paper_bgcolor='#0e1729',
        plot_bgcolor='#0e1729',
        font=dict(color='#d3d4d6')
    )

    return fig
```

### Emphasis Techniques

| Technique | Implementation | Use Case |
|-----------|----------------|----------|
| Color saturation | Full color vs gray | Highlight one of many |
| Opacity | 1.0 vs 0.2-0.3 | Background context |
| Line width | 3px vs 1px | Emphasize trend line |
| Marker size | Large vs small | Highlight data points |
| Z-order | Plot last (on top) | Ensure visibility |

### De-emphasis Patterns

```python
# De-emphasis color palette (grays with varying opacity)
DE_EMPHASIS_COLORS = [
    'rgba(156, 163, 175, 0.5)',  # Gray 50% opacity
    'rgba(156, 163, 175, 0.4)',
    'rgba(156, 163, 175, 0.3)',
    'rgba(156, 163, 175, 0.2)',
]

# Alternative: Single gray with varying opacity
def get_deemphasis_color(opacity=0.3):
    """Return gray color with specified opacity."""
    return f'rgba(156, 163, 175, {opacity})'

# For bar charts: lighter fill with subtle border
def get_deemphasis_bar_style():
    return dict(
        color='rgba(156, 163, 175, 0.2)',
        line=dict(color='rgba(156, 163, 175, 0.4)', width=1)
    )
```

### Focus + Context Pattern

```python
def create_focus_context_chart(df, all_series, focus_series, date_col='date'):
    """
    Create chart with focus series prominent, context series faded.

    Args:
        df: DataFrame with data
        all_series: List of all series names
        focus_series: Name of series to focus on
        date_col: Date column name

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    context_series = [s for s in all_series if s != focus_series]

    # Context (background) - plot first
    for i, series in enumerate(context_series):
        fig.add_trace(go.Scatter(
            x=df[date_col],
            y=df[series],
            name=series,
            mode='lines',
            line=dict(
                color=f'rgba(156, 163, 175, {0.3 - i * 0.05})',  # Graduated fade
                width=1
            ),
            legendgroup='context',
            legendgrouptitle_text='Context'
        ))

    # Focus - plot last (on top)
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[focus_series],
        name=focus_series,
        mode='lines',
        line=dict(color='#60A5FA', width=3),
        legendgroup='focus',
        legendgrouptitle_text='Focus'
    ))

    return fig
```

---

## Consistent Color Mapping

### Why Consistency Matters

Users build mental models associating colors with meanings. Inconsistent colors force re-learning and cause confusion.

**Rules:**
1. Same entity = same color across all charts
2. Define mappings once, apply everywhere
3. Document color assignments

### Creating a Color Map

```python
# Central color mapping for an application
COLOR_MAP = {
    # Assets
    'BTC': '#60A5FA',
    'ETH': '#818CF8',
    'SOL': '#34D399',
    'AVAX': '#F87171',
    'DOT': '#E879F9',
    'MATIC': '#FBBF24',
    'LINK': '#FB923C',
    'UNI': '#22D3EE',

    # Exchanges
    'Binance': '#FBBF24',
    'Coinbase': '#60A5FA',
    'Kraken': '#818CF8',
    'FTX': '#22D3EE',

    # Categories
    'DeFi': '#34D399',
    'NFT': '#E879F9',
    'L1': '#60A5FA',
    'L2': '#818CF8',
}

def get_color(name, default='#60A5FA'):
    """Get consistent color for an entity."""
    return COLOR_MAP.get(name, default)

def get_colors(names):
    """Get colors for a list of entities."""
    return [get_color(name) for name in names]
```

### Applying Color Maps

```python
# In a line chart
fig = go.Figure()
for asset in ['BTC', 'ETH', 'SOL']:
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[asset],
        name=asset,
        mode='lines',
        line=dict(color=get_color(asset))
    ))

# In a bar chart
fig = go.Figure(data=go.Bar(
    x=assets,
    y=values,
    marker=dict(color=get_colors(assets))
))

# In a pie/donut chart
fig = go.Figure(data=go.Pie(
    labels=categories,
    values=values,
    marker=dict(colors=get_colors(categories))
))
```

### Fallback Strategy

When entities don't have predefined colors:

```python
def get_color_with_fallback(name, used_colors=None):
    """
    Get color for entity, with fallback to unused colorway colors.

    Args:
        name: Entity name
        used_colors: Set of already used colors

    Returns:
        Color hex string
    """
    if name in COLOR_MAP:
        return COLOR_MAP[name]

    used_colors = used_colors or set()

    # Find first unused color from colorway
    for color in COLORWAY:
        if color not in used_colors:
            return color

    # All colors used, return first from colorway
    return COLORWAY[0]
```

---

## Accessibility Requirements

### Colorblind-Safe Design

Approximately 8% of men and 0.5% of women have some form of color vision deficiency.

**Types to consider:**
- Deuteranopia (red-green, most common)
- Protanopia (red-green)
- Tritanopia (blue-yellow, rare)

### Problematic Color Combinations

| Avoid Pairing | Reason | Alternative |
|---------------|--------|-------------|
| Red + Green | Indistinguishable for deuteranopes | Blue + Orange |
| Green + Brown | Can appear similar | Green + Purple |
| Blue + Purple | Similar for tritanopes | Blue + Orange |
| Light green + Yellow | Low contrast | Dark green + Yellow |

### Colorblind-Safe Palette

```python
# Colorblind-safe categorical palette
# Optimized for deuteranopia and protanopia
COLORBLIND_SAFE_PALETTE = [
    '#60A5FA',  # Blue
    '#FB923C',  # Orange
    '#818CF8',  # Purple/Indigo
    '#FBBF24',  # Yellow/Amber
    '#22D3EE',  # Cyan
    '#F472B6',  # Pink
    '#A3E635',  # Yellow-green (distinct from orange)
    '#9CA3AF',  # Gray
]

# For positive/negative that's colorblind-safe
COLORBLIND_POS_NEG = {
    'positive': '#60A5FA',  # Blue (instead of green)
    'negative': '#FB923C',  # Orange (instead of red)
}
```

### Redundant Encoding

Never rely on color alone. Add a second visual channel:

```python
# Redundant encoding: color + pattern/shape
def create_accessible_bar_chart(df, categories, values, statuses):
    """
    Create bar chart with redundant encoding for accessibility.

    Args:
        df: DataFrame
        categories: Category names
        values: Bar values
        statuses: Status for each bar ('positive', 'negative', 'neutral')

    Returns:
        Plotly figure
    """
    # Color mapping
    color_map = {
        'positive': '#34D399',
        'negative': '#F87171',
        'neutral': '#9CA3AF'
    }

    # Pattern mapping (redundant encoding)
    pattern_map = {
        'positive': '',       # Solid
        'negative': '/',      # Diagonal lines
        'neutral': 'x'        # Cross-hatch
    }

    colors = [color_map[s] for s in statuses]
    patterns = [pattern_map[s] for s in statuses]

    fig = go.Figure(data=go.Bar(
        x=categories,
        y=values,
        marker=dict(
            color=colors,
            pattern=dict(shape=patterns)
        )
    ))

    return fig

# Redundant encoding: color + marker shape
def create_accessible_scatter(df, x_col, y_col, category_col):
    """
    Create scatter plot with color + shape encoding.
    """
    symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up']

    fig = go.Figure()

    for i, cat in enumerate(df[category_col].unique()):
        subset = df[df[category_col] == cat]
        fig.add_trace(go.Scatter(
            x=subset[x_col],
            y=subset[y_col],
            name=cat,
            mode='markers',
            marker=dict(
                color=COLORWAY[i % len(COLORWAY)],
                symbol=symbols[i % len(symbols)],
                size=10
            )
        ))

    return fig
```

### Contrast Requirements

Ensure sufficient contrast between:
- Text and background
- Data elements and plot area
- Legend symbols and background

```python
# Minimum contrast ratios (WCAG 2.1 guidelines)
# Normal text: 4.5:1
# Large text (18pt+): 3:1
# UI components: 3:1

# Template colors meet contrast requirements:
# #d3d4d6 on #0e1729 = 10.5:1 (exceeds 4.5:1)

# Verify with: https://webaim.org/resources/contrastchecker/

# For chart elements, ensure opacity doesn't reduce contrast too much
MINIMUM_OPACITY = 0.5  # For primary data elements
CONTEXT_OPACITY = 0.3  # For de-emphasized elements (acceptable for context)
```

---

## Special Cases

### Heatmaps

**Sequential heatmaps (magnitude only):**

```python
# Single-direction data (all positive)
fig = go.Figure(data=go.Heatmap(
    z=magnitude_data,
    colorscale='Viridis',
    colorbar=dict(title='Value')
))
```

**Diverging heatmaps (centered data):**

```python
# Data with meaningful center (zero, average, etc.)
fig = go.Figure(data=go.Heatmap(
    z=deviation_data,
    colorscale=[
        [0.0, '#F87171'],   # Below center (red)
        [0.5, '#4B5563'],   # At center (gray)
        [1.0, '#34D399']    # Above center (green)
    ],
    zmid=0,  # Center at zero
    colorbar=dict(title='Deviation')
))
```

**Decision table:**

| Data Type | Colorscale | zmid |
|-----------|------------|------|
| Volume, counts, intensity | Sequential (Viridis) | Not set |
| Returns, P&L, change | Diverging (RdGn) | 0 |
| Correlation | Diverging (RdBu) | 0 |
| Temperature anomaly | Diverging | Average |
| Percentage of target | Diverging | 100% or 1 |

### Positive/Negative Values

Standard convention for financial data:

```python
# Standard: Green = positive, Red = negative
POSITIVE_COLOR = '#34D399'
NEGATIVE_COLOR = '#F87171'
NEUTRAL_COLOR = '#9CA3AF'

def get_value_color(value):
    """Get color based on positive/negative value."""
    if value > 0:
        return POSITIVE_COLOR
    elif value < 0:
        return NEGATIVE_COLOR
    else:
        return NEUTRAL_COLOR

# Apply to bar chart
colors = [get_value_color(v) for v in values]
fig = go.Figure(data=go.Bar(
    x=categories,
    y=values,
    marker=dict(color=colors)
))
```

**Waterfall charts:**

```python
fig = go.Figure(data=go.Waterfall(
    x=categories,
    y=values,
    measure=['relative'] * len(values),
    increasing=dict(marker=dict(color='#34D399')),
    decreasing=dict(marker=dict(color='#F87171')),
    totals=dict(marker=dict(color='#60A5FA')),
    connector=dict(line=dict(color='rgba(255,255,255,0.2)'))
))
```

### Candlestick/OHLC Charts

```python
# Standard financial convention
CANDLE_UP_COLOR = '#34D399'    # Green for price increase
CANDLE_DOWN_COLOR = '#F87171'  # Red for price decrease

fig = go.Figure(data=go.Candlestick(
    x=df['date'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    increasing=dict(
        line=dict(color=CANDLE_UP_COLOR),
        fillcolor=CANDLE_UP_COLOR
    ),
    decreasing=dict(
        line=dict(color=CANDLE_DOWN_COLOR),
        fillcolor=CANDLE_DOWN_COLOR
    )
))
```

### Area Charts with Fill

```python
# Single area: use primary color with transparency
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['value'],
    mode='lines',
    fill='tozeroy',
    line=dict(color='#60A5FA'),
    fillcolor='rgba(96, 165, 250, 0.3)'  # Same color, 30% opacity
))

# Stacked areas: use colorway with transparency
def get_fill_color(hex_color, opacity=0.7):
    """Convert hex to rgba with opacity."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f'rgba({r}, {g}, {b}, {opacity})'

for i, col in enumerate(columns):
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[col],
        name=col,
        mode='lines',
        stackgroup='one',
        line=dict(color=COLORWAY[i], width=0.5),
        fillcolor=get_fill_color(COLORWAY[i], 0.7)
    ))
```

---

## Color Selection Functions

### Complete Color Selector

```python
def select_colors(
    data_type='categorical',
    num_categories=None,
    has_midpoint=False,
    highlight_index=None,
    colorblind_safe=False
):
    """
    Select appropriate colors based on data characteristics.

    Args:
        data_type: 'categorical', 'sequential', or 'diverging'
        num_categories: Number of categories (for categorical)
        has_midpoint: Whether data has meaningful midpoint (for diverging)
        highlight_index: Index of category to highlight (others de-emphasized)
        colorblind_safe: Use colorblind-safe palette

    Returns:
        Dict with colors, colorscale, or other color configuration
    """
    if data_type == 'categorical':
        palette = COLORBLIND_SAFE_PALETTE if colorblind_safe else COLORWAY

        if highlight_index is not None:
            # Emphasis pattern
            colors = []
            for i in range(num_categories):
                if i == highlight_index:
                    colors.append(palette[0])
                else:
                    colors.append(f'rgba(156, 163, 175, 0.3)')
            return {'colors': colors, 'type': 'emphasis'}

        return {'colors': palette[:num_categories], 'type': 'categorical'}

    elif data_type == 'sequential':
        return {
            'colorscale': 'Viridis',
            'type': 'sequential'
        }

    elif data_type == 'diverging':
        return {
            'colorscale': [
                [0.0, '#F87171'],
                [0.5, '#4B5563'],
                [1.0, '#34D399']
            ],
            'zmid': 0 if has_midpoint else None,
            'type': 'diverging'
        }

    return {'colors': COLORWAY, 'type': 'default'}
```

### Value-to-Color Mapper

```python
def create_value_color_mapper(
    values,
    palette_type='diverging',
    center=0,
    colorblind_safe=False
):
    """
    Create a function that maps values to colors.

    Args:
        values: Array of values to determine range
        palette_type: 'diverging' or 'sequential'
        center: Center value for diverging palette
        colorblind_safe: Use colorblind-safe colors

    Returns:
        Function that maps value to color
    """
    import numpy as np

    min_val, max_val = np.min(values), np.max(values)

    if palette_type == 'diverging':
        if colorblind_safe:
            neg_color = '#FB923C'  # Orange
            pos_color = '#60A5FA'  # Blue
        else:
            neg_color = '#F87171'  # Red
            pos_color = '#34D399'  # Green

        neutral_color = '#4B5563'  # Gray

        def mapper(value):
            if value < center:
                # Negative range
                ratio = (center - value) / (center - min_val) if center != min_val else 0
                return interpolate_color(neutral_color, neg_color, ratio)
            else:
                # Positive range
                ratio = (value - center) / (max_val - center) if max_val != center else 0
                return interpolate_color(neutral_color, pos_color, ratio)

        return mapper

    else:  # sequential
        def mapper(value):
            ratio = (value - min_val) / (max_val - min_val) if max_val != min_val else 0
            return interpolate_color('#0e1729', '#60A5FA', ratio)

        return mapper


def interpolate_color(color1, color2, ratio):
    """
    Interpolate between two hex colors.

    Args:
        color1: Start color (hex)
        color2: End color (hex)
        ratio: Interpolation ratio (0-1)

    Returns:
        Interpolated color (hex)
    """
    def hex_to_rgb(hex_color):
        return tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))

    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(*rgb)

    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)

    r = int(r1 + (r2 - r1) * ratio)
    g = int(g1 + (g2 - g1) * ratio)
    b = int(b1 + (b2 - b1) * ratio)

    return rgb_to_hex((r, g, b))
```

---

## Plotly Configuration

### Applying Template Colors

```python
import plotly.graph_objects as go
import plotly.io as pio

# Create custom template
custom_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='#0e1729',
        plot_bgcolor='#0e1729',
        font=dict(color='#d3d4d6'),
        colorway=COLORWAY,
        xaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)',
            linecolor='rgba(255, 255, 255, 0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(255, 255, 255, 0.1)',
            linecolor='rgba(255, 255, 255, 0.2)'
        )
    )
)

# Register template
pio.templates['custom_dark'] = custom_template
pio.templates.default = 'custom_dark'

# Or apply to individual figure
fig = go.Figure()
fig.update_layout(template='custom_dark')
```

### Color Constants Module

```python
# colors.py - Central color definitions

# Base colors
BACKGROUND = '#0e1729'
TEXT = '#d3d4d6'
GRID = 'rgba(255, 255, 255, 0.1)'
AXIS = 'rgba(255, 255, 255, 0.2)'

# Categorical colorway
COLORWAY = [
    '#60A5FA', '#F87171', '#34D399', '#FBBF24', '#E879F9',
    '#818CF8', '#FB923C', '#22D3EE', '#A3E635', '#F472B6'
]

# Semantic colors
POSITIVE = '#34D399'
NEGATIVE = '#F87171'
NEUTRAL = '#9CA3AF'
PRIMARY = '#60A5FA'

# De-emphasis
DEEMPHASIS = 'rgba(156, 163, 175, 0.3)'
DEEMPHASIS_STRONG = 'rgba(156, 163, 175, 0.5)'

# Sequential scale
SEQUENTIAL = 'Viridis'

# Diverging scale
DIVERGING = [
    [0.0, '#F87171'],
    [0.5, '#4B5563'],
    [1.0, '#34D399']
]

# Colorblind-safe alternatives
COLORBLIND_POSITIVE = '#60A5FA'
COLORBLIND_NEGATIVE = '#FB923C'
```

---

## Quick Reference

### Palette Selection Decision Tree

```
What type of data?
|
+-- Categorical (discrete groups)
|   |
|   +-- How many categories?
|       +-- 1-6: Standard colorway
|       +-- 7-10: Full colorway
|       +-- 11+: Consider filtering/grouping
|
+-- Sequential (magnitude/intensity)
|   |
|   +-- Use Viridis or Blues
|
+-- Diverging (above/below center)
    |
    +-- What's the center?
        +-- Zero: zmid=0
        +-- Average: zmid=mean
        +-- Target: zmid=target
```

### Color Usage Summary

| Scenario | Palette Type | Recommended |
|----------|--------------|-------------|
| Multiple series comparison | Categorical | Colorway |
| Single series emphasis | Emphasis | Blue + gray |
| Volume/intensity | Sequential | Viridis |
| Returns/P&L | Diverging | Red-Gray-Green |
| Correlation matrix | Diverging | Red-Gray-Green, zmid=0 |
| Positive/negative bars | Semantic | Green/Red |
| Time series with bands | Sequential opacity | Primary color + 30% fill |

### Template Colors Summary

| Element | Color Code | Usage |
|---------|------------|-------|
| Background | #0e1729 | Paper and plot background |
| Text | #d3d4d6 | All text, labels, titles |
| Primary | #60A5FA | Main data, emphasis |
| Positive | #34D399 | Gains, increases |
| Negative | #F87171 | Losses, decreases |
| Neutral | #9CA3AF | Zero, baseline |
| Grid | rgba(255,255,255,0.1) | Gridlines |
| De-emphasis | rgba(156,163,175,0.3) | Context series |

---

## Color Checklist

Before finalizing colors, verify:

**Palette Selection:**
- [ ] Palette type matches data type (categorical/sequential/diverging)
- [ ] Number of colors appropriate for category count (max 10)
- [ ] Diverging scales have explicit midpoint (zmid)

**Consistency:**
- [ ] Same entities use same colors across all charts
- [ ] Color mapping documented and centralized
- [ ] Semantic colors used correctly (green=positive, red=negative)

**Emphasis:**
- [ ] Focus elements are clearly highlighted
- [ ] Context elements are appropriately de-emphasized
- [ ] Z-order correct (emphasized elements on top)

**Accessibility:**
- [ ] Colorblind-safe palette used if needed
- [ ] Not relying on color alone (redundant encoding)
- [ ] Sufficient contrast against background
- [ ] Red-green not used together without redundant encoding

**Special Cases:**
- [ ] Heatmaps use correct scale (sequential vs diverging)
- [ ] Financial charts follow convention (green=up, red=down)
- [ ] Area fills use appropriate opacity (30-70%)

**Template Compliance:**
- [ ] Background is #0e1729
- [ ] Text is #d3d4d6
- [ ] Grid uses rgba(255, 255, 255, 0.1)
- [ ] Primary color is #60A5FA
