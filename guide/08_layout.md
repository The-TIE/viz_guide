# 08 - Layout

> Rules for margins, aspect ratios, and subplot arrangements in Plotly visualizations.
> Proper layout ensures readability and prevents clipped content.

---

## Margins

When including bottom annotations (source, timestamps), increase bottom margin to at least 100px to prevent clipping: `margin=dict(l=60, r=30, t=80, b=100)`. Adjust based on annotation text length and positioning.

### Base Margins (in pixels)

| Position | Base Value | With Element | Notes |
|----------|------------|--------------|-------|
| Top | 60 | 80 | Add 20px for chart title |
| Bottom | 60 | 90 | Add 30px for x-axis title |
| Left | 60 | Variable | Adjust for long y-tick labels |
| Right | 30 | 150 | Add 120px for right-side legend |

**Default margin configuration:**

```python
fig.update_layout(
    margin=dict(
        t=60,   # Top: 60px base
        b=60,   # Bottom: 60px base
        l=60,   # Left: 60px base
        r=30    # Right: 30px base
    )
)
```

### Auto-Adjustment Rules

#### Title Accommodation

```python
def calculate_top_margin(has_title=False, has_subtitle=False):
    """Calculate top margin based on title elements."""
    base = 60
    if has_title:
        base += 20
    if has_subtitle:
        base += 20  # Subtitle adds another line
    return base

# With title
fig.update_layout(margin=dict(t=80))

# With title + subtitle
fig.update_layout(margin=dict(t=100))
```

#### Long Tick Label Adjustment

Estimate label width and add to left margin. Use approximately 7 pixels per character for standard font sizes.

```python
def calculate_left_margin(y_tick_labels, base=60, char_width=7):
    """
    Calculate left margin based on y-axis tick label length.

    Args:
        y_tick_labels: List of tick label strings
        base: Base left margin in pixels
        char_width: Approximate width per character (7px for 12px font)

    Returns:
        Calculated left margin in pixels
    """
    if not y_tick_labels:
        return base

    max_label_length = max(len(str(label)) for label in y_tick_labels)
    label_width = max_label_length * char_width

    # Add padding and base margin
    return base + max(0, label_width - 40)  # 40px already accounted in base
```

**Common scenarios:**

| Label Type | Example | Estimated Width | Left Margin |
|------------|---------|-----------------|-------------|
| Short numbers | 1.2M | ~4 chars | 60px (base) |
| Medium numbers | $12.34M | ~8 chars | 70px |
| Long categories | "Cryptocurrency Exchange" | ~24 chars | 120px |
| Date labels | "Jan 2025" | ~8 chars | 70px |

#### X-Axis Title Accommodation

```python
def calculate_bottom_margin(has_xaxis_title=False):
    """Calculate bottom margin based on x-axis title."""
    base = 60
    if has_xaxis_title:
        base += 30
    return base

# With x-axis title
fig.update_layout(margin=dict(b=90))
```

#### Legend Placement Impact

```python
def calculate_margins_with_legend(legend_position='top'):
    """
    Calculate margins based on legend placement.

    Args:
        legend_position: 'top', 'right', 'bottom', or 'inside'

    Returns:
        Margin dict for update_layout
    """
    margins = {'t': 60, 'b': 60, 'l': 60, 'r': 30}

    if legend_position == 'top':
        margins['t'] = 100  # Space above chart for horizontal legend
    elif legend_position == 'right':
        margins['r'] = 150  # Space for vertical legend
    elif legend_position == 'bottom':
        margins['b'] = 100  # Space below chart
    # 'inside' requires no additional margins

    return margins
```

### Complete Margin Calculator

```python
def calculate_chart_margins(
    has_title=False,
    has_subtitle=False,
    has_xaxis_title=False,
    legend_position='top',
    max_ylabel_length=6
):
    """
    Calculate all margins for a chart.

    Args:
        has_title: Chart has a title
        has_subtitle: Chart has a subtitle
        has_xaxis_title: X-axis has a title
        legend_position: 'top', 'right', 'bottom', 'inside', or None
        max_ylabel_length: Maximum character length of y-axis labels

    Returns:
        Dict with t, b, l, r margin values
    """
    # Top margin
    top = 60
    if has_title:
        top += 20
    if has_subtitle:
        top += 20
    if legend_position == 'top':
        top += 40

    # Bottom margin
    bottom = 60
    if has_xaxis_title:
        bottom += 30
    if legend_position == 'bottom':
        bottom += 40

    # Left margin
    left = 60
    if max_ylabel_length > 6:
        left += (max_ylabel_length - 6) * 7

    # Right margin
    right = 30
    if legend_position == 'right':
        right += 120

    return {'t': top, 'b': bottom, 'l': left, 'r': right}
```

---

## Aspect Ratios

### By Chart Type

| Chart Type | Recommended Ratio | Width:Height | Reasoning |
|------------|-------------------|--------------|-----------|
| Time series | Wide | 16:9 or 2:1 | Horizontal time progression |
| Bar chart (horizontal) | Varies | Based on categories | Height scales with category count |
| Bar chart (vertical) | Near-square | 4:3 | Categories need horizontal space |
| Scatter plot | Square | 1:1 | Equal weight to both variables |
| Heatmap | Match data | Data-driven | Rows:Cols ratio |
| Donut/Pie | Square | 1:1 | Circular shape |
| Small multiples | Grid-based | Panel-dependent | Consistent panel sizes |

### Time Series Charts

Time flows left-to-right; wider formats emphasize temporal patterns.

```python
# Standard time series dimensions
TIME_SERIES_WIDTH = 900
TIME_SERIES_HEIGHT = 500  # ~16:9 ratio

# Extended time series (long date ranges)
WIDE_TIME_SERIES_WIDTH = 1200
WIDE_TIME_SERIES_HEIGHT = 400  # ~3:1 ratio

fig.update_layout(
    width=TIME_SERIES_WIDTH,
    height=TIME_SERIES_HEIGHT
)
```

### Bar Charts

Height depends on category count for horizontal bars:

```python
def calculate_bar_chart_height(num_categories, bar_height=25, min_height=300, max_height=800):
    """
    Calculate height for horizontal bar chart.

    Args:
        num_categories: Number of bars
        bar_height: Approximate height per bar (including gap)
        min_height: Minimum chart height
        max_height: Maximum chart height

    Returns:
        Chart height in pixels
    """
    calculated = num_categories * bar_height + 100  # 100px for margins/labels
    return max(min_height, min(calculated, max_height))

# Example usage
num_categories = 15
height = calculate_bar_chart_height(num_categories)
# Result: 15 * 25 + 100 = 475px

fig.update_layout(
    width=700,
    height=height
)
```

### Scatter Plots

Square or near-square to give equal visual weight to both axes:

```python
# Standard scatter plot
SCATTER_WIDTH = 600
SCATTER_HEIGHT = 600  # 1:1 ratio

# When x-axis has more range
SCATTER_WIDE_WIDTH = 700
SCATTER_WIDE_HEIGHT = 600  # Slightly wide

fig.update_layout(
    width=SCATTER_WIDTH,
    height=SCATTER_HEIGHT
)
```

### Heatmaps

Match aspect ratio to data dimensions:

```python
def calculate_heatmap_dimensions(num_rows, num_cols, cell_size=40, max_width=1000, max_height=800):
    """
    Calculate heatmap dimensions based on data shape.

    Args:
        num_rows: Number of rows in the matrix
        num_cols: Number of columns in the matrix
        cell_size: Target size per cell in pixels
        max_width: Maximum chart width
        max_height: Maximum chart height

    Returns:
        Tuple of (width, height)
    """
    # Calculate ideal dimensions
    width = num_cols * cell_size + 100   # Add margin for labels
    height = num_rows * cell_size + 100

    # Apply constraints
    width = min(width, max_width)
    height = min(height, max_height)

    return width, height

# Example: 10x8 correlation matrix
width, height = calculate_heatmap_dimensions(10, 8)
# Result: (420, 500)
```

### Responsive Considerations

#### Minimum Viable Widths

| Chart Type | Minimum Width | Notes |
|------------|---------------|-------|
| Line chart | 400px | Labels become cramped below |
| Bar chart (horizontal) | 400px | Need space for values |
| Bar chart (vertical) | 300px | Labels may need rotation |
| Scatter plot | 350px | Point density suffers |
| Heatmap | 300px | Cell labels unreadable |
| Small multiples | 250px per panel | Minimum per subplot |

#### Mobile Adaptations

For narrow viewports, consider:

1. **Stack instead of side-by-side**: Convert 2-column layouts to single column
2. **Reduce margins**: Mobile-specific margin values
3. **Simplify legends**: Move to bottom, reduce to icons
4. **Rotate labels**: Use angled x-axis labels

```python
def get_responsive_layout(viewport_width):
    """
    Get layout parameters based on viewport width.

    Args:
        viewport_width: Available width in pixels

    Returns:
        Dict with layout parameters
    """
    if viewport_width < 500:  # Mobile
        return {
            'margin': {'t': 50, 'b': 80, 'l': 50, 'r': 20},
            'legend': {
                'orientation': 'h',
                'yanchor': 'top',
                'y': -0.15,
                'xanchor': 'center',
                'x': 0.5
            },
            'xaxis_tickangle': -45
        }
    elif viewport_width < 800:  # Tablet
        return {
            'margin': {'t': 60, 'b': 70, 'l': 60, 'r': 30},
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'left',
                'x': 0
            },
            'xaxis_tickangle': 0
        }
    else:  # Desktop
        return {
            'margin': {'t': 80, 'b': 60, 'l': 60, 'r': 30},
            'legend': {
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'left',
                'x': 0
            },
            'xaxis_tickangle': 0
        }
```

---

## Subplot Arrangements

### Grid Layout Calculation

Determine optimal rows and columns based on subplot count:

```python
def calculate_grid_layout(num_subplots, max_cols=3):
    """
    Calculate optimal grid layout for subplots.

    Args:
        num_subplots: Total number of subplots
        max_cols: Maximum columns (default 3)

    Returns:
        Tuple of (rows, cols)

    Guidelines:
        - 1-2 subplots: 1 row
        - 3-4 subplots: 2 columns
        - 5-9 subplots: 3 columns
        - 10+ subplots: Consider filtering data
    """
    if num_subplots <= 2:
        return 1, num_subplots
    elif num_subplots <= 4:
        cols = 2
    else:
        cols = min(3, max_cols)

    rows = (num_subplots + cols - 1) // cols  # Ceiling division
    return rows, cols
```

**Layout recommendations by subplot count:**

| Subplots | Rows x Cols | Notes |
|----------|-------------|-------|
| 1 | 1x1 | Single chart |
| 2 | 1x2 | Side-by-side comparison |
| 3 | 1x3 or 2x2 | 2x2 with empty cell acceptable |
| 4 | 2x2 | Clean grid |
| 5-6 | 2x3 | Three-column layout |
| 7-9 | 3x3 | Maximum recommended |
| 10-12 | 3x4 or 4x3 | Consider filtering |
| 12+ | Reconsider | Too many for effective display |

### When to Use Different Column Counts

**1-Column Layout:**
- Sequential story (top-to-bottom narrative)
- Comparing very different chart types
- Mobile/narrow viewports

**2-Column Layout:**
- Direct A/B comparison
- Before/after scenarios
- Related but distinct metrics

**3-Column Layout:**
- Small multiples (same metric, different categories)
- Dashboard overview panels
- Maximum detail without scrolling

### Shared Axes Patterns

#### Shared X-Axes for Time Series

Use when comparing multiple time series with the same date range:

```python
from plotly.subplots import make_subplots

# Time series small multiples with shared x-axis
categories = ['BTC', 'ETH', 'SOL', 'AVAX']
rows, cols = calculate_grid_layout(len(categories))

fig = make_subplots(
    rows=rows,
    cols=cols,
    shared_xaxes=True,  # All subplots share x-axis
    shared_yaxes=False,  # Each has own y-scale
    subplot_titles=categories,
    vertical_spacing=0.1,
    horizontal_spacing=0.06
)

# Add traces
for i, cat in enumerate(categories):
    row = (i // cols) + 1
    col = (i % cols) + 1

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

# Only bottom row shows x-axis ticks
fig.update_xaxes(
    tickformat='%b %Y',
    row=rows,  # Only configure bottom row
    showticklabels=True
)
```

#### Shared Y-Axes for Scale Comparison

Use when comparing values that should be on the same scale:

```python
# Comparing distributions across categories
fig = make_subplots(
    rows=1,
    cols=3,
    shared_xaxes=False,
    shared_yaxes=True,  # Same y-scale for comparison
    subplot_titles=['Group A', 'Group B', 'Group C'],
    horizontal_spacing=0.05
)

# All subplots now share the same y-axis range
# Only leftmost shows y-axis labels by default
```

#### Mixed Shared Axes

```python
# Shared x-axes per row, shared y-axes per column
fig = make_subplots(
    rows=2,
    cols=2,
    shared_xaxes='rows',  # Each row shares x-axis
    shared_yaxes='columns',  # Each column shares y-axis
    vertical_spacing=0.1,
    horizontal_spacing=0.05
)
```

### Spacing Guidelines

| Parameter | Typical Range | Default | Notes |
|-----------|---------------|---------|-------|
| `vertical_spacing` | 0.08 - 0.12 | 0.3 / rows | Space between rows |
| `horizontal_spacing` | 0.05 - 0.08 | 0.2 / cols | Space between columns |

**Spacing selection rules:**

```python
def calculate_subplot_spacing(rows, cols, has_subplot_titles=True):
    """
    Calculate optimal spacing for subplot grid.

    Args:
        rows: Number of rows
        cols: Number of columns
        has_subplot_titles: Whether subplots have titles

    Returns:
        Dict with vertical_spacing and horizontal_spacing
    """
    # Vertical spacing
    if has_subplot_titles:
        v_spacing = 0.12 if rows <= 2 else 0.10
    else:
        v_spacing = 0.08 if rows <= 2 else 0.06

    # Horizontal spacing
    h_spacing = 0.08 if cols <= 2 else 0.05

    return {
        'vertical_spacing': v_spacing,
        'horizontal_spacing': h_spacing
    }
```

### Height Scaling for Subplots

Scale total height with row count to maintain readable panel sizes:

```python
def calculate_subplot_height(
    num_rows,
    base_height_per_row=250,
    min_height=300,
    max_height=1200
):
    """
    Calculate total figure height based on subplot rows.

    Args:
        num_rows: Number of subplot rows
        base_height_per_row: Height per row in pixels
        min_height: Minimum figure height
        max_height: Maximum figure height

    Returns:
        Figure height in pixels
    """
    calculated = num_rows * base_height_per_row
    return max(min_height, min(calculated, max_height))

# Examples:
# 1 row: max(300, min(250, 1200)) = 300px
# 2 rows: max(300, min(500, 1200)) = 500px
# 3 rows: max(300, min(750, 1200)) = 750px
# 4 rows: max(300, min(1000, 1200)) = 1000px
```

**Complete subplot dimension calculator:**

```python
def calculate_subplot_dimensions(
    num_subplots,
    base_width=800,
    base_height_per_row=250,
    max_cols=3
):
    """
    Calculate complete dimensions for subplot figure.

    Args:
        num_subplots: Total number of subplots
        base_width: Base figure width
        base_height_per_row: Height per row
        max_cols: Maximum columns

    Returns:
        Dict with rows, cols, width, height, spacing
    """
    rows, cols = calculate_grid_layout(num_subplots, max_cols)
    height = calculate_subplot_height(rows, base_height_per_row)
    spacing = calculate_subplot_spacing(rows, cols)

    return {
        'rows': rows,
        'cols': cols,
        'width': base_width,
        'height': height,
        **spacing
    }
```

---

## Plotly Implementation

### Setting Margins

```python
import plotly.graph_objects as go

# Basic margin configuration
fig = go.Figure()
fig.update_layout(
    margin=dict(
        t=80,   # Top margin (with title)
        b=90,   # Bottom margin (with x-axis title)
        l=70,   # Left margin (moderate y-labels)
        r=30,   # Right margin (no right legend)
        pad=4   # Padding between plot and axes
    ),
    paper_bgcolor='#0e1729',
    plot_bgcolor='#0e1729',
    font=dict(color='#d3d4d6')
)
```

### Responsive Sizing with Autosize

```python
# Let Plotly auto-size to container
fig.update_layout(
    autosize=True,
    # Don't set explicit width/height when using autosize
    margin=dict(t=60, b=60, l=60, r=30)
)

# Or set explicit dimensions for static output
fig.update_layout(
    autosize=False,
    width=900,
    height=500,
    margin=dict(t=60, b=60, l=60, r=30)
)
```

### Complete make_subplots Pattern

```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def create_small_multiples(df, categories, date_col='date', value_col='value'):
    """
    Create small multiples chart for time series data.

    Args:
        df: DataFrame with data
        categories: List of category names (column names or filter values)
        date_col: Column name for dates
        value_col: Column name for values

    Returns:
        Plotly figure
    """
    # Calculate layout
    dims = calculate_subplot_dimensions(len(categories))

    # Create subplot grid
    fig = make_subplots(
        rows=dims['rows'],
        cols=dims['cols'],
        shared_xaxes=True,
        shared_yaxes=True,
        subplot_titles=categories,
        vertical_spacing=dims['vertical_spacing'],
        horizontal_spacing=dims['horizontal_spacing']
    )

    # Add traces to subplots
    for i, cat in enumerate(categories):
        row = (i // dims['cols']) + 1
        col = (i % dims['cols']) + 1

        fig.add_trace(
            go.Scatter(
                x=df[date_col],
                y=df[cat] if cat in df.columns else df[df['category'] == cat][value_col],
                mode='lines',
                name=cat,
                line=dict(width=2, color='#60A5FA'),
                showlegend=False
            ),
            row=row,
            col=col
        )

    # Apply consistent styling
    fig.update_layout(
        width=dims['width'],
        height=dims['height'],
        paper_bgcolor='#0e1729',
        plot_bgcolor='#0e1729',
        font=dict(color='#d3d4d6'),
        margin=dict(t=80, b=60, l=60, r=30)
    )

    # Style all axes
    fig.update_xaxes(
        tickformat='%b %Y',
        gridcolor='rgba(255, 255, 255, 0.1)',
        linecolor='rgba(255, 255, 255, 0.2)'
    )

    fig.update_yaxes(
        tickformat=',.2s',
        gridcolor='rgba(255, 255, 255, 0.1)',
        linecolor='rgba(255, 255, 255, 0.2)'
    )

    return fig
```

### Height and Width Calculations

```python
# Standard dimensions by chart type
CHART_DIMENSIONS = {
    'time_series': {'width': 900, 'height': 500},
    'time_series_wide': {'width': 1200, 'height': 400},
    'bar_horizontal': {'width': 700, 'height': None},  # Height calculated
    'bar_vertical': {'width': 700, 'height': 450},
    'scatter': {'width': 600, 'height': 600},
    'heatmap': {'width': None, 'height': None},  # Both calculated
    'donut': {'width': 500, 'height': 500},
}

def get_chart_dimensions(chart_type, num_categories=None, num_rows=None, num_cols=None):
    """
    Get appropriate dimensions for chart type.

    Args:
        chart_type: Type of chart
        num_categories: Number of categories (for bar charts)
        num_rows: Number of data rows (for heatmaps)
        num_cols: Number of data columns (for heatmaps)

    Returns:
        Dict with width and height
    """
    base = CHART_DIMENSIONS.get(chart_type, {'width': 700, 'height': 450})

    if chart_type == 'bar_horizontal' and num_categories:
        height = calculate_bar_chart_height(num_categories)
        return {'width': base['width'], 'height': height}

    elif chart_type == 'heatmap' and num_rows and num_cols:
        width, height = calculate_heatmap_dimensions(num_rows, num_cols)
        return {'width': width, 'height': height}

    return base
```

---

## Complete Layout Example

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sample data setup
categories = ['Exchange A', 'Exchange B', 'Exchange C', 'Exchange D']
num_subplots = len(categories)

# Calculate layout parameters
dims = calculate_subplot_dimensions(num_subplots)
margins = calculate_chart_margins(
    has_title=True,
    has_subtitle=True,
    legend_position=None,  # Small multiples don't need legend
    max_ylabel_length=10
)

# Create figure
fig = make_subplots(
    rows=dims['rows'],
    cols=dims['cols'],
    shared_xaxes=True,
    shared_yaxes=True,
    subplot_titles=categories,
    vertical_spacing=dims['vertical_spacing'],
    horizontal_spacing=dims['horizontal_spacing']
)

# Add traces (example)
for i, cat in enumerate(categories):
    row = (i // dims['cols']) + 1
    col = (i % dims['cols']) + 1

    fig.add_trace(
        go.Scatter(
            x=['2025-01-01', '2025-01-02', '2025-01-03'],
            y=[100, 120, 115],
            mode='lines',
            name=cat,
            line=dict(color='#60A5FA'),
            showlegend=False
        ),
        row=row,
        col=col
    )

# Apply complete layout
fig.update_layout(
    title=dict(
        text='Trading Volume by Exchange<br><sup>Daily totals, last 7 days</sup>',
        font=dict(size=16, color='#d3d4d6'),
        x=0,
        xanchor='left'
    ),
    width=dims['width'],
    height=dims['height'],
    margin=margins,
    paper_bgcolor='#0e1729',
    plot_bgcolor='#0e1729',
    font=dict(color='#d3d4d6'),
    showlegend=False
)

# Update all axes
fig.update_xaxes(
    tickformat='%b %d',
    gridcolor='rgba(255, 255, 255, 0.1)'
)

fig.update_yaxes(
    tickformat='$,.2s',
    gridcolor='rgba(255, 255, 255, 0.1)'
)
```

---

## Quick Reference

### Margin Defaults

| Configuration | top | bottom | left | right |
|---------------|-----|--------|------|-------|
| Minimal | 40 | 40 | 50 | 20 |
| Standard | 60 | 60 | 60 | 30 |
| With title | 80 | 60 | 60 | 30 |
| With title + x-axis title | 80 | 90 | 60 | 30 |
| With right legend | 60 | 60 | 60 | 150 |
| Full (all elements) | 100 | 90 | 80 | 150 |

### Aspect Ratio Quick Reference

| Chart Type | Ratio | Example Dimensions |
|------------|-------|-------------------|
| Time series | 16:9 | 900 x 500 |
| Scatter | 1:1 | 600 x 600 |
| Heatmap | Data-driven | Varies |
| Bar (horizontal) | Category-driven | 700 x (n*25+100) |

### Subplot Grid Reference

| Subplots | Layout | Spacing (v/h) |
|----------|--------|---------------|
| 2 | 1x2 | 0.08 / 0.08 |
| 4 | 2x2 | 0.10 / 0.06 |
| 6 | 2x3 | 0.10 / 0.05 |
| 9 | 3x3 | 0.08 / 0.05 |

### Template Colors

| Element | Color |
|---------|-------|
| Background | #0e1729 |
| Text | #d3d4d6 |
| Grid (subtle) | rgba(255, 255, 255, 0.1) |
| Axis lines | rgba(255, 255, 255, 0.2) |

---

## Layout Checklist

Before finalizing layout, verify:

**Margins:**
- [ ] Title has adequate top margin (80px with title)
- [ ] X-axis title has adequate bottom margin (90px with title)
- [ ] Y-axis labels are not clipped (adjust left margin)
- [ ] Legend has space (right: 150px, top: extra 40px)

**Dimensions:**
- [ ] Aspect ratio matches chart type
- [ ] Height scales with content (bar categories, subplot rows)
- [ ] Width is at least minimum viable (400px for most charts)
- [ ] Heatmap dimensions match data shape

**Subplots:**
- [ ] Grid layout is optimal for subplot count
- [ ] Shared axes are configured correctly
- [ ] Vertical spacing accounts for subplot titles
- [ ] Total height scales with row count

**Responsive:**
- [ ] Autosize enabled for container-filling
- [ ] Explicit dimensions for static exports
- [ ] Mobile breakpoints considered if applicable

**Styling:**
- [ ] Background color matches template (#0e1729)
- [ ] Text color is readable (#d3d4d6)
- [ ] Grid lines are subtle but visible

## Common Property Naming Patterns

Plotly uses nested `dict()` structures for styling. For titles and labels, use `title=dict(text='...', font=dict(...))` rather than flat properties like `titlefont`. Similarly, use `tickfont=dict(...)` not `tickfontsize`. Check the Plotly documentation for the current property structure as some older examples use deprecated flat properties.

## Title Alignment

Left-align titles by default using `title=dict(x=0, xanchor='left')`. This follows standard data visualization conventions and provides better visual hierarchy than center-aligned titles.

## Default Settings to Override

Common default settings that often need adjustment: (1) Grid lines are enabled - usually should be disabled with `showgrid=False`, (2) Titles are centered - often better left-aligned with `x=0, xanchor='left'`, (3) Margins may be too tight - adjust with `margin=dict()` for proper spacing.

## Title Positioning

For dashboard-style visualizations, left-align titles using `title=dict(x=0, xanchor='left')` rather than center alignment. This creates a cleaner, more professional appearance consistent with modern data visualization practices.
