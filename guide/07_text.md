# 07 - Text Formatting

> Rules and patterns for formatting numbers, dates, and text labels in visualizations.
> Consistent formatting improves readability and reduces cognitive load.

---

## Number Formatting Rules

For hover values, use 3 significant figures with k/M/B notation. Implement with `format_with_B(value, decimals=0)` which produces values like '158k', '2.5M', '1.2B'. The `decimals=0` parameter ensures whole numbers after the decimal point, giving effectively 3 significant figures. Use `customdata` with this formatting and reference in `hovertemplate='%{customdata}'`.

For financial and volume data, format numbers with 3 significant figures using custom formatting functions. Scale large numbers with suffixes (B for billions, M for millions, k for thousands) and maintain consistent significant figures across scales. Example: $1.23B, $456M, $78.9k.

For financial data in billions, use 'B' suffix (not Plotly's default 'G' for giga). Create a custom formatting function that handles significant figures: `format_with_B(value * 1e9, prefix='$', sig_figs=3)` to display values like '$3.62B'. Apply this to tick labels, bar labels, and hover text consistently. Default to 3 significant figures for readability unless more precision is needed.

### Human-Readable Scaling (CRITICAL)

Large numbers must use SI notation for readability. Raw numbers like `816890000` are unreadable; `816.89M` is not.

**Standard: 3 significant figures with SI suffixes**

| Value Range | Suffix | Example Input | Formatted Output |
|-------------|--------|---------------|------------------|
| < 1,000 | none | 742 | 742 |
| 1,000 - 999,999 | k | 7,860 | 7.86k |
| 1,000,000 - 999,999,999 | M | 816,890,000 | 816.89M |
| 1,000,000,000 - 999,999,999,999 | B | 1,200,000,000 | 1.2B |
| 1,000,000,000,000+ | T | 1,230,000,000,000 | 1.23T |

> **⚠️ CRITICAL: Plotly uses "G" not "B" for billions!**
>
> Plotly's `,.2s` format produces "12.3G" not "12.3B". This is incorrect for financial data.
> **You MUST use the `format_with_B()` function below for any values that might reach billions.**

**Examples from audit:**
- Market cap: `816.89M`, `297.32M`, `1.2B`
- Trading volume: `7.86M`, `12.3M`
- Supply figures: `21M`, `115.4B`

### Currency Formatting

Dollar sign prefix, combined with SI notation for large values.

| Context | Format | Example |
|---------|--------|---------|
| Exact price | `$X.XX` | $488.02, $0.9998 |
| Large currency | `$X.XXs` | $1.23M, $816.89M |
| Very large currency | `$X.XXs` | $1.2B, $2.34T |
| Small currency | `$X.XXXX` | $0.0001234 |

**Rules:**
- Prices always show 2 decimal places minimum: `$488.02` not `$488`
- Stablecoins near $1: show 4 decimals: `$0.9998`, `$1.0002`
- Market caps/volumes: use SI notation: `$816.89M`

### Percentage Formatting

**Sign indication is mandatory for changes:**

| Type | Format | Examples |
|------|--------|----------|
| Positive change | `+X.X%` | +12.3%, +5.8%, +0.01% |
| Negative change | `-X.X%` | -5.2%, -1.2%, -0.09% |
| Static percentage | `X.X%` | 45.2%, 12.3% |
| Small percentage | `X.XX%` | 0.09%, 0.01% |

**Color coding for percentage changes:**
- Positive: Green (`#34D399`)
- Negative: Red (`#F87171`)
- Zero/neutral: Default text color (`#d3d4d6`)

### Decimal Precision by Context

| Context | Precision | Example |
|---------|-----------|---------|
| Prices (USD) | 2 decimals | $488.02 |
| Stablecoin prices | 4 decimals | $0.9998 |
| Crypto prices < $1 | 4-6 decimals | $0.001234 |
| Percentages | 1-2 decimals | 12.3%, -0.09% |
| Large numbers | SI notation (3 sig figs) | 816.89M |
| Ratios/multipliers | 2 decimals | 1.23x, 0.85x |
| Correlation coefficients | 2-3 decimals | 0.87, -0.923 |

---

## Date/Time Formatting

### Axis Tick Formats by Timeframe

| Timeframe | Plotly Format | Example Output |
|-----------|---------------|----------------|
| Intraday (hours) | `%H:%M` | 14:30 |
| Intraday (minutes) | `%H:%M` | 09:45 |
| Daily | `%b %d` | Jan 15 |
| Weekly | `%b %d` | Jan 15 |
| Monthly | `%b %Y` | Jan 2025 |
| Quarterly | `Q%q %Y` | Q1 2025 |
| Multi-year | `%Y` | 2025 |

**Selection logic:**

```python
def get_date_tick_format(date_range_days):
    """Select appropriate tick format based on date range."""
    if date_range_days <= 1:
        return '%H:%M'  # Intraday
    elif date_range_days <= 14:
        return '%b %d'  # Daily/weekly
    elif date_range_days <= 90:
        return '%b %d'  # Weekly view
    elif date_range_days <= 365:
        return '%b %Y'  # Monthly
    else:
        return '%Y'  # Multi-year
```

### Hover Date Formats

Hover templates should show full, unambiguous dates:

| Context | Format Pattern | Example Output |
|---------|---------------|----------------|
| Daily data | `%b %d, %Y` | Jan 15, 2025 |
| Intraday data | `%b %d, %Y %H:%M` | Jan 15, 2025 14:30 |
| With timezone | `%b %d, %Y %H:%M %Z` | Jan 15, 2025 14:30 UTC |

---

## Title Hierarchy

### Chart Title

**Purpose:** What the chart shows + essential context

**Rules:**
- Short and descriptive (5-10 words max)
- Include the subject and timeframe
- Avoid redundant words like "Chart of" or "Graph showing"

| Good | Bad |
|------|-----|
| BTC Price (30 Days) | Chart Showing Bitcoin Price Over the Last 30 Days |
| Exchange Volume Comparison | Volume |
| Market Cap Distribution | A Pie Chart of Market Caps |

### Subtitle

**Purpose:** Additional context, filters, or data source

**When to use:**
- Filters are applied (e.g., "Top 10 by Volume")
- Date range needs clarification
- Data source attribution

```python
fig.update_layout(
    title=dict(
        text='Exchange Trading Volume<br><sup>Top 10 exchanges, last 24 hours</sup>',
        font=dict(size=16, color='#d3d4d6'),
        x=0,
        xanchor='left'
    )
)
```

### Axis Titles

**Rules:**
- Include units when not obvious from the data
- Omit when axis values are self-explanatory (dates, percentages)
- Keep concise

| Include | Omit |
|---------|------|
| Price (USD) | Date (obvious from axis) |
| Volume (24h) | % (obvious from values) |
| Market Cap (USD) | Time (obvious from axis) |

---

## Plotly Implementation

### Tick Format Patterns

Plotly uses d3-format for number formatting:

| Pattern | Description | Input | Output |
|---------|-------------|-------|--------|
| `,.2s` | SI notation, 2 decimals | 1234567 | 1.2M |
| `,.3s` | SI notation, 3 decimals | 1234567 | 1.23M |
| `$,.2f` | Currency, 2 decimals | 488.02 | $488.02 |
| `$,.2s` | Currency + SI notation | 1234567 | $1.2M |
| `.1%` | Percentage, 1 decimal | 0.123 | 12.3% |
| `.2%` | Percentage, 2 decimals | 0.1234 | 12.34% |
| `+.1%` | Percentage with sign | 0.123 | +12.3% |
| `,.0f` | Integer with commas | 1234567 | 1,234,567 |
| `,.2f` | Float, 2 decimals | 1234.567 | 1,234.57 |

### ⚠️ MANDATORY: Use "B" Not "G" for Billions

**Plotly's d3-format uses "G" (giga) for 10^9. In finance, we MUST use "B" (billions).**

This is a persistent issue. **NEVER use `,.2s` or `$,.2s` format for values that might reach billions.** Instead, use this function:

```python
def format_with_B(value, prefix='$', decimals=1):
    """
    Format large numbers with B/T instead of G/T.

    ALWAYS use this instead of Plotly's ,.2s format for financial data.

    Args:
        value: The numeric value to format
        prefix: Currency prefix (default '$', use '' for no prefix)
        decimals: Number of decimal places (default 1)

    Returns:
        Formatted string like '$1.2B', '$345.6M', '$12.3k'
    """
    fmt = f'.{decimals}f'
    abs_val = abs(value)
    sign = '-' if value < 0 else ''

    if abs_val >= 1e12:
        return f'{sign}{prefix}{abs_val/1e12:{fmt}}T'
    elif abs_val >= 1e9:
        return f'{sign}{prefix}{abs_val/1e9:{fmt}}B'  # B not G!
    elif abs_val >= 1e6:
        return f'{sign}{prefix}{abs_val/1e6:{fmt}}M'
    elif abs_val >= 1e3:
        return f'{sign}{prefix}{abs_val/1e3:{fmt}}k'
    else:
        return f'{sign}{prefix}{abs_val:{fmt}}'
```

**Usage for axis tick labels:**

```python
# Calculate appropriate tick values
max_val = df['volume'].max()
tick_vals = [0, max_val * 0.25, max_val * 0.5, max_val * 0.75, max_val]

fig.update_yaxes(
    tickmode='array',
    tickvals=tick_vals,
    ticktext=[format_with_B(v) for v in tick_vals]
)
```

**Usage for bar chart text labels:**

```python
fig.add_trace(go.Bar(
    x=df['exchange'],
    y=df['volume'],
    text=df['volume'].apply(format_with_B),
    textposition='outside'
))
```

**Usage in hover templates:**

```python
# Add formatted values as customdata
df['volume_fmt'] = df['volume'].apply(format_with_B)

fig.add_trace(go.Bar(
    x=df['exchange'],
    y=df['volume'],
    customdata=df[['volume_fmt']],
    hovertemplate='<b>%{x}</b><br>Volume: %{customdata[0]}<extra></extra>'
))
```

**When is `,.2s` acceptable?**
- Values guaranteed to stay under 1 billion (most won't exceed millions)
- Non-financial contexts where "G" is acceptable

### Axis Tick Formatting

```python
# SI notation for large numbers (default for most y-axes)
fig.update_yaxes(tickformat=',.2s')  # 1.23M, 456k

# Currency formatting
fig.update_yaxes(tickformat='$,.2f')  # $1,234.56
fig.update_yaxes(tickformat='$,.2s')  # $1.23M

# Percentage formatting
fig.update_yaxes(tickformat='.1%')  # 12.3%

# Date formatting
fig.update_xaxes(
    tickformat='%b %Y',  # Jan 2025
    dtick='M1'  # One tick per month
)
```

### Hover Template Patterns

```python
# Basic number with SI notation
hovertemplate='Value: %{y:,.2s}<extra></extra>'

# Currency with full precision
hovertemplate='Price: $%{y:,.2f}<extra></extra>'

# Percentage with sign
hovertemplate='Change: %{y:+.2%}<extra></extra>'

# Date + value combination
hovertemplate='%{x|%b %d, %Y}<br>%{y:$,.2f}<extra></extra>'

# Multi-field hover
hovertemplate=(
    '<b>%{customdata[0]}</b><br>'
    'Price: $%{y:,.2f}<br>'
    'Volume: %{customdata[1]:,.2s}<br>'
    'Change: %{customdata[2]:+.2%}'
    '<extra></extra>'
)
```

**Note:** The `<extra></extra>` tag removes the trace name from the hover box.

### Custom Formatting Functions

#### Human-Readable Number Formatter

```python
def format_number(value, prefix='', suffix='', decimals=2):
    """
    Format number with SI notation (k, M, B, T).

    Args:
        value: Number to format
        prefix: String prefix (e.g., '$')
        suffix: String suffix (e.g., '%')
        decimals: Decimal places (default 2)

    Returns:
        Formatted string (e.g., '$1.23M', '456k')

    Examples:
        >>> format_number(1234567, prefix='$')
        '$1.23M'
        >>> format_number(7860, suffix=' users')
        '7.86k users'
        >>> format_number(0.123, suffix='%', decimals=1)
        '12.3%'
    """
    if value is None or (isinstance(value, float) and (value != value)):  # Check for NaN
        return 'N/A'

    abs_value = abs(value)
    sign = '-' if value < 0 else ''

    if abs_value >= 1_000_000_000_000:
        formatted = f'{abs_value / 1_000_000_000_000:.{decimals}f}'.rstrip('0').rstrip('.')
        return f'{sign}{prefix}{formatted}T{suffix}'
    elif abs_value >= 1_000_000_000:
        formatted = f'{abs_value / 1_000_000_000:.{decimals}f}'.rstrip('0').rstrip('.')
        return f'{sign}{prefix}{formatted}B{suffix}'
    elif abs_value >= 1_000_000:
        formatted = f'{abs_value / 1_000_000:.{decimals}f}'.rstrip('0').rstrip('.')
        return f'{sign}{prefix}{formatted}M{suffix}'
    elif abs_value >= 1_000:
        formatted = f'{abs_value / 1_000:.{decimals}f}'.rstrip('0').rstrip('.')
        return f'{sign}{prefix}{formatted}k{suffix}'
    else:
        formatted = f'{abs_value:.{decimals}f}'.rstrip('0').rstrip('.')
        return f'{sign}{prefix}{formatted}{suffix}'
```

#### Currency Formatter

```python
def format_currency(value, decimals=2, si_threshold=1_000_000):
    """
    Format currency with appropriate precision.

    Args:
        value: Number to format
        decimals: Decimal places for small values
        si_threshold: Value above which to use SI notation

    Returns:
        Formatted currency string

    Examples:
        >>> format_currency(488.02)
        '$488.02'
        >>> format_currency(816890000)
        '$816.89M'
        >>> format_currency(0.9998)
        '$0.9998'
    """
    if value is None:
        return 'N/A'

    abs_value = abs(value)

    # Use SI notation for large values
    if abs_value >= si_threshold:
        return format_number(value, prefix='$')

    # Stablecoin precision (values near 1)
    if 0.9 <= abs_value <= 1.1:
        return f'${value:,.4f}'

    # Small values need more precision
    if abs_value < 0.01:
        return f'${value:,.6f}'

    # Standard 2-decimal precision
    return f'${value:,.{decimals}f}'
```

#### Percentage Formatter

```python
def format_percentage(value, decimals=1, show_sign=True):
    """
    Format percentage with optional sign indicator.

    Args:
        value: Decimal value (0.123 = 12.3%)
        decimals: Decimal places
        show_sign: Include + for positive values

    Returns:
        Formatted percentage string

    Examples:
        >>> format_percentage(0.123)
        '+12.3%'
        >>> format_percentage(-0.052)
        '-5.2%'
        >>> format_percentage(0.123, show_sign=False)
        '12.3%'
    """
    if value is None:
        return 'N/A'

    pct_value = value * 100

    if show_sign and value > 0:
        return f'+{pct_value:.{decimals}f}%'
    else:
        return f'{pct_value:.{decimals}f}%'
```

#### Date Range Formatter

```python
def format_date_range(start_date, end_date):
    """
    Format date range for display in titles/subtitles.

    Args:
        start_date: Start datetime
        end_date: End datetime

    Returns:
        Formatted date range string

    Examples:
        >>> format_date_range(datetime(2025, 1, 1), datetime(2025, 1, 31))
        'Jan 1 - Jan 31, 2025'
        >>> format_date_range(datetime(2024, 12, 1), datetime(2025, 1, 31))
        'Dec 1, 2024 - Jan 31, 2025'
    """
    if start_date.year == end_date.year:
        return f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}"
    else:
        return f"{start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}"
```

---

## Color-Coded Text

### Percentage Change Colors

```python
def get_change_color(value):
    """Return color based on positive/negative value."""
    if value > 0:
        return '#34D399'  # Green
    elif value < 0:
        return '#F87171'  # Red
    else:
        return '#d3d4d6'  # Neutral (text color)


def format_colored_change(value, decimals=1):
    """Format percentage change with color HTML."""
    color = get_change_color(value)
    formatted = format_percentage(value, decimals=decimals, show_sign=True)
    return f'<span style="color: {color}">{formatted}</span>'
```

### Applying Colors in Plotly

```python
# Colored text annotations
fig.add_annotation(
    x=x_pos,
    y=y_pos,
    text=f'<span style="color: #34D399">+12.3%</span>',
    showarrow=False,
    font=dict(size=14)
)

# Conditional bar colors
colors = ['#34D399' if v > 0 else '#F87171' for v in df['change']]
fig.add_trace(go.Bar(
    x=df['category'],
    y=df['change'],
    marker=dict(color=colors)
))
```

---

## Complete Example

### Chart with Properly Formatted Elements

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sample data
dates = ['2025-01-01', '2025-01-02', '2025-01-03', '2025-01-04', '2025-01-05']
prices = [45000, 46200, 45800, 47500, 48200]
volumes = [1_234_000_000, 1_456_000_000, 1_123_000_000, 1_789_000_000, 1_567_000_000]

# Create dual-axis chart
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Volume bars (primary y-axis)
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

# Price line (secondary y-axis)
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

# Layout with proper formatting
fig.update_layout(
    title=dict(
        text='BTC Price & Volume<br><sup>Last 5 days, hourly data</sup>',
        font=dict(size=16, color='#d3d4d6'),
        x=0,
        xanchor='left'
    ),
    paper_bgcolor='#0e1729',
    plot_bgcolor='#0e1729',
    font=dict(color='#d3d4d6'),
    hovermode='x unified',
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='left',
        x=0
    )
)

# Y-axis formatting
fig.update_yaxes(
    title_text='Volume (24h)',
    tickformat='$,.2s',  # SI notation with $ prefix
    secondary_y=False,
    gridcolor='rgba(255, 255, 255, 0.1)'
)

fig.update_yaxes(
    title_text='Price (USD)',
    tickformat='$,.0f',  # Currency, no decimals for large values
    secondary_y=True,
    showgrid=False
)

# X-axis formatting
fig.update_xaxes(
    tickformat='%b %d',  # "Jan 01" format
    gridcolor='rgba(255, 255, 255, 0.1)'
)
```

---

## Quick Reference

### Number Format Cheat Sheet

| Value Type | Plotly tickformat | Example Output |
|------------|-------------------|----------------|
| Large number | `,.2s` | 1.23M |
| Currency (exact) | `$,.2f` | $1,234.56 |
| Currency (large) | `$,.2s` | $1.23M |
| Percentage | `.1%` | 12.3% |
| Percentage (signed) | `+.1%` | +12.3% |
| Integer | `,.0f` | 1,234,567 |

### Date Format Cheat Sheet

| Timeframe | tickformat | dtick | Example |
|-----------|------------|-------|---------|
| Hourly | `%H:%M` | `3600000` | 14:30 |
| Daily | `%b %d` | `86400000` | Jan 15 |
| Weekly | `%b %d` | `604800000` | Jan 15 |
| Monthly | `%b %Y` | `M1` | Jan 2025 |
| Yearly | `%Y` | `M12` | 2025 |

### Template Colors

| Element | Color |
|---------|-------|
| Text | #d3d4d6 |
| Background | #0e1729 |
| Positive/Green | #34D399 |
| Negative/Red | #F87171 |
| Primary Blue | #60A5FA |
| Grid (subtle) | rgba(255, 255, 255, 0.1) |

---

## Formatting Checklist

Before finalizing any chart, verify:

- [ ] Large numbers use SI notation (k, M, B, T)
- [ ] Currency values have `$` prefix
- [ ] Percentage changes show sign (`+` or `-`)
- [ ] Percentage changes are color-coded (green/red)
- [ ] Dates match the timeframe granularity
- [ ] Hover templates show full, readable information
- [ ] Axis tick labels are not overcrowded
- [ ] Title clearly describes the chart content
- [ ] Units are included where not obvious

## Title Positioning

By default, Plotly centers titles. For a more professional, report-style appearance, left-align titles using `title=dict(text='Your Title', x=0, xanchor='left')`. This creates better visual hierarchy and follows common data journalism conventions.

## Number Formatting for Large Values

For financial data in billions, use clear formatting like '$50B' rather than mixing units (e.g., '$184' when the axis label says 'billions'). When values are already in billions in the data, format tick labels as `format_with_B(val * 1e9, decimals=0)` to show '$20B', '$40B', etc. Alternatively, show plain numbers like '50' with an axis label '(USD, billions)'. Never show raw large numbers without context.

## Bar Labels

**When to use bar labels:** Only add direct labels when there are few bars (roughly ≤12) and the values are important to show precisely. Skip labels for:
- Time series with many bars (e.g., daily data over weeks/months)
- Stacked bar charts (always skip - use hover for segment values)
- Any chart where labels would overlap or require tiny font sizes

**When labels are appropriate**, position them intelligently based on bar length. Use `textposition=['inside' if abs(val) > threshold else 'outside' for val in values]` where threshold is determined by the data range. This ensures labels are readable without overlapping.

```python
# Only add labels if bar count is reasonable
if len(df) <= 12:
    fig.update_traces(
        text=df['value'].apply(format_with_B),
        textposition='outside'
    )
```

For charts with many bars, rely on hover tooltips and clear axis labels instead of direct bar labels.

## Number Formatting with Units

When formatting axis labels with abbreviated units (k/M/B/T), always include both the currency prefix AND the unit suffix. Use custom formatting functions that accept both parameters: `format_with_B(value * 1e9, prefix='$', decimals=0)` for axis labels. The prefix ('$') and suffix ('B') must both be present in the final output (e.g., '$50B').

## Significant Figures

For hover values displaying abbreviated numbers, implement 3 significant figures by adjusting decimal places based on magnitude: 0 decimals for values ≥100 in their unit (e.g., '150B'), 1 decimal for ≥10 (e.g., '15.5B'), 2 decimals for <10 (e.g., '1.55B'). Create a dedicated formatting function that handles this logic across all magnitude ranges (k/M/B/T).

## Decimal Precision for Axis Labels

For axis tick labels displaying large rounded values (billions, millions), omit decimal places entirely by setting `decimals=0` in your formatting function. This is especially important when tick intervals are whole numbers (e.g., '$50B' not '$50.0B').

## Number Formatting Functions

Create two separate formatting functions: (1) `format_with_B()` for axis ticks with configurable decimals (often 0 or 1), and (2) `format_sig_figs()` for hover tooltips with significant figures (typically 3). The sig_figs function should calculate decimals dynamically based on the scaled value (e.g., 123B needs 0 decimals, 12.3B needs 1, 1.23B needs 2) to maintain consistent precision across different magnitudes.

## Significant Figures Formatting

When formatting to significant figures (not fixed decimals), calculate decimal places dynamically based on the scaled value's magnitude. For N significant figures: if scaled_value ≥ 10^(N-1), use 0 decimals; if scaled_value ≥ 10^(N-2), use 1 decimal; etc. This ensures consistent precision across different magnitudes (e.g., 1.23B, 456M, 78.9k all have 3 sig figs).
