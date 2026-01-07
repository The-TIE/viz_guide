# Plotly Visualization Guide

> A comprehensive decision framework for AI-assisted chart generation.
> Given a dataset and visualization intent, produce publication-ready Plotly charts.

---

## How to Use This Guide

This guide is structured as a decision framework. When generating a visualization:

1. **Analyze the data** (Section 1) - Understand what you're working with
2. **Clarify the intent** (Section 2) - What question/story does this serve?
3. **Select chart type** (Section 3) - Match data + intent to visualization
4. **Apply encoding rules** (Section 4) - Map data to visual properties
5. **Configure axes** (Section 5) - Scales, ranges, formatting
6. **Apply color** (Section 6) - Palette selection, emphasis
7. **Format text** (Section 7) - Numbers, labels, titles
8. **Design layout** (Section 8) - Margins, spacing, arrangement
9. **Configure legends** (Section 9) - Placement, necessity
10. **Set up hover** (Section 10) - Tooltips, unified vs individual
11. **Configure interactions** (Section 11) - Zoom, pan, controls
12. **Add annotations** (Section 12) - Sources, timestamps, markers
13. **Validate accessibility** (Section 13) - Color, contrast, clarity
14. **Optimize performance** (Section 14) - Large data handling

---

## Section Index

### Part I: Analysis & Selection

- [01 - Data Analysis](01_data_analysis.md)
  - Data types (temporal, categorical, numerical, hierarchical)
  - Dimensionality (univariate, bivariate, multivariate)
  - Data characteristics (volume, density, outliers, missing values)
  - Aggregation requirements

- [02 - Intent Classification](02_intent.md)
  - Question types (comparison, composition, distribution, relationship, change over time)
  - Audience considerations
  - Exploratory vs explanatory
  - Dashboard context vs standalone

- [03 - Chart Type Selection](03_chart_selection.md)
  - Decision tree: data type + intent → chart type
  - Single chart vs subplots vs small multiples
  - Combining chart types (bar + line, etc.)
  - When NOT to use certain charts
  - Chart type reference with use cases

### Part II: Visual Encoding

- [04 - Data Encoding](04_encoding.md)
  - Position encoding (x, y, z)
  - Color encoding (hue, saturation, lightness)
  - Size encoding (area, length, radius)
  - Shape encoding (markers, symbols)
  - Line encoding (style, width)
  - Opacity encoding
  - Encoding priorities and limits

- [05 - Axis Configuration](05_axes.md)
  - Scale types (linear, log, date, categorical)
  - Range determination (auto, fixed, include zero?)
  - Tick configuration (count, format, rotation)
  - Axis labels (placement, formatting)
  - Secondary/dual axes (when appropriate, alignment)
  - Shared axes in subplots
  - Reversed axes

- [06 - Color](06_color.md)
  - Categorical palettes (when, which)
  - Sequential palettes (when, which)
  - Diverging palettes (when, which)
  - Emphasis colors (highlighting specific series)
  - De-emphasis (graying out, reducing opacity)
  - Consistent color mapping across related charts
  - Accessibility requirements

### Part III: Text & Labels

- [07 - Text Formatting](07_text.md)
  - Number formatting rules
    - Significant figures (3 sig figs standard)
    - Human-readable scaling (1.23k, 12.3M, 123B, 1.23T)
    - Currency formatting ($1.23M)
    - Percentage formatting (12.3%)
    - Decimal precision by context
  - Date/time formatting
    - Axis tick formats by timeframe
    - Hover date formats
  - Title hierarchy
    - Chart title (what + context)
    - Subtitle (additional context, filters applied)
    - Axis titles (unit indication)
  - Label truncation and wrapping

### Part IV: Layout & Composition

- [08 - Layout](08_layout.md)
  - Margins
    - Base margins
    - Auto-adjustment for long tick labels
    - Title/legend accommodation
  - Aspect ratios
    - Chart type defaults
    - Wide vs square vs tall
  - Subplot arrangements
    - Grid layouts (rows x cols)
    - Shared axes patterns
    - Vertical spacing
    - Horizontal spacing
  - Responsive considerations
    - Minimum viable sizes
    - Mobile adaptations

- [09 - Legends](09_legends.md)
  - When to include legends
    - Required: multiple series without inline labels
    - Optional: single series (can omit)
    - Never: when inline labels are clearer
  - Placement
    - Horizontal above chart (default for ≤5 items)
    - Vertical right side (for many items)
    - Inside chart (when space permits)
  - Legend formatting
    - Item order (match visual order)
    - Grouping related items
  - Interactive legend behavior (click to toggle)

### Part V: Interactivity

- [10 - Hover & Tooltips](10_hover.md)
  - Hover mode selection
    - `x unified`: comparing multiple series at same x
    - `closest`: detailed single-point inspection
    - `x`: time series default
  - Hover template design
    - Essential information hierarchy
    - Formatting within hover
    - Custom hover templates
  - Spike lines (when useful)

- [11 - Interaction Modes](11_interactions.md)
  - Zoom behavior by chart type
    - Time series: x-axis zoom (horizontal)
    - Scatter: box zoom (2D)
    - Bar charts: often disable zoom
  - Pan behavior
    - When to enable
    - Constrained panning
  - Range sliders
    - Long time series
    - Default visible range
  - Buttons and dropdowns
    - Trace visibility toggles
    - Time range presets
    - Metric switching
  - Reset behavior

### Part VI: Annotations & Context

- [12 - Annotations](12_annotations.md)
  - Required annotations
    - Data source attribution
    - Last updated timestamp
    - Generation timestamp (for static exports)
  - Reference lines
    - Horizontal (thresholds, averages, targets)
    - Vertical (events, dates)
    - Styling (dashed, labeled)
  - Event markers
    - Point annotations
    - Range annotations (shaded regions)
    - Text placement strategies
  - Callouts and notes
    - When to use
    - Positioning

### Part VII: Quality & Performance

- [13 - Accessibility](13_accessibility.md)
  - Color blindness considerations
    - Safe palette selection
    - Redundant encoding (color + shape)
  - Contrast requirements
    - Text on backgrounds
    - Line visibility
  - Screen reader considerations
    - Alt text patterns
    - Semantic structure

- [14 - Performance](14_performance.md)
  - Large dataset handling
    - When to aggregate
    - Downsampling strategies
    - WebGL mode (scattergl, etc.)
  - Animation considerations
    - When to animate
    - Performance limits
  - Static export optimization

---

## Quick Reference

### Chart Type Decision Matrix

| Data Type | Intent | Recommended Chart |
|-----------|--------|-------------------|
| Time series, single | Show trend | Line chart |
| Time series, multiple | Compare trends | Multi-line or small multiples |
| Time series + volume | Trend + magnitude | Line + bar combo |
| Categorical, single measure | Compare categories | Horizontal bar |
| Categorical, multiple measures | Compare across measures | Grouped bar or small multiples |
| Part-to-whole, few categories | Show composition | Donut (not pie) |
| Part-to-whole, over time | Show composition change | Stacked area |
| Distribution, single | Show spread | Histogram or box plot |
| Distribution, compare | Compare spreads | Ridgeline or violin |
| Two numerical variables | Show relationship | Scatter plot |
| Three numerical variables | Relationship + magnitude | Bubble chart |
| Hierarchical | Show structure | Treemap or sunburst |

### Number Formatting Quick Reference

| Value | Formatted | Context |
|-------|-----------|---------|
| 1234 | 1.23k | General |
| 1234567 | 1.23M | General |
| 1234567890 | 1.23B | General |
| 0.1234 | 12.3% | Percentage |
| 1234567 | $1.23M | Currency |
| 0.001234 | 0.00123 | Small decimals |

### Standard Margins (pixels)

| Element | Margin |
|---------|--------|
| Base top | 60 |
| Base bottom | 60 |
| Base left | 60 |
| Base right | 30 |
| With title | top + 40 |
| With x-axis title | bottom + 30 |
| With long y-labels | left + (max_label_length * 7) |
| With legend right | right + 120 |

---

## Template Reference

Base template location: `~/repositories/token_labs_python/token_labs/visualization/plotly.py`

Standard colorway:
```python
["#60A5FA", "#F87171", "#34D399", "#FBBF24", "#E879F9", ...]
```

Background: `#0e1729`
Text: `#d3d4d6`

---

## Sections Status

- [ ] 01 - Data Analysis
- [ ] 02 - Intent Classification
- [ ] 03 - Chart Type Selection
- [ ] 04 - Data Encoding
- [ ] 05 - Axis Configuration
- [ ] 06 - Color
- [ ] 07 - Text Formatting
- [ ] 08 - Layout
- [ ] 09 - Legends
- [ ] 10 - Hover & Tooltips
- [ ] 11 - Interaction Modes
- [ ] 12 - Annotations
- [ ] 13 - Accessibility
- [ ] 14 - Performance
