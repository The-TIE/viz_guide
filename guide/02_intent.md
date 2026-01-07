# 02 - Intent Classification

> Understand what question a visualization needs to answer.
> Matching visualization to intent is the foundation of effective data communication.

---

## Why Intent Matters

Before selecting a chart type, you must understand:

1. **What question are we answering?** - The analytical purpose
2. **Who is the audience?** - Technical depth and familiarity
3. **What is the context?** - Exploration vs presentation, standalone vs dashboard

Getting intent wrong leads to charts that technically display data but fail to communicate insights.

---

## Question Types (Visualization Intents)

### The Seven Core Intents

Every visualization serves one of these analytical purposes:

```
INTENT:
  - COMPARISON: How does A compare to B?
  - COMPOSITION: What are the parts of the whole?
  - DISTRIBUTION: How is the data spread?
  - RELATIONSHIP: How are variables related?
  - CHANGE_OVER_TIME: What is the trend?
  - RANKING: What is the order?
  - DEVIATION: How far from a reference?
  - GEOSPATIAL: Where?
```

---

### Comparison

**Question:** "How does A compare to B?"

| Signal Words | Data Pattern | Primary Charts |
|--------------|--------------|----------------|
| compare, versus, vs, difference, between, relative, contrast | Multiple categories or series with shared metric | Bar chart, Grouped bar, Multi-line |

**Examples:**
- "Compare trading volumes across exchanges"
- "How does BTC performance differ from ETH?"
- "Contrast Q1 vs Q2 revenue"

**Key Considerations:**
- Same scale required for valid comparison
- Order categories meaningfully (alphabetical, by value, or by logical grouping)
- Limit to 5-7 items for direct comparison; use small multiples for more

**Decision Logic:**

```
COMPARISON
├── Across categories (static)
│   ├── Single metric → Horizontal Bar
│   └── Multiple metrics → Grouped Bar or Small Multiples
│
├── Across categories (over time)
│   ├── Few series (2-5) → Multi-line Chart
│   └── Many series (6+) → Small Multiples
│
└── Two specific values
    └── → Bullet Chart or Paired Bar
```

---

### Composition

**Question:** "What makes up the whole?"

| Signal Words | Data Pattern | Primary Charts |
|--------------|--------------|----------------|
| breakdown, share, portion, percentage, allocation, composition, part of, consists of | Parts that sum to a meaningful total | Donut, Stacked bar, Stacked area, Treemap |

**Examples:**
- "What's the market share breakdown?"
- "How is the portfolio allocated?"
- "What portion of volume comes from each exchange?"

**Key Considerations:**
- Parts must logically sum to 100% or a meaningful total
- Maximum 6-7 categories before aggregating to "Other"
- Order slices by size (largest first) unless there's a logical order

**Decision Logic:**

```
COMPOSITION
├── Single point in time
│   ├── Few categories (2-6) → Donut Chart
│   ├── Many categories (7+) → Stacked Bar (horizontal) or Treemap
│   └── Hierarchical → Treemap or Sunburst
│
├── Over time
│   ├── Show absolute values → Stacked Area
│   └── Show percentages (100%) → 100% Stacked Area
│
└── Nested/hierarchical
    ├── 2 levels → Treemap
    └── 3+ levels → Sunburst
```

---

### Distribution

**Question:** "How is the data spread?"

| Signal Words | Data Pattern | Primary Charts |
|--------------|--------------|----------------|
| distribution, spread, range, frequency, density, histogram, outliers, variance, typical, normal | Continuous numeric values | Histogram, Box plot, Violin, Density |

**Examples:**
- "What's the distribution of trade sizes?"
- "How are returns spread?"
- "Are there outliers in the pricing data?"

**Key Considerations:**
- Bin width significantly affects histogram interpretation
- Consider whether showing outliers explicitly is important
- Box plots hide multimodal distributions; violin plots reveal them

**Decision Logic:**

```
DISTRIBUTION
├── Single variable
│   ├── Show shape/frequency → Histogram
│   ├── Show quartiles/outliers → Box Plot
│   └── Show full density → Violin Plot
│
├── Compare distributions
│   ├── Few groups (2-4) → Overlapping Histograms or Grouped Box Plots
│   ├── Many groups (5+) → Ridgeline (Joy Plot) or Small Multiples
│   └── Compare statistical summaries → Multiple Box Plots
│
└── Over time
    └── → Box Plots by Time Period or Heatmap
```

---

### Relationship

**Question:** "How are variables related?"

| Signal Words | Data Pattern | Primary Charts |
|--------------|--------------|----------------|
| correlation, relationship, association, connected, linked, impact, affect, cause, driver | Two or more numeric variables | Scatter, Bubble, Heatmap (correlation matrix) |

**Examples:**
- "Is there a relationship between volume and price?"
- "How correlated are these assets?"
- "Does market cap affect volatility?"

**Key Considerations:**
- Correlation is not causation - label appropriately
- Consider adding trend/regression lines
- For many pairwise relationships, use correlation heatmaps

**Decision Logic:**

```
RELATIONSHIP
├── Two numeric variables
│   └── → Scatter Plot
│
├── Two numeric + one categorical
│   └── → Scatter with Color Encoding
│
├── Two numeric + one numeric (third dimension)
│   └── → Bubble Chart (size encoding)
│
├── Many pairwise relationships
│   └── → Correlation Matrix Heatmap
│
└── Hierarchical/network relationships
    └── → Network Graph or Sankey Diagram
```

---

### Change Over Time (Trend)

**Question:** "How has it changed?"

| Signal Words | Data Pattern | Primary Charts |
|--------------|--------------|----------------|
| trend, over time, change, growth, decline, history, evolution, timeline, forecast | Time-indexed data | Line, Area, Bar (time-based) |

**Examples:**
- "How has the price trended?"
- "Show the growth over the past year"
- "What's the historical pattern?"

**Key Considerations:**
- Time should almost always be on x-axis
- Use area charts only when magnitude from zero matters
- Consider whether to show actual values vs percentage change

**Decision Logic:**

```
CHANGE_OVER_TIME
├── Single series
│   ├── Emphasize trend → Line Chart
│   └── Emphasize magnitude from zero → Area Chart
│
├── Multiple series
│   ├── Same scale, few series (2-5) → Multi-line Chart
│   ├── Different scales, 2 series → Dual-axis Chart
│   └── Many series (6+) → Small Multiples
│
├── Show composition over time
│   ├── Absolute values → Stacked Area
│   └── Percentages → 100% Stacked Area
│
├── Discrete time periods
│   └── → Vertical Bar Chart
│
└── Two time points (before/after)
    └── → Slope Chart
```

---

### Ranking

**Question:** "What's the order?"

| Signal Words | Data Pattern | Primary Charts |
|--------------|--------------|----------------|
| top, bottom, rank, best, worst, highest, lowest, leading, lagging, leaderboard | Categorical with sortable values | Horizontal bar (sorted), Dot plot |

**Examples:**
- "What are the top 10 exchanges by volume?"
- "Rank the assets by market cap"
- "Which categories perform worst?"

**Key Considerations:**
- Sort by the ranking metric (not alphabetically)
- Consider showing only top/bottom N for long lists
- Include the ranking number if position matters

**Decision Logic:**

```
RANKING
├── Single metric
│   ├── Few items (≤15) → Horizontal Bar (sorted)
│   └── Many items (15+) → Show Top N + "Other"
│
├── With change indication
│   └── → Bar with arrows or Bump Chart (rank over time)
│
└── Multiple metrics per item
    └── → Dot Plot (Cleveland) or Parallel Coordinates
```

---

### Deviation

**Question:** "How far from the reference?"

| Signal Words | Data Pattern | Primary Charts |
|--------------|--------------|----------------|
| above, below, deviation, variance, versus target, benchmark, expected, actual vs, difference from | Values compared against a baseline | Diverging bar, Bullet, Deviation line |

**Examples:**
- "How does performance compare to the benchmark?"
- "Which regions are above/below target?"
- "Show deviation from the average"

**Key Considerations:**
- Clearly mark the reference point/line
- Use diverging colors (e.g., green above, red below)
- Consider both absolute deviation and percentage deviation

**Decision Logic:**

```
DEVIATION
├── Categories vs reference
│   ├── Single reference value → Diverging Bar Chart
│   └── Individual targets → Bullet Chart
│
├── Time series vs reference
│   ├── Fixed reference → Line with Reference Line
│   └── Rolling reference (e.g., moving average) → Line with Shaded Band
│
└── Actual vs expected
    └── → Bullet Chart or Variance Bar
```

---

### Geospatial

**Question:** "Where?"

| Signal Words | Data Pattern | Primary Charts |
|--------------|--------------|----------------|
| where, location, geographic, map, region, country, state, city, coordinates | Data with geographic dimension | Choropleth, Bubble map, Point map |

**Examples:**
- "Where are users located?"
- "Which regions have highest adoption?"
- "Show trading volume by country"

**Key Considerations:**
- Maps can be misleading (large areas dominate visually)
- Consider if a bar chart by region might be clearer
- Ensure color scales are accessible

**Decision Logic:**

```
GEOSPATIAL
├── Regional aggregates
│   └── → Choropleth Map
│
├── Point locations
│   ├── Few points → Point Map with Labels
│   └── Many points → Heatmap or Clustered Points
│
├── Flows between locations
│   └── → Flow Map or Connection Map
│
└── Location + magnitude
    └── → Bubble Map (size = magnitude)
```

---

## Audience Considerations

### Technical vs Non-Technical

| Audience | Characteristics | Visualization Approach |
|----------|-----------------|----------------------|
| **Technical** (analysts, developers) | Comfortable with complex charts, wants detail, can interpret statistics | More data-dense charts, include statistical measures, advanced chart types acceptable |
| **Non-Technical** (executives, general audience) | Needs immediate clarity, limited time, unfamiliar with statistics | Simpler charts, clear titles, explicit labels, limit to 2-3 key insights |

**Adjustments for Non-Technical Audiences:**

```
SIMPLIFICATION_RULES:
  - Prefer bar charts over complex alternatives
  - Avoid: box plots, violin plots, parallel coordinates
  - Use: simple lines, bars, donuts
  - Add explicit annotations pointing out key insights
  - Remove statistical notation (p-values, confidence intervals)
  - Use absolute numbers, not percentages/ratios when possible
```

**Adjustments for Technical Audiences:**

```
ENHANCEMENT_RULES:
  - Can include: box plots, violin plots, scatter matrices
  - Show confidence intervals, error bars
  - Include statistical measures in tooltips
  - Enable more detailed zoom/exploration
  - Provide access to underlying data
```

---

### Exploratory vs Explanatory

| Mode | Purpose | Visualization Approach |
|------|---------|----------------------|
| **Exploratory** (analyst working) | Find patterns, investigate hypotheses, iterate quickly | Interactive, multiple views, raw data visible, less polish |
| **Explanatory** (presentation) | Communicate specific finding, tell a story | Focused message, annotations, refined design, guided attention |

**Exploratory Visualization:**

```python
# Exploratory: Interactive, multi-view, all data visible
EXPLORATORY_SETTINGS:
  - Enable all interactions (zoom, pan, select)
  - Show all data points (don't aggregate prematurely)
  - Include detailed tooltips with all fields
  - Use small multiples for comparison
  - Allow toggling traces on/off
  - Include range sliders for time navigation
```

**Explanatory Visualization:**

```python
# Explanatory: Focused, guided, polished
EXPLANATORY_SETTINGS:
  - Title states the insight, not just the topic
  - Add annotations pointing to key findings
  - Remove non-essential elements
  - Use consistent, branded colors
  - Limit interactivity (or make it optional)
  - Ensure works well as static export (for slides)
```

---

### Dashboard Context vs Standalone Chart

| Context | Requirements | Design Approach |
|---------|--------------|-----------------|
| **Dashboard** | Space-constrained, part of larger story, needs to work with other charts | Compact, consistent with neighbors, avoid redundant legends, shared color schemes |
| **Standalone** | Self-contained, single focus, can use full space | More detail, complete context, full legends, comprehensive tooltips |

**Dashboard Chart Adjustments:**

```
DASHBOARD_RULES:
  - Reduce margins (charts sit closer together)
  - Use shared legends when possible
  - Coordinate colors across related charts
  - Smaller titles (hierarchy managed by dashboard layout)
  - Consider linked interactions (cross-filtering)
  - Ensure charts answer different questions (avoid redundancy)
```

**Standalone Chart Considerations:**

```
STANDALONE_RULES:
  - Include full context in title/subtitle
  - Complete legends (chart must be self-explanatory)
  - Add data source attribution
  - Consider both interactive and static export use cases
  - More generous margins and spacing
```

---

## Context Factors

### Purpose Decision Tree

```
Is this for exploration or communication?
├── EXPLORATION
│   ├── Enable zoom, pan, selection
│   ├── Show all data points
│   ├── Include detailed tooltips
│   ├── Use linked views if multiple charts
│   └── Prioritize flexibility over aesthetics
│
└── COMMUNICATION
    ├── What's the one key message?
    │   ├── State it in the title
    │   ├── Use annotations to guide attention
    │   └── Remove everything that doesn't support the message
    │
    ├── Will it be presented live or shared asynchronously?
    │   ├── Live presentation → Can explain verbally, simpler chart ok
    │   └── Async (email, report) → Must be self-explanatory
    │
    └── What's the emotional impact needed?
        ├── Urgency/alarm → Red accents, clear deviation markers
        ├── Success/achievement → Green accents, goal lines
        └── Neutral/informational → Standard palette
```

### Single Chart vs Dashboard Component

```
Is this a single chart or part of a dashboard?
├── SINGLE_CHART
│   ├── Full title with context
│   ├── Complete legend
│   ├── Source attribution
│   ├── Standard margins
│   └── Can use full-page dimensions
│
└── DASHBOARD_COMPONENT
    ├── Where does this chart fit in the story?
    │   ├── Primary (hero chart) → Larger, more prominent
    │   ├── Supporting → Smaller, less detail
    │   └── Reference/context → Minimal, just essential info
    │
    ├── What other charts is it adjacent to?
    │   ├── Share color schemes with related charts
    │   ├── Coordinate time ranges
    │   └── Consider linked interactions
    │
    └── Space allocation
        ├── Tight space → Remove axis titles, shrink margins
        ├── Generous space → Standard formatting
        └── Very small (sparkline) → Minimal, trend-only
```

### Interactive vs Static

```
Will this be interactive or static?
├── INTERACTIVE
│   ├── Hover tooltips essential
│   ├── Consider zoom/pan needs
│   │   ├── Time series → x-axis zoom, range slider
│   │   ├── Scatter plots → Box select, zoom
│   │   └── Bar charts → Usually disable zoom
│   │
│   ├── Legend interactivity
│   │   ├── Many series → Enable click-to-toggle
│   │   └── Few series → Keep always visible
│   │
│   └── Performance considerations
│       ├── >10k points → Use WebGL (scattergl)
│       └── Complex animations → Test on target devices
│
└── STATIC (export for PDF, slide, image)
    ├── All essential info must be visible
    │   ├── Key values labeled directly on chart
    │   ├── No hidden information in tooltips
    │   └── Complete legend always shown
    │
    ├── Resolution considerations
    │   ├── Print → High DPI export (300+)
    │   └── Screen → Standard resolution ok
    │
    └── Remove unnecessary interactive elements
        ├── Mode bar → Hide
        ├── Range slider → Usually remove
        └── Zoom controls → Remove
```

### Screen Size Considerations

```
What screen size is the primary target?
├── DESKTOP (1920x1080 or larger)
│   ├── Full chart dimensions (800-1200px width)
│   ├── Standard font sizes
│   ├── Detailed tooltips acceptable
│   └── Multiple charts can share screen
│
├── LAPTOP (1366x768 typical)
│   ├── Medium chart dimensions (600-900px width)
│   ├── Slightly condensed layout
│   ├── Consider chart height for visibility
│   └── Scrolling may be needed for dashboards
│
├── TABLET (768-1024px width)
│   ├── Single chart focus
│   ├── Touch-friendly interactions
│   │   ├── Larger tap targets
│   │   └── Tap instead of hover
│   └── Simplified legends
│
└── MOBILE (320-480px width)
    ├── STRONGLY consider if chart is necessary
    │   └── Text summaries often better
    │
    ├── If chart needed:
    │   ├── Single series only
    │   ├── Minimal annotations
    │   ├── Larger fonts
    │   ├── Vertical orientation
    │   ├── Swipe for time navigation
    │   └── Numbers as primary display, chart as support
    │
    └── Mobile-hostile chart types:
        ├── Scatter plots (too dense)
        ├── Small multiples (too small)
        ├── Heatmaps (labels unreadable)
        └── Complex interactivity (hover doesn't work)
```

---

## AI Agent Intent Classification

### Parsing Natural Language Requests

When processing a visualization request, extract these elements:

```
REQUEST_PARSING:
  1. EXPLICIT_INTENT: Keywords directly stating intent
     - "compare", "trend", "distribution", "breakdown", "correlation"

  2. IMPLIED_INTENT: Inferred from question structure
     - "How has X changed?" → CHANGE_OVER_TIME
     - "What are the top X?" → RANKING
     - "What makes up X?" → COMPOSITION

  3. DATA_REFERENCES: What data is mentioned
     - Time periods ("last 30 days", "YTD", "Q1")
     - Entities ("exchanges", "assets", "regions")
     - Metrics ("volume", "price", "returns")

  4. AUDIENCE_SIGNALS: Who will see this
     - "for the board" → Non-technical, explanatory
     - "for my analysis" → Technical, exploratory
     - "for the dashboard" → Dashboard context
```

### Intent Detection Patterns

| Pattern | Detected Intent | Confidence |
|---------|-----------------|------------|
| "compare X to Y" | COMPARISON | High |
| "how does X compare" | COMPARISON | High |
| "X vs Y" | COMPARISON | High |
| "difference between" | COMPARISON | High |
| "breakdown of X" | COMPOSITION | High |
| "what makes up X" | COMPOSITION | High |
| "share of" | COMPOSITION | High |
| "portion/percentage of" | COMPOSITION | High |
| "how is X distributed" | DISTRIBUTION | High |
| "spread of X" | DISTRIBUTION | High |
| "range of X" | DISTRIBUTION | Medium |
| "outliers in X" | DISTRIBUTION | High |
| "correlation between" | RELATIONSHIP | High |
| "relationship between" | RELATIONSHIP | High |
| "how does X affect Y" | RELATIONSHIP | High |
| "trend of X" | CHANGE_OVER_TIME | High |
| "how has X changed" | CHANGE_OVER_TIME | High |
| "X over time" | CHANGE_OVER_TIME | High |
| "historical X" | CHANGE_OVER_TIME | High |
| "growth of X" | CHANGE_OVER_TIME | High |
| "top N" | RANKING | High |
| "rank X by" | RANKING | High |
| "best/worst X" | RANKING | High |
| "leading/lagging" | RANKING | High |
| "X above/below Y" | DEVIATION | High |
| "deviation from" | DEVIATION | High |
| "versus target" | DEVIATION | High |
| "where is X" | GEOSPATIAL | High |
| "by location/region/country" | GEOSPATIAL | Medium |

### Ambiguous Request Handling

When intent is unclear, prioritize by data characteristics:

```
DISAMBIGUATION_PRIORITY:
  1. If data has datetime index → Default to CHANGE_OVER_TIME
  2. If data has geographic column → Consider GEOSPATIAL
  3. If request mentions "by category" → Default to COMPARISON
  4. If single numeric variable mentioned → Default to DISTRIBUTION
  5. If two numeric variables mentioned → Default to RELATIONSHIP
  6. If percentages/shares mentioned → Default to COMPOSITION
```

### Example Request Classifications

**Request:** "Show me BTC price over the last 30 days"

```
CLASSIFICATION:
  Intent: CHANGE_OVER_TIME
  Confidence: High
  Signals: "over the last 30 days" (explicit time reference)
  Data: BTC price (single time series)
  Recommended: Line chart
```

**Request:** "What's the trading volume breakdown by exchange?"

```
CLASSIFICATION:
  Intent: COMPOSITION
  Confidence: High
  Signals: "breakdown by" (composition keyword)
  Data: Volume by exchange (categorical)
  Recommended: Donut chart (if few exchanges) or Horizontal bar (if many)
```

**Request:** "How do the top exchanges compare?"

```
CLASSIFICATION:
  Intent: COMPARISON (primary) + RANKING (secondary)
  Confidence: High
  Signals: "compare" (comparison), "top" (ranking)
  Data: Multiple exchanges (categorical)
  Recommended: Horizontal bar chart (sorted by value)
```

**Request:** "Is there a correlation between volume and volatility?"

```
CLASSIFICATION:
  Intent: RELATIONSHIP
  Confidence: High
  Signals: "correlation between" (explicit relationship keyword)
  Data: Two numeric variables
  Recommended: Scatter plot with trend line
```

**Request:** "Show the data for all assets"

```
CLASSIFICATION:
  Intent: AMBIGUOUS
  Confidence: Low
  Action: Ask clarifying questions
  Questions:
    - "What aspect of the assets? (price, volume, market cap)"
    - "Over what time period?"
    - "Are you looking to compare them or see individual trends?"
```

---

## Intent-to-Chart Quick Reference

### Primary Intent Mapping

| Intent | First Choice | Alternative | Avoid |
|--------|--------------|-------------|-------|
| COMPARISON | Horizontal bar | Grouped bar, Multi-line | Pie chart |
| COMPOSITION | Donut | Stacked bar, Treemap | 3D pie |
| DISTRIBUTION | Histogram | Box plot, Violin | Bar chart |
| RELATIONSHIP | Scatter | Bubble, Heatmap | Line chart |
| CHANGE_OVER_TIME | Line | Area, Bar (discrete) | Scatter |
| RANKING | Horizontal bar (sorted) | Dot plot, Lollipop | Unsorted bar |
| DEVIATION | Diverging bar | Bullet chart | Standard bar |
| GEOSPATIAL | Choropleth | Bubble map | Table |

### Multi-Intent Combinations

| Combined Intents | Recommended Approach |
|------------------|----------------------|
| Comparison + Time | Multi-line chart or Small multiples |
| Composition + Time | Stacked area or 100% stacked area |
| Ranking + Time | Bump chart (rank changes over time) |
| Distribution + Comparison | Grouped box plots or Ridgeline |
| Relationship + Categories | Scatter with color encoding |
| Deviation + Time | Line with reference line/band |

---

## Validation Checklist

Before generating a visualization, verify:

- [ ] Primary intent is clearly identified
- [ ] Intent matches the question being asked
- [ ] Audience level (technical/non-technical) is considered
- [ ] Context (exploratory/explanatory, dashboard/standalone) is appropriate
- [ ] Chart type aligns with intent (see mapping above)
- [ ] Interactivity level matches use case
- [ ] Screen size constraints are considered

---

## Cross-Reference

- **Next step:** [03 - Chart Type Selection](03_chart_selection.md) - Detailed chart selection rules
- **Data context:** [01 - Data Analysis](01_data_analysis.md) - Understanding your data first
- **Color by intent:** [06 - Color](06_color.md) - Choosing appropriate color schemes
