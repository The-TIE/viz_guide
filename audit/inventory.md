# Dashboard Audit Inventory

> Systematic catalog of dashboards and their visualization patterns.
> Goal: Identify all chart types, layouts, and styling decisions in use.

---

## Summary Statistics

| Metric | Count |
|--------|-------|
| Dashboards audited | 5 |
| Total widgets cataloged | 30+ |
| Unique chart types | 12 |

---

## Dashboards to Audit

### Priority 1: Diverse Sample (6 dashboards)
- [ ] **Futures and Perpetual Swaps Analytics** - Complex derivatives dashboard
- [ ] **Options Analytics** - Already viewed, has term structure, skew charts
- [ ] **Macro Dashboard** - Likely broad market indicators
- [ ] **Market Liquidity** - Liquidity metrics
- [ ] **Sentiment Dashboard** - Social/sentiment data
- [ ] **Factor Monitor** - Quantitative factors

### Priority 2: Specialized Dashboards
- [ ] CFTC Commitment of Traders Report (already viewed)
- [ ] Options Overview
- [ ] Options Volatility
- [ ] Portfolio Overview
- [ ] Social Competitive dashboards (zkVerify, Horizon, Stellar)
- [ ] NEAR Liquidity
- [ ] [Tezos] APAC

---

## Audit Template

For each dashboard, capture:

```markdown
## [Dashboard Name]

**URL:**
**Last Updated:**
**Tags:**

### Widgets

| Widget Name | Chart Type | Subplots | Dual Axis | Data Type | Notes |
|-------------|------------|----------|-----------|-----------|-------|
| Example     | Line       | No       | No        | Time series | |

### Layout Patterns
- Grid: [rows x cols]
- Notable spacing/margins:

### Formatting Observed
- Number formats:
- Date formats:
- Hover mode:

### Good Patterns to Preserve
-

### Issues / Improvements Needed
-

### Screenshots
- `audit/screenshots/[dashboard-name]-overview.png`
```

---

## Completed Audits

### Futures and Perpetual Swaps Analytics

**URL:** https://terminal.thetie.io/dashboard/09c5938d-1c1f-4689-b7ee-a6e3d2160e9c
**Tags:** personal, team, Derivatives, Futures

#### Widgets Observed

| Widget Name | Chart Type | Inputs | Data Type | Notes |
|-------------|------------|--------|-----------|-------|
| [Unnamed - Top Left] | Stacked Area | - | Time series | Exchange breakdown, 6+ colors |
| [Unnamed - Top Right Upper] | Horizontal Bar | - | Categorical | Red bars, ranking/comparison |
| Global Futures Historical Volumes | Multi-line | - | Time series | White + gray lines, high volatility |
| Aggregated Open Interest / Liquidations | Dual-axis Line+Bar | Coin, granularity (4hr) | Time series | Green/red bars for long/short liquidations |
| Perpetual Swaps Summary | Data Table | - | Snapshot | Coin icons, conditional formatting (green/red) |
| Perpetual Swaps Currency Summary by Instrument | Data Table | exchanges (multi-select), Coin | Snapshot | Exchange comparison grid |
| Futures Annualized Basis | Chart (not visible) | Coin | - | |
| Futures+Perps Open Interest by Exchange | Data Table | Coin | Snapshot | Exchange, Open Interest $, Exchange Type |
| Futures Basis Table | Data Table | Coin | Matrix | Expiry x Exchange grid |
| Futures Curve | Chart (not visible) | Coin | - | |

#### Input Patterns
- Single-select dropdowns: `Coin:`, `granularity:`
- Multi-select dropdowns: `exchanges:` (shows "9 Selected")
- All use "Show popup" button pattern

#### Layout Patterns
- 2x2 grid layout for charts
- Tables span full width
- Section headers (e.g., "Basis", "Funding") with descriptions

#### Formatting Observed
- Human-readable numbers: 816.89M, 297.32M, 1.2B
- Currency: $488.02
- Percentages with color: green for positive, red for negative (-0.09%, +0.01%)
- Conditional cell formatting in tables

#### Chart Library Mix
- Highcharts ("Created with Highcharts 12.3.0")
- Custom tables

---

### Macro Dashboard

**URL:** https://terminal.thetie.io/dashboard/d21df0c5-cd78-461f-bfa8-90302e97bf01
**Tags:** personal

#### Widgets Observed

| Widget Name | Chart Type | Inputs | Data Type | Notes |
|-------------|------------|--------|-----------|-------|
| Economic Calendar (TradingView) | Third-party embed | - | Events | TradingView calendar widget, country flags, importance indicators |
| News Feed | Feed widget | Source dropdown (Firehose), search | News | Headlines from Watcher Guru, JD Supra, Galaxy Digital, etc. |
| Inflation Rates | Bar chart | - | Time series | PCE and CPI, red/coral bars, legend with dashed/solid line styles |
| US Gross Domestic Product | Line chart | - | Time series | Nominal GDP vs Real GDP, blue lines, Y-axis: % Change YoY |
| Asset Classes | (section) | - | - | Header visible in snapshot |

#### Layout Patterns
- Section-based layout with header titles ("Macro Events", "Macro Economy", "Asset Classes")
- 2-column layout within sections
- Third-party embeds alongside custom widgets

#### Formatting Observed
- Economic data: Currency values ($7,288.9M), percentages (+7.7%)
- Importance indicators (red/green dots or markers)
- GDP percentages on Y-axis (5%, 10%, 15%)

#### Key Patterns
- Third-party widget integration (TradingView)
- News feed with source filtering
- Traditional economic chart types
- Section headers for content organization

---

### Market Liquidity

**URL:** https://terminal.thetie.io/dashboard/31215421-f0db-4df9-bb2e-6f0f9bbbc6a9
**Tags:** personal

#### Widgets Observed

| Widget Name | Chart Type | Inputs | Data Type | Notes |
|-------------|------------|--------|-----------|-------|
| Aggregate Spot Liquidity, Top Assets | Data Table | - | Ranking | Asset list with values |
| Order Book Depth Heatmap (USD) | **Heatmap** | - | Matrix | Rows: assets, Cols: depth levels ($5M, $10M, $25M), green/teal gradient |
| Predicted Slippage Heatmap | **Heatmap** | - | Matrix | Rows: assets, Cols: trade sizes, white-to-orange-to-red gradient |
| Order Book Imbalance Heatmap by Instrument and Depth | **Heatmap** | - | Matrix | Diverging colors (red/green for negative/positive imbalance) |
| Spot Market Liquidity by Pair | Chart | Depth dropdown | Time series | |
| Market Liquidity, Spot vs Perps | Chart | Slippage, Depth dropdowns | Time series | Comparing spot vs perpetuals |

#### Sections
- "Top 5 Assets"
- "Spot vs Perps"
- "Individual Spot Asset"

#### NEW Chart Type: Heatmaps
This is the first dashboard with heatmaps. Three distinct heatmap patterns:

1. **Sequential Heatmap** (Order Book Depth)
   - Single color gradient (teal/green)
   - Shows magnitude only
   - Use case: depth/liquidity levels

2. **Sequential Heatmap** (Predicted Slippage)
   - White → Orange → Red gradient
   - Higher values = more red (bad)
   - Use case: risk/cost metrics

3. **Diverging Heatmap** (Order Book Imbalance)
   - Red (negative) ↔ White (neutral) ↔ Green (positive)
   - Shows directional imbalance
   - Use case: buy/sell pressure, long/short ratios

#### Formatting Observed
- Matrix cells show percentages: -1.2%, +5.8%, +14.8%
- Color intensity indicates magnitude
- Row labels: instrument names (BNB-USD, BTC-USD, etc.)
- Column labels: depth percentages from mid price

#### Key Patterns to Document
- **Heatmap color scale selection**: Sequential vs Diverging
- **Matrix orientation**: Assets as rows, metrics as columns
- **Cell text**: When to show values in cells vs rely on color only

---

### CFTC Commitment of Traders Report (Partial)

**URL:** https://terminal.thetie.io/dashboard/ffd40b85-9cd6-4aa0-828e-448d0e4ce70d
**Tags:** personal

#### Widgets Observed

| Widget Name | Chart Type | Subplots | Dual Axis | Data Type | Notes |
|-------------|------------|----------|-----------|-----------|-------|
| CFTC Open Interest | Stacked Bar + Line | No | Yes | Time series | Bar for components, line for total |
| CFTC Commitment of Traders Net Positioning | Multi-line (Highcharts) | No | No | Time series | 5 category series |
| Commitment of Traders Position Share by Category | Stacked Area | Yes (5 subplots) | No | Time series | Long/Short split |
| Commitment of Traders Net Positioning Percentile | Multi-line | Yes (5 subplots) | No | Time series | Percentile 0-100% |
| NOTE widget | Text/Markdown | N/A | N/A | N/A | Informational note |

#### Layout Patterns
- 2x2 grid with 1 full-width note
- Some charts use small multiples (5 category subplots)

#### Formatting Observed
- Y-axis: Human readable (5B, 10B, 15B)
- Dates: "Apr 09, 18" format on x-axis
- Percentages: "0%", "50%", "100%"

#### Notes
- Mix of Plotly and Highcharts
- Uses subplot small multiples for category breakdown
- Consistent color coding across related charts

---

### Options Analytics (Partial)

**URL:** https://terminal.thetie.io/dashboard/8ff97c10-16d9-4e85-bef5-5119c3bb9882
**Tags:** personal, team, Derivatives, Options

#### Widgets Observed

| Widget Name | Chart Type | Subplots | Dual Axis | Data Type | Notes |
|-------------|------------|----------|-----------|-----------|-------|
| ETH 90 Term Structure Time-lapse | Multi-line | No | No | Term structure | Shows evolution over time periods |
| Skrift Expiry Skew | Line | No | No | Options data | |
| Historical Gamma Exposure | Line + Bar | No | Yes | Time series | Volume bars + line |
| Options Historical Volumes Put-Call Ratio | Multi-line | No | No | Time series | Multiple ratio series |
| Options Interest Open Interest Put-Call Ratio | Multi-line | No | No | Time series | |

#### Formatting Observed
- Time comparisons: "1 Week Ago", "2 Weeks Ago", "4 Weeks Ago"
- Consistent color palette across charts

---

## Chart Type Reference (Discovered)

| Chart Type | Count | Example Dashboard | Notes |
|------------|-------|-------------------|-------|
| Multi-line time series | 5+ | Options Analytics, CFTC | Most common |
| Stacked area | 2 | CFTC, Futures | Exchange/category breakdown |
| Stacked bar | 1 | CFTC | |
| Line + Bar combo (dual-axis) | 3 | Options Analytics, CFTC, Futures | Volume/liquidations overlay |
| Small multiples (subplots) | 2 | CFTC | Category breakdown |
| Data tables | 4+ | Futures, Liquidity | With conditional formatting |
| Horizontal bar | 1 | Futures | Rankings |
| Bar chart | 1 | Macro | Inflation rates |
| **Heatmap - Sequential** | 2 | Liquidity | Depth, slippage |
| **Heatmap - Diverging** | 1 | Liquidity | Imbalance (red/green) |
| Third-party embeds | 2 | Macro (TradingView), Futures (Highcharts) | |
| News/Feed widgets | 1 | Macro | With source filtering |

---

## Pattern Observations

### What's Working Well
1. Consistent dark theme
2. Human-readable number formatting (B, M, k)
3. Color consistency within dashboards
4. Small multiples for category breakdowns

### Potential Issues
1. Mix of Plotly and Highcharts (inconsistent styling?)
2. Need to verify hover formatting consistency
3. Legend placement varies

### Patterns to Document
- [ ] Term structure visualization
- [ ] Time-lapse comparisons (same chart, different time periods)
- [ ] Ratio charts (put-call ratios)
- [ ] Positioning breakdown (long/short)
- [ ] Percentile visualizations
