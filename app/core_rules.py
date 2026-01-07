"""Core rules always loaded into agent context (~2k tokens)."""

from pathlib import Path

# Dark theme configuration
DARK_THEME = {
    "background": "#0e1729",
    "paper_background": "#0e1729",
    "text_color": "#d3d4d6",
    "grid_color": "rgba(255,255,255,0.1)",
}

# Watermark configuration
WATERMARK_OPTIONS = {
    "none": None,
    "labs": "tie_labs_logo_transparent.png",
    "qf": "QF_Logo_big_background.png",
    "tie": "TT_KO.png",
}

REFERENCES_DIR = Path(__file__).parent.parent / "references"

# Standard colorway for consistent styling
COLORWAY = [
    "#60A5FA",  # Blue
    "#F87171",  # Red
    "#34D399",  # Green
    "#FBBF24",  # Yellow
    "#E879F9",  # Pink
    "#818CF8",  # Indigo
    "#FB923C",  # Orange
    "#22D3EE",  # Cyan
    "#A78BFA",  # Purple
    "#F472B6",  # Rose
]

# The format_with_B function - CRITICAL for financial data
FORMAT_WITH_B_CODE = '''
def format_with_B(value, prefix='$', decimals='auto'):
    """Format large numbers with B for billions (NOT G).

    CRITICAL: Plotly's tickformat=',.2s' uses SI notation where
    1 billion = 1G (giga). For financial/business contexts, use this
    function instead to display 1B (billion).

    Args:
        value: The number to format
        prefix: Currency/unit prefix (default '$')
        decimals: 'auto' for smart decimals, or int for fixed decimals
            - auto: 0 decimals if scaled >= 5, else 1 decimal
            - Strips trailing .0 for cleaner display
    """
    abs_val = abs(value)
    sign = '-' if value < 0 else ''

    # Determine scale and suffix
    if abs_val >= 1e12:
        scaled, suffix = abs_val / 1e12, 'T'
    elif abs_val >= 1e9:
        scaled, suffix = abs_val / 1e9, 'B'
    elif abs_val >= 1e6:
        scaled, suffix = abs_val / 1e6, 'M'
    elif abs_val >= 1e3:
        scaled, suffix = abs_val / 1e3, 'k'
    else:
        scaled, suffix = abs_val, ''

    # Auto-decimals: 0 if scaled >= 5, else 1 (but strip .0)
    if decimals == 'auto':
        if scaled >= 5 or scaled == int(scaled):
            formatted = f'{scaled:.0f}'
        else:
            formatted = f'{scaled:.1f}'.rstrip('0').rstrip('.')
    else:
        formatted = f'{scaled:.{decimals}f}'

    return f'{sign}{prefix}{formatted}{suffix}'
'''

# Critical anti-patterns to always avoid
ANTI_PATTERNS = """
## CRITICAL ANTI-PATTERNS (NEVER DO THESE)

1. **NEVER use tickformat=',.2s' for values that may reach billions**
   - Plotly uses SI notation: 1B = 1G (giga)
   - Use format_with_B() function instead for financial data
   - Apply via ticktext/tickvals or hover templates

2. **NEVER add "Date" label to time series x-axis**
   - Dates are self-evident
   - Omit xaxis_title for time series charts

3. **NEVER use 3D charts**
   - Distorts perception, hides data relationships
   - Use small multiples, heatmaps, or animation instead
   - Even for 3D surfaces (like IV surfaces), use 2D heatmaps

4. **NEVER use tickformat='%b %Y, %H:%M' in unified hover**
   - This repeats the date for every series
   - Use hovertemplate with %{y} only for unified hover

5. **NEVER place legend overlapping with subtitle**
   - Legend y=1.02 conflicts with subtitle at y=0.95
   - Adjust legend y position and top margin accordingly

6. **NEVER use deprecated 'titlefont' or 'tickfont' at axis level**
   - WRONG: `xaxis=dict(titlefont=dict(size=14))`
   - CORRECT: `xaxis=dict(title=dict(font=dict(size=14)))`
   - WRONG: `yaxis_titlefont=dict(size=14)`
   - CORRECT: `yaxis=dict(title=dict(font=dict(size=14)))`

7. **NEVER add bar labels to stacked charts or charts with many bars (>12)**
   - Stacked bars: labels clutter segments, use hover instead
   - Dense bar charts (daily data, etc.): labels overlap and become unreadable
   - Use clear axis labels + hover tooltips for these cases

8. **NEVER generate more than ~10 tick values**
   - Too many ticks crashes browsers and is unreadable
   - WRONG: `range(0, max_val, interval)` - unbounded, can create thousands
   - WRONG: `np.arange(0, max_val, interval)` - same problem
   - CORRECT: `np.linspace(min_val, max_val, 6)` - exactly 6 ticks, guaranteed
   - Always use linspace with a fixed count, never range/arange with intervals

9. **NEVER pass pre-scaled values directly to format_with_B**
   - If column is `market_cap_billions`, values are ALREADY in billions (e.g., 42.83 = $42.83B)
   - WRONG: `format_with_B(df['market_cap_billions'].max())` → returns "$43" not "$43B"
   - CORRECT: `format_with_B(df['market_cap_billions'].max() * 1e9)` → returns "$43B"
   - OR skip format_with_B: `f"${v:.0f}B"` for pre-scaled billions data
   - Check column names! _billions, _millions, _thousands indicate pre-scaled data
"""

# Mandatory defaults that must be applied to ALL charts
MANDATORY_DEFAULTS = """
## MANDATORY DEFAULTS (ALWAYS APPLY THESE)

### 1. Grid Lines: OFF by default
```python
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
```

### 2. Title: LEFT-ALIGNED (not centered)
```python
title=dict(
    text='Your Title',
    x=0,
    xanchor='left',
    font=dict(size=18, color='#d3d4d6')
)
```

### 3. Source & Updated Annotations: ALWAYS ADD
```python
# Source annotation (bottom-left)
fig.add_annotation(
    text='Source: [Your Source]',
    xref='paper', yref='paper',
    x=0, y=-0.12,
    showarrow=False,
    font=dict(size=10, color='#9ca3af'),
    xanchor='left'
)

# Updated annotation (bottom-right)
fig.add_annotation(
    text=f'Updated: {{datetime.now().strftime("%b %d, %Y")}}',
    xref='paper', yref='paper',
    x=1, y=-0.12,
    showarrow=False,
    font=dict(size=10, color='#9ca3af'),
    xanchor='right'
)
```

### 4. Hover: Use unified mode (unless >10 traces)
```python
# For ≤10 traces: use unified hover
fig.update_layout(hovermode='x unified')
# In traces, use: hovertemplate='<b>Series</b>: %{y:,.0f}<extra></extra>'
# Do NOT repeat the x-axis value - unified hover shows it automatically

# For >10 traces: use closest instead (unified would fill the screen)
fig.update_layout(hovermode='x')  # or 'closest'
```

### 5. Bottom Margin: Increase for annotations
```python
margin=dict(l=60, r=30, t=80, b=80)  # b=80 minimum for source/updated
```

### 6. Tick Labels: Use NICE round intervals
```python
# For financial data needing B notation, use tickvals/ticktext
# CRITICAL: Use nice round intervals (10, 20, 25, 50) NOT linspace on data range

import numpy as np

# Step 1: Find a "nice" interval based on data range
max_val = df['value'].max()
magnitude = 10 ** np.floor(np.log10(max_val))  # e.g., 10B for 44B
nice_intervals = [1, 2, 2.5, 5, 10]  # Standard nice intervals
interval = magnitude * min(n for n in nice_intervals if max_val / (magnitude * n) <= 6)

# Step 2: Generate ticks at nice round values
tickvals = np.arange(0, max_val + interval, interval)
tickvals = tickvals[:8]  # SAFETY: never more than 8 ticks
ticktext = [format_with_B(v) for v in tickvals]
fig.update_xaxes(tickmode='array', tickvals=tickvals, ticktext=ticktext)

# Example: max_val=44B → interval=10B → ticks: $0, $10B, $20B, $30B, $40B, $50B
# NOT: $0, $9B, $18B, $27B, $35B, $44B (ugly linspace result)
```
"""

# Agent workflow instructions
AGENT_INSTRUCTIONS = """
## Your Workflow

You are a visualization expert with access to tools that search the visualization guide.
Follow this workflow for EVERY request:

1. **ANALYZE THE DATA FIRST**
   - Examine the data sample carefully
   - Note exact column names (case-sensitive)
   - Identify data types (datetime, numeric, categorical)
   - ONLY use columns that exist

2. **CLASSIFY THE REQUEST**
   - Use classify_intent tool to understand what type of visualization is needed
   - Consider: comparison, composition, distribution, relationship, trend

3. **SEARCH FOR GUIDANCE**
   - Use search_guide to find relevant sections
   - Use get_chart_config for specific chart type guidance
   - Multiple searches are encouraged for comprehensive context

4. **GET CRITICAL RULES**
   - Use get_critical_rules for must-follow rules for your chart type
   - These override any conflicting guidance

5. **GENERATE CODE**
   - Apply dark theme consistently
   - Use format_with_B for financial data with large numbers
   - Follow all formatting guidelines from searched content
   - Create a `fig` variable with the complete Plotly figure
   - Return ONLY Python code wrapped in ```python``` markers
"""

# Watermark code template
WATERMARK_CODE = '''
from PIL import Image
from pathlib import Path

def add_watermark(fig, watermark_type="{watermark_type}", references_dir="{references_dir}"):
    """Add watermark to a Plotly figure."""
    watermark_files = {{
        "labs": "tie_labs_logo_transparent.png",
        "qf": "QF_Logo_big_background.png",
        "tie": "TT_KO.png",
    }}

    if watermark_type.lower() not in watermark_files:
        return fig

    img_file = watermark_files[watermark_type.lower()]
    img_path = Path(references_dir) / img_file

    if img_path.exists():
        fig.add_layout_image(
            dict(
                source=Image.open(img_path),
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                sizex=0.4,
                sizey=0.4,
                sizing="contain",
                opacity=0.15,
                layer="above",
                xanchor="center",
                yanchor="middle",
            )
        )
    return fig
'''


def get_watermark_instructions(watermark_type: str, references_dir: Path | str) -> str:
    """Get watermark instructions for the agent."""
    if watermark_type == "none":
        return ""

    return f"""
### Watermark (MANDATORY)
Add a watermark to the figure using this code pattern:

```python
from PIL import Image
from pathlib import Path

# Add watermark - MUST be called after creating fig
img_path = Path("{references_dir}") / "{WATERMARK_OPTIONS.get(watermark_type, 'tie_labs_logo_transparent.png')}"
if img_path.exists():
    fig.add_layout_image(
        dict(
            source=Image.open(img_path),
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            sizex=0.4,
            sizey=0.4,
            sizing="contain",
            opacity=0.15,
            layer="above",
            xanchor="center",
            yanchor="middle",
        )
    )
```
"""


# Compile the full core rules prompt
CORE_RULES = f"""
## Always-Apply Rules

### Dark Theme (MANDATORY)
Apply to ALL charts:
- plot_bgcolor: {DARK_THEME['background']}
- paper_bgcolor: {DARK_THEME['paper_background']}
- Font color: {DARK_THEME['text_color']}
- Grid color: {DARK_THEME['grid_color']}

### Standard Colorway
Use these colors in order for multiple series:
{COLORWAY}

### format_with_B Function (MANDATORY FOR FINANCIAL DATA)
{FORMAT_WITH_B_CODE}

Usage examples:
- Y-axis: Create custom tickvals/ticktext using format_with_B
- Hover: f"{{format_with_B(value)}}" in hovertemplate
- Annotations: format_with_B(value) for text labels

{ANTI_PATTERNS}

{MANDATORY_DEFAULTS}

{AGENT_INSTRUCTIONS}
"""


def get_core_rules(watermark_type: str = "none") -> str:
    """Return the core rules prompt for the agent.

    Args:
        watermark_type: One of 'none', 'labs', 'qf', 'tie'
    """
    watermark_instructions = get_watermark_instructions(watermark_type, REFERENCES_DIR)
    return CORE_RULES + watermark_instructions


def get_theme_config() -> dict:
    """Return theme configuration for direct use."""
    return DARK_THEME.copy()


def get_colorway() -> list[str]:
    """Return the standard colorway."""
    return COLORWAY.copy()


if __name__ == "__main__":
    # Print stats about the core rules
    rules = get_core_rules()
    token_estimate = len(rules) // 4
    print(f"Core rules length: {len(rules):,} characters")
    print(f"Estimated tokens: {token_estimate:,}")
    print(f"\nDark theme: {DARK_THEME}")
    print(f"Colorway: {len(COLORWAY)} colors")
    print(f"\n--- Preview (first 500 chars) ---\n{rules[:500]}...")
