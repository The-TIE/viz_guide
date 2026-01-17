"""VizGuide Agent - Agentic workflow using Claude Agent SDK (keyless mode).

Note: MCP tool integration in the SDK is currently broken, so we use a
simplified approach that includes relevant guide context directly in the prompt.
"""

import asyncio
import re
from pathlib import Path

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

try:
    from agent_tools import classify_intent, search_guide, get_chart_config, get_critical_rules
    from core_rules import get_core_rules
except ImportError:
    from .agent_tools import classify_intent, search_guide, get_chart_config, get_critical_rules
    from .core_rules import get_core_rules


# Maximum number of turns
MAX_TURNS = 1  # Single turn since we're not using tools


def extract_code(response_text: str) -> str:
    """Extract Python code from the response."""
    # Look for code blocks
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Also try generic code blocks
    pattern = r"```\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        # Check if it looks like Python code
        code = matches[0].strip()
        if "import " in code or "fig" in code or "df" in code:
            return code

    # If no code blocks, try to find code-like content
    lines = response_text.strip().split("\n")
    code_lines = []
    in_code = False

    for line in lines:
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            in_code = True
        if in_code:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)

    # Don't return raw conversational text - raise error instead
    raise ValueError(f"No Python code found in response. Response started with: {response_text[:100]}...")


def _classify_template(description: str, data_sample: str, intent: dict) -> dict:
    """Classify which template should be used based on description and data.

    Returns dict with template recommendation and reasoning.
    """
    desc_lower = description.lower()
    suggested_charts = intent.get("suggested_charts", [])
    primary_chart = suggested_charts[0] if suggested_charts else "line"

    # Parse data columns from sample
    columns = []
    if "Columns:" in data_sample:
        col_line = data_sample.split("Columns:")[1].split("\n")[0]
        try:
            columns = eval(col_line.strip())
        except:
            columns = []

    # Detect time column
    time_cols = [c for c in columns if any(t in c.lower() for t in ['date', 'time', 'week', 'month', 'year', 'day', 'period'])]
    has_time = len(time_cols) > 0

    # Detect category columns (for grouping/series identification)
    category_keywords = ['name', 'category', 'type', 'id', 'label', 'team', 'token', 'asset', 'exchange', 'symbol', 'ticker', 'group']
    category_cols = [c for c in columns if c not in time_cols and any(cat in c.lower() for cat in category_keywords)]

    # Detect numeric columns (potential y-axis values) - columns that aren't time or category
    non_time_cols = [c for c in columns if c not in time_cols]
    potential_series = [c for c in non_time_cols if c not in category_cols]
    num_series = len(potential_series)

    # Decision logic
    template = None
    reason = None
    params_hint = {}
    data_format = "wide"  # Track if data needs pivoting

    # Detect "long" format: time + category + single value column
    # Example: date, team, epa (each team appears multiple times per date)
    is_long_format = has_time and len(category_cols) >= 1 and num_series == 1

    # Time series patterns
    if has_time:
        if is_long_format:
            # Long format data - needs pivot for multi_line_chart
            template = "multi_line_chart"
            data_format = "long"
            category_col = category_cols[0]
            value_col = potential_series[0] if potential_series else "value"
            reason = f"Long format data - pivot {category_col} to columns, values from {value_col}"
            params_hint = {
                "x_column": time_cols[0],
                "y_columns": f"<columns from pivoting '{category_col}'>",
                "pivot_required": True,
                "pivot_code": f"df_wide = df.pivot(index='{time_cols[0]}', columns='{category_col}', values='{value_col}').reset_index()",
            }
        elif num_series == 1:
            template = "line_chart"
            reason = f"Single time series (1 value column: {potential_series[0] if potential_series else 'detected'})"
            params_hint = {"x_column": time_cols[0], "y_column": potential_series[0] if potential_series else "value"}
        elif 2 <= num_series <= 5:
            template = "multi_line_chart"
            reason = f"Multiple time series ({num_series} series: {potential_series[:3]}{'...' if num_series > 3 else ''})"
            params_hint = {"x_column": time_cols[0], "y_columns": potential_series}
        elif num_series > 5:
            template = "small_multiples_chart"
            reason = f"Many time series ({num_series} series) - use grid layout"
            params_hint = {"x_column": time_cols[0], "y_columns": potential_series}

    # Bar chart patterns
    if not template:
        if "bar" in primary_chart or "rank" in desc_lower or "top" in desc_lower or "compar" in desc_lower:
            cat_cols = [c for c in non_time_cols if any(cat in c.lower() for cat in ['name', 'category', 'type', 'team', 'exchange', 'token'])]
            value_cols = [c for c in non_time_cols if c not in cat_cols]

            if cat_cols and value_cols:
                if "horizontal" in desc_lower or "rank" in desc_lower or len(cat_cols[0]) > 10:
                    template = "horizontal_bar_chart"
                    reason = f"Ranking/comparison with categories ({cat_cols[0]})"
                    params_hint = {"category_column": cat_cols[0], "value_column": value_cols[0]}
                elif len(value_cols) > 1 and "stack" in desc_lower:
                    template = "stacked_bar_chart"
                    reason = f"Stacked composition ({len(value_cols)} value columns)"
                    params_hint = {"x_column": cat_cols[0], "stack_columns": value_cols}
                elif len(value_cols) > 1:
                    template = "grouped_bar_chart"
                    reason = f"Grouped comparison ({len(value_cols)} measures)"
                    params_hint = {"x_column": cat_cols[0], "group_columns": value_cols}
                else:
                    template = "bar_chart"
                    reason = f"Simple bar chart ({cat_cols[0]} vs {value_cols[0]})"
                    params_hint = {"x_column": cat_cols[0], "y_column": value_cols[0]}

    # Fallback: use intent suggestion
    if not template:
        if primary_chart in ["line", "area"]:
            template = "line_chart"
            reason = "Default to line chart based on intent"
        elif primary_chart in ["bar", "horizontal_bar"]:
            template = "horizontal_bar_chart"
            reason = "Default to horizontal bar based on intent"
        else:
            template = None
            reason = f"No template match - use custom code for {primary_chart}"

    return {
        "template": template,
        "reason": reason,
        "params_hint": params_hint,
        "columns_detected": columns,
        "data_format": data_format,
    }


def _gather_context(description: str) -> dict:
    """Pre-gather relevant context using the tool functions directly.

    Returns dict with intent classification and relevant guide sections.
    """
    # Classify intent
    intent_result = classify_intent(description)

    # Get chart config for suggested charts
    chart_configs = []
    for chart_type in intent_result.get("suggested_charts", [])[:2]:
        config = get_chart_config(chart_type)
        if config and "No specific guidance" not in config:
            chart_configs.append(f"### {chart_type} Chart Config\n{config}")

    # Get critical rules for primary chart type
    primary_chart = intent_result.get("suggested_charts", ["line"])[0]
    critical = get_critical_rules(primary_chart)

    # Search for relevant guide sections
    search_queries = [
        f"{intent_result['intent']} {intent_result['data_type']}",
        "hover template formatting",
    ]
    search_results = []
    for q in search_queries:
        result = search_guide(q, limit=3)
        if result and "No results" not in result:
            search_results.append(result)

    return {
        "intent": intent_result,
        "chart_configs": chart_configs,
        "critical_rules": critical,
        "search_results": search_results,
    }


def _build_system_prompt(watermark: str, context: dict) -> str:
    """Build the system prompt with core rules and gathered context."""
    core_rules = get_core_rules(watermark_type=watermark)

    # Format intent classification
    intent = context["intent"]
    intent_section = f"""## Request Analysis
- Intent: {intent['intent']}
- Data Type: {intent['data_type']}
- Suggested Charts: {', '.join(intent['suggested_charts'])}
- Confidence: {intent['confidence']}
"""

    # Format chart configs
    chart_section = ""
    if context["chart_configs"]:
        chart_section = "\n## Chart Type Guidance\n" + "\n\n".join(context["chart_configs"][:2])

    # Template functions section - STRONG preference for templates
    template_section = """
## Template Functions - USE THESE FIRST

**IMPORTANT: Always use a template function if one fits. Only write custom code if no template matches.**

### Decision Tree:
1. Comparing 2-5 time series? → `multi_line_chart()`
   - Raw values: `normalize=False`
   - Indexed to 100: `normalize=True` or `normalize='indexed'`
   - Percentage returns from 0%: `normalize='returns'`
2. Single time series trend? → `line_chart()`
3. 6+ time series? → `small_multiples_chart()`
4. Ranking or categorical comparison? → `horizontal_bar_chart(sort=True)`
5. Composition over time? → `stacked_bar_chart()`
6. Comparing groups side-by-side? → `grouped_bar_chart()`
7. Simple value comparison? → `bar_chart()`

### Function Signatures:

```python
from app.templates import multi_line_chart, line_chart, horizontal_bar_chart

# BTC vs ETH cumulative returns - template handles ALL the calculation and formatting
fig = multi_line_chart(
    df,
    x_column='date',
    y_columns=['BTC_price', 'ETH_price'],  # Use raw price columns, template calculates returns
    title='BTC vs ETH Cumulative Returns',
    normalize='returns',  # This converts to % returns from 0% automatically
    source='CoinGecko'  # ALWAYS include source attribution
)
# That's it! No manual calculation needed. Template adds reference line at 0%, proper axis labels, etc.

# Single series
fig = line_chart(df, x_column='date', y_column='price', title='Title', source='Data Provider')

# Rankings
fig = horizontal_bar_chart(df, category_column='name', value_column='volume', title='Title', sort=True, source='CoinGecko')
```

**ALWAYS pass `source` parameter** - every chart needs attribution (e.g., 'CoinGecko', 'DeFiLlama', 'Glassnode').

### When to Use Custom Code (ONLY these cases):
- Specialized chart type not in templates (candlestick, heatmap, sankey, etc.)
- Complex multi-chart dashboard layouts
- Highly custom interactivity requirements

**DEFAULT TO TEMPLATES** - they handle dark theme, colors, hover, legends, and all formatting automatically.
**DO NOT pass colors parameter** - templates use the standard colorway by default.
"""

    # Format critical rules
    critical_section = f"\n## Critical Rules for {intent['suggested_charts'][0]}\n{context['critical_rules']}"

    # Format search results
    search_section = ""
    if context["search_results"]:
        search_section = "\n## Additional Guidance\n" + "\n---\n".join(context["search_results"][:2])

    return f"""You are a Plotly visualization expert. Generate Python code for publication-ready charts.

{template_section}

{intent_section}
{chart_section}
{critical_section}
{search_section}

## Styling Rules (for custom code only - templates handle this automatically)
{core_rules}

## Output Format (CRITICAL)
Return ONLY Python code wrapped in ```python``` markers. No explanations, no analysis, no conversation.
The code must define a `fig` variable containing a Plotly figure.
If a template matches, use it - the code should be ~5 lines.

Example output format:
```python
from app.templates import multi_line_chart
fig = multi_line_chart(df, 'date', ['col1', 'col2'], 'Title', source='Source')
```
"""


async def generate_visualization_async(
    description: str,
    data_sample: str,
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.3,
    watermark: str = "none",
) -> dict:
    """
    Generate visualization code using Claude Agent SDK (keyless mode).

    Args:
        description: User's visualization description
        data_sample: String representation of the data
        model: Model to use for generation
        temperature: Temperature for generation
        watermark: Watermark type ('none', 'labs', 'qf', 'tie')

    Returns:
        dict with "code", "raw_response", "tool_calls", and "turns" keys
    """
    # Pre-gather context using tool functions directly
    context = _gather_context(description)

    # Classify which template to use
    template_decision = _classify_template(description, data_sample, context["intent"])

    # Build template instruction
    if template_decision["template"]:
        params_hint = template_decision["params_hint"]

        # Check if data needs pivoting (long format)
        if params_hint.get("pivot_required"):
            pivot_code = params_hint.get("pivot_code", "")
            template_instruction = f"""## TEMPLATE DECISION (MANDATORY)
You MUST use this template: `{template_decision["template"]}`
Reason: {template_decision["reason"]}

**DATA IS IN LONG FORMAT - PIVOT FIRST:**
```python
{pivot_code}
# Then use df_wide with template:
fig = {template_decision["template"]}(df_wide, x_column='{params_hint.get("x_column")}', y_columns=list(df_wide.columns[1:]), ...)
```

Import from app.templates and call the function. Only write custom Plotly code if the template literally cannot work for this data."""
        else:
            template_instruction = f"""## TEMPLATE DECISION (MANDATORY)
You MUST use this template: `{template_decision["template"]}`
Reason: {template_decision["reason"]}
Suggested parameters: {params_hint}

Import from app.templates and call the function. Only write custom Plotly code if the template literally cannot work for this data."""
    else:
        template_instruction = f"""## TEMPLATE DECISION
No template match found. Reason: {template_decision["reason"]}
Write custom Plotly code following the styling rules below."""

    # Build user message
    user_message = f"""Generate a Plotly visualization for the following:

{template_instruction}

DESCRIPTION:
{description}

DATA AVAILABLE (inspect carefully before writing code):
{data_sample}

Only reference columns that are explicitly listed in the data above.
Do not assume or invent column names."""

    options = ClaudeAgentOptions(
        system_prompt=_build_system_prompt(watermark, context),
        max_turns=MAX_TURNS,
        model=model,
        cwd=str(Path(__file__).parent.parent),
    )

    raw_response = ""
    turns = 0

    async for message in query(prompt=user_message, options=options):
        if isinstance(message, AssistantMessage):
            turns += 1
            for block in message.content:
                if isinstance(block, TextBlock):
                    raw_response += block.text

    code = extract_code(raw_response)

    # Return tool_calls showing what context was gathered (for debugging)
    tool_calls = [
        {"name": "classify_intent", "input": {"description": description[:100]}},
        {"name": "classify_template", "input": {"template": template_decision["template"], "reason": template_decision["reason"]}},
        {"name": "get_chart_config", "input": {"chart_type": context["intent"]["suggested_charts"][0]}},
    ]

    return {
        "code": code,
        "raw_response": raw_response,
        "tool_calls": tool_calls,
        "turns": turns,
        "template_decision": template_decision,
    }


def generate_visualization(
    description: str,
    data_sample: str,
    api_key: str | None = None,  # Kept for backwards compat, ignored
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.3,
    watermark: str = "none",
) -> dict:
    """
    Generate visualization code using the VizGuide Agent.

    This is the main entry point that replaces the old generator.
    Now uses Claude Agent SDK in keyless mode (api_key parameter ignored).

    Args:
        description: User's visualization description
        data_sample: String representation of the data
        api_key: DEPRECATED - Ignored, uses Claude Code CLI auth
        model: Model to use
        temperature: Temperature for generation
        watermark: Watermark type ('none', 'labs', 'qf', 'tie')

    Returns:
        dict with "code", "raw_response", "tool_calls", and "turns" keys
    """
    return asyncio.run(
        generate_visualization_async(
            description,
            data_sample,
            model,
            temperature,
            watermark,
        )
    )


# Keep the old class interface for backwards compatibility
class VizGuideAgent:
    """Agent for generating visualizations with progressive context loading.

    DEPRECATED: Use generate_visualization() directly instead.
    This class is kept for backwards compatibility.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        watermark: str = "none",
    ):
        """Initialize the agent (api_key is ignored, uses Claude Code CLI auth)."""
        self.model = model
        self.watermark = watermark

    def generate(
        self,
        description: str,
        data_sample: str,
        temperature: float = 0.3,
    ) -> dict:
        """Generate visualization code using agentic workflow."""
        return generate_visualization(
            description=description,
            data_sample=data_sample,
            model=self.model,
            temperature=temperature,
            watermark=self.watermark,
        )


if __name__ == "__main__":
    # Test the agent
    import sys

    # Simple test
    test_description = "Compare Bitcoin and Ethereum prices over the last year"
    test_data = """DataFrame (365 rows x 3 columns):
   date        BTC_price    ETH_price
0  2024-01-01  42500.00    2250.00
1  2024-01-02  43100.00    2280.00
2  2024-01-03  42800.00    2265.00
...
364 2024-12-31  98500.00   3850.00

Columns:
- date: datetime64[ns]
- BTC_price: float64
- ETH_price: float64"""

    print("Testing VizGuide Agent (SDK mode)...")
    print(f"Description: {test_description}")
    print(f"Data sample:\n{test_data}\n")

    try:
        result = generate_visualization(test_description, test_data)
        print(f"Tool calls made: {len(result['tool_calls'])}")
        for tc in result["tool_calls"]:
            print(f"  - {tc['name']}: {tc['input']}")
        print(f"\nTurns: {result['turns']}")
        print(f"\nGenerated code preview:\n{result['code'][:500]}...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
