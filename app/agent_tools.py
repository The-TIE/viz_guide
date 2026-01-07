"""Agent tools for progressive disclosure of visualization guide."""

import re
from typing import Any

try:
    from guide_index import get_guide_index
except ImportError:
    from .guide_index import get_guide_index

# Intent classification keywords
INTENT_KEYWORDS = {
    "comparison": [
        "compare", "versus", "vs", "difference", "between", "against",
        "relative", "benchmark", "better", "worse", "more", "less"
    ],
    "composition": [
        "breakdown", "composition", "proportion", "percentage", "share",
        "part of", "makes up", "consists of", "distribution of"
    ],
    "distribution": [
        "distribution", "spread", "range", "histogram", "frequency",
        "density", "variance", "outlier", "quartile", "percentile"
    ],
    "trend": [
        "over time", "trend", "growth", "change", "increase", "decrease",
        "historical", "timeline", "evolution", "progression", "forecast"
    ],
    "relationship": [
        "correlation", "relationship", "scatter", "regression", "between",
        "affect", "impact", "influence", "depends on", "related to"
    ],
    "ranking": [
        "top", "bottom", "best", "worst", "rank", "leading", "highest",
        "lowest", "most", "least", "order by"
    ],
}

# Data type keywords
DATA_TYPE_KEYWORDS = {
    "time_series": [
        "date", "time", "daily", "monthly", "yearly", "quarterly",
        "historical", "over time", "period", "year", "month", "week"
    ],
    "categorical": [
        "category", "type", "group", "segment", "region", "country",
        "product", "department", "sector", "industry"
    ],
    "numerical": [
        "value", "amount", "count", "sum", "average", "total",
        "number", "quantity", "metric", "measure"
    ],
    "hierarchical": [
        "hierarchy", "parent", "child", "tree", "nested", "level",
        "drill down", "breakdown by"
    ],
}

# Chart type suggestions based on intent + data type
CHART_SUGGESTIONS = {
    ("comparison", "time_series"): ["multi_line", "area"],
    ("comparison", "categorical"): ["grouped_bar", "bar"],
    ("comparison", "numerical"): ["bar", "scatter"],
    ("composition", "categorical"): ["stacked_bar", "donut"],
    ("composition", "time_series"): ["stacked_area", "stacked_bar"],
    ("distribution", "numerical"): ["histogram", "box", "violin"],
    ("distribution", "categorical"): ["box", "violin"],
    ("trend", "time_series"): ["line", "area"],
    ("relationship", "numerical"): ["scatter", "bubble"],
    ("ranking", "categorical"): ["bar", "horizontal_bar"],
}


def classify_intent(description: str) -> dict[str, Any]:
    """
    Classify visualization intent from description.

    Args:
        description: User's visualization description

    Returns:
        dict with intent, data_type, suggested_charts, and confidence
    """
    desc_lower = description.lower()

    # Score each intent
    intent_scores = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in desc_lower)
        if score > 0:
            intent_scores[intent] = score

    # Get primary intent
    if intent_scores:
        primary_intent = max(intent_scores, key=intent_scores.get)
        confidence = min(intent_scores[primary_intent] / 3, 1.0)
    else:
        primary_intent = "comparison"  # Default
        confidence = 0.3

    # Score each data type
    data_type_scores = {}
    for dtype, keywords in DATA_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in desc_lower)
        if score > 0:
            data_type_scores[dtype] = score

    # Get primary data type
    if data_type_scores:
        primary_data_type = max(data_type_scores, key=data_type_scores.get)
    else:
        primary_data_type = "categorical"  # Default

    # Get chart suggestions
    key = (primary_intent, primary_data_type)
    suggested_charts = CHART_SUGGESTIONS.get(key, ["bar", "line"])

    return {
        "intent": primary_intent,
        "data_type": primary_data_type,
        "suggested_charts": suggested_charts,
        "confidence": round(confidence, 2),
        "all_intents": intent_scores,
        "all_data_types": data_type_scores,
    }


def search_guide(query: str, limit: int = 5) -> str:
    """
    Search visualization guide for relevant sections.

    Args:
        query: Search query string
        limit: Maximum number of chunks to return

    Returns:
        Formatted string with relevant guide sections
    """
    index = get_guide_index()
    chunks = index.search(query, limit=limit)

    if not chunks:
        return f"No results found for query: '{query}'"

    return index.format_chunks(chunks)


def get_chart_config(chart_type: str) -> str:
    """
    Get configuration and examples for specific chart type.

    Args:
        chart_type: Chart type key (line, bar, scatter, heatmap, etc.)

    Returns:
        Formatted string with chart-specific guidance
    """
    index = get_guide_index()

    # Get chunks tagged with this chart type
    chunks = index.get_by_chart_type(chart_type)

    if not chunks:
        # Try searching for the chart type as a query
        chunks = index.search(f"{chart_type} chart", limit=5)

    if not chunks:
        return f"No specific guidance found for chart type: '{chart_type}'"

    return index.format_chunks(chunks, max_tokens=6000)


# Critical rules by chart type
CRITICAL_RULES_BY_TYPE = {
    "line": """
## Critical Rules for Line Charts

1. **Time series X-axis**: Do NOT add "Date" label - it's self-evident
2. **Multi-series**: Use `hovermode='x unified'` for synchronized hover
3. **Unified hover format**: Use `hovertemplate='%{y:,.0f}'` - don't repeat date
4. **Legend**: Place at top, avoid overlap with subtitle
5. **Grid**: Use subtle grid lines (rgba(255,255,255,0.1))
6. **Financial values**: Use format_with_B() for billions, NOT tickformat=',.2s'
""",
    "multi_line": """
## Critical Rules for Multi-Line Charts

1. **Unified hover**: MANDATORY - use `hovermode='x unified'`
2. **Hover template**: `hovertemplate='<b>%{data.name}</b>: %{y:,.0f}<extra></extra>'`
3. **Legend placement**: `legend=dict(orientation='h', yanchor='bottom', y=1.02)`
4. **Normalize if needed**: For comparing % changes, normalize to starting point
5. **Color consistency**: Use COLORWAY in order for series
6. **Financial values**: Use format_with_B() for billions
""",
    "bar": """
## Critical Rules for Bar Charts

1. **Orientation**: Use horizontal for long labels or many categories
2. **Sorting**: Sort by value (descending) for ranking visualizations
3. **Text labels**: Add `text=values, textposition='outside'` for small datasets
4. **Bar gap**: Use `bargap=0.2` for clear separation
5. **Financial values**: Use format_with_B() for y-axis and hover
6. **Color**: Single color for single series, colorway for grouped
""",
    "stacked_bar": """
## Critical Rules for Stacked Bar Charts

1. **Use when**: Showing composition/parts of a whole over categories
2. **Legend**: Always include - users can't identify segments without it
3. **Order**: Order segments consistently (largest at bottom usually)
4. **Hover**: Show both segment value and total
5. **100% stacked**: Use when comparing proportions across categories
""",
    "scatter": """
## Critical Rules for Scatter Plots

1. **Axis labels**: ALWAYS label both axes with units
2. **Marker size**: Use consistent size unless encoding a variable (bubble)
3. **Overplotting**: Use transparency (`opacity=0.6`) for many points
4. **Trendline**: Add regression line if relationship is the focus
5. **Hover**: Show x, y values and any identifying information
""",
    "heatmap": """
## Critical Rules for Heatmaps

1. **Cell sizing**: Use equal width cells - set `xgap` and `ygap`
2. **Color scale**: Use diverging scale for data with meaningful midpoint
3. **Annotations**: Add cell values for small heatmaps (<100 cells)
4. **Axis labels**: Ensure readable without rotation if possible
5. **Missing data**: Use distinct color (gray) for NaN values
""",
    "histogram": """
## Critical Rules for Histograms

1. **Bins**: Use appropriate bin count (sqrt(n) rule or Sturges)
2. **Y-axis**: Label as "Frequency" or "Count"
3. **Overlapping**: Use `opacity=0.7` and `barmode='overlay'`
4. **Distribution comparison**: Consider violin or box plots instead
""",
    "donut": """
## Critical Rules for Donut/Pie Charts

1. **Use sparingly**: Only for parts of a whole (must sum to 100%)
2. **Max slices**: Limit to 5-7 segments; combine smaller into "Other"
3. **Labels**: Use `textinfo='percent+label'` for direct labeling
4. **Sort**: Order slices by size (largest first)
5. **Hole**: Use `hole=0.4` for donut style
""",
    "candlestick": """
## Critical Rules for Candlestick Charts

1. **OHLC required**: Must have open, high, low, close columns
2. **Colors**: Green for up days, red for down (or use increasing/decreasing)
3. **Volume**: Often add volume bars as subplot below
4. **Range slider**: Disable with `xaxis_rangeslider_visible=False`
5. **Date gaps**: Handle weekends/holidays appropriately
""",
    "dual_axis": """
## Critical Rules for Dual-Axis Charts

1. **Use with caution**: Can mislead if scales are cherry-picked
2. **Make secondary axis obvious**: Different color, clear label
3. **Use make_subplots**: `make_subplots(specs=[[{"secondary_y": True}]])`
4. **Legend**: Clearly indicate which series is on which axis
5. **Consider alternatives**: Small multiples often clearer
""",
}

# Default rules for unspecified chart types
DEFAULT_CRITICAL_RULES = """
## Critical Rules (General)

1. **Dark theme**: Apply background #0e1729, text #d3d4d6
2. **Financial values**: Use format_with_B() for billions (NOT tickformat=',.2s')
3. **Hover**: Enable appropriate hover mode
4. **Legend**: Place at top, avoid overlap with title/subtitle
5. **Grid**: Use subtle rgba(255,255,255,0.1)
6. **Margins**: Leave adequate margins for labels and legend
"""


def get_critical_rules(chart_type: str) -> str:
    """
    Get must-follow rules for chart type.

    Args:
        chart_type: Chart type key

    Returns:
        Critical rules as formatted string
    """
    # Normalize chart type
    chart_type_lower = chart_type.lower().replace(" ", "_").replace("-", "_")

    # Check for direct match
    if chart_type_lower in CRITICAL_RULES_BY_TYPE:
        return CRITICAL_RULES_BY_TYPE[chart_type_lower]

    # Check for partial matches
    for key, rules in CRITICAL_RULES_BY_TYPE.items():
        if key in chart_type_lower or chart_type_lower in key:
            return rules

    # Return default rules
    return DEFAULT_CRITICAL_RULES


# Tool definitions for the Claude API
TOOL_DEFINITIONS = [
    {
        "name": "classify_intent",
        "description": """Classify the visualization intent from the user's description.
Returns the likely intent (comparison, composition, distribution, trend, relationship, ranking),
data type (time_series, categorical, numerical, hierarchical), and suggested chart types.
Use this FIRST to understand what the user wants.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The user's visualization description",
                }
            },
            "required": ["description"],
        },
    },
    {
        "name": "search_guide",
        "description": """Search the visualization guide for relevant sections.
Use this to find specific guidance on topics like hover templates, formatting,
color schemes, accessibility, etc. Returns formatted guide content.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., 'unified hover template', 'format billions', 'dark theme')",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_chart_config",
        "description": """Get configuration guidance and examples for a specific chart type.
Use this after determining the chart type to get detailed setup instructions.
Chart types: line, multi_line, bar, stacked_bar, grouped_bar, scatter, bubble,
heatmap, histogram, box, violin, donut, candlestick, dual_axis, area, subplots.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "description": "Chart type key (e.g., 'line', 'bar', 'heatmap')",
                }
            },
            "required": ["chart_type"],
        },
    },
    {
        "name": "get_critical_rules",
        "description": """Get the must-follow rules for a specific chart type.
These are critical rules that should NOT be violated. Use this before generating
code to ensure compliance with visualization best practices.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "description": "Chart type key (e.g., 'line', 'bar', 'heatmap')",
                }
            },
            "required": ["chart_type"],
        },
    },
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """
    Execute a tool by name with given input.

    Args:
        tool_name: Name of the tool to execute
        tool_input: Dictionary of tool arguments

    Returns:
        Tool result as string
    """
    if tool_name == "classify_intent":
        result = classify_intent(tool_input["description"])
        return (
            f"Intent: {result['intent']}\n"
            f"Data Type: {result['data_type']}\n"
            f"Suggested Charts: {', '.join(result['suggested_charts'])}\n"
            f"Confidence: {result['confidence']}"
        )

    elif tool_name == "search_guide":
        return search_guide(
            tool_input["query"],
            limit=tool_input.get("limit", 5)
        )

    elif tool_name == "get_chart_config":
        return get_chart_config(tool_input["chart_type"])

    elif tool_name == "get_critical_rules":
        return get_critical_rules(tool_input["chart_type"])

    else:
        return f"Unknown tool: {tool_name}"


if __name__ == "__main__":
    # Test the tools
    print("=== Testing classify_intent ===")
    test_descriptions = [
        "How does Bitcoin compare to Ethereum over the past year?",
        "Show me the distribution of transaction amounts",
        "What's the breakdown of sales by region?",
        "Track monthly revenue growth over time",
    ]

    for desc in test_descriptions:
        result = classify_intent(desc)
        print(f"\n'{desc[:50]}...'")
        print(f"  Intent: {result['intent']}, Data: {result['data_type']}")
        print(f"  Charts: {result['suggested_charts']}")

    print("\n=== Testing search_guide ===")
    result = search_guide("format billions hover", limit=2)
    print(f"Search 'format billions hover':\n{result[:500]}...")

    print("\n=== Testing get_chart_config ===")
    result = get_chart_config("line")
    print(f"Line chart config:\n{result[:500]}...")

    print("\n=== Testing get_critical_rules ===")
    result = get_critical_rules("multi_line")
    print(f"Multi-line rules:\n{result}")
