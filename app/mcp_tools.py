"""MCP tools for visualization guide - SDK pattern."""

from claude_agent_sdk import tool, create_sdk_mcp_server

try:
    from agent_tools import (
        classify_intent as _classify_intent,
        search_guide as _search_guide,
        get_chart_config as _get_chart_config,
        get_critical_rules as _get_critical_rules,
    )
except ImportError:
    from .agent_tools import (
        classify_intent as _classify_intent,
        search_guide as _search_guide,
        get_chart_config as _get_chart_config,
        get_critical_rules as _get_critical_rules,
    )


@tool(
    name="classify_intent",
    description="""Classify the visualization intent from the user's description.
Returns the likely intent (comparison, composition, distribution, trend, relationship, ranking),
data type (time_series, categorical, numerical, hierarchical), and suggested chart types.
Use this FIRST to understand what the user wants.""",
    input_schema={"description": str},
)
async def classify_intent(args: dict) -> dict:
    """Classify visualization intent."""
    result = _classify_intent(args["description"])
    text = (
        f"Intent: {result['intent']}\n"
        f"Data Type: {result['data_type']}\n"
        f"Suggested Charts: {', '.join(result['suggested_charts'])}\n"
        f"Confidence: {result['confidence']}"
    )
    return {"content": [{"type": "text", "text": text}]}


@tool(
    name="search_guide",
    description="""Search the visualization guide for relevant sections.
Use this to find specific guidance on topics like hover templates, formatting,
color schemes, accessibility, etc. Returns formatted guide content.""",
    input_schema={"query": str, "limit": int},
)
async def search_guide(args: dict) -> dict:
    """Search the visualization guide."""
    result = _search_guide(args["query"], limit=args.get("limit", 5))
    return {"content": [{"type": "text", "text": result}]}


@tool(
    name="get_chart_config",
    description="""Get configuration guidance and examples for a specific chart type.
Use this after determining the chart type to get detailed setup instructions.
Chart types: line, multi_line, bar, stacked_bar, grouped_bar, scatter, bubble,
heatmap, histogram, box, violin, donut, candlestick, dual_axis, area, subplots.""",
    input_schema={"chart_type": str},
)
async def get_chart_config(args: dict) -> dict:
    """Get chart configuration guidance."""
    result = _get_chart_config(args["chart_type"])
    return {"content": [{"type": "text", "text": result}]}


@tool(
    name="get_critical_rules",
    description="""Get the must-follow rules for a specific chart type.
These are critical rules that should NOT be violated. Use this before generating
code to ensure compliance with visualization best practices.""",
    input_schema={"chart_type": str},
)
async def get_critical_rules(args: dict) -> dict:
    """Get critical rules for chart type."""
    result = _get_critical_rules(args["chart_type"])
    return {"content": [{"type": "text", "text": result}]}


def create_vizguide_mcp_server():
    """Create the MCP server with all visualization guide tools."""
    return create_sdk_mcp_server(
        name="vizguide",
        version="1.0.0",
        tools=[classify_intent, search_guide, get_chart_config, get_critical_rules],
    )
