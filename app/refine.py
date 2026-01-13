"""Refinement agent for iterative visualization improvements using Claude Agent SDK.

Note: MCP tool integration in the SDK is currently broken, so we use a
simplified approach that includes relevant guide context directly in the prompt.
"""

import re
from pathlib import Path

import asyncio

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
)

try:
    from agent_tools import search_guide, get_critical_rules
    from session import RefinementSession
except ImportError:
    from .agent_tools import search_guide, get_critical_rules
    from .session import RefinementSession


# Maximum number of turns
MAX_TURNS = 1  # Single turn since we're not using tools


def extract_code(response_text: str) -> str:
    """Extract Python code from the response."""
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)

    if matches:
        return matches[0].strip()

    # Fallback: find code-like content
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

    return response_text


def _gather_refinement_context(feedback: str) -> dict:
    """Pre-gather relevant context for refinement based on feedback.

    Returns dict with search results and critical rules.
    """
    # Keywords to detect chart type from feedback/code
    feedback_lower = feedback.lower()

    # Determine likely chart type from feedback keywords
    chart_type = "line"  # Default
    if "bar" in feedback_lower:
        chart_type = "bar"
    elif "scatter" in feedback_lower:
        chart_type = "scatter"
    elif "heatmap" in feedback_lower:
        chart_type = "heatmap"
    elif "donut" in feedback_lower or "pie" in feedback_lower:
        chart_type = "donut"

    # Get critical rules for the chart type
    critical = get_critical_rules(chart_type)

    # Search for relevant guide sections based on feedback keywords
    search_queries = []

    if "hover" in feedback_lower:
        search_queries.append("hover template formatting")
    if "legend" in feedback_lower:
        search_queries.append("legend positioning placement")
    if "color" in feedback_lower or "theme" in feedback_lower:
        search_queries.append("dark theme colors colorway")
    if "format" in feedback_lower or "billion" in feedback_lower or "million" in feedback_lower:
        search_queries.append("format billions format_with_B")
    if "axis" in feedback_lower or "label" in feedback_lower:
        search_queries.append("axis formatting labels")
    if "title" in feedback_lower or "subtitle" in feedback_lower:
        search_queries.append("title subtitle formatting")
    if "grid" in feedback_lower:
        search_queries.append("grid lines configuration")
    if "margin" in feedback_lower:
        search_queries.append("layout margins spacing")

    # If no specific keywords, do a general search based on the feedback
    if not search_queries:
        search_queries.append(feedback[:100])

    search_results = []
    for q in search_queries[:3]:  # Limit to 3 searches
        result = search_guide(q, limit=3)
        if result and "No results" not in result:
            search_results.append(result)

    return {
        "chart_type": chart_type,
        "critical_rules": critical,
        "search_results": search_results,
    }


def _build_refinement_system_prompt(context: dict) -> str:
    """Build the system prompt with gathered context."""

    # Format search results
    search_section = ""
    if context["search_results"]:
        search_section = "\n## Guide References\n" + "\n---\n".join(context["search_results"][:2])

    return f"""You are a Plotly visualization expert making TARGETED FIXES to an existing chart.

## Theme Rules (ALWAYS APPLY)
- plot_bgcolor: #0e1729
- paper_bgcolor: #0e1729
- Font color: #d3d4d6
- Use format_with_B() for financial values (billions = B, not G)

## Refinement Mode (NOT Initial Generation)

You are REFINING existing code, not creating from scratch. The current code already:
- Has the correct chart type selected
- Has the basic structure in place
- May have fixes from previous feedback iterations

Your job is to make SURGICAL CHANGES to address the new feedback while keeping everything else intact.

## CRITICAL: How to Refine

1. **START with the CURRENT CODE as your base** - copy it, then modify
2. **Read PREVIOUS ITERATIONS** - these issues are ALREADY FIXED, do not revert them
3. **Make MINIMAL changes** to address ONLY the new feedback
4. **Keep ALL existing styling** - theme colors, colorway, fonts, etc.
5. **Keep ALL existing features** - hover templates, legends, annotations, etc.

## Common Regression Bugs (AVOID THESE)
- Removing hover templates that were added
- Changing axis formatting that was corrected
- Reverting legend positioning
- Losing the dark theme colors
- Removing watermark code
- Changing from format_with_B back to tickformat

{context['critical_rules']}

{search_section}

## Output Format
Return ONLY Python code wrapped in ```python``` markers.
The code MUST be the current code with targeted modifications - NOT a rewrite.
"""


async def refine_visualization_async(
    session: RefinementSession,
    feedback: str,
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.3,
) -> dict:
    """
    Refine a visualization based on user feedback using Claude Agent SDK.

    Args:
        session: The current refinement session with iteration history
        feedback: User's feedback on what to fix
        model: Model to use
        temperature: Temperature for generation

    Returns:
        dict with "code", "raw_response", "tool_calls", and "turns" keys
    """
    # Get current code
    current_code = session.get_current_code()
    if not current_code:
        raise ValueError("No current code in session to refine.")

    # Build iteration history context
    iteration_history = session.get_iteration_history()

    # Pre-gather context using tool functions directly
    context = _gather_refinement_context(feedback)

    # Build user message
    user_message = f"""Refine this Plotly visualization based on user feedback.

ORIGINAL REQUEST:
{session.description}

DATA AVAILABLE:
{session.data_sample}

CURRENT CODE (version {len(session.iterations)}):
```python
{current_code}
```

USER FEEDBACK:
{feedback}

PREVIOUS ITERATIONS:
{iteration_history}

Please fix the issue described in the feedback. Return the complete updated code."""

    options = ClaudeAgentOptions(
        system_prompt=_build_refinement_system_prompt(context),
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
        {"name": "get_critical_rules", "input": {"chart_type": context["chart_type"]}},
    ]
    for i, q in enumerate(context["search_results"][:2]):
        tool_calls.append({"name": "search_guide", "input": {"query": f"(context search {i+1})"}})

    return {
        "code": code,
        "raw_response": raw_response,
        "tool_calls": tool_calls,
        "turns": turns,
    }


def refine_visualization(
    session: RefinementSession,
    feedback: str,
    api_key: str | None = None,  # Kept for backwards compat, ignored
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.3,
) -> dict:
    """
    Refine a visualization based on user feedback.

    Now uses Claude Agent SDK in keyless mode (api_key parameter ignored).

    Args:
        session: The current refinement session with iteration history
        feedback: User's feedback on what to fix
        api_key: DEPRECATED - Ignored, uses Claude Code CLI auth
        model: Model to use
        temperature: Temperature for generation

    Returns:
        dict with "code", "raw_response", "tool_calls", and "turns" keys
    """
    return asyncio.run(
        refine_visualization_async(
            session,
            feedback,
            model,
            temperature,
        )
    )


if __name__ == "__main__":
    # Test refinement
    from session import RefinementSession

    # Create a mock session
    session = RefinementSession.create(
        description="Compare BTC and ETH prices over time",
        data_sample="""DataFrame (365 rows x 3 columns):
   date        BTC_price    ETH_price
0  2024-01-01  42500.00    2250.00
1  2024-01-02  43100.00    2280.00

Columns:
- date: datetime64[ns]
- BTC_price: float64
- ETH_price: float64""",
        watermark="none",
    )

    # Add initial "generated" code
    session.add_iteration(
        code="""import plotly.graph_objects as go
import pandas as pd

fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['BTC_price'], name='BTC'))
fig.add_trace(go.Scatter(x=df['date'], y=df['ETH_price'], name='ETH'))
fig.update_layout(
    title='BTC vs ETH Price',
    template='plotly_dark'
)
"""
    )

    print("Testing refinement agent (SDK mode)...")
    print(f"Session ID: {session.id}")
    print(f"Current iteration: {len(session.iterations)}")

    # Test with feedback
    feedback = "Use the dark theme colors from the guide and add a unified hover"

    try:
        result = refine_visualization(
            session=session,
            feedback=feedback,
        )
        print(f"\nTool calls: {len(result['tool_calls'])}")
        for tc in result["tool_calls"]:
            print(f"  - {tc['name']}")
        print(f"\nRefined code preview:\n{result['code'][:300]}...")
    except Exception as e:
        print(f"Error: {e}")
