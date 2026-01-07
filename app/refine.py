"""Refinement agent for iterative visualization improvements."""

import os
import re

import anthropic

try:
    from agent_tools import TOOL_DEFINITIONS, execute_tool
    from session import RefinementSession
except ImportError:
    from .agent_tools import TOOL_DEFINITIONS, execute_tool
    from .session import RefinementSession


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


REFINEMENT_SYSTEM_PROMPT = """You are a Plotly visualization expert making TARGETED FIXES to an existing chart.

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

## Output Format
Return ONLY Python code wrapped in ```python``` markers.
The code MUST be the current code with targeted modifications - NOT a rewrite.
"""


def refine_visualization(
    session: RefinementSession,
    feedback: str,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.3,
) -> dict:
    """
    Refine a visualization based on user feedback.

    Args:
        session: The current refinement session with iteration history
        feedback: User's feedback on what to fix
        api_key: Anthropic API key (or uses env var)
        model: Model to use
        temperature: Temperature for generation

    Returns:
        dict with "code", "raw_response", and "tool_calls" keys
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API key required.")

    client = anthropic.Anthropic(api_key=api_key)

    # Get current code
    current_code = session.get_current_code()
    if not current_code:
        raise ValueError("No current code in session to refine.")

    # Build iteration history context
    iteration_history = session.get_iteration_history()

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

Please fix the issue described in the feedback. Search the guide if you need specific formatting or styling guidance. Return the complete updated code."""

    # Use the focused refinement prompt (no need for full core_rules which has generation workflow)
    system_prompt = REFINEMENT_SYSTEM_PROMPT

    messages = [{"role": "user", "content": user_message}]
    all_tool_calls = []
    max_turns = 5

    # Agent loop
    for turn in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            system=system_prompt,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        # Handle tool calls
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    all_tool_calls.append({
                        "name": block.name,
                        "input": block.input,
                    })
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            # Extract code from response
            raw_response = ""
            for block in response.content:
                if hasattr(block, "text"):
                    raw_response += block.text

            code = extract_code(raw_response)

            return {
                "code": code,
                "raw_response": raw_response,
                "tool_calls": all_tool_calls,
                "turns": turn + 1,
            }

    # Max turns reached
    return {
        "code": "",
        "raw_response": "Max turns reached without generating code",
        "tool_calls": all_tool_calls,
        "turns": max_turns,
    }


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

    print("Testing refinement agent...")
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
