"""Legacy generator for visualization code generation using Claude Agent SDK.

This is the "legacy mode" generator that loads the entire guide into context
rather than using progressive disclosure via tools. It's kept for comparison
purposes and as a fallback.
"""

import re
from pathlib import Path

import asyncio

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock


GUIDE_DIR = Path(__file__).parent.parent / "guide"


def load_guide_context() -> str:
    """Load all guide .md files into a single context string."""
    sections = sorted(GUIDE_DIR.glob("*.md"))

    context_parts = []
    for section in sections:
        content = section.read_text()
        # Add section header
        context_parts.append(f"### {section.stem}\n\n{content}")

    return "\n\n---\n\n".join(context_parts)


SYSTEM_PROMPT = """You are a Plotly visualization expert. Generate Python code that creates
publication-ready charts following the visualization guide provided below.

IMPORTANT GUIDELINES:

1. FIRST, carefully analyze the DATA PROVIDED:
   - Read the column names exactly as shown
   - Note the data types (datetime, numeric, categorical)
   - Understand the shape and structure
   - ONLY use columns that exist in the data - never assume or invent column names

2. Then analyze the user's description to determine:
   - Data type (temporal, categorical, numerical, hierarchical)
   - Intent (comparison, composition, distribution, relationship, trend)
   - Appropriate chart type based on the guide's decision trees

3. Generate complete, runnable Python code that:
   - Imports required libraries (plotly.graph_objects as go, pandas as pd, numpy as np)
   - Uses ONLY the exact column names from the provided data sample
   - CRITICAL: For values that might reach billions, use format_with_B() function, NOT tickformat=',.2s'
     (Plotly uses "G" for giga, but finance uses "B" for billions)
   - Applies the dark theme consistently:
     * Background: #0e1729
     * Paper background: #0e1729
     * Text color: #d3d4d6
     * Grid color: rgba(255,255,255,0.1)
   - Uses the standard colorway: ["#60A5FA", "#F87171", "#34D399", "#FBBF24", "#E879F9", "#818CF8", "#FB923C", "#22D3EE", "#A78BFA", "#F472B6"]
   - Follows all formatting guidelines from the guide:
     * Human-readable numbers (1.23k, 12.3M, 1.23B)
     * Appropriate date formats
     * Proper hover templates
     * Correct legend placement
     * Appropriate margins

3. The code MUST:
   - Define a variable `fig` containing the Plotly figure
   - Be complete and runnable (no placeholders like "...")
   - Handle the actual column names from the provided data
   - Include fig.update_layout() with all necessary styling

4. For financial data with large numbers, ALWAYS include and use this formatter:

def format_with_B(value, prefix='$', decimals=1):
    abs_val = abs(value)
    sign = '-' if value < 0 else ''
    if abs_val >= 1e12: return f'{sign}{prefix}{abs_val/1e12:.{decimals}f}T'
    elif abs_val >= 1e9: return f'{sign}{prefix}{abs_val/1e9:.{decimals}f}B'
    elif abs_val >= 1e6: return f'{sign}{prefix}{abs_val/1e6:.{decimals}f}M'
    elif abs_val >= 1e3: return f'{sign}{prefix}{abs_val/1e3:.{decimals}f}k'
    return f'{sign}{prefix}{abs_val:.{decimals}f}'

5. Return ONLY Python code wrapped in ```python``` markers.
   Do not include explanations outside the code block.
   Include brief comments in the code explaining key decisions.

VISUALIZATION GUIDE:
{guide_context}
"""


def extract_code(response_text: str) -> str:
    """Extract Python code from the response."""
    # Look for code blocks
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, response_text, re.DOTALL)

    if matches:
        return matches[0].strip()

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

    return response_text


async def generate_visualization_async(
    description: str,
    data_sample: str,
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.3,
) -> dict:
    """
    Generate visualization code using Claude Agent SDK (keyless mode).

    This is the legacy generator that loads the entire guide into context.

    Args:
        description: User's visualization description
        data_sample: String representation of the data
        model: Model to use
        temperature: Temperature for generation

    Returns:
        dict with "code" and "raw_response" keys
    """
    # Load guide context
    guide_context = load_guide_context()

    # Build system prompt
    system = SYSTEM_PROMPT.format(guide_context=guide_context)

    # Build user message
    user_message = f"""Generate a Plotly visualization for the following:

DESCRIPTION:
{description}

DATA AVAILABLE (inspect carefully before writing code):
{data_sample}

CRITICAL: Before writing any code, verify which columns exist in the data above.
Only reference columns that are explicitly listed. Do not assume or invent column names.

Generate the complete Python code to create this visualization. The code should:
1. Use ONLY the exact column names shown in the data above (case-sensitive)
2. Create a `fig` variable with the Plotly figure
3. Apply all styling according to the guide (dark theme, proper formatting, etc.)
4. If the data doesn't have the columns needed for the requested visualization, adapt the visualization to what's possible with the available data
"""

    options = ClaudeAgentOptions(
        system_prompt=system,
        max_turns=1,  # Single response, no tool calls
        model=model,
        cwd=str(Path(__file__).parent.parent),
    )

    raw_response = ""

    async for message in query(prompt=user_message, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    raw_response += block.text

    code = extract_code(raw_response)

    return {
        "code": code,
        "raw_response": raw_response,
    }


def generate_visualization(
    description: str,
    data_sample: str,
    api_key: str | None = None,  # Kept for backwards compat, ignored
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.3,
) -> dict:
    """
    Generate visualization code using Claude API (legacy mode).

    Now uses Claude Agent SDK in keyless mode (api_key parameter ignored).

    Args:
        description: User's visualization description
        data_sample: String representation of the data
        api_key: DEPRECATED - Ignored, uses Claude Code CLI auth
        model: Model to use
        temperature: Temperature for generation

    Returns:
        dict with "code" and "raw_response" keys
    """
    return asyncio.run(
        generate_visualization_async(
            description,
            data_sample,
            model,
            temperature,
        )
    )
