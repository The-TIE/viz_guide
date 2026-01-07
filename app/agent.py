"""VizGuide Agent - Agentic workflow for visualization code generation."""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

try:
    from agent_tools import TOOL_DEFINITIONS, execute_tool
    from core_rules import get_core_rules
except ImportError:
    from .agent_tools import TOOL_DEFINITIONS, execute_tool
    from .core_rules import get_core_rules


# Maximum number of tool-calling turns to prevent infinite loops
MAX_TURNS = 10


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


class VizGuideAgent:
    """Agent for generating visualizations with progressive context loading."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250929",
        watermark: str = "none",
    ):
        """
        Initialize the agent.

        Args:
            api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
            model: Model to use for generation
            watermark: Watermark type ('none', 'labs', 'qf', 'tie')
        """
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("API key required. Set ANTHROPIC_API_KEY or pass api_key parameter.")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.watermark = watermark
        self.core_rules = get_core_rules(watermark_type=watermark)

    def _build_system_prompt(self) -> str:
        """Build the system prompt with core rules."""
        return f"""You are a Plotly visualization expert. Generate Python code that creates
publication-ready charts following the visualization guidelines.

{self.core_rules}

## Output Format
Return ONLY Python code wrapped in ```python``` markers.
The code MUST:
- Define a variable `fig` containing the Plotly figure
- Be complete and runnable (no placeholders)
- Include all necessary imports
- Apply dark theme and formatting from core rules
"""

    def _execute_tools_parallel(self, response) -> list[dict]:
        """Execute all tool calls from a response in parallel."""
        tool_calls = [block for block in response.content if block.type == "tool_use"]

        if not tool_calls:
            return []

        results = []

        # Execute tools in parallel
        with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as executor:
            futures = {}
            for tc in tool_calls:
                future = executor.submit(execute_tool, tc.name, tc.input)
                futures[future] = tc

            for future in as_completed(futures):
                tc = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = f"Error executing {tc.name}: {str(e)}"

                results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result,
                })

        return results

    def generate(
        self,
        description: str,
        data_sample: str,
        temperature: float = 0.3,
    ) -> dict:
        """
        Generate visualization code using agentic workflow.

        Args:
            description: User's visualization description
            data_sample: String representation of the data
            temperature: Temperature for generation

        Returns:
            dict with "code", "raw_response", and "tool_calls" keys
        """
        # Build user message
        user_message = f"""Generate a Plotly visualization for the following:

DESCRIPTION:
{description}

DATA AVAILABLE (inspect carefully before writing code):
{data_sample}

CRITICAL: Before writing any code:
1. Use classify_intent to understand what visualization is needed
2. Use get_chart_config to get guidance for the chart type
3. Use search_guide to find relevant formatting/styling guidance
4. Use get_critical_rules to get must-follow rules

Only reference columns that are explicitly listed in the data above.
Do not assume or invent column names."""

        messages = [{"role": "user", "content": user_message}]
        all_tool_calls = []

        # Agent loop
        for turn in range(MAX_TURNS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=temperature,
                system=self._build_system_prompt(),
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )

            # Handle tool calls
            if response.stop_reason == "tool_use":
                # Execute tools in parallel
                tool_results = self._execute_tools_parallel(response)

                # Track tool calls for debugging
                for tc in response.content:
                    if tc.type == "tool_use":
                        all_tool_calls.append({
                            "name": tc.name,
                            "input": tc.input,
                        })

                # Add assistant response and tool results to messages
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

            else:
                # End of conversation - extract code
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

        # If we hit max turns, return what we have
        return {
            "code": "",
            "raw_response": "Max turns reached without generating code",
            "tool_calls": all_tool_calls,
            "turns": MAX_TURNS,
        }


def generate_visualization(
    description: str,
    data_sample: str,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-5-20250929",
    temperature: float = 0.3,
    watermark: str = "none",
) -> dict:
    """
    Generate visualization code using the VizGuide Agent.

    This is the main entry point that replaces the old generator.

    Args:
        description: User's visualization description
        data_sample: String representation of the data
        api_key: Anthropic API key (or uses ANTHROPIC_API_KEY env var)
        model: Model to use
        temperature: Temperature for generation
        watermark: Watermark type ('none', 'labs', 'qf', 'tie')

    Returns:
        dict with "code", "raw_response", and "tool_calls" keys
    """
    agent = VizGuideAgent(api_key=api_key, model=model, watermark=watermark)
    return agent.generate(description, data_sample, temperature=temperature)


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

    print("Testing VizGuide Agent...")
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
        sys.exit(1)
