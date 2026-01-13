"""Template feedback analysis and application module (Phase 2.5).

Analyzes refinement sessions to determine if feedback should update
template Python code rather than guide markdown.
"""

import ast
import inspect
import json
import re
import shutil
from pathlib import Path

import asyncio

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

try:
    from session import TemplateSuggestion, RefinementSession
except ImportError:
    from .session import TemplateSuggestion, RefinementSession


TEMPLATES_DIR = Path(__file__).parent / "templates"

# Mapping from template function name to file
TEMPLATE_FILES = {
    "line_chart": "line.py",
    "multi_line_chart": "line.py",
    "small_multiples_chart": "line.py",
    "bar_chart": "bar.py",
    "horizontal_bar_chart": "bar.py",
    "stacked_bar_chart": "bar.py",
    "grouped_bar_chart": "bar.py",
}

# Set of valid template function names
TEMPLATE_FUNCTIONS = set(TEMPLATE_FILES.keys())


def detect_template_usage(code: str) -> tuple[str | None, str | None]:
    """Detect if code imports and uses a template function.

    Args:
        code: Generated Python code

    Returns:
        Tuple of (template_id, template_file) or (None, None) if no template used
    """
    # Pattern to match template imports
    # Handles: from app.templates import X, Y
    # Handles: from app.templates.line import X
    pattern = r"from\s+app\.templates(?:\.(\w+))?\s+import\s+(.+)"
    match = re.search(pattern, code)

    if not match:
        return None, None

    imports_str = match.group(2)
    # Parse comma-separated imports, handle "X as Y" aliases
    imports = []
    for item in imports_str.split(","):
        item = item.strip()
        # Handle "X as Y" - take just X
        if " as " in item:
            item = item.split(" as ")[0].strip()
        imports.append(item)

    # Find the first template function that was imported
    for func_name in imports:
        if func_name in TEMPLATE_FUNCTIONS:
            return func_name, TEMPLATE_FILES[func_name]

    return None, None


def get_template_source(template_id: str) -> str | None:
    """Get the source code of a template function.

    Args:
        template_id: Template function name (e.g., "multi_line_chart")

    Returns:
        Source code string or None if not found
    """
    if template_id not in TEMPLATE_FILES:
        return None

    template_file = TEMPLATE_FILES[template_id]
    template_path = TEMPLATES_DIR / template_file

    if not template_path.exists():
        return None

    content = template_path.read_text()

    # Use AST to find the function
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == template_id:
                # Extract source lines
                start_line = node.lineno - 1
                end_line = node.end_lineno
                lines = content.split("\n")
                return "\n".join(lines[start_line:end_line])
    except SyntaxError:
        pass

    return None


def get_template_signature(template_id: str) -> str | None:
    """Get just the function signature of a template.

    Args:
        template_id: Template function name

    Returns:
        Function signature string or None
    """
    source = get_template_source(template_id)
    if not source:
        return None

    # Extract up to the first docstring or first line of body
    lines = source.split("\n")
    sig_lines = []
    for line in lines:
        sig_lines.append(line)
        if line.strip().endswith("):"):
            break
        if '"""' in line or "'''" in line:
            # Found docstring, stop before it
            sig_lines.pop()
            break

    return "\n".join(sig_lines)


CATEGORIZE_PROMPT = """Analyze this refinement feedback for a visualization that used the `{template_name}` template.

## Template Function Signature
```python
{template_signature}
```

## Template Source Code
```python
{template_code}
```

## Feedback History
{feedback_history}

## Final Refined Code
```python
{final_code}
```

## Task

For each piece of feedback that required changes, categorize it:

1. **parameter**: Could be fixed by adding/changing a template parameter
   - Example: "need option to disable grid" → add `show_grid: bool = True` parameter
   - Example: "default should sort bars" → change `sort=False` to `sort=True`

2. **template_code**: Requires changing the template function logic
   - Example: "hover shows wrong format" → modify hovertemplate in function body
   - Example: "legend position conflicts with title" → adjust layout logic

3. **guide**: Conceptual issue better documented in guide markdown
   - Example: "when should I normalize?" → document in guide, not code change

## Output Format

Return a JSON array. For each feedback item that should update the template:

```json
[
  {{
    "feedback": "the original feedback text",
    "category": "parameter",
    "description": "Add show_grid parameter to control grid visibility",
    "suggested_code": "show_grid: bool = True",
    "reason": "User needed to disable grid lines",
    "param_name": "show_grid",
    "param_type": "bool",
    "param_default": "True"
  }},
  {{
    "feedback": "the original feedback text",
    "category": "template_code",
    "description": "Fix hover format to use format_with_B",
    "suggested_code": "hovertemplate=f'{{format_with_B(y)}}' instead of {{y:,.2f}}",
    "reason": "Hover was showing raw numbers instead of B/M format"
  }}
]
```

Only include items with category "parameter" or "template_code".
Skip items that are "guide" category or one-off adjustments.
Return empty array [] if no template changes are needed.
"""


async def analyze_template_feedback_async(
    session: RefinementSession,
    model: str = "claude-sonnet-4-5-20250929",
) -> list[TemplateSuggestion]:
    """Analyze session feedback to generate template improvement suggestions.

    Args:
        session: Approved refinement session with template usage
        model: Model to use for analysis

    Returns:
        List of TemplateSuggestion objects
    """
    if not session.used_template:
        return []

    if len(session.iterations) < 2:
        # No refinements needed
        return []

    template_id = session.used_template
    template_file = session.template_file or TEMPLATE_FILES.get(template_id, "")

    # Get template info
    template_source = get_template_source(template_id)
    template_sig = get_template_signature(template_id)

    if not template_source:
        return []

    # Build feedback history
    feedback_history = []
    for it in session.iterations:
        if it.feedback:
            feedback_history.append(f"- Iteration {it.version}: \"{it.feedback}\"")

    if not feedback_history:
        return []

    final_code = session.get_current_code() or ""

    # Build prompt
    prompt = CATEGORIZE_PROMPT.format(
        template_name=template_id,
        template_signature=template_sig or "N/A",
        template_code=template_source,
        feedback_history="\n".join(feedback_history),
        final_code=final_code[:3000],  # Truncate if too long
    )

    options = ClaudeAgentOptions(
        max_turns=1,
        model=model,
        cwd=str(Path(__file__).parent.parent),
    )

    raw_response = ""

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    raw_response += block.text

    # Extract JSON from response
    json_match = re.search(r"\[.*\]", raw_response, re.DOTALL)
    if not json_match:
        return []

    try:
        suggestions_data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    # Convert to TemplateSuggestion objects
    suggestions = []
    for item in suggestions_data:
        if not all(k in item for k in ["category", "description", "suggested_code", "reason"]):
            continue

        category = item["category"]
        if category not in ("parameter", "template_code"):
            continue

        suggestions.append(
            TemplateSuggestion(
                template_id=template_id,
                template_file=template_file,
                category=category,
                description=item["description"],
                suggested_code=item["suggested_code"],
                reason=item["reason"],
                applied=False,
                param_name=item.get("param_name"),
                param_type=item.get("param_type"),
                param_default=item.get("param_default"),
            )
        )

    return suggestions


def analyze_template_feedback(
    session: RefinementSession,
    model: str = "claude-sonnet-4-5-20250929",
) -> list[TemplateSuggestion]:
    """Sync wrapper for analyze_template_feedback_async.

    Args:
        session: Approved refinement session
        model: Model to use

    Returns:
        List of TemplateSuggestion objects
    """
    return asyncio.run(analyze_template_feedback_async(session, model))


def apply_template_suggestion(suggestion: TemplateSuggestion) -> tuple[bool, str]:
    """Apply a template suggestion by editing the template file.

    For parameter additions:
    - Adds parameter to function signature
    - Notes that function body may need manual update

    For template_code changes:
    - Creates backup
    - Provides guidance (manual application recommended for safety)

    Args:
        suggestion: The TemplateSuggestion to apply

    Returns:
        Tuple of (success, message)
    """
    template_path = TEMPLATES_DIR / suggestion.template_file

    if not template_path.exists():
        return False, f"Template file not found: {suggestion.template_file}"

    # Create backup
    backup_path = template_path.with_suffix(".py.bak")
    shutil.copy(template_path, backup_path)

    try:
        content = template_path.read_text()

        if suggestion.category == "parameter":
            content = _add_parameter_to_function(
                content,
                suggestion.template_id,
                suggestion.param_name,
                suggestion.param_type,
                suggestion.param_default,
            )
            message = f"Added parameter `{suggestion.param_name}` to {suggestion.template_id}()"

        elif suggestion.category == "template_code":
            # For code changes, we provide guidance rather than auto-applying
            # This is safer as template code changes can be complex
            backup_path.unlink()  # Remove backup since we're not changing
            return False, (
                f"Template code change suggested for {suggestion.template_id}:\n"
                f"  {suggestion.description}\n"
                f"  Suggested: {suggestion.suggested_code}\n"
                f"  Please apply manually to {suggestion.template_file}"
            )

        else:
            backup_path.unlink()
            return False, f"Unknown category: {suggestion.category}"

        # Validate syntax
        try:
            ast.parse(content)
        except SyntaxError as e:
            # Rollback
            shutil.copy(backup_path, template_path)
            backup_path.unlink()
            return False, f"Syntax error after change: {e}"

        # Write changes
        template_path.write_text(content)
        backup_path.unlink()  # Remove backup on success

        return True, message

    except Exception as e:
        # Rollback on any error
        if backup_path.exists():
            shutil.copy(backup_path, template_path)
            backup_path.unlink()
        return False, f"Error applying suggestion: {e}"


def _add_parameter_to_function(
    content: str,
    func_name: str,
    param_name: str | None,
    param_type: str | None,
    param_default: str | None,
) -> str:
    """Add a new parameter to a function signature.

    Args:
        content: File content
        func_name: Function name to modify
        param_name: New parameter name
        param_type: Parameter type annotation
        param_default: Default value

    Returns:
        Modified content
    """
    if not param_name:
        return content

    # Build the parameter string
    if param_type and param_default:
        new_param = f"{param_name}: {param_type} = {param_default}"
    elif param_type:
        new_param = f"{param_name}: {param_type}"
    elif param_default:
        new_param = f"{param_name}={param_default}"
    else:
        new_param = param_name

    # Find the function and add parameter before the closing )
    # Pattern matches the last parameter before ) -> type:
    # We'll add the new parameter after watermark (which is typically last)

    # Regex to find function definition ending
    func_pattern = rf"(def {func_name}\([^)]+)(,?\s*\)\s*->\s*go\.Figure:)"
    match = re.search(func_pattern, content, re.DOTALL)

    if match:
        # Insert new parameter before the closing )
        before = match.group(1)
        after = match.group(2)

        # Ensure proper comma
        if not before.rstrip().endswith(","):
            before = before.rstrip() + ","

        # Add with proper indentation (assuming 4 spaces)
        new_content = before + f"\n    {new_param}" + after
        content = content[:match.start()] + new_content + content[match.end():]

    return content


def validate_template_syntax(template_file: str) -> tuple[bool, str]:
    """Validate that a template file has valid Python syntax.

    Args:
        template_file: Filename in templates directory

    Returns:
        Tuple of (valid, error_message)
    """
    template_path = TEMPLATES_DIR / template_file

    if not template_path.exists():
        return False, f"File not found: {template_file}"

    try:
        content = template_path.read_text()
        ast.parse(content)
        return True, "Syntax valid"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"


if __name__ == "__main__":
    # Test template detection
    test_code = """
from app.templates import multi_line_chart, format_with_B

fig = multi_line_chart(df, 'date', ['BTC', 'ETH'], 'Price Comparison')
"""
    template_id, template_file = detect_template_usage(test_code)
    print(f"Detected template: {template_id} from {template_file}")

    # Test getting template source
    if template_id:
        source = get_template_source(template_id)
        print(f"\nTemplate source ({len(source) if source else 0} chars):")
        if source:
            print(source[:200] + "...")

        sig = get_template_signature(template_id)
        print(f"\nSignature:\n{sig}")
