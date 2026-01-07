"""Guide suggestion and improvement module."""

import json
import os
import re
from pathlib import Path

import anthropic

try:
    from session import ConflictFix, GuideSuggestion, RefinementSession
    from agent import generate_visualization
except ImportError:
    from .session import ConflictFix, GuideSuggestion, RefinementSession
    from .agent import generate_visualization


GUIDE_DIR = Path(__file__).parent.parent / "guide"


SUGGESTION_PROMPT = """Analyze this visualization refinement session to identify improvements for the visualization guide.

## Session Details

**Original Request:**
{description}

**Data Sample:**
{data_sample}

**Refinement History:**
{iteration_history}

**Final Approved Code:**
```python
{final_code}
```

## Task

For each piece of feedback that required code changes, determine:
1. What underlying issue or pattern caused the problem
2. Which guide file should document this (see available files below)
3. What specific text should be added to prevent this issue

## Available Guide Files (USE EXACT FILENAMES)
- 01_data_analysis.md - Data types, dimensionality, characteristics
- 02_intent.md - Intent classification (comparison, composition, distribution, etc.)
- 03_chart_selection.md - Chart type selection and configuration
- 04_encoding.md - How data maps to visual properties
- 05_axes.md - Axis configuration and formatting
- 06_color.md - Color scales, themes, accessibility
- 07_text.md - Number formatting, labels, titles
- 08_layout.md - Overall layout, margins, spacing
- 09_legends.md - Legend positioning and formatting
- 10_hover.md - Hover templates and tooltips
- 11_interactions.md - Zoom, pan, hover interactions
- 12_annotations.md - Text annotations and shapes
- 13_accessibility.md - Accessibility considerations
- 14_performance.md - Performance optimization

## Output Format

Return a JSON array of suggestions. Each suggestion should have:
- "file": The guide filename (e.g., "09_legends.md")
- "section": The section name where this should be added
- "content": The specific text to add (1-3 sentences with code if needed)
- "reason": Why this is needed based on the feedback
- "conflicts": (optional) Array of conflicting code patterns in that file that should be fixed

For conflicts, include:
- "pattern": A regex pattern to find the conflict (e.g., "showgrid=True")
- "replacement": Simple replacement string, or null if too complex for auto-fix
- "description": Human-readable description of what conflicts

Example:
[
  {{
    "file": "05_axes.md",
    "section": "Grid Configuration",
    "content": "Grid lines should be OFF by default. Use `showgrid=False` unless gridlines add clarity for the specific chart type.",
    "reason": "User feedback indicated gridlines were appearing when they should be off",
    "conflicts": [
      {{
        "pattern": "showgrid=True",
        "replacement": "showgrid=False",
        "description": "Code examples showing showgrid=True contradict the default-off rule"
      }}
    ]
  }},
  {{
    "file": "09_legends.md",
    "section": "Legend Positioning",
    "content": "When using a subtitle, position the legend at y=1.08 or higher to avoid overlap.",
    "reason": "User feedback indicated legend was overlapping with subtitle",
    "conflicts": null
  }}
]

Only suggest additions for issues that were actually reported in the feedback.
Return an empty array [] if no guide improvements are needed.
"""


def analyze_session(
    session: RefinementSession,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-5-20250929",
) -> list[GuideSuggestion]:
    """
    Analyze a completed session to suggest guide improvements.

    Args:
        session: The approved refinement session
        api_key: Anthropic API key
        model: Model to use

    Returns:
        List of GuideSuggestion objects
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API key required.")

    if session.status != "approved":
        return []

    if len(session.iterations) < 2:
        # No refinements were needed
        return []

    client = anthropic.Anthropic(api_key=api_key)

    # Build iteration history with feedback
    iteration_history = []
    for it in session.iterations:
        entry = f"**Version {it.version}:**"
        if it.feedback:
            entry += f"\n  Feedback: \"{it.feedback}\""
        iteration_history.append(entry)

    # Get final code
    final_code = session.get_current_code() or ""

    # Build prompt
    prompt = SUGGESTION_PROMPT.format(
        description=session.description,
        data_sample=session.data_sample[:1000],  # Truncate if too long
        iteration_history="\n".join(iteration_history),
        final_code=final_code,
    )

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=0.3,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse response
    response_text = response.content[0].text

    # Extract JSON from response
    json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if not json_match:
        return []

    try:
        suggestions_data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return []

    # Convert to GuideSuggestion objects
    suggestions = []
    for item in suggestions_data:
        if all(k in item for k in ["file", "section", "content", "reason"]):
            # Parse conflicts if present
            conflicts = None
            if item.get("conflicts"):
                conflicts = []
                for c in item["conflicts"]:
                    if all(k in c for k in ["pattern", "description"]):
                        conflicts.append(ConflictFix(
                            pattern=c["pattern"],
                            replacement=c.get("replacement"),  # Can be None
                            description=c["description"],
                        ))

            suggestions.append(
                GuideSuggestion(
                    file=item["file"],
                    section=item["section"],
                    content=item["content"],
                    reason=item["reason"],
                    applied=False,
                    conflicts=conflicts if conflicts else None,
                )
            )

    return suggestions


def apply_suggestion(suggestion: GuideSuggestion) -> tuple[bool, str]:
    """
    Apply a guide suggestion by editing the relevant file.

    Handles both:
    - Adding new content to the appropriate section
    - Fixing conflicting code patterns (auto-fix simple patterns, flag complex ones)

    Args:
        suggestion: The suggestion to apply

    Returns:
        Tuple of (success, message)
    """
    guide_path = GUIDE_DIR / suggestion.file

    if not guide_path.exists():
        return False, f"File not found: {suggestion.file}"

    content = guide_path.read_text()
    messages = []

    # Step 1: Handle conflicts (hybrid approach)
    conflicts_fixed = 0
    conflicts_flagged = []

    if suggestion.conflicts:
        for conflict in suggestion.conflicts:
            try:
                # Count matches before replacing
                matches = re.findall(conflict.pattern, content)
                if matches:
                    if conflict.replacement is not None:
                        # Simple pattern - auto-fix
                        content = re.sub(conflict.pattern, conflict.replacement, content)
                        conflicts_fixed += len(matches)
                    else:
                        # Complex pattern - flag for manual review
                        conflicts_flagged.append(
                            f"  - Pattern '{conflict.pattern}': {conflict.description} ({len(matches)} occurrences)"
                        )
            except re.error as e:
                conflicts_flagged.append(f"  - Invalid regex '{conflict.pattern}': {e}")

    if conflicts_fixed > 0:
        messages.append(f"Auto-fixed {conflicts_fixed} conflicting pattern(s)")

    if conflicts_flagged:
        messages.append("Manual review needed:\n" + "\n".join(conflicts_flagged))

    # Step 2: Add new content to the section
    section_pattern = rf"(##\s*{re.escape(suggestion.section)}.*?)(?=\n##|\Z)"
    match = re.search(section_pattern, content, re.DOTALL | re.IGNORECASE)

    if match:
        # Add content at the end of the section
        section_end = match.end()
        content = (
            content[:section_end].rstrip()
            + "\n\n"
            + suggestion.content
            + "\n"
            + content[section_end:]
        )
        messages.append(f"Added to section '{suggestion.section}'")
    else:
        # Section not found - add to end of file
        content = content.rstrip() + f"\n\n## {suggestion.section}\n\n{suggestion.content}\n"
        messages.append(f"Created new section '{suggestion.section}'")

    guide_path.write_text(content)
    suggestion.applied = True

    return True, "; ".join(messages)


def validate_improvement(
    session: RefinementSession,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-5-20250929",
    watermark: str = "none",
) -> dict:
    """
    Re-run the original task with the current guide to verify improvements.

    Args:
        session: The original session to validate
        api_key: Anthropic API key
        model: Model to use
        watermark: Watermark setting

    Returns:
        dict with "code", "success", and "comparison" keys
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("API key required.")

    # Generate new visualization with current guide
    result = generate_visualization(
        description=session.description,
        data_sample=session.data_sample,
        api_key=api_key,
        model=model,
        watermark=watermark,
    )

    new_code = result.get("code", "")
    original_code = session.iterations[0].code if session.iterations else ""
    final_code = session.get_current_code() or ""

    # Simple comparison: check if known issues are addressed
    # This could be enhanced with more sophisticated comparison
    comparison = {
        "original_code_length": len(original_code),
        "final_code_length": len(final_code),
        "new_code_length": len(new_code),
        "iterations_needed_before": len(session.iterations),
    }

    return {
        "code": new_code,
        "success": bool(new_code),
        "comparison": comparison,
        "tool_calls": result.get("tool_calls", []),
    }


def get_feedback_patterns(sessions: list[RefinementSession]) -> dict[str, int]:
    """
    Analyze multiple sessions to find common feedback patterns.

    Args:
        sessions: List of sessions to analyze

    Returns:
        dict mapping feedback keywords to counts
    """
    patterns = {}
    keywords = [
        "legend", "hover", "color", "axis", "label", "format",
        "billion", "title", "subtitle", "margin", "grid", "theme",
        "overlap", "position", "size", "font", "annotation"
    ]

    for session in sessions:
        for iteration in session.iterations:
            if iteration.feedback:
                feedback_lower = iteration.feedback.lower()
                for keyword in keywords:
                    if keyword in feedback_lower:
                        patterns[keyword] = patterns.get(keyword, 0) + 1

    return dict(sorted(patterns.items(), key=lambda x: -x[1]))


if __name__ == "__main__":
    # Test the suggestion module
    from session import RefinementSession, Iteration

    # Create a mock approved session
    session = RefinementSession(
        id="test123",
        started_at="2024-01-01T00:00:00",
        description="Compare BTC and ETH prices over time",
        data_sample="date, BTC_price, ETH_price",
        watermark="none",
        iterations=[
            Iteration(
                version=1,
                code="fig = go.Figure()\nfig.update_layout(title='BTC vs ETH')",
                feedback="The legend is overlapping with the subtitle",
                timestamp="2024-01-01T00:01:00",
            ),
            Iteration(
                version=2,
                code="fig = go.Figure()\nfig.update_layout(title='BTC vs ETH', legend=dict(y=1.08))",
                feedback="Use B for billions not G",
                timestamp="2024-01-01T00:02:00",
            ),
            Iteration(
                version=3,
                code="fig = go.Figure()\nfig.update_layout(title='BTC vs ETH', legend=dict(y=1.08))\n# Using format_with_B",
                feedback=None,
                timestamp="2024-01-01T00:03:00",
            ),
        ],
        status="approved",
        approved_at="2024-01-01T00:03:00",
    )

    print("Testing guide suggestion analysis...")
    print(f"Session: {session.id}")
    print(f"Iterations: {len(session.iterations)}")

    # Test pattern analysis (doesn't need API)
    patterns = get_feedback_patterns([session])
    print(f"\nFeedback patterns: {patterns}")

    # Test suggestion analysis (needs API)
    try:
        suggestions = analyze_session(session)
        print(f"\nSuggestions generated: {len(suggestions)}")
        for s in suggestions:
            print(f"  - {s.file} / {s.section}")
            print(f"    Content: {s.content[:80]}...")
            print(f"    Reason: {s.reason}")
    except Exception as e:
        print(f"Suggestion analysis skipped (needs API): {e}")
