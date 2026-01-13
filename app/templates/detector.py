"""Chart type detection from existing Plotly code using AST parsing."""

import ast
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionResult:
    """Result of chart type detection from code."""

    template_id: str
    confidence: float
    trace_types: list[str]
    trace_count: int
    layout_properties: dict
    detection_method: str  # "ast" or "regex"


def detect_chart_type(code: str) -> DetectionResult:
    """Detect chart type from Plotly code.

    Uses AST parsing for accurate detection with regex fallback.

    Args:
        code: Python code containing Plotly figure

    Returns:
        DetectionResult with template_id and confidence
    """
    try:
        return _detect_with_ast(code)
    except SyntaxError:
        return _detect_with_regex(code)


def _detect_with_ast(code: str) -> DetectionResult:
    """AST-based detection for valid Python code."""
    tree = ast.parse(code)

    trace_types = []
    trace_kwargs = []
    layout_props = {}
    uses_make_subplots = False

    # Trace types to detect (exclude Figure which is the container)
    TRACE_TYPES = {"Scatter", "Bar", "Heatmap", "Pie", "Candlestick", "Box", "Histogram", "Violin", "Funnel"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Detect go.Scatter, go.Bar, etc.
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "go":
                    trace_type = node.func.attr
                    # Only include actual trace types, not Figure
                    if trace_type in TRACE_TYPES:
                        trace_types.append(trace_type)

                        # Extract kwargs
                        kwargs = {}
                        for kw in node.keywords:
                            if kw.arg and isinstance(kw.value, ast.Constant):
                                kwargs[kw.arg] = kw.value.value
                        trace_kwargs.append(kwargs)

            # Detect make_subplots
            if isinstance(node.func, ast.Name) and node.func.id == "make_subplots":
                uses_make_subplots = True

            # Detect update_layout
            if isinstance(node.func, ast.Attribute) and node.func.attr == "update_layout":
                for kw in node.keywords:
                    if kw.arg and isinstance(kw.value, ast.Constant):
                        layout_props[kw.arg] = kw.value.value

    return _resolve_template(
        trace_types=trace_types,
        trace_kwargs=trace_kwargs,
        layout_props=layout_props,
        uses_make_subplots=uses_make_subplots,
        method="ast",
    )


def _detect_with_regex(code: str) -> DetectionResult:
    """Regex fallback for malformed code."""
    trace_types = []
    trace_kwargs = [{}]
    layout_props = {}

    # Detect trace types
    trace_patterns = {
        "Scatter": r"go\.Scatter\(",
        "Bar": r"go\.Bar\(",
        "Heatmap": r"go\.Heatmap\(",
        "Pie": r"go\.Pie\(",
        "Candlestick": r"go\.Candlestick\(",
        "Box": r"go\.Box\(",
        "Histogram": r"go\.Histogram\(",
    }

    for trace_type, pattern in trace_patterns.items():
        if re.search(pattern, code):
            count = len(re.findall(pattern, code))
            trace_types.extend([trace_type] * count)

    # Detect mode
    mode_match = re.search(r"mode=['\"](\w+)['\"]", code)
    if mode_match:
        trace_kwargs[0]["mode"] = mode_match.group(1)

    # Detect orientation
    orient_match = re.search(r"orientation=['\"](\w)['\"]", code)
    if orient_match:
        trace_kwargs[0]["orientation"] = orient_match.group(1)

    # Detect barmode
    barmode_match = re.search(r"barmode=['\"](\w+)['\"]", code)
    if barmode_match:
        layout_props["barmode"] = barmode_match.group(1)

    # Detect make_subplots
    uses_make_subplots = "make_subplots" in code

    return _resolve_template(
        trace_types=trace_types,
        trace_kwargs=trace_kwargs,
        layout_props=layout_props,
        uses_make_subplots=uses_make_subplots,
        method="regex",
    )


def _resolve_template(
    trace_types: list[str],
    trace_kwargs: list[dict],
    layout_props: dict,
    uses_make_subplots: bool,
    method: str,
) -> DetectionResult:
    """Resolve detected patterns to a template ID."""

    if not trace_types:
        return DetectionResult(
            template_id="line",
            confidence=0.3,
            trace_types=[],
            trace_count=0,
            layout_properties=layout_props,
            detection_method=method,
        )

    primary_trace = trace_types[0]
    trace_count = len(trace_types)
    first_kwargs = trace_kwargs[0] if trace_kwargs else {}

    # Small multiples detection (highest priority)
    if uses_make_subplots:
        return DetectionResult(
            template_id="small_multiples",
            confidence=0.9,
            trace_types=trace_types,
            trace_count=trace_count,
            layout_properties=layout_props,
            detection_method=method,
        )

    # Bar chart family
    if primary_trace == "Bar":
        orientation = first_kwargs.get("orientation")
        barmode = layout_props.get("barmode")

        if barmode == "stack":
            return DetectionResult(
                template_id="stacked_bar",
                confidence=0.95,
                trace_types=trace_types,
                trace_count=trace_count,
                layout_properties=layout_props,
                detection_method=method,
            )
        elif barmode == "group":
            return DetectionResult(
                template_id="grouped_bar",
                confidence=0.95,
                trace_types=trace_types,
                trace_count=trace_count,
                layout_properties=layout_props,
                detection_method=method,
            )
        elif orientation == "h":
            return DetectionResult(
                template_id="horizontal_bar",
                confidence=0.95,
                trace_types=trace_types,
                trace_count=trace_count,
                layout_properties=layout_props,
                detection_method=method,
            )
        else:
            return DetectionResult(
                template_id="bar",
                confidence=0.9,
                trace_types=trace_types,
                trace_count=trace_count,
                layout_properties=layout_props,
                detection_method=method,
            )

    # Line/Scatter family
    if primary_trace == "Scatter":
        mode = first_kwargs.get("mode", "lines")

        # Scatter plot (markers only)
        if "markers" in mode and "lines" not in mode:
            return DetectionResult(
                template_id="scatter",
                confidence=0.9,
                trace_types=trace_types,
                trace_count=trace_count,
                layout_properties=layout_props,
                detection_method=method,
            )

        # Line charts
        if trace_count == 1:
            return DetectionResult(
                template_id="line",
                confidence=0.9,
                trace_types=trace_types,
                trace_count=trace_count,
                layout_properties=layout_props,
                detection_method=method,
            )
        elif trace_count <= 5:
            return DetectionResult(
                template_id="multi_line",
                confidence=0.85,
                trace_types=trace_types,
                trace_count=trace_count,
                layout_properties=layout_props,
                detection_method=method,
            )
        else:
            # Many traces without subplots - recommend small multiples
            return DetectionResult(
                template_id="multi_line",
                confidence=0.6,
                trace_types=trace_types,
                trace_count=trace_count,
                layout_properties=layout_props,
                detection_method=method,
            )

    # Other specific types
    if primary_trace == "Heatmap":
        return DetectionResult(
            template_id="heatmap",
            confidence=0.95,
            trace_types=trace_types,
            trace_count=trace_count,
            layout_properties=layout_props,
            detection_method=method,
        )

    if primary_trace == "Pie":
        return DetectionResult(
            template_id="donut",
            confidence=0.9,
            trace_types=trace_types,
            trace_count=trace_count,
            layout_properties=layout_props,
            detection_method=method,
        )

    if primary_trace == "Candlestick":
        return DetectionResult(
            template_id="candlestick",
            confidence=0.95,
            trace_types=trace_types,
            trace_count=trace_count,
            layout_properties=layout_props,
            detection_method=method,
        )

    if primary_trace == "Histogram":
        return DetectionResult(
            template_id="histogram",
            confidence=0.9,
            trace_types=trace_types,
            trace_count=trace_count,
            layout_properties=layout_props,
            detection_method=method,
        )

    if primary_trace == "Box":
        return DetectionResult(
            template_id="box",
            confidence=0.9,
            trace_types=trace_types,
            trace_count=trace_count,
            layout_properties=layout_props,
            detection_method=method,
        )

    # Fallback
    return DetectionResult(
        template_id="line",
        confidence=0.3,
        trace_types=trace_types,
        trace_count=trace_count,
        layout_properties=layout_props,
        detection_method=method,
    )


if __name__ == "__main__":
    # Test detection
    test_codes = [
        # Multi-line chart
        """
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['date'], y=df['BTC'], mode='lines', name='BTC'))
fig.add_trace(go.Scatter(x=df['date'], y=df['ETH'], mode='lines', name='ETH'))
fig.update_layout(hovermode='x unified')
""",
        # Horizontal bar
        """
fig = go.Figure()
fig.add_trace(go.Bar(y=df['exchange'], x=df['volume'], orientation='h'))
""",
        # Stacked bar
        """
fig = go.Figure()
fig.add_trace(go.Bar(x=df['month'], y=df['A'], name='A'))
fig.add_trace(go.Bar(x=df['month'], y=df['B'], name='B'))
fig.update_layout(barmode='stack')
""",
        # Small multiples
        """
from plotly.subplots import make_subplots
fig = make_subplots(rows=2, cols=2)
fig.add_trace(go.Scatter(x=df['date'], y=df['A']), row=1, col=1)
""",
    ]

    for i, code in enumerate(test_codes, 1):
        result = detect_chart_type(code)
        print(f"Test {i}: {result.template_id} (confidence: {result.confidence}, method: {result.detection_method})")
