# Plotly Visualization Style Guide

A comprehensive style guide for creating consistent, publication-ready Plotly visualizations. Designed to be used directly by humans or as context for AI code generation.

## Why This Guide?

Data visualization is deceptively hard to get right. Default Plotly outputs often have:
- Ugly tick labels (SI notation showing "G" instead of "B" for billions)
- Poor dark theme styling
- Cluttered legends and annotations
- Inconsistent formatting across charts

This guide codifies best practices into actionable rules that produce clean, professional visualizations every time.

## The Guide

The guide lives in [`guide/`](guide/) as 14 markdown sections:

| Section | What It Covers |
|---------|----------------|
| [00 Index](guide/00_index.md) | Quick reference and templates |
| [01 Data Analysis](guide/01_data_analysis.md) | Understanding your data before charting |
| [02 Intent](guide/02_intent.md) | Choosing the right visualization type |
| [03 Chart Selection](guide/03_chart_selection.md) | When to use each chart type |
| [04 Encoding](guide/04_encoding.md) | Mapping data to visual properties |
| [05 Axes](guide/05_axes.md) | Axis configuration and tick formatting |
| [06 Color](guide/06_color.md) | Color scales, themes, accessibility |
| [07 Text](guide/07_text.md) | Labels, titles, number formatting |
| [08 Layout](guide/08_layout.md) | Margins, spacing, positioning |
| [09 Legends](guide/09_legends.md) | Legend placement and styling |
| [10 Hover](guide/10_hover.md) | Tooltips and hover templates |
| [11 Interactions](guide/11_interactions.md) | Zoom, pan, selection |
| [12 Annotations](guide/12_annotations.md) | Text annotations and shapes |
| [13 Accessibility](guide/13_accessibility.md) | Making charts accessible |
| [14 Performance](guide/14_performance.md) | Optimizing large datasets |

## Using the Guide

### For Humans
Read the guide sections directly. Each includes code examples and anti-patterns to avoid.

### For AI Code Generation
Include relevant guide sections as context when prompting an LLM to generate Plotly code. The guide is structured for easy retrieval and includes explicit rules the model can follow.

### With the Tester App
The included Streamlit app generates guide-compliant visualizations and helps refine the guide itself through iterative feedback.

## Tester App

The app serves two purposes:
1. **Generate visualizations** that follow the guide
2. **Refine the guide** by capturing feedback on what works and what doesn't

### Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone and run:
```bash
git clone git@github.com:The-TIE/viz_guide.git
cd viz_guide
uv sync
cp .env.example .env  # Add your Anthropic API key
uv run streamlit run app/main.py
```

### Workflow

1. Describe a visualization and paste sample data
2. Review the generated chart, provide feedback
3. Iterate until the chart meets standards
4. Approve to capture the result
5. Review suggested guide improvements, apply via PR

## Contributing

Guide improvements go through PR review. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Project Structure

```
guide/           # The style guide (14 markdown sections)
app/             # Streamlit tester application
feedback/captures/  # Gallery of approved visualizations
references/      # Logo and watermark images
```
