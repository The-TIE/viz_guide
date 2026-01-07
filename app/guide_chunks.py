"""Guide chunking utilities for the VizGuide Agent."""

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GuideChunk:
    """A searchable chunk from the visualization guide."""

    id: str  # e.g., "03_chart_selection__line_charts"
    source_file: str  # e.g., "03_chart_selection.md"
    section: str  # e.g., "Line Charts"
    content: str  # The actual text
    keywords: list[str] = field(default_factory=list)  # For keyword search
    chart_types: list[str] = field(default_factory=list)  # ["line", "multi_line", "area"]

    def __len__(self) -> int:
        """Return approximate token count (chars / 4)."""
        return len(self.content) // 4


# Chart type keywords for tagging chunks
CHART_TYPE_KEYWORDS = {
    "line": ["line chart", "line plot", "go.scatter", "mode='lines'", "time series"],
    "multi_line": ["multi-line", "multiple series", "multi-series", "comparing series"],
    "area": ["area chart", "go.scatter.*fill", "stacked area", "fill='tozeroy'"],
    "bar": ["bar chart", "go.bar", "horizontal bar", "vertical bar"],
    "stacked_bar": ["stacked bar", "barmode='stack'"],
    "grouped_bar": ["grouped bar", "barmode='group'"],
    "scatter": ["scatter plot", "scatter chart", "go.scatter", "mode='markers'"],
    "bubble": ["bubble chart", "bubble plot", "marker size"],
    "heatmap": ["heatmap", "go.heatmap", "correlation matrix", "color scale"],
    "histogram": ["histogram", "go.histogram", "distribution", "bin"],
    "box": ["box plot", "go.box", "quartile", "whisker"],
    "violin": ["violin plot", "go.violin", "distribution"],
    "donut": ["donut chart", "pie chart", "go.pie", "hole="],
    "candlestick": ["candlestick", "ohlc", "go.candlestick", "open high low close"],
    "dual_axis": ["dual axis", "secondary_y", "two y-axes", "dual y"],
    "subplots": ["subplot", "make_subplots", "small multiples", "facet"],
}


def extract_keywords(content: str) -> list[str]:
    """Extract keywords from content for search indexing."""
    keywords = set()

    # Extract words from headers
    headers = re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)
    for header in headers:
        words = re.findall(r"\b\w+\b", header.lower())
        keywords.update(words)

    # Extract key terms (capitalized phrases, technical terms)
    technical_terms = re.findall(r"`([^`]+)`", content)
    for term in technical_terms:
        keywords.add(term.lower())

    # Extract Plotly-specific terms
    plotly_terms = re.findall(r"go\.\w+|fig\.\w+|update_\w+|add_\w+", content, re.IGNORECASE)
    keywords.update(t.lower() for t in plotly_terms)

    # Common visualization terms
    viz_terms = [
        "axis", "legend", "hover", "tooltip", "color", "theme", "format",
        "label", "title", "margin", "grid", "tick", "annotation", "layout"
    ]
    content_lower = content.lower()
    for term in viz_terms:
        if term in content_lower:
            keywords.add(term)

    return list(keywords)


def extract_chart_types(content: str) -> list[str]:
    """Identify which chart types a chunk is relevant to."""
    content_lower = content.lower()
    chart_types = []

    for chart_type, keywords in CHART_TYPE_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in content_lower:
                chart_types.append(chart_type)
                break

    return chart_types


def chunk_markdown(content: str, source_file: str) -> list[GuideChunk]:
    """Split a markdown file into chunks based on ## headers."""
    chunks = []
    file_stem = Path(source_file).stem

    # Split on ## headers (level 2)
    # Keep ### and deeper headers with their parent ##
    sections = re.split(r"(?=^## )", content, flags=re.MULTILINE)

    for section in sections:
        if not section.strip():
            continue

        # Extract section title
        title_match = re.match(r"^##\s+(.+?)(?:\n|$)", section)
        if title_match:
            section_title = title_match.group(1).strip()
        else:
            # First section before any ## header
            section_title = "Introduction"

        # Create chunk ID
        chunk_id = f"{file_stem}__{section_title.lower().replace(' ', '_').replace('-', '_')}"
        chunk_id = re.sub(r"[^a-z0-9_]", "", chunk_id)

        # Extract keywords and chart types
        keywords = extract_keywords(section)
        chart_types = extract_chart_types(section)

        chunk = GuideChunk(
            id=chunk_id,
            source_file=source_file,
            section=section_title,
            content=section.strip(),
            keywords=keywords,
            chart_types=chart_types,
        )
        chunks.append(chunk)

    return chunks


def chunk_guide(guide_dir: Path | str | None = None) -> list[GuideChunk]:
    """
    Load and chunk all guide markdown files.

    Args:
        guide_dir: Path to guide directory. Defaults to ../guide relative to this file.

    Returns:
        List of GuideChunk objects ready for indexing.
    """
    if guide_dir is None:
        guide_dir = Path(__file__).parent.parent / "guide"
    else:
        guide_dir = Path(guide_dir)

    all_chunks = []

    # Process each markdown file in sorted order
    for md_file in sorted(guide_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        chunks = chunk_markdown(content, md_file.name)
        all_chunks.extend(chunks)

    return all_chunks


def get_chunk_stats(chunks: list[GuideChunk]) -> dict:
    """Get statistics about the chunked guide."""
    total_tokens = sum(len(c) for c in chunks)
    chart_type_counts = {}
    for chunk in chunks:
        for ct in chunk.chart_types:
            chart_type_counts[ct] = chart_type_counts.get(ct, 0) + 1

    return {
        "total_chunks": len(chunks),
        "total_tokens": total_tokens,
        "avg_tokens_per_chunk": total_tokens // len(chunks) if chunks else 0,
        "chunks_by_file": len(set(c.source_file for c in chunks)),
        "chart_type_coverage": chart_type_counts,
    }


if __name__ == "__main__":
    # Test chunking
    chunks = chunk_guide()
    stats = get_chunk_stats(chunks)

    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Avg tokens/chunk: {stats['avg_tokens_per_chunk']}")
    print(f"Files processed: {stats['chunks_by_file']}")
    print(f"\nChart type coverage:")
    for ct, count in sorted(stats["chart_type_coverage"].items(), key=lambda x: -x[1]):
        print(f"  {ct}: {count} chunks")

    print(f"\nSample chunks:")
    for chunk in chunks[:5]:
        print(f"  - {chunk.id} ({len(chunk)} tokens)")
