"""Fuzzy search index for the visualization guide."""

import re
from difflib import SequenceMatcher
from pathlib import Path

try:
    from guide_chunks import GuideChunk, chunk_guide
except ImportError:
    from .guide_chunks import GuideChunk, chunk_guide


class GuideIndex:
    """Searchable index over guide chunks with fuzzy matching."""

    def __init__(self, chunks: list[GuideChunk] | None = None):
        """
        Initialize the index.

        Args:
            chunks: List of GuideChunks. If None, loads from guide directory.
        """
        if chunks is None:
            chunks = chunk_guide()
        self.chunks = chunks
        self._build_indices()

    def _build_indices(self) -> None:
        """Build lookup indices for fast access."""
        # Index by chart type
        self.chart_type_index: dict[str, list[GuideChunk]] = {}
        for chunk in self.chunks:
            for ct in chunk.chart_types:
                if ct not in self.chart_type_index:
                    self.chart_type_index[ct] = []
                self.chart_type_index[ct].append(chunk)

        # Index by source file
        self.file_index: dict[str, list[GuideChunk]] = {}
        for chunk in self.chunks:
            if chunk.source_file not in self.file_index:
                self.file_index[chunk.source_file] = []
            self.file_index[chunk.source_file].append(chunk)

        # Build keyword to chunk mapping for fast lookup
        self.keyword_index: dict[str, list[GuideChunk]] = {}
        for chunk in self.chunks:
            for kw in chunk.keywords:
                kw_lower = kw.lower()
                if kw_lower not in self.keyword_index:
                    self.keyword_index[kw_lower] = []
                self.keyword_index[kw_lower].append(chunk)

    def search(self, query: str, limit: int = 5) -> list[GuideChunk]:
        """
        Fuzzy search with grep-like matching.

        Args:
            query: Search query string
            limit: Maximum number of results

        Returns:
            List of matching chunks, sorted by relevance score
        """
        query_terms = query.lower().split()
        scored_chunks: list[tuple[float, GuideChunk]] = []

        for chunk in self.chunks:
            content_lower = chunk.content.lower()
            score = 0.0

            # Exact substring matches (grep-like) - highest weight
            for term in query_terms:
                if term in content_lower:
                    # Count occurrences, cap at 5 to avoid over-weighting
                    count = min(content_lower.count(term), 5)
                    score += count * 3

            # Fuzzy matching on section titles
            title_ratio = SequenceMatcher(
                None, query.lower(), chunk.section.lower()
            ).ratio()
            score += title_ratio * 15

            # Keyword tag matches
            for kw in chunk.keywords:
                kw_lower = kw.lower()
                for term in query_terms:
                    if term in kw_lower:
                        score += 5
                    # Partial match bonus
                    elif len(term) >= 4 and term[:4] in kw_lower:
                        score += 2

            # Boost for matching chart types in query
            for ct in chunk.chart_types:
                ct_words = ct.replace("_", " ").split()
                for ct_word in ct_words:
                    if ct_word in query.lower():
                        score += 8

            if score > 0:
                scored_chunks.append((score, chunk))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:limit]]

    def get_by_chart_type(self, chart_type: str) -> list[GuideChunk]:
        """
        Get all chunks tagged with a specific chart type.

        Args:
            chart_type: One of the chart type keys (line, bar, scatter, etc.)

        Returns:
            List of chunks relevant to that chart type
        """
        return self.chart_type_index.get(chart_type, [])

    def get_by_file(self, filename: str) -> list[GuideChunk]:
        """
        Get all chunks from a specific guide file.

        Args:
            filename: The source filename (e.g., "03_chart_selection.md")

        Returns:
            List of chunks from that file
        """
        return self.file_index.get(filename, [])

    def grep(self, pattern: str, limit: int = 10) -> list[GuideChunk]:
        """
        Regex grep across all chunks.

        Args:
            pattern: Regex pattern to search for
            limit: Maximum number of results

        Returns:
            List of chunks matching the pattern
        """
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # Invalid regex, fall back to literal search
            pattern_escaped = re.escape(pattern)
            regex = re.compile(pattern_escaped, re.IGNORECASE)

        matches = []
        for chunk in self.chunks:
            if regex.search(chunk.content):
                matches.append(chunk)
                if len(matches) >= limit:
                    break

        return matches

    def get_chunk_by_id(self, chunk_id: str) -> GuideChunk | None:
        """Get a specific chunk by its ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None

    def format_chunks(self, chunks: list[GuideChunk], max_tokens: int = 8000) -> str:
        """
        Format chunks into a string for the agent context.

        Args:
            chunks: List of chunks to format
            max_tokens: Maximum approximate tokens to include

        Returns:
            Formatted string with chunk contents
        """
        result_parts = []
        total_tokens = 0

        for chunk in chunks:
            chunk_tokens = len(chunk)
            if total_tokens + chunk_tokens > max_tokens:
                # Add truncation notice
                result_parts.append(f"\n[Truncated: {len(chunks) - len(result_parts)} more chunks available]")
                break

            result_parts.append(f"### {chunk.section} (from {chunk.source_file})\n\n{chunk.content}")
            total_tokens += chunk_tokens

        return "\n\n---\n\n".join(result_parts)


# Singleton instance for reuse
_index_instance: GuideIndex | None = None


def get_guide_index() -> GuideIndex:
    """Get or create the singleton guide index."""
    global _index_instance
    if _index_instance is None:
        _index_instance = GuideIndex()
    return _index_instance


def reload_guide_index() -> GuideIndex:
    """Force reload the guide index from disk."""
    global _index_instance
    _index_instance = GuideIndex()
    return _index_instance


if __name__ == "__main__":
    # Test the search index
    index = GuideIndex()

    print("=== Testing Search ===")
    test_queries = [
        "hover template unified",
        "bar chart text labels",
        "format billions",
        "dark theme colors",
        "line chart time series",
    ]

    for query in test_queries:
        results = index.search(query, limit=3)
        print(f"\nQuery: '{query}'")
        for r in results:
            print(f"  - {r.section} ({r.source_file}) [{len(r)} tokens]")

    print("\n=== Testing Chart Type Lookup ===")
    for ct in ["line", "bar", "heatmap"]:
        chunks = index.get_by_chart_type(ct)
        print(f"{ct}: {len(chunks)} chunks")

    print("\n=== Testing Grep ===")
    results = index.grep(r"format_with_B|format_billions", limit=5)
    print(f"Grep 'format_with_B|format_billions': {len(results)} matches")
    for r in results:
        print(f"  - {r.section} ({r.source_file})")
