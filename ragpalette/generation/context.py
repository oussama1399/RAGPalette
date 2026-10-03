from ragpalette.core.models import SearchResult


class ContextBuilder:
    """Format retrieved search results as LLM context."""

    def build(self, results: list[SearchResult]) -> str:
        if not results:
            return ""

        formatted_sources: list[str] = []

        for position, result in enumerate(results, start=1):
            # TODO 1:
            # Format one source using:
            # - position
            # - result.source
            # - result.chunk.text
            formatted_source = f"Source {position}: {result.source}\n{result.chunk.text}"

            # TODO 2:
            # Add the formatted source to formatted_sources.
            formatted_sources.append(formatted_source)

        # TODO 3:
        # Join all formatted sources with blank lines.
        return "\n\n".join(formatted_sources)