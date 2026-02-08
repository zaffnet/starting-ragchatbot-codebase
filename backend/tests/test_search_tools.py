from unittest.mock import MagicMock
import pytest
from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(**overrides):
    """Return a MagicMock VectorStore with sensible defaults."""
    store = MagicMock()
    store.search.return_value = overrides.get(
        "search_return",
        SearchResults(documents=[], metadata=[], distances=[]),
    )
    store.get_lesson_link.return_value = overrides.get("lesson_link", None)
    return store


# ---------------------------------------------------------------------------
# CourseSearchTool.execute – success / empty / error
# ---------------------------------------------------------------------------

class TestCourseSearchToolExecute:
    def test_successful_search_returns_formatted_results(self):
        store = _make_store(
            search_return=SearchResults(
                documents=["chunk text"],
                metadata=[{"course_title": "Intro to AI", "lesson_number": 1}],
                distances=[0.3],
            )
        )
        tool = CourseSearchTool(store)
        result = tool.execute(query="neural networks")

        assert "chunk text" in result
        assert "[Intro to AI - Lesson 1]" in result
        store.search.assert_called_once_with(
            query="neural networks", course_name=None, lesson_number=None
        )

    def test_successful_search_tracks_sources(self):
        store = _make_store(
            search_return=SearchResults(
                documents=["text"],
                metadata=[{"course_title": "ML", "lesson_number": 2}],
                distances=[0.1],
            ),
            lesson_link="https://example.com/ml/2",
        )
        tool = CourseSearchTool(store)
        tool.execute(query="gradient descent")

        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["name"] == "ML - Lesson 2"
        assert tool.last_sources[0]["url"] == "https://example.com/ml/2"

    def test_empty_results_no_filter(self):
        tool = CourseSearchTool(_make_store())
        result = tool.execute(query="unknown topic")
        assert result == "No relevant content found."

    def test_empty_results_with_course_filter(self):
        tool = CourseSearchTool(_make_store())
        result = tool.execute(query="topic", course_name="Physics 101")
        assert "Physics 101" in result

    def test_empty_results_with_lesson_filter(self):
        tool = CourseSearchTool(_make_store())
        result = tool.execute(query="topic", lesson_number=3)
        assert "lesson 3" in result

    def test_empty_results_with_both_filters(self):
        tool = CourseSearchTool(_make_store())
        result = tool.execute(query="topic", course_name="Physics", lesson_number=5)
        assert "Physics" in result
        assert "lesson 5" in result

    def test_error_propagated(self):
        store = _make_store(
            search_return=SearchResults.empty("Search error: something broke")
        )
        tool = CourseSearchTool(store)
        result = tool.execute(query="anything")
        assert result == "Search error: something broke"

    # ---- Bug reproduction ----
    def test_execute_max_results_zero_causes_search_error(self):
        """Proves that when VectorStore is configured with max_results=0,
        ChromaDB raises an error which propagates as a string result."""
        store = _make_store(
            search_return=SearchResults.empty(
                "Search error: Number of requested results 0, cannot be less than 1."
            )
        )
        tool = CourseSearchTool(store)
        result = tool.execute(query="what is machine learning")
        assert "Search error" in result
        assert "cannot be less than 1" in result


# ---------------------------------------------------------------------------
# _format_results – header formatting and source URL tracking
# ---------------------------------------------------------------------------

class TestFormatResults:
    def test_header_without_lesson(self):
        store = _make_store()
        tool = CourseSearchTool(store)
        results = SearchResults(
            documents=["content"],
            metadata=[{"course_title": "Docker Basics"}],
            distances=[0.2],
        )
        formatted = tool._format_results(results)
        assert "[Docker Basics]" in formatted
        assert "Lesson" not in formatted

    def test_header_with_lesson(self):
        store = _make_store()
        tool = CourseSearchTool(store)
        results = SearchResults(
            documents=["content"],
            metadata=[{"course_title": "Docker Basics", "lesson_number": 4}],
            distances=[0.2],
        )
        formatted = tool._format_results(results)
        assert "[Docker Basics - Lesson 4]" in formatted

    def test_source_url_with_lesson_link(self):
        store = _make_store(lesson_link="https://example.com/lesson/4")
        tool = CourseSearchTool(store)
        results = SearchResults(
            documents=["content"],
            metadata=[{"course_title": "Docker Basics", "lesson_number": 4}],
            distances=[0.2],
        )
        tool._format_results(results)
        assert tool.last_sources[0]["url"] == "https://example.com/lesson/4"

    def test_source_url_none_without_lesson(self):
        store = _make_store()
        tool = CourseSearchTool(store)
        results = SearchResults(
            documents=["content"],
            metadata=[{"course_title": "Docker Basics"}],
            distances=[0.2],
        )
        tool._format_results(results)
        assert tool.last_sources[0]["url"] is None


# ---------------------------------------------------------------------------
# ToolManager
# ---------------------------------------------------------------------------

class TestToolManager:
    def _registered_manager(self):
        store = _make_store(
            search_return=SearchResults(
                documents=["data"],
                metadata=[{"course_title": "X", "lesson_number": 1}],
                distances=[0.1],
            )
        )
        tm = ToolManager()
        tm.register_tool(CourseSearchTool(store))
        return tm

    def test_register_and_get_definitions(self):
        tm = self._registered_manager()
        defs = tm.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "search_course_content"

    def test_execute_known_tool(self):
        tm = self._registered_manager()
        result = tm.execute_tool("search_course_content", query="test")
        assert "data" in result

    def test_execute_unknown_tool(self):
        tm = self._registered_manager()
        result = tm.execute_tool("nonexistent", query="test")
        assert "not found" in result

    def test_get_and_reset_sources(self):
        tm = self._registered_manager()
        tm.execute_tool("search_course_content", query="test")
        sources = tm.get_last_sources()
        assert len(sources) == 1

        tm.reset_sources()
        assert tm.get_last_sources() == []
