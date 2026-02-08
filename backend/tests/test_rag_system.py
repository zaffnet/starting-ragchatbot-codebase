from unittest.mock import patch, MagicMock, call
import pytest
from vector_store import SearchResults
from config import Config


# ---------------------------------------------------------------------------
# Config propagation
# ---------------------------------------------------------------------------

class TestRAGSystemInit:
    def test_default_max_results_is_positive(self):
        """Regression guard: MAX_RESULTS must be >= 1 for ChromaDB."""
        cfg = Config()
        assert cfg.MAX_RESULTS >= 1, (
            f"MAX_RESULTS={cfg.MAX_RESULTS} would cause ChromaDB query errors"
        )

    @patch("rag_system.SessionManager")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    def test_max_results_passed_to_vector_store(
        self, MockVS, MockAI, MockDP, MockSM
    ):
        from rag_system import RAGSystem

        config = MagicMock()
        config.MAX_RESULTS = 5
        config.CHROMA_PATH = "/tmp/chroma"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.ANTHROPIC_API_KEY = "fake"
        config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.MAX_HISTORY = 2

        RAGSystem(config)

        MockVS.assert_called_once_with("/tmp/chroma", "all-MiniLM-L6-v2", 5)

    # ---- Bug reproduction ----
    @patch("rag_system.SessionManager")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    def test_init_max_results_zero_propagates(
        self, MockVS, MockAI, MockDP, MockSM
    ):
        """Proves that MAX_RESULTS=0 from config is passed directly to VectorStore."""
        from rag_system import RAGSystem

        config = MagicMock()
        config.MAX_RESULTS = 0
        config.CHROMA_PATH = "/tmp/chroma"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.ANTHROPIC_API_KEY = "fake"
        config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.MAX_HISTORY = 2

        RAGSystem(config)

        MockVS.assert_called_once_with("/tmp/chroma", "all-MiniLM-L6-v2", 0)


# ---------------------------------------------------------------------------
# Query flow
# ---------------------------------------------------------------------------

class TestRAGSystemQuery:
    def _make_rag(self, max_results=5):
        """Build a RAGSystem with all heavy deps mocked out."""
        with patch("rag_system.VectorStore") as MockVS, \
             patch("rag_system.AIGenerator") as MockAI, \
             patch("rag_system.DocumentProcessor"), \
             patch("rag_system.SessionManager") as MockSM:

            config = MagicMock()
            config.MAX_RESULTS = max_results
            config.CHROMA_PATH = "/tmp"
            config.EMBEDDING_MODEL = "m"
            config.ANTHROPIC_API_KEY = "k"
            config.ANTHROPIC_MODEL = "model"
            config.CHUNK_SIZE = 800
            config.CHUNK_OVERLAP = 100
            config.MAX_HISTORY = 2

            from rag_system import RAGSystem
            rag = RAGSystem(config)
            return rag, MockAI, MockSM

    def test_query_calls_ai_with_tools(self):
        rag, MockAI, MockSM = self._make_rag()
        rag.session_manager.get_conversation_history.return_value = None
        rag.ai_generator.generate_response.return_value = "AI answer"

        response, sources = rag.query("What is ML?")

        assert response == "AI answer"
        rag.ai_generator.generate_response.assert_called_once()
        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_manager"] is rag.tool_manager

    def test_query_retrieves_history_for_session(self):
        rag, _, _ = self._make_rag()
        rag.session_manager.get_conversation_history.return_value = "User: hi\nAssistant: hello"
        rag.ai_generator.generate_response.return_value = "answer"

        rag.query("follow up", session_id="s1")

        rag.session_manager.get_conversation_history.assert_called_with("s1")
        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] == "User: hi\nAssistant: hello"

    def test_query_records_exchange(self):
        rag, _, _ = self._make_rag()
        rag.session_manager.get_conversation_history.return_value = None
        rag.ai_generator.generate_response.return_value = "answer"

        rag.query("question", session_id="s1")

        rag.session_manager.add_exchange.assert_called_once()
        args = rag.session_manager.add_exchange.call_args[0]
        assert args[0] == "s1"
        assert "question" in args[1]
        assert args[2] == "answer"

    def test_query_resets_sources_after_retrieval(self):
        rag, _, _ = self._make_rag()
        rag.session_manager.get_conversation_history.return_value = None
        rag.ai_generator.generate_response.return_value = "answer"

        # Pre-populate sources on the search tool
        rag.search_tool.last_sources = [{"name": "X", "url": None}]

        response, sources = rag.query("q")

        # Sources should have been retrieved then reset
        assert sources == [{"name": "X", "url": None}]
        assert rag.search_tool.last_sources == []


# ---------------------------------------------------------------------------
# End-to-end bug reproduction
# ---------------------------------------------------------------------------

class TestRAGSystemBugE2E:
    @patch("rag_system.SessionManager")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    def test_max_results_zero_produces_search_error(
        self, MockVS, MockAI, MockDP, MockSM
    ):
        """End-to-end: MAX_RESULTS=0 → VectorStore.search returns error →
        CourseSearchTool returns error string → AI receives error as tool result →
        response contains unhelpful answer."""
        from rag_system import RAGSystem

        config = MagicMock()
        config.MAX_RESULTS = 0
        config.CHROMA_PATH = "/tmp"
        config.EMBEDDING_MODEL = "m"
        config.ANTHROPIC_API_KEY = "k"
        config.ANTHROPIC_MODEL = "model"
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.MAX_HISTORY = 2

        # VectorStore.search returns the error that ChromaDB would produce
        mock_vs_instance = MockVS.return_value
        mock_vs_instance.search.return_value = SearchResults.empty(
            "Search error: Number of requested results 0, cannot be less than 1."
        )

        # AI generator: simulate tool use flow
        # First call: Claude decides to use the search tool
        # We mock generate_response to actually call tool_manager.execute_tool
        # so we can verify the error propagates
        def fake_generate(query, conversation_history=None, tools=None, tool_manager=None):
            if tool_manager:
                result = tool_manager.execute_tool(
                    "search_course_content", query="test query"
                )
                # The AI would see this error and produce an unhelpful response
                if "Search error" in result:
                    return "I'm sorry, I encountered an error while searching."
            return "normal response"

        mock_ai = MockAI.return_value
        mock_ai.generate_response.side_effect = fake_generate

        mock_sm = MockSM.return_value
        mock_sm.get_conversation_history.return_value = None

        rag = RAGSystem(config)
        response, sources = rag.query("What is machine learning?")

        assert "error" in response.lower()
        # Verify the search was called with n_results=0 path
        mock_vs_instance.search.assert_called_once()
