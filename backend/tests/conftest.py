import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add backend/ to sys.path so bare imports (e.g. `from vector_store import ...`) resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    """A MagicMock with all Config fields pre-set to test values."""
    cfg = MagicMock()
    cfg.ANTHROPIC_API_KEY = "test-key"
    cfg.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    cfg.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    cfg.CHUNK_SIZE = 800
    cfg.CHUNK_OVERLAP = 100
    cfg.MAX_RESULTS = 5
    cfg.MAX_HISTORY = 2
    cfg.CHROMA_PATH = "/tmp/test_chroma"
    return cfg


@pytest.fixture
def mock_rag_system():
    """A MagicMock mimicking RAGSystem with sensible defaults."""
    rag = MagicMock()
    rag.query.return_value = ("Test answer", [{"source": "test.txt"}])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Course A", "Course B"],
    }
    rag.session_manager.create_session.return_value = "new-session-123"
    rag.session_manager.delete_session.return_value = None
    return rag


def _build_test_app(mock_rag):
    """Create a lightweight FastAPI app with the same endpoints as app.py,
    but using mock_rag instead of importing the real RAGSystem."""

    app = FastAPI()

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[dict]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag.session_manager.create_session()
            answer, sources = mock_rag.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        mock_rag.session_manager.delete_session(session_id)
        return {"status": "ok"}

    return app


@pytest.fixture
def test_app(mock_rag_system):
    """A FastAPI test app wired to mock_rag_system, returned as (app, mock_rag)."""
    app = _build_test_app(mock_rag_system)
    return app, mock_rag_system


@pytest.fixture
def client(test_app):
    """A TestClient wrapping the test app."""
    app, _ = test_app
    return TestClient(app)
