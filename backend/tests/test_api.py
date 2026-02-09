# ---------------------------------------------------------------------------
# Tests — POST /api/query
# ---------------------------------------------------------------------------


class TestQueryEndpoint:
    def test_query_success(self, client, test_app):
        """200 response with correct JSON shape."""
        _, mock_rag = test_app
        mock_rag.query.return_value = ("The answer is 42", [{"source": "guide.txt"}])

        resp = client.post("/api/query", json={"query": "What is the answer?", "session_id": "s1"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "The answer is 42"
        assert data["session_id"] == "s1"
        assert isinstance(data["sources"], list)

    def test_query_creates_session_when_missing(self, client, test_app):
        """When no session_id is provided, one is auto-created."""
        _, mock_rag = test_app
        mock_rag.session_manager.create_session.return_value = "auto-abc"

        resp = client.post("/api/query", json={"query": "hello"})

        assert resp.status_code == 200
        assert resp.json()["session_id"] == "auto-abc"
        mock_rag.session_manager.create_session.assert_called_once()

    def test_query_uses_provided_session_id(self, client, test_app):
        """Passes through an existing session_id without creating a new one."""
        _, mock_rag = test_app

        resp = client.post("/api/query", json={"query": "hi", "session_id": "existing-1"})

        assert resp.status_code == 200
        assert resp.json()["session_id"] == "existing-1"
        mock_rag.session_manager.create_session.assert_not_called()

    def test_query_returns_sources(self, client, test_app):
        """Sources array is included in response."""
        _, mock_rag = test_app
        mock_rag.query.return_value = (
            "answer",
            [{"source": "a.txt", "page": 1}, {"source": "b.txt", "page": 2}],
        )

        resp = client.post("/api/query", json={"query": "q", "session_id": "s"})

        sources = resp.json()["sources"]
        assert len(sources) == 2
        assert sources[0]["source"] == "a.txt"
        assert sources[1]["source"] == "b.txt"

    def test_query_internal_error(self, client, test_app):
        """When rag.query raises, the endpoint returns 500."""
        _, mock_rag = test_app
        mock_rag.query.side_effect = RuntimeError("db connection lost")

        resp = client.post("/api/query", json={"query": "q", "session_id": "s"})

        assert resp.status_code == 500
        assert "db connection lost" in resp.json()["detail"]

    def test_query_missing_body(self, client):
        """No request body → 422 validation error."""
        resp = client.post("/api/query")

        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests — GET /api/courses
# ---------------------------------------------------------------------------


class TestCoursesEndpoint:
    def test_courses_success(self, client, test_app):
        """200 response with correct shape."""
        _, mock_rag = test_app
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["ML 101", "NLP 201", "DL 301"],
        }

        resp = client.get("/api/courses")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "ML 101" in data["course_titles"]

    def test_courses_empty(self, client, test_app):
        """Zero courses returns empty list."""
        _, mock_rag = test_app
        mock_rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        resp = client.get("/api/courses")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_courses_internal_error(self, client, test_app):
        """When analytics raises, the endpoint returns 500."""
        _, mock_rag = test_app
        mock_rag.get_course_analytics.side_effect = RuntimeError("storage unavailable")

        resp = client.get("/api/courses")

        assert resp.status_code == 500
        assert "storage unavailable" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Tests — DELETE /api/sessions/{session_id}
# ---------------------------------------------------------------------------


class TestDeleteSessionEndpoint:
    def test_delete_session_success(self, client):
        """200 response with {"status": "ok"}."""
        resp = client.delete("/api/sessions/sess-42")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_delete_session_calls_manager(self, client, test_app):
        """Verifies delete_session is called with the correct session_id."""
        _, mock_rag = test_app

        client.delete("/api/sessions/sess-99")

        mock_rag.session_manager.delete_session.assert_called_once_with("sess-99")
