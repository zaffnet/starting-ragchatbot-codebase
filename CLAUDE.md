# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about course materials. FastAPI backend with vanilla JS frontend, using ChromaDB for vector storage and Claude for AI generation.

## Commands

### Setup

```bash
uv sync                    # Install Python dependencies
cp .env.example .env       # Then add your ANTHROPIC_API_KEY
```

### Run

```bash
./run.sh                   # Quick start (creates docs/ dir, starts server)
# Or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app serves at `http://localhost:8000` (API docs at `/docs`).

### Package Management

Always use `uv`, never `pip`. Install new dependencies with `uv add <package>`.
To run Python commands from the project root: `source ~/.zshrc && uv run python ...`

### No tests, linting, or CI exist yet

## Architecture

### Request Flow (tool-based RAG)

1. Frontend sends `POST /api/query` with `{query, session_id}`
2. `RAGSystem` fetches conversation history from `SessionManager` (in-memory, max 2 exchanges)
3. `AIGenerator` calls Claude API with the query and a `search_course_content` tool definition
4. Claude **decides** whether to search or answer directly (tool use is optional, not forced)
5. If Claude calls the tool: `CourseSearchTool` → `VectorStore` → ChromaDB → results formatted and sent back to Claude in a **second API call without tools** (single-shot, no recursive tool loops)
6. Sources are tracked via mutable state on `CourseSearchTool.last_sources` (not thread-safe)

### Key Design Decisions

- **Two ChromaDB collections**: `course_catalog` (metadata, used for fuzzy course name resolution via semantic search) and `course_content` (chunked text for content search)
- **Local embeddings, remote LLM**: `all-MiniLM-L6-v2` runs locally via sentence-transformers; only generation hits the Anthropic API
- **Document format**: Course files in `docs/` follow a structured text format with `Course Title:`, `Course Link:`, `Course Instructor:`, and `Lesson N:` markers. Supported extensions: `.pdf`, `.docx`, `.txt`
- **Chunking**: 800-char chunks with 100-char overlap, split on sentence boundaries
- **Sessions are ephemeral**: In-memory dict, lost on server restart
- **Frontend serves from backend**: Static files mounted at `/` via FastAPI's `StaticFiles`

### Backend Modules (`backend/`)

| Module | Role |
|---|---|
| `app.py` | FastAPI app, endpoints, static file serving |
| `rag_system.py` | Orchestrator wiring all components together |
| `ai_generator.py` | Claude API client with tool-use handling |
| `vector_store.py` | ChromaDB wrapper (two collections, filtered search) |
| `document_processor.py` | Parses course files, extracts metadata, chunks text |
| `search_tools.py` | `Tool` ABC, `CourseSearchTool`, `ToolManager` |
| `session_manager.py` | In-memory conversation history |
| `config.py` | Dataclass config (model, chunk size, paths) |
| `models.py` | Pydantic models: `Course`, `Lesson`, `CourseChunk` |

### Frontend (`frontend/`)

Vanilla HTML/CSS/JS. Uses `marked.js` for markdown rendering. No build step.
