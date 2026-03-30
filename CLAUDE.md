# AgenTel

Personal multi-agent AI system built with LangGraph, MCP, and FastAPI.

## Architecture

### Tech Stack
- **Orchestration:** LangGraph (StateGraph with supervisor routing)
- **Tool integration:** Model Context Protocol (MCP) servers
- **LLM:** Provider-agnostic via `build_llm()` — OpenAI gpt-4o default, Ollama fallback
- **Backend API:** FastAPI + WebSocket for streaming
- **Frontend:** Next.js
- **Database:** SQLite (knowledge graph persistence)
- **Code execution:** E2B sandboxed environments

### Project Structure
```
agentel/
├── backend/
│   ├── agents/          # One module per agent
│   │   ├── websearch.py
│   │   ├── code_executor.py
│   │   ├── filesystem.py
│   │   └── rag.py
│   ├── core/            # Orchestration layer
│   │   ├── graph.py     # LangGraph topology and conditional routing
│   │   ├── supervisor.py # Supervisor agent + RouteDecision schema
│   │   ├── state.py     # AgentState TypedDict
│   │   └── llm.py       # Provider-agnostic LLM factory
│   ├── memory/          # Three-tier memory system
│   │   ├── thread.py    # Sliding window + LLM summarization
│   │   ├── shared.py    # Scratchpad dict for inter-agent data passing
│   │   └── knowledge.py # SQLite entity-relationship store + LLM extraction
│   ├── security/        # Cross-cutting guards
│   │   └── guards.py    # Turn limits, token budget, path allowlist
│   ├── api/             # API layer
│   │   ├── server.py    # FastAPI app entry
│   │   ├── routes.py    # REST endpoints (history, memory CRUD, sessions)
│   │   └── ws.py        # WebSocket handler for streaming
│   ├── prompts.py
│   └── main.py
├── frontend/            # Next.js app
├── CLAUDE.md
├── .env
└── .gitignore
```

### Agent Roster

| Agent | Tools | Status |
|---|---|---|
| **websearch** | Brave Search MCP, Playwright MCP | Implemented |
| **filesystem** | Filesystem MCP (path-allowlisted) | Planned |
| **code_executor** | E2B sandboxed execution | Planned |
| **rag** | Vector embeddings, document ingestion, retrieval | Planned |

Agents are isolated: each agent can ONLY access its own assigned tools.

### Graph Topology
```
Supervisor (entry)
    ├→ websearch → Supervisor
    ├→ filesystem → Supervisor
    ├→ code_executor → Supervisor
    ├→ rag → Supervisor
    └→ FINISH (end)
```
Supervisor uses structured output (RouteDecision pydantic model) to route. Loop prevention via turn_count and checking if agent already completed its subtask.

## Three-Tier Memory System

### Tier 1: Thread Memory
- Sliding window of recent messages kept in full
- Older messages compressed via LLM-generated summary
- Lifespan: single session

### Tier 2: Shared State
- Scratchpad `dict` on `AgentState` for structured inter-agent data passing
- Example: websearch finds a URL, code_executor uses it — without polluting message history
- Lifespan: single graph execution

### Tier 3: Knowledge Graph
- SQLite with entity-relationship schema
- After each conversation turn, a lightweight LLM call extracts entities, relationships, facts with importance score (1-5)
- Only persists above confidence threshold
- Periodic consolidation pass merges duplicates and decays stale facts
- Lifespan: persistent across all sessions

## Security Model

### Agent Isolation
Each agent is scoped to its own MCP tools. Websearch cannot write files. Filesystem cannot execute code.

### Human-in-the-Loop
Uses LangGraph `interrupt()` for approval gates:
- **Mandatory approval:** File writes, file deletes, code execution
- **Auto-execute:** Web search, file reads, RAG ingestion (but notify user what was ingested)
- **Auto-execute + observable:** Memory persistence (surface what was stored, user can correct/delete)

### Guardrails
- `turn_count` hard cap prevents infinite agent loops
- Per-session token budget triggers graceful shutdown
- Filesystem agent restricted to explicit path allowlist
- Code execution sandboxed via E2B (ephemeral containers)
- API keys never passed through message history — env var isolation only

## Frontend (Next.js)

Four views, single-page app with sidebar navigation:
1. **Chat** — message thread with streaming, approval modals for gated actions
2. **Agent Activity** — live indicator of active agent, tool calls, routing breadcrumbs
3. **Conversation History** — sidebar with past sessions, click to reload
4. **Memory Viewer** — browse, search, delete knowledge graph entries

## Build Phases

1. **Restructure** — migrate flat files into backend/ package structure, verify everything runs
2. **Filesystem agent** — MCP filesystem server + path allowlist guard
3. **Code executor** — E2B integration + human-in-the-loop interrupt
4. **Context management** — thread summarization, shared state scratchpad, turn/token limits
5. **Knowledge graph** — SQLite persistence, LLM extraction with scoring, consolidation
6. **RAG agent** — document ingestion, vector embeddings, retrieval
7. **API layer** — FastAPI + WebSocket, REST endpoints
8. **Frontend** — Next.js chat, agent activity, history, memory viewer

## Conventions

- Async-first: all agent nodes and tool calls are async
- Type-safe: TypedDict for state, Pydantic for structured LLM output, Literal types for routing
- MCP for tool integration: new tools = new MCP server config, not custom code
- Provider-agnostic LLM layer: swap models by changing `build_llm()` provider arg
- No over-engineering: build what's needed now, design interfaces so you can swap implementations later
