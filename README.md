# AgenTel

A personal multi-agent AI system built with LangGraph, MCP, and FastAPI. A supervisor agent routes user requests to specialized sub-agents, each scoped to its own tools via the Model Context Protocol.

## Architecture

```
Supervisor (entry)
    |-> websearch -> Supervisor
    |-> filesystem -> Supervisor
    |-> code_executor -> Supervisor
    |-> rag -> Supervisor
    |-> FINISH (end)
```

The supervisor uses structured output (Pydantic `RouteDecision`) to decide which agent handles each request. Loop prevention via turn counting and subtask completion checks.

### Tech Stack

- **Orchestration:** LangGraph (StateGraph with supervisor routing)
- **Tool Integration:** Model Context Protocol (MCP) servers
- **LLM:** Provider-agnostic via `build_llm()` — OpenAI gpt-4o default for testing, will integrate open source models with ollama once the project is complete for more privacy
- **Backend API:** FastAPI + WebSocket streaming
- **Frontend:** Next.js (planned)
- **Database:** SQLite (knowledge graph persistence)
- **Code Execution:** E2B sandboxed environments

### Project Structure

```
backend/
├── agents/           # One module per agent
│   ├── websearch.py      # Brave Search + Playwright MCP
│   ├── filesystem.py     # Filesystem MCP (path-allowlisted)
│   └── code_executor.py  # E2B sandboxed execution
├── core/             # Orchestration layer
│   ├── graph.py          # LangGraph topology + conditional routing
│   ├── supervisor.py     # Supervisor agent + RouteDecision schema
│   ├── state.py          # AgentState TypedDict
│   └── llm.py            # Provider-agnostic LLM factory
├── memory/           # Three-tier memory system (planned)
├── security/
│   └── guards.py         # Turn limits, token budget, path allowlist
├── api/              # FastAPI layer (planned)
├── prompts.py
└── main.py
```

## Agents

| Agent | Tools | Status |
|---|---|---|
| **websearch** | Brave Search MCP, Playwright MCP | Implemented |
| **filesystem** | Filesystem MCP (path-allowlisted) | Planned |
| **code_executor** | E2B sandboxed execution | Planned |
| **rag** | Vector embeddings, document retrieval | Planned |

Each agent is isolated — it can only access its own assigned MCP tools.

## Three-Tier Memory System

| Tier | Scope | Lifespan |
|---|---|---|
| **Thread Memory** | Sliding window + LLM summarization of older messages | Single session |
| **Shared State** | Scratchpad dict for inter-agent data passing | Single graph execution |
| **Knowledge Graph** | SQLite entity-relationship store with LLM extraction and importance scoring | Persistent |

## Security

- **Agent isolation:** Each agent scoped to its own MCP tools
- **Human-in-the-loop:** LangGraph `interrupt()` gates on file writes, deletes, and code execution
- **Guardrails:** Turn count caps, per-session token budgets, filesystem path allowlist, E2B sandboxing
- **Secrets:** API keys loaded from env vars, never passed through message history

## Setup

```bash
# Clone
git clone https://github.com/telsaabii/AgenTel.git
cd AgenTel

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your API keys: OPENAI_API_KEY, BRAVE_API_KEY, E2B_API_KEY

# Run
python -m backend.main
```

## Roadmap

- [ ] **Filesystem agent** — MCP filesystem server with path allowlist guard
- [ ] **Code executor agent** — E2B sandboxed execution with human-in-the-loop approval
- [ ] **Context management** — Thread summarization, shared state scratchpad, turn/token limits
- [ ] **Knowledge graph** — SQLite persistence, LLM entity extraction with scoring, consolidation
- [ ] **RAG agent** — Document ingestion, vector embeddings, semantic retrieval
- [ ] **API layer** — FastAPI + WebSocket endpoints for streaming
- [ ] **Frontend** — Next.js app with chat, agent activity view, conversation history, and memory viewer

## License

MIT
