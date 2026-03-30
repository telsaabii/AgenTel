import json
from datetime import datetime, timezone
from langchain_core.messages import BaseMessage, SystemMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from backend.core.state import AgentState
from backend.core.llm import build_llm
from backend.security.guards import count_message_tokens, needs_summarization

# ── Summary JSON schema ──────────────────────────────────────────────
EMPTY_SUMMARY: dict = {
    "key_facts": [],
    "decisions_made": [],
    "files_touched": [],
    "urls_found": [],
    "open_questions": [],
    "user_preferences": {},
}

# ── Summary history store (observability, NOT on AgentState) ─────────
_summary_history: dict[str, list[dict]] = {}


def get_summary_history(session_id: str) -> list[dict]:
    """Return the summary event log for a session."""
    return _summary_history.get(session_id, [])


def _record_summary_event(
    session_id: str,
    previous_summary: dict,
    new_summary: dict,
    messages_compressed: int,
    tokens_before: int,
    tokens_after: int,
) -> None:
    if session_id not in _summary_history:
        _summary_history[session_id] = []
    _summary_history[session_id].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "previous_summary": previous_summary,
        "new_summary": new_summary,
        "messages_compressed": messages_compressed,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
    })


# ── Summarization prompt ─────────────────────────────────────────────
SUMMARIZE_SYSTEM_PROMPT = """\
You are a conversation summarizer. You will receive:
1. A previous summary (JSON) — may be empty on the first pass.
2. A list of conversation messages to compress.

Your job: produce an UPDATED summary JSON that merges the new information \
into the previous summary.

## Output schema (respond with ONLY this JSON, no extra text):
{{
    "key_facts": ["fact 1", "fact 2"],
    "decisions_made": ["decision 1"],
    "files_touched": ["/path/to/file"],
    "urls_found": ["https://example.com"],
    "open_questions": ["unanswered question"],
    "user_preferences": {{"key": "value"}}
}}

## Smart merge rules:
- If new information CONTRADICTS an existing fact, REPLACE the old fact. \
Do NOT keep both. Example: if the previous summary says "user prefers Django" \
but the new messages show they switched to FastAPI, only keep "user prefers FastAPI".
- Remove facts that are no longer relevant or have been superseded.
- Deduplicate entries — no repeated facts, URLs, or files.
- Keep entries concise — one line per fact, no lengthy explanations.
- user_preferences is a flat key-value dict. Overwrite keys on conflict.\
"""


def _format_messages_for_summary(messages: list[BaseMessage]) -> str:
    """Format messages into a readable text block for the summarizer."""
    lines = []
    for msg in messages:
        role = getattr(msg, "type", "unknown")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        # Truncate very long tool outputs to avoid blowing up the summarization context
        if len(content) > 2000:
            content = content[:2000] + "... [truncated]"
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)


async def summarize_messages(state: AgentState, session_id: str) -> dict:
    """
    Compress older messages into a structured JSON summary.

    Returns a state update dict with:
    - "summary": the new structured summary
    - "messages": RemoveMessage objects for each compressed message
    """
    messages = state["messages"]
    tokens_before = count_message_tokens(messages)

    # Find split point: keep the most recent ~30% of tokens, compress the rest
    running = 0
    split_idx = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        content = messages[i].content if isinstance(messages[i].content, str) else str(messages[i].content)
        running += len(content.split()) * 1.3  # rough token estimate for split finding
        if running > 20_000:  # keep ~20k tokens of recent messages
            split_idx = i
            break

    # Never compress the very first message (usually the user's initial request)
    split_idx = max(split_idx, 1)

    messages_to_compress = messages[:split_idx]
    if not messages_to_compress:
        return {}

    previous_summary = state.get("summary", {}) or EMPTY_SUMMARY.copy()

    # Build the summarization input
    llm = build_llm(provider="openai-mini")
    formatted_messages = _format_messages_for_summary(messages_to_compress)

    summary_input = [
        SystemMessage(content=SUMMARIZE_SYSTEM_PROMPT),
        SystemMessage(content=f"Previous summary:\n{json.dumps(previous_summary, indent=2)}"),
        SystemMessage(content=f"Messages to compress:\n{formatted_messages}"),
    ]

    response = await llm.ainvoke(summary_input)

    # Parse the JSON response
    try:
        new_summary = json.loads(response.content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response if it has extra text
        content = response.content
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            new_summary = json.loads(content[start:end])
        else:
            # Fallback: keep previous summary unchanged
            new_summary = previous_summary

    # Record the event for observability
    tokens_after = count_message_tokens(messages[split_idx:])
    _record_summary_event(
        session_id=session_id,
        previous_summary=previous_summary,
        new_summary=new_summary,
        messages_compressed=len(messages_to_compress),
        tokens_before=tokens_before,
        tokens_after=tokens_after,
    )

    # Build RemoveMessage objects for compressed messages
    remove_messages = [RemoveMessage(id=msg.id) for msg in messages_to_compress if msg.id]

    return {
        "summary": new_summary,
        "messages": remove_messages,
    }


def build_messages_with_summary(state: AgentState) -> list[BaseMessage]:
    """
    Construct the message list that agents/supervisor see.

    If a summary exists, prepend it as a SystemMessage.
    The summary SystemMessage is never stored in state — built on the fly.
    """
    summary = state.get("summary", {})
    messages = list(state["messages"])

    if summary:
        summary_msg = SystemMessage(
            content=(
                "## Conversation Summary (compressed from earlier messages)\n"
                f"```json\n{json.dumps(summary, indent=2)}\n```"
            )
        )
        return [summary_msg] + messages

    return messages


async def context_manager_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Graph node inserted between every agent and the supervisor.
    Checks if summarization is needed. If yes, runs it. If no, no-op.
    """
    if not needs_summarization(state):
        return {}

    session_id = config.get("configurable", {}).get("thread_id", "default")
    print(f"[context_manager] high-water mark reached, summarizing (session: {session_id})")

    return await summarize_messages(state, session_id)
