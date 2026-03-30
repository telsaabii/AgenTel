from fastapi import APIRouter, HTTPException
from backend.core.graph import graph
from backend.memory.thread import summarize_messages, get_summary_history
from backend.security.guards import count_message_tokens, HIGH_WATER_MARK

router = APIRouter(prefix="/sessions", tags=["context"])


@router.get("/{session_id}/context")
async def get_context_stats(session_id: str):
    """Return current token usage and context window stats."""
    config = {"configurable": {"thread_id": session_id}}
    state = graph.get_state(config)

    if not state.values:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    messages = state.values.get("messages", [])
    summary = state.values.get("summary", {})
    scratchpad = state.values.get("scratchpad", {})

    message_tokens = count_message_tokens(messages)

    return {
        "session_id": session_id,
        "message_tokens": message_tokens,
        "high_water_mark": HIGH_WATER_MARK,
        "usage_percent": round((message_tokens / HIGH_WATER_MARK) * 100, 1),
        "message_count": len(messages),
        "has_summary": bool(summary),
        "summary": summary,
        "scratchpad_namespaces": list(scratchpad.keys()),
    }


@router.post("/{session_id}/compact")
async def compact_session(session_id: str):
    """Manually trigger summarization regardless of high-water mark."""
    config = {"configurable": {"thread_id": session_id}}
    state = graph.get_state(config)

    if not state.values:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    messages = state.values.get("messages", [])
    if len(messages) < 2:
        raise HTTPException(status_code=400, detail="Not enough messages to summarize")

    state_update = await summarize_messages(state.values, session_id)

    if state_update:
        await graph.aupdate_state(config, state_update)

    return {
        "session_id": session_id,
        "summary": state_update.get("summary", {}),
        "messages_removed": len([m for m in state_update.get("messages", [])]),
    }


@router.get("/{session_id}/summary-history")
async def get_session_summary_history(session_id: str):
    """Return the summary change log for observability."""
    history = get_summary_history(session_id)
    return {
        "session_id": session_id,
        "events": history,
        "total_compressions": len(history),
    }
