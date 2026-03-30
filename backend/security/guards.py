import tiktoken
from langchain_core.messages import BaseMessage
from backend.core.state import AgentState

MAX_TURNS = 15
HIGH_WATER_MARK = 60_000  # tokens

_encoder = tiktoken.encoding_for_model("gpt-4o")


def count_message_tokens(messages: list[BaseMessage]) -> int:
    """Count total tokens across all messages using tiktoken."""
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total += len(_encoder.encode(content))
        total += 4  # per-message overhead (role, separators)
    return total


def needs_summarization(state: AgentState) -> bool:
    """Return True if message tokens exceed the high-water mark."""
    return count_message_tokens(state["messages"]) >= HIGH_WATER_MARK


def check_turn_limit(state: AgentState) -> None:
    """Raise RuntimeError if the supervisor has exceeded the turn limit."""
    if state.get("turn_count", 0) >= MAX_TURNS:
        raise RuntimeError(f"Turn limit ({MAX_TURNS}) exceeded")
