from backend.core.state import AgentState

MAX_TURNS = 15


def check_turn_limit(state: AgentState) -> None:
    """Raise RuntimeError if the supervisor has exceeded the turn limit."""
    if state.get("turn_count", 0) >= MAX_TURNS:
        raise RuntimeError(f"Turn limit ({MAX_TURNS}) exceeded")
