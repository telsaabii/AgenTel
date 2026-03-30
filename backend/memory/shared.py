from backend.core.state import AgentState


def scratchpad_write(state: AgentState, agent_name: str, data: dict) -> None:
    """Write structured data to an agent's scratchpad namespace."""
    if "scratchpad" not in state or state["scratchpad"] is None:
        state["scratchpad"] = {}
    state["scratchpad"][agent_name] = data


def scratchpad_read(state: AgentState, agent_name: str) -> dict:
    """Read another agent's scratchpad namespace."""
    return state.get("scratchpad", {}).get(agent_name, {})


def scratchpad_read_all(state: AgentState) -> dict:
    """Read all scratchpad namespaces."""
    return state.get("scratchpad", {})
