from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from backend.core.state import AgentState
from backend.core.supervisor import supervisor_node
from backend.agents.websearch import websearch_node
from backend.agents.filesystem import filesystem_node
from backend.agents.code_executor import code_executor_node
from backend.memory.thread import context_manager_node

def route_next(state: AgentState) -> str:
    """
    Reads the routing decision the supervisor wrote into state["next"] and
    returns the name of the next node (or END sentinel).
    """
    destination = state.get("next", "FINISH")
    print(f"[graph] routing → {destination}")
    return destination

def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("websearch", websearch_node)
    builder.add_node("filesystem", filesystem_node)
    builder.add_node("code_executor", code_executor_node)
    builder.add_node("context_manager", context_manager_node)
    builder.set_entry_point("supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "websearch": "websearch",
            "filesystem": "filesystem",
            "code_executor": "code_executor",
            # "memory":        "memory",
            # "imagegen":      "imagegen",
            "FINISH": END,
        },
    )

    builder.add_edge("websearch", "context_manager")
    builder.add_edge("filesystem", "context_manager")
    builder.add_edge("code_executor", "context_manager")
    builder.add_edge("context_manager", "supervisor")

    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()
