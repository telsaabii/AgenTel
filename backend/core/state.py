from typing import Annotated,Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

#typeddict best used for tracking agent state
#basemodel is used for strict data validation
class AgentState(TypedDict):
    """
    Docstring for AgentState:
    Shared state that flows through every node in the graph.
    Fields:
    -  messages : Full conversation history.  The `add_messages` reducer appends
                new messages rather than overwriting the list, so every node
                simply returns the messages it wants to add.

    -  next : Routing signal written by the supervisor.  One of the registered
            agent names (e.g. "websearch") or the sentinel "FINISH".

    - metadata : Arbitrary key-value bag for thread/run context.  Used to pass
              LangSmith trace metadata, session IDs, and any future per-run
              config without polluting the message list.

    """
    messages : Annotated[list[BaseMessage],add_messages]
    next : str
    metadata : dict[str,Any]
    turn_count : int
