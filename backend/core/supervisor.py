from backend.core.state import AgentState
from backend.security.guards import check_turn_limit
from backend.memory.thread import build_messages_with_summary
from backend.memory.shared import scratchpad_read_all
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from backend.core.llm import build_llm
from backend.prompts import supervisor_prompt
import json

AGENTS = Literal[
    "websearch",
    "filesystem",
    "code_executor",
    # "memory",
    # "imagegen",
    "FINISH",
]

AGENTS_DESCRIPTION = """\
- **websearch** — searches the web (Brave Search) and can browse pages \
(Playwright). Use whenever the answer requires up-to-date information, \
URLs, news, or live data.
- **filesystem** — reads and writes local files via the filesystem MCP. \
Can also parse and extract text from PDF files. \
Use for any task involving reading, creating, editing, moving, or searching \
local files and directories, including PDF content extraction.
- **code_executor** — writes and runs Python code safely in an isolated \
E2B sandbox. Can also save generated files (PDFs, images, etc.) from the \
sandbox to the local filesystem. Use for computation, data analysis, \
plotting, file generation, or any task requiring code execution.\
"""


class RouteDecision(BaseModel):
    """Structured output schema for the supervisor's routing decision."""

    next: AGENTS = Field(
        description=(
            "The name of the agent to invoke next, or 'FINISH' if the "
            "task is fully complete."
        )
    )
    reasoning: str = Field(
        description="One-sentence explanation of why this agent was chosen.",
        default="",
    )



llm = build_llm()
structured_llm = llm.with_structured_output(RouteDecision)
supervisor_chain = supervisor_prompt.partial(agents=AGENTS_DESCRIPTION) | structured_llm

async def supervisor_node(state: AgentState) -> dict:
    """
    Inspect the current state and decide which agent should run next.

    Returns a state update dict containing `next` and incremented `turn_count`.
    The graph's conditional edge reads `state["next"]` to route.
    """
    new_turn_count = state.get("turn_count", 0) + 1

    try:
        check_turn_limit(state)
    except RuntimeError as e:
        print(f"[supervisor] turn limit reached: {e}")
        return {
            "next": "FINISH",
            "turn_count": new_turn_count,
            "messages": [SystemMessage(content=f"Turn limit reached ({new_turn_count}). Finishing.")],
        }

    messages = build_messages_with_summary(state)

    # Inject scratchpad context so supervisor can see inter-agent data
    scratchpad = scratchpad_read_all(state)
    if scratchpad:
        scratchpad_msg = SystemMessage(
            content=f"## Agent Scratchpad (structured data from agents)\n```json\n{json.dumps(scratchpad, indent=2)}\n```"
        )
        messages = [messages[0], scratchpad_msg] + messages[1:] if messages else [scratchpad_msg]

    decision: RouteDecision = await supervisor_chain.ainvoke({"messages": messages})

    print(f"[supervisor] → {decision.next}  ({decision.reasoning})")

    return {"next": decision.next, "turn_count": new_turn_count}
