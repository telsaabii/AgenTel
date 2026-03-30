import os
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from backend.core.state import AgentState
from backend.core.llm import build_llm
from backend.prompts import websearch_prompt
from backend.memory.thread import build_messages_with_summary
from dotenv import load_dotenv
load_dotenv()

MCP_CONFIG: dict = {
    "brave-search": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-brave-search"],
        # The Brave API key must be set in your shell environment.
        "env": {"BRAVE_API_KEY": os.environ.get("BRAVE_API_KEY", "")},
        "transport": "stdio",
    },
    "playwright": {
        "command": "npx",
        "args": ["@playwright/mcp@latest", "--headless"],   # drop --headless for debugging
        "transport": "stdio",
    },
}

async def websearch_node(state: AgentState,config: RunnableConfig) -> dict:
    """
    LangGraph node — runs the Brave + Playwright ReAct agent.

    Accepts the full AgentState, returns a dict with only the *new* messages
    so the `add_messages` reducer appends (not overwrites) them.
    """
    tools = config["configurable"].get("tools_websearch")
    if not tools:
        raise RuntimeError(
            "No tools found in config. Did you forget to pass them at graph invocation? "
            "See main.py for the expected startup pattern."
        )


    system_prompt : str = websearch_prompt.messages[0].prompt.template
    agent = create_agent(
            model=build_llm(),
            tools=tools,
            # state_modifier injects the system prompt before the messages list
            system_prompt=system_prompt,
        )

    # Pass summary-aware messages so the agent has compressed context
    messages_for_agent = build_messages_with_summary(state)
    agent_result = await agent.ainvoke({"messages": messages_for_agent})

    # Slice off only the messages the agent produced in this turn
    new_messages = agent_result["messages"][len(messages_for_agent):]

    print(f"[websearch] produced {len(new_messages)} new message(s)")

    return {"messages": new_messages}
