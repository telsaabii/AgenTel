import asyncio
import uuid
from dotenv import load_dotenv
load_dotenv()

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command
from e2b_code_interpreter import AsyncSandbox

from backend.agents.websearch import MCP_CONFIG as WEBSEARCH_MCP
from backend.agents.filesystem import MCP_CONFIG as FILESYSTEM_MCP
from backend.core.graph import graph


async def main():
    # Per-agent MCP clients
    ws_client = MultiServerMCPClient(WEBSEARCH_MCP)
    fs_client = MultiServerMCPClient(FILESYSTEM_MCP)

    ws_tools = await ws_client.get_tools()
    fs_tools = await fs_client.get_tools()
    print("[debug] websearch tools:", [t.name for t in ws_tools])
    print("[debug] filesystem tools:", [t.name for t in fs_tools])

    # E2B sandbox — persists for the session
    sandbox = await AsyncSandbox.create()
    print(f"[debug] E2B sandbox created: {sandbox.sandbox_id}")

    try:
        config = {
            "configurable": {
                "tools_websearch": ws_tools,
                "tools_filesystem": fs_tools,
                "e2b_sandbox": sandbox,
                "thread_id": str(uuid.uuid4()),
            }
        }

        initial_state = {
            "messages": [{"role": "user", "content": "Can you write a one page report on screen addiction and then save it as a pdf into FileSys"}],
            "next": "",
            "metadata": {"session_id": "abc123"},
            "turn_count": 0,
        }

        # Invoke graph — may return early due to interrupt()
        result = await graph.ainvoke(initial_state, config=config)

        # Interactive interrupt handling loop
        while True:
            state = graph.get_state(config)

            # Check for pending interrupts
            if not state.tasks or not any(
                hasattr(t, "interrupts") and t.interrupts for t in state.tasks
            ):
                break

            # Display interrupt info
            for task in state.tasks:
                if hasattr(task, "interrupts"):
                    for intr in task.interrupts:
                        print("\n--- APPROVAL REQUIRED ---")
                        payload = intr.value
                        if isinstance(payload, dict):
                            if "code" in payload:
                                print(f"Agent: {payload.get('agent', '?')}")
                                print(f"Language: {payload.get('language', '?')}")
                                print(f"Code:\n{payload['code']}")
                                if payload.get("pip_packages"):
                                    print(f"Packages: {payload['pip_packages']}")
                            else:
                                print(f"Agent: {payload.get('agent', '?')}")
                                print(f"Action: {payload.get('action', '?')}")
                                print(f"Args: {payload.get('args', {})}")
                        else:
                            print(payload)
                        print("-------------------------")

            user_input = input("Your response (yes/no/alternative): ").strip()
            result = await graph.ainvoke(
                Command(resume=user_input), config=config
            )

        print("\n[final answer]")
        print(result["messages"][-1].content)

    finally:
        await sandbox.kill()
        print("[debug] E2B sandbox killed")


if __name__ == "__main__":
    asyncio.run(main())
