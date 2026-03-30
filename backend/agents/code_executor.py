import os
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from backend.core.state import AgentState
from backend.core.llm import build_llm
from backend.agents.filesystem import ALLOWED_DIRS
from backend.prompts import code_executor_prompt
from backend.memory.thread import build_messages_with_summary
from dotenv import load_dotenv
load_dotenv()


class CodeExecutionInput(BaseModel):
    """Input schema for the execute_python tool."""
    code: str = Field(description="Python code to execute in the sandbox")
    pip_packages: list[str] = Field(
        default=[],
        description="pip packages to install before execution",
    )


async def _execute_python(sandbox, code: str, pip_packages: list[str]) -> str:
    """Run Python code in an E2B sandbox. Returns formatted output."""
    # Install requested packages
    for pkg in pip_packages:
        await sandbox.commands.run(f"pip install {pkg}")

    execution = await sandbox.run_code(code)

    # Format results
    parts = []
    if execution.logs.stdout:
        stdout = "".join(execution.logs.stdout)
        if stdout.strip():
            parts.append(f"STDOUT:\n{stdout}")
    if execution.logs.stderr:
        stderr = "".join(execution.logs.stderr)
        if stderr.strip():
            parts.append(f"STDERR:\n{stderr}")
    if execution.error:
        parts.append(
            f"ERROR:\n{execution.error.name}: {execution.error.value}\n"
            f"{execution.error.traceback}"
        )
    if execution.results:
        for result in execution.results:
            if result.text:
                parts.append(f"RESULT:\n{result.text}")

    return "\n\n".join(parts) if parts else "Code executed successfully (no output)."


def _build_gated_tool(sandbox):
    """Build a StructuredTool that gates execution with interrupt()."""

    async def _gated_execute(code: str, pip_packages: list[str] = []) -> str:
        response = interrupt({
            "agent": "code_executor",
            "action": "execute_python",
            "language": "python",
            "code": code,
            "pip_packages": pip_packages,
            "message": "Code executor wants to run the above code. Approve?",
        })

        if isinstance(response, str) and response.lower() in ("yes", "y", "approve"):
            return await _execute_python(sandbox, code, pip_packages)
        if isinstance(response, dict) and response.get("action") == "approve":
            return await _execute_python(sandbox, code, pip_packages)

        return f"User denied execution. User said: {response}"

    return StructuredTool.from_function(
        coroutine=_gated_execute,
        name="execute_python",
        description=(
            "Execute Python code in an isolated E2B sandbox. "
            "Can install pip packages. Every execution requires user approval."
        ),
        args_schema=CodeExecutionInput,
    )


class SaveSandboxFileInput(BaseModel):
    """Input schema for the save_sandbox_file_locally tool."""
    sandbox_path: str = Field(description="Path to the file inside the E2B sandbox (e.g. /tmp/report.pdf)")
    local_path: str = Field(description="Absolute local filesystem path to save the file to")


def _build_save_file_tool(sandbox):
    """Build a gated tool that downloads a file from E2B and writes it locally."""

    async def _save_file(sandbox_path: str, local_path: str) -> str:
        # Validate against allowed directories
        if not any(local_path.startswith(d) for d in ALLOWED_DIRS):
            return (
                f"Denied: {local_path!r} is not under any allowed directory. "
                f"Allowed: {ALLOWED_DIRS}"
            )

        response = interrupt({
            "agent": "code_executor",
            "action": "save_sandbox_file_locally",
            "sandbox_path": sandbox_path,
            "local_path": local_path,
            "message": (
                f"Code executor wants to save sandbox file {sandbox_path!r} "
                f"to local path {local_path!r}. Approve?"
            ),
        })

        if isinstance(response, str) and response.lower() in ("yes", "y", "approve"):
            pass  # approved, continue
        elif isinstance(response, dict) and response.get("action") == "approve":
            pass  # approved, continue
        else:
            return f"User denied file save. User said: {response}"

        # Download from sandbox and write locally
        content = await sandbox.files.read(sandbox_path, format="bytes")
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

        size = dest.stat().st_size
        return f"Saved {sandbox_path} → {local_path} ({size:,} bytes)"

    return StructuredTool.from_function(
        coroutine=_save_file,
        name="save_sandbox_file_locally",
        description=(
            "Download a file from the E2B sandbox and save it to the local filesystem. "
            "The local path must be under an allowed directory. Requires user approval."
        ),
        args_schema=SaveSandboxFileInput,
    )


async def code_executor_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    LangGraph node — runs the code executor ReAct agent.

    All code execution is gated with interrupt() so the user must approve
    before any code runs in the sandbox.
    """
    sandbox = config["configurable"].get("e2b_sandbox")
    if not sandbox:
        raise RuntimeError(
            "No E2B sandbox found in config. "
            "Did you forget to pass e2b_sandbox at graph invocation?"
        )

    execute_tool = _build_gated_tool(sandbox)
    save_tool = _build_save_file_tool(sandbox)

    system_prompt: str = code_executor_prompt.messages[0].prompt.template
    agent = create_agent(
        model=build_llm(),
        tools=[execute_tool, save_tool],
        system_prompt=system_prompt,
    )

    messages_for_agent = build_messages_with_summary(state)
    agent_result = await agent.ainvoke({"messages": messages_for_agent})

    new_messages = agent_result["messages"][len(messages_for_agent):]
    print(f"[code_executor] produced {len(new_messages)} new message(s)")

    return {"messages": new_messages}
