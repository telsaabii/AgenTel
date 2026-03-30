import os
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel, Field
from backend.core.state import AgentState
from backend.core.llm import build_llm
from backend.prompts import filesystem_prompt
from backend.memory.thread import build_messages_with_summary
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Path allowlist from env
# ---------------------------------------------------------------------------
_raw = os.environ.get("ALLOWED_PATHS", "")
ALLOWED_DIRS: list[str] = [d.strip() for d in _raw.split(",") if d.strip()]
if not ALLOWED_DIRS:
    raise ValueError(
        "ALLOWED_PATHS env var is empty or missing. "
        "Set it to a comma-separated list of absolute directory paths."
    )

# ---------------------------------------------------------------------------
# MCP server configuration
# ---------------------------------------------------------------------------
MCP_CONFIG: dict = {
    "filesystem": {
        "command": "npx",
        "args": [
            "-y",
            "@modelcontextprotocol/server-filesystem",
            *ALLOWED_DIRS,
        ],
        "transport": "stdio",
    },
}

# ---------------------------------------------------------------------------
# Dangerous operations that require user approval
# ---------------------------------------------------------------------------
DANGEROUS_OPS = {"write_file", "create_directory", "move_file", "edit_file"}


def _gate_tool(tool):
    """Wrap a dangerous MCP tool so it calls interrupt() before executing."""

    async def _gated(**kwargs):
        response = interrupt({
            "agent": "filesystem",
            "action": tool.name,
            "args": kwargs,
            "message": f"Filesystem agent wants to {tool.name}. Approve?",
        })

        if isinstance(response, str) and response.lower() in ("yes", "y", "approve"):
            return await tool.ainvoke(kwargs)
        if isinstance(response, dict) and response.get("action") == "approve":
            return await tool.ainvoke(kwargs)

        return f"User denied this operation. User said: {response}"

    return StructuredTool.from_function(
        coroutine=_gated,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def _prepare_tools(raw_tools: list) -> list:
    """Gate dangerous tools, pass through safe ones unchanged."""
    result = []
    for t in raw_tools:
        if t.name in DANGEROUS_OPS:
            result.append(_gate_tool(t))
        else:
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# PDF parsing tool (uses E2B sandbox for extraction)
# ---------------------------------------------------------------------------
class ParsePdfInput(BaseModel):
    """Input schema for the parse_pdf tool."""
    file_path: str = Field(description="Absolute path to a local PDF file to parse and extract text from")


def _validate_path(file_path: str) -> None:
    """Raise ValueError if file_path is not under an allowed directory."""
    if not any(file_path.startswith(d) for d in ALLOWED_DIRS):
        raise ValueError(
            f"Path {file_path!r} is not under any allowed directory: {ALLOWED_DIRS}"
        )


def _build_parse_pdf_tool(sandbox):
    """Build a tool that reads a local PDF and extracts text via E2B + pdfplumber."""

    async def _parse_pdf(file_path: str) -> str:
        _validate_path(file_path)

        local_path = Path(file_path)
        if not local_path.exists():
            return f"File not found: {file_path}"
        if not local_path.suffix.lower() == ".pdf":
            return f"Not a PDF file: {file_path}"

        file_bytes = local_path.read_bytes()
        remote_path = "/tmp/parse_target.pdf"

        # Upload to sandbox
        await sandbox.files.write(remote_path, file_bytes)

        # Install pdfplumber and extract text
        await sandbox.commands.run("pip install pdfplumber", timeout=60)
        execution = await sandbox.run_code(
            'import pdfplumber\n'
            'with pdfplumber.open("/tmp/parse_target.pdf") as pdf:\n'
            '    text = ""\n'
            '    for i, page in enumerate(pdf.pages):\n'
            '        text += f"--- Page {i+1} ---\\n"\n'
            '        text += (page.extract_text() or "[no text found]") + "\\n"\n'
            'print(text)\n'
        )

        # Extract stdout
        if execution.error:
            return (
                f"PDF parsing failed: {execution.error.name}: {execution.error.value}\n"
                f"{execution.error.traceback}"
            )

        stdout = "".join(execution.logs.stdout) if execution.logs.stdout else ""
        return stdout.strip() if stdout.strip() else "PDF parsed but no text content found."

    return StructuredTool.from_function(
        coroutine=_parse_pdf,
        name="parse_pdf",
        description=(
            "Parse a local PDF file and extract its text content. "
            "Handles multi-page PDFs. Auto-approved (read-only operation)."
        ),
        args_schema=ParsePdfInput,
    )


async def filesystem_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    LangGraph node — runs the filesystem ReAct agent.

    Dangerous operations (write, delete, move) are gated with interrupt()
    so the user must approve before they execute.
    """
    raw_tools = config["configurable"].get("tools_filesystem")
    if not raw_tools:
        raise RuntimeError(
            "No filesystem tools found in config. "
            "Did you forget to pass tools_filesystem at graph invocation?"
        )

    tools = _prepare_tools(raw_tools)

    # Add PDF parsing tool if E2B sandbox is available
    sandbox = config["configurable"].get("e2b_sandbox")
    if sandbox:
        tools.append(_build_parse_pdf_tool(sandbox))

    system_prompt: str = filesystem_prompt.messages[0].prompt.template
    agent = create_agent(
        model=build_llm(),
        tools=tools,
        system_prompt=system_prompt,
    )

    messages_for_agent = build_messages_with_summary(state)
    agent_result = await agent.ainvoke({"messages": messages_for_agent})

    new_messages = agent_result["messages"][len(messages_for_agent):]
    print(f"[filesystem] produced {len(new_messages)} new message(s)")

    return {"messages": new_messages}
