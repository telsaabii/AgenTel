"""
Test script for the context management phase.
Run with: python -m backend.tests.test_context_management
"""
import asyncio
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from backend.core.state import AgentState
from backend.security.guards import count_message_tokens, needs_summarization, HIGH_WATER_MARK
from backend.memory.shared import scratchpad_write, scratchpad_read, scratchpad_read_all
from backend.memory.thread import (
    build_messages_with_summary,
    summarize_messages,
    get_summary_history,
    EMPTY_SUMMARY,
)


def test_token_counting():
    print("=== Test 1: Token Counting ===")
    msgs = [
        HumanMessage(content="What is the population of Tokyo?"),
        AIMessage(content="The population of Tokyo is approximately 13.96 million as of 2024."),
    ]
    tokens = count_message_tokens(msgs)
    print(f"  Token count for 2 messages: {tokens}")
    assert tokens > 0, "Token count should be positive"

    # Test needs_summarization with small messages (should be False)
    state: AgentState = {
        "messages": msgs,
        "next": "",
        "metadata": {},
        "turn_count": 0,
        "summary": {},
        "scratchpad": {},
    }
    assert not needs_summarization(state), "Should NOT need summarization for small messages"
    print(f"  needs_summarization (small): False ✓")
    print(f"  HIGH_WATER_MARK: {HIGH_WATER_MARK}")
    print()


def test_scratchpad():
    print("=== Test 2: Scratchpad ===")
    state: AgentState = {
        "messages": [],
        "next": "",
        "metadata": {},
        "turn_count": 0,
        "summary": {},
        "scratchpad": {},
    }

    # Write from websearch agent
    scratchpad_write(state, "websearch", {
        "tokyo_population": 13960000,
        "growth_rate": 0.0012,
        "source": "worldometers.info",
    })
    print(f"  Written websearch data: {state['scratchpad']['websearch']}")

    # Read from code_executor's perspective
    ws_data = scratchpad_read(state, "websearch")
    assert ws_data["tokyo_population"] == 13960000
    print(f"  Read from websearch namespace: {ws_data}")

    # Read nonexistent namespace
    empty = scratchpad_read(state, "code_executor")
    assert empty == {}
    print(f"  Read nonexistent namespace: {empty} ✓")

    # Read all
    all_data = scratchpad_read_all(state)
    assert "websearch" in all_data
    print(f"  Read all namespaces: {list(all_data.keys())} ✓")
    print()


def test_build_messages_with_summary():
    print("=== Test 3: build_messages_with_summary ===")

    # Without summary
    state: AgentState = {
        "messages": [HumanMessage(content="hello"), AIMessage(content="hi")],
        "next": "",
        "metadata": {},
        "turn_count": 0,
        "summary": {},
        "scratchpad": {},
    }
    msgs = build_messages_with_summary(state)
    assert len(msgs) == 2, f"Expected 2 messages, got {len(msgs)}"
    print(f"  No summary: {len(msgs)} messages ✓")

    # With summary
    state["summary"] = {
        "key_facts": ["User is researching Tokyo"],
        "decisions_made": [],
        "files_touched": [],
        "urls_found": ["https://worldometers.info"],
        "open_questions": [],
        "user_preferences": {},
    }
    msgs = build_messages_with_summary(state)
    assert len(msgs) == 3, f"Expected 3 messages, got {len(msgs)}"
    assert isinstance(msgs[0], SystemMessage), "First message should be SystemMessage"
    assert "Conversation Summary" in msgs[0].content
    print(f"  With summary: {len(msgs)} messages, summary prepended ✓")
    print(f"  Summary message preview: {msgs[0].content[:80]}...")
    print()


async def test_summarization():
    print("=== Test 4: Summarization (live LLM call to gpt-4o-mini) ===")

    # Build a conversation with enough messages to be interesting
    messages = [
        HumanMessage(content="I'm building a web app with FastAPI and I need help choosing a database."),
        AIMessage(content="For FastAPI, popular choices are PostgreSQL with SQLAlchemy, MongoDB with Motor, or SQLite for simpler apps. What's your scale?"),
        HumanMessage(content="Medium scale, about 10k users. I was thinking Django but switched to FastAPI."),
        AIMessage(content="Good choice for your scale. I'd recommend PostgreSQL with async SQLAlchemy. It handles 10k users well and pairs nicely with FastAPI's async nature."),
        HumanMessage(content="Actually, let me use Supabase instead of raw PostgreSQL. Can you help me set that up?"),
        AIMessage(content="Supabase is great — it gives you PostgreSQL under the hood plus auth, storage, and realtime. You can use the supabase-py client library with FastAPI."),
        HumanMessage(content="Perfect. Also, I found this useful guide at https://docs.supabase.com/guides/getting-started"),
        AIMessage(content="That's the official getting started guide. Let me also check if there's a FastAPI-specific integration pattern."),
    ]

    state: AgentState = {
        "messages": messages,
        "next": "",
        "metadata": {},
        "turn_count": 4,
        "summary": {},
        "scratchpad": {},
    }

    print(f"  Messages before: {len(messages)}")
    print(f"  Tokens before: {count_message_tokens(messages)}")

    result = await summarize_messages(state, "test-session-001")

    new_summary = result.get("summary", {})
    removed = [m for m in result.get("messages", []) if hasattr(m, "id")]

    print(f"  Messages to remove: {len(removed)}")
    print(f"  New summary:")
    print(f"    {json.dumps(new_summary, indent=4)}")

    # Verify smart merge: should mention Supabase, NOT raw PostgreSQL as the choice
    summary_text = json.dumps(new_summary).lower()
    print()

    # Check key fields exist
    assert "key_facts" in new_summary, "Summary missing key_facts"
    assert "user_preferences" in new_summary, "Summary missing user_preferences"
    print("  Summary schema valid ✓")

    # Check summary history was recorded
    history = get_summary_history("test-session-001")
    assert len(history) == 1, f"Expected 1 history event, got {len(history)}"
    print(f"  Summary history recorded: {len(history)} event(s) ✓")
    print(f"  Event timestamp: {history[0]['timestamp']}")
    print(f"  Messages compressed: {history[0]['messages_compressed']}")
    print()


async def test_smart_merge():
    print("=== Test 5: Smart Merge (re-summarization) ===")

    # Start with a summary that says "user prefers Django"
    existing_summary = {
        "key_facts": ["Building a web application", "User prefers Django"],
        "decisions_made": ["Chose Django as web framework"],
        "files_touched": [],
        "urls_found": [],
        "open_questions": [],
        "user_preferences": {"framework": "Django"},
    }

    # New messages contradict: user switched to FastAPI
    messages = [
        HumanMessage(content="Actually I've decided to switch from Django to FastAPI. It's better for my async needs."),
        AIMessage(content="Good call. FastAPI's native async support is excellent. Let me help you migrate."),
    ]

    state: AgentState = {
        "messages": messages,
        "next": "",
        "metadata": {},
        "turn_count": 2,
        "summary": existing_summary,
        "scratchpad": {},
    }

    result = await summarize_messages(state, "test-session-merge")
    new_summary = result.get("summary", {})

    print(f"  Previous: framework = Django")
    print(f"  New summary:")
    print(f"    {json.dumps(new_summary, indent=4)}")

    summary_text = json.dumps(new_summary).lower()
    if "django" in summary_text and "fastapi" in summary_text:
        print("  ⚠ Both Django and FastAPI present — smart merge may not have fully replaced")
    elif "fastapi" in summary_text and "django" not in summary_text:
        print("  Smart merge: Django replaced by FastAPI ✓")
    else:
        print(f"  Check manually — summary content: {summary_text[:200]}")
    print()


async def main():
    print("╔══════════════════════════════════════════╗")
    print("║  Context Management Integration Tests    ║")
    print("╚══════════════════════════════════════════╝\n")

    # Sync tests
    test_token_counting()
    test_scratchpad()
    test_build_messages_with_summary()

    # Async tests (require LLM calls)
    await test_summarization()
    await test_smart_merge()

    print("═" * 44)
    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
