from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
load_dotenv()

def build_llm(provider: str = "openai"):
    """
    Returns the appropriate chat model

    Args:
    provider: "openai" or "ollama"
    """

    if provider == "openai":
        return ChatOpenAI(
            model = "gpt-4o",
            temperature=0,#DETERMINISTIC MODEL -> better for strict classification
            api_key = os.environ["OPENAI_API_KEY"],
        )

    elif provider == "ollama":
        return ChatOllama(
            model = "qwen2:7b",
            temperature=0,
        )

    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose 'openai' or 'ollama'.")
