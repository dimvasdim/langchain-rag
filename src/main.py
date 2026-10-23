"""Setup check: prints library versions and makes one call to the configured model."""

from importlib.metadata import version
from config import PROVIDER, get_llm


def main() -> None:
    print(f"langchain-core version: {version('langchain-core')}")
    print(f"langgraph version: {version('langgraph')}")
    print(f"Provider: {PROVIDER}")

    llm = get_llm(temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response: {response.content}")


if __name__ == "__main__":
    main()
