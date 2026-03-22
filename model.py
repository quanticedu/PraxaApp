from langchain_community.chat_models import ChatOpenAI
from typing import Optional, Any
import os

DEFAULT_MODEL_CANDIDATES = [
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-3-27b-it:free",
    "openai/gpt-4o-mini",
]

class ChatModel(ChatOpenAI):
    """
    Creates a chat model from openrouter.ai using the OpenAI API
    """
    def __init__(
            self,
            model_name: str,
            openai_api_key: Optional[str] = None,
            openai_api_base: str="https://openrouter.ai/api/v1",
            **kwargs: Any):
        openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY')
        if not openai_api_key:
            raise RuntimeError(
                "Missing OPENROUTER_API_KEY. Set it in your shell before running model.py"
            )
        super().__init__(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            model_name=model_name,
            **kwargs
        )

def get_model(model_name: str = DEFAULT_MODEL_CANDIDATES[0]) -> ChatModel:
    """
    Gets a reference to a model
    
    :param model_name: Name of the model
    :type model_name: str
    :return: the model
    :rtype: ChatModel
    """
    return ChatModel(
        model_name=model_name,
        max_tokens=512,
        temperature=0
    )


def invoke_with_model_fallback(messages: list[Any], model_candidates: Optional[list[str]] = None) -> tuple[str, Any]:
    """
    Invoke the chat model with fallback candidates when endpoints are unavailable or rate-limited.

    :param messages: Chat messages to send
    :type messages: list[Any]
    :param model_candidates: Ordered model names to try
    :type model_candidates: Optional[list[str]]
    :return: Tuple of (model_name_used, response)
    :rtype: tuple[str, Any]
    """
    candidates = model_candidates or DEFAULT_MODEL_CANDIDATES
    errors: list[str] = []

    for model_name in candidates:
        try:
            model = get_model(model_name)
            response = model.invoke(messages)
            return model_name, response
        except Exception as exc:
            exc_text = str(exc)
            if "401" in exc_text or "User not found" in exc_text:
                raise RuntimeError(
                    "OpenRouter authentication failed (401). Verify OPENROUTER_API_KEY is valid and belongs to your OpenRouter account."
                ) from exc
            errors.append(f"{model_name}: {exc}")
            continue

    joined_errors = "\n".join(errors)
    raise RuntimeError(f"All model candidates failed.\n{joined_errors}")

if __name__ == "__main__":
# when run as a script, run some tests to demonstrate capabilities
    from langchain_core.messages import HumanMessage
    from langchain.prompts import ChatPromptTemplate

    prompt_template = ChatPromptTemplate([
        ("human", "You are a helpful assistant."),
        ("human", "What is {playwright}'s most recent play?")
    ])

    model_used, response = invoke_with_model_fallback(
        [HumanMessage("You are a helpful assistant."),
        HumanMessage("What are some plays by Tawfiq al-Hakim?")])
    print(f"[model={model_used}]")
    print(response.content)
    print("----------")
    model_used, response = invoke_with_model_fallback(
        [HumanMessage("You are a helpful assistant."),
         HumanMessage("What is Ryan Calais Camerons's most recent play?")])
    print(f"[model={model_used}]")
    print(response.content)
    print("----------")
    model_used, response = invoke_with_model_fallback(
        [HumanMessage("You are a helpful assistant."),
         HumanMessage("What Broadway shows have more than 10,000 performances?")])
    print(f"[model={model_used}]")
    print(response.content)

#    print(prompt_template.invoke({"playwright": "Ryan Calais Cameron"}))
#    response = model.invoke(prompt_template.invoke({"playwright": "Ryan Calais Cameron"}))
#    print(response.content)

#    chain = ???
#    response = ???{"playwright": "Ryan Calais Cameron"})
#    print(response.content)

    pass