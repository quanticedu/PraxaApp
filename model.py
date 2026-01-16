from langchain_community.chat_models import ChatOpenAI
from typing import Optional, Any
import os

os.environ["OPENROUTER_API_KEY"] = "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"

class ChatModel(ChatOpenAI):
    """
    Creates a chat model from openrouter.ai using the OpenAI API
    """
    def __init__(
            self,
            model_name: str,
            openai_api_key: Optional[str]=None,
            openai_api_base: str="https://openrouter.ai/api/v1",
            **kwargs: Any):
        openai_api_key = openai_api_key or os.getenv('OPENROUTER_API_KEY')
        super().__init__(
            openai_api_base=openai_api_base,
            openai_api_key=openai_api_key,
            model_name=model_name,
            **kwargs
        )

def get_model(model_name: str = "google/gemma-3-27b-it:free"):
    """
    Docstring for get_model
    
    :param model_name: Name of the model (default is google/gemma-3-27b-it:free)
    """
    return ChatModel(
        model_name=model_name,
        max_tokens=512,
        temperature=0
    )

if __name__ == "__main__":
# when run as a script, run some tests to demonstrate capabilities
#    model = get_model()
#    from langchain_core.messages import SystemMessage, HumanMessage
#    from langchain.prompts import ChatPromptTemplate

#    prompt_template = ChatPromptTemplate([
#        ("system", "You are a helpful assistant."),
#        ("user", "What is {playwright}'s most recent play?")
#    ])

#    response = model.invoke(
#        [SystemMessage("You are a helpful assistant."),
#         HumanMessage("What are some plays by Tawfiq al-Hakim?")])
#    print(response.content)
#    print("----------")
#    response = model.invoke(
#        [SystemMessage("You are a helpful assistant."),
#         HumanMessage("What is Ryan Calais Camerons's most recent play?")])
#    print(response.content)
#    print("----------")
#    response = model.invoke(
#        [SystemMessage("You are a helpful assistant."),
#         HumanMessage("What Broadway shows have more than 10,000 performances?")])
#    print(response.content)

#    print(prompt_template.invoke({"playwright": "Ryan Calais Cameron"}))
#    response = model.invoke(prompt_template.invoke({"playwright": "Ryan Calais Cameron"}))
#    print(response.content)

#    chain = prompt_template | model
#    response = chain.invoke({"playwright": "Ryan Calais Cameron"})
#    print(response.content)

    pass