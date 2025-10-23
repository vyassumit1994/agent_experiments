from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
import os
from dotenv import load_dotenv

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc."
        ),
        MessagesPlaceholder(variable_name='messages')
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts."
        ),
        MessagesPlaceholder(variable_name='messages')
    ]
)

# 1. load .env into environment variables
load_dotenv()  # reads .env from current working directory

# 2. read variables
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
AZURE_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")


llm = AzureChatOpenAI(
    azure_deployment=AZURE_DEPLOYMENT,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_KEY,
    model=AZURE_MODEL,
    api_version=AZURE_API_VERSION
)

generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm



