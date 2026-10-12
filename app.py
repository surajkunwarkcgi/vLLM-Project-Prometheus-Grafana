import os
import time
import mlflow
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

SERVER_API = ""
API_URL = f"http://{SERVER_API}:40836/v1"
TRACKING_URL = f"http://{SERVER_API}:40685"
API_KEY = "MLEXPERT_RULES"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

os.environ["MLFLOW_TRACKING_URI"] = TRACKING_URL
mlflow.langchain.autolog()

model = init_chat_model(
    model = MODEL,
    model_provider = "openai",
    base_url = API_URL,
    api_key = API_KEY
)

mlflow.set_experiment("conversation")

system_msg = SystemMessage("You are an expert LLMOps engineer.")
human_msg = HumanMessage(
    "What are the top 3 best practices for LLMOps? Summarize each in a few words."
)

messages = [system_msg, human_msg]

%%time
response = model.invoke(messages)
print(response.content)

messages.append(response)
messages.append(
    HumanMessage(
        "So what are the top 3 tools for LLMOps? Summarize each in a few words."
    )
)