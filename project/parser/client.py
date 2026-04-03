from dotenv import load_dotenv
import json
from anthropic import Client
from project.parser.prompt import SYSTEM_PROMPT
from schema import ProblemSpec
import os

load_dotenv()  # reads .env
api_key = os.environ["ANTHROPIC_API_KEY"]

# print("API Key loaded:", bool(api_key))

client = Client(api_key=api_key)

def parse_problem(prompt: str) -> ProblemSpec:
    response = client.completions.create(
        model="claude-sonnet-4-0",
        prompt=f"{SYSTEM_PROMPT}\nInput: {prompt}\nOutput:",
        max_tokens_to_sample=500
    )
    text = response.completion.strip()
    data = json.loads(text)

    return ProblemSpec(**data)