# client.py
from dotenv import load_dotenv
import json
from anthropic import Anthropic
from project.parser.prompt import SYSTEM_PROMPT
from schema import ProblemSpec
import os

load_dotenv()  # reads .env
api_key = os.environ["ANTHROPIC_API_KEY"]

# print("API Key loaded:", bool(api_key))

client = Anthropic(api_key=api_key)

def parse_problem(prompt: str) -> ProblemSpec:
    response = client.messages.create(
        model="claude-sonnet-4-20250514", 
        max_tokens=500, # Replaced max_tokens_to_sample (deprecated)
        system=SYSTEM_PROMPT, # Anthropic handles system prompts at the top level
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    text = response.content[0].text.strip()
    # Find the first { and the last } in the response
    start_idx = text.find('{')
    end_idx = text.rfind('}') + 1
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON:\n{text}") from e
    
    try:
        problem = ProblemSpec(**data)
    except Exception as e:
        raise ValueError("JSON does not match ProblemSpec schema: {data}") from e

    return problem