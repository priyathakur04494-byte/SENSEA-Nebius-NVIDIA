import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env
load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError("NVIDIA_API_KEY not found in .env")

# NVIDIA API client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

# Test transcript
transcript = """
Today we are going to learn neural networks.
We will first discuss artificial neurons.
Then we will learn about backpropagation.
The assignment is due on Friday.
"""

response = client.chat.completions.create(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",

    messages=[
        {
            "role": "system",
            "content": """
You are the AI assistant for Deaf Assist.

Analyze only the information provided by the user.
Do not invent information.
Do not show your reasoning or thinking process.

Return:
1. A short summary
2. Three key points
"""
        },
        {
            "role": "user",
            "content": f"Transcript:\n{transcript}"
        }
    ],

    temperature=0.2,
    max_tokens=500,

    # IMPORTANT:
    # Disable Nemotron's reasoning for faster,
    # direct responses.
    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    },

    stream=False
)

print("\n===== NVIDIA RESPONSE =====\n")
print(response.choices[0].message.content)

print("\n===== FINISH REASON =====")
print(response.choices[0].finish_reason)