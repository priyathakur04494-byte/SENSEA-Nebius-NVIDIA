from ai_engine import ask_nvidia


prompt = """
Analyze this transcript.

Transcript:
Today we are going to learn neural networks.
We will first discuss artificial neurons.
Then we will learn about backpropagation.
The assignment is due on Friday.

Give:
1. A short summary
2. Three key points
"""

result = ask_nvidia(prompt)

print("\n===== AI ENGINE RESPONSE =====\n")
print(result)