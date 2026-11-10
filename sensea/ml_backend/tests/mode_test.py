from ai_engine import ask_nvidia

from prompts import (
    GENERAL_PROMPT,
    HOSPITAL_PROMPT,
    MEETING_PROMPT,
    CLASSROOM_PROMPT
)


transcript = """
Teacher: Today we are going to learn neural networks.
Teacher: We will first discuss artificial neurons.
Student: Will we also learn backpropagation?
Teacher: Yes. We will cover backpropagation today.
Teacher: The assignment is due on Friday.
Student: Okay, I will submit it before Friday.
"""


modes = {
    "GENERAL": GENERAL_PROMPT,
    "HOSPITAL": HOSPITAL_PROMPT,
    "MEETING": MEETING_PROMPT,
    "CLASSROOM": CLASSROOM_PROMPT
}


for mode, prompt in modes.items():

    print("\n" + "=" * 60)
    print(f"              {mode} MODE")
    print("=" * 60)

    full_prompt = f"""
{prompt}

Here is the transcript:

{transcript}
"""

    try:
        result = ask_nvidia(
            full_prompt,
            max_tokens=700
        )

        print(result)

    except Exception as e:
        print(f"ERROR in {mode} mode:")
        print(e)