import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")

if not API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in .env")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=API_KEY
)

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


# ============================================================
# NVIDIA / NEMOTRON
# ============================================================

def ask_nvidia(prompt, max_tokens=1500):

    response = client.chat.completions.create(
        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": """
You are the AI analysis engine for Deaf Assist.

Your job is to analyze the provided conversation.

IMPORTANT RULES:

1. Use ONLY information explicitly present in the conversation.
2. Never invent facts.
3. Never guess missing information.
4. Never provide medical diagnosis.
5. Never change medication dosage or instructions.
6. Return EXACTLY ONE JSON object.
7. The response MUST begin with { and end with }.
8. Do NOT output anything before the JSON object.
9. Do NOT output anything after the JSON object.
10. Do NOT output <think> or </think>.
11. Do NOT output Markdown.
12. Do NOT use ``` or code fences.
13. Do NOT generate multiple JSON objects.
14. Follow the requested JSON schema exactly.
"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.1,
        max_tokens=max_tokens,

        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        },

        stream=False
    )

    return response.choices[0].message.content


# ============================================================
# JSON SCHEMAS
# ============================================================

GENERAL_SCHEMA = {
    "mode": "general",
    "summary": "",
    "key_points": [],
    "important_information": [],
    "questions": []
}


HOSPITAL_SCHEMA = {
    "mode": "hospital",
    "medical_summary": "",
    "instructions": [],
    "medications": [],
    "tests_procedures": [],
    "appointments_follow_up": [],
    "important_information": []
}


MEETING_SCHEMA = {
    "mode": "meeting",
    "summary": "",
    "key_discussion_points": [],
    "decisions": [],
    "action_items": [],
    "deadlines": [],
    "open_questions": []
}


CLASSROOM_SCHEMA = {
    "mode": "classroom",
    "summary": "",
    "topic": "",
    "key_concepts": [],
    "definitions": [],
    "notes": [],
    "assignments": [],
    "deadlines": [],

    "flashcards": [
        {
            "question": "",
            "answer": ""
        }
    ],

    "quiz": [
        {
            "question": "",
            "options": [],
            "answer": ""
        }
    ]
}


# ============================================================
# GET SCHEMA
# ============================================================

def get_schema(mode):

    schemas = {
        "general": GENERAL_SCHEMA,
        "hospital": HOSPITAL_SCHEMA,
        "meeting": MEETING_SCHEMA,
        "classroom": CLASSROOM_SCHEMA
    }

    if mode not in schemas:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    return schemas[mode]


# ============================================================
# BUILD CONVERSATION
# ============================================================

def build_conversation(segments):

    conversation = ""

    for segment in segments:

        speaker = segment.get(
            "speaker_id",
            "Unknown Speaker"
        )

        text = segment.get(
            "text",
            ""
        )

        translation = segment.get(
            "translation",
            ""
        )

        conversation += (
            f"{speaker}:\n"
            f"Transcription: {text}\n"
            f"Translation: {translation}\n\n"
        )

    return conversation


# ============================================================
# EXTRACT FIRST COMPLETE JSON OBJECT
# ============================================================

def extract_json_object(text):

    start = text.find("{")

    if start == -1:
        raise ValueError(
            "No JSON object found in Nemotron response."
        )

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):

        char = text[i]

        # Handle escaped characters inside JSON strings
        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        # Detect beginning/end of JSON string
        if char == '"':
            in_string = not in_string
            continue

        # Ignore braces inside strings
        if in_string:
            continue

        if char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                return text[start:i + 1]

    raise ValueError(
        "Nemotron returned incomplete JSON."
    )


# ============================================================
# ANALYZE CONVERSATION
# ============================================================

def analyze_conversation(mode, segments):

    mode = mode.lower().strip()

    schema = get_schema(mode)

    conversation = build_conversation(segments)

    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    prompt = f"""
Analyze the following Deaf Assist conversation.

SELECTED MODE:
{mode}

IMPORTANT:
Use ONLY information contained in the conversation.

If information required for a field is not present,
return an empty string or empty list.

Do NOT invent information.

Return EXACTLY ONE JSON object.

Return JSON matching EXACTLY this structure:

{json.dumps(schema, indent=2)}

CONVERSATION:

{conversation}

IMPORTANT FINAL INSTRUCTION:

Return ONLY ONE JSON object.

Do not write explanations.

Do not write Markdown.

Do not write <think> or </think>.

Do not generate a second answer.

Stop immediately after the closing }} of the JSON object.
"""

    # --------------------------------------------------------
    # CALL NEMOTRON
    # --------------------------------------------------------

    raw_result = ask_nvidia(
        prompt,
        max_tokens=1500
    )

    # --------------------------------------------------------
    # CLEAN RESPONSE
    # --------------------------------------------------------

    raw_result = raw_result.strip()

    # Remove accidental thinking tags
    raw_result = raw_result.replace(
        "<think>",
        ""
    )

    raw_result = raw_result.replace(
        "</think>",
        ""
    )

    # Remove accidental Markdown fences
    raw_result = raw_result.replace(
        "```json",
        ""
    )

    raw_result = raw_result.replace(
        "```",
        ""
    )

    raw_result = raw_result.strip()

    # --------------------------------------------------------
    # EXTRACT JSON
    # --------------------------------------------------------

    try:

        json_text = extract_json_object(
            raw_result
        )

        result = json.loads(
            json_text
        )

    except (json.JSONDecodeError, ValueError) as e:

        print("\n❌ Nemotron returned invalid JSON.")

        print("\nRaw response:")
        print(raw_result)

        raise ValueError(
            f"Invalid JSON returned by Nemotron: {e}"
        )

    # --------------------------------------------------------
    # VALIDATE JSON STRUCTURE
    # --------------------------------------------------------

    expected_fields = set(
        schema.keys()
    )

    missing_fields = (
        expected_fields
        - set(result.keys())
    )

    if missing_fields:

        print(
            "\n⚠️ Nemotron response is missing fields:"
        )

        print(
            missing_fields
        )

        # Fill missing fields with
        # the expected empty structure
        for field in missing_fields:

            result[field] = schema[field]

    # --------------------------------------------------------
    # FORCE CORRECT MODE
    # --------------------------------------------------------

    result["mode"] = mode

    # --------------------------------------------------------
    # RETURN FINAL RESULT
    # --------------------------------------------------------

    return result