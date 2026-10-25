
import json

from app.llm import ask_llm


ALLOWED_MODES = {
    "general",
    "classroom",
    "meeting",
    "hospital",
}


MODE_INSTRUCTIONS = {
    "general": """
Focus on useful everyday information:
- Important events and reminders
- Personal tasks and deadlines
- Appointments and plans
- Important facts explicitly mentioned
Do not treat every ordinary sentence as important.
""",

    "classroom": """
Focus on educational information:
- Exams, tests, subjects, and exam schedules
- Assignments, projects, and submission deadlines
- Lectures, concepts, and academic announcements
- Study instructions and academic tasks
Classify information according to what the speaker actually says.
""",

    "meeting": """
Focus on meeting-related information:
- Decisions actually made
- Tasks and action items
- The person responsible for a task, if stated
- Deadlines and follow-up actions
- Meeting outcomes and commitments
Do not invent a decision, owner, or commitment.
""",

    "hospital": """
Focus on healthcare-related information explicitly stated:
- Appointments and follow-up visits
- Doctor or healthcare professional instructions
- Medicines and medical tests mentioned
- Dates, times, and preparation instructions
Do not invent diagnoses, medicines, dosages, or medical advice.
""",
}


def normalize_mode(mode):
    if not isinstance(mode, str):
        raise ValueError("Mode must be a string.")

    mode = mode.strip().lower()

    if mode not in ALLOWED_MODES:
        raise ValueError(
            f"Unsupported mode: {mode}. "
            f"Choose one of: {', '.join(sorted(ALLOWED_MODES))}"
        )

    return mode


def process_information(data):
    mode = normalize_mode(data.mode)
    transcript = data.transcript.strip()

    if not transcript:
        raise ValueError("Transcript cannot be empty.")

    mode_instructions = MODE_INSTRUCTIONS[mode]

    prompt = f"""
You are an AI accessibility assistant.

SELECTED MODE: {mode}

REFERENCE TIMESTAMP: {data.timestamp}

MODE-SPECIFIC CONTEXT RULES:
{mode_instructions}

CONVERSATION:
{json.dumps(transcript, ensure_ascii=False)}

Your tasks:

1. CONTEXT UNDERSTANDING
Understand the meaning of the conversation in its selected mode.
Identify the event, task, instruction, or situation described.
Use the mode to guide classification and prioritization.
Do not change the meaning of the conversation.

2. SIMPLIFICATION
Rewrite the conversation in clear, simple language.
Preserve names, numbers, dates, times, deadlines, conditions,
negations, and instructions.
Do not add facts that are absent from the conversation.

3. SUMMARY
Write a short summary containing the main point.

4. IMPORTANT INFORMATION
Extract only information worth remembering in the selected mode.
For each item, provide:
- text: the important fact or instruction
- memory_type: a concise category, such as exam, assignment,
  task, decision, appointment, reminder, or important_event
- importance: "high", "medium", or "low"
- date: a date explicitly stated, or the relative date wording
  exactly as expressed, or null if no date is given
- time: a time explicitly stated, or null if no time is given

MODE RULES:
- Classroom: prioritize exams, academic tasks, assignments,
  lectures, projects, and deadlines.
- Meeting: prioritize decisions, action items, responsibilities,
  deadlines, and follow-ups.
- Hospital: prioritize appointments, stated healthcare instructions,
  medicines, tests, and follow-ups.
- General: prioritize useful events, tasks, reminders, appointments,
  deadlines, and other important facts.

DATE AND ACCURACY RULES:
- Never invent a date, time, name, or fact.
- For phrases such as "next Wednesday", preserve that wording
  instead of guessing an absolute date.
- Do not interpret a relative date using an unrelated year.
- Use null when a date or time is not stated.
- Do not classify an event as completed unless the conversation
  says it has happened.
- If no important information is present, return [].
- Do not create multiple items that repeat the same fact.

Return ONLY valid JSON using exactly this structure:

{{
  "simplified_text": "Simple version of the conversation",
  "summary": "Short summary",
  "important_information": [
    {{
      "text": "Important information",
      "memory_type": "category",
      "importance": "high",
      "date": null,
      "time": null
    }}
  ]
}}
"""

    result = ask_llm(prompt)

    try:
        output = json.loads(result)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("LLM returned invalid JSON.") from exc

    required_keys = {
        "simplified_text",
        "summary",
        "important_information",
    }

    if not isinstance(output, dict):
        raise ValueError("LLM output must be a JSON object.")

    if not required_keys.issubset(output):
        raise ValueError("LLM output is missing required fields.")

    if not isinstance(output["simplified_text"], str):
        raise ValueError("simplified_text must be a string.")

    if not isinstance(output["summary"], str):
        raise ValueError("summary must be a string.")

    items = output["important_information"]

    if not isinstance(items, list):
        raise ValueError("important_information must be a list.")

    item_keys = {
        "text",
        "memory_type",
        "importance",
        "date",
        "time",
    }

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each important information item must be an object.")

        if not item_keys.issubset(item):
            raise ValueError("An important information item is missing fields.")

        for key in ("text", "memory_type", "importance"):
            if not isinstance(item[key], str):
                raise ValueError(f"{key} must be a string.")

        if item["importance"] not in {"high", "medium", "low"}:
            raise ValueError("Invalid importance value.")

        for key in ("date", "time"):
            if item[key] is not None and not isinstance(item[key], str):
                raise ValueError(f"{key} must be a string or null.")

    return output
