from app.database import initialize_database
from app.memory_service import save_temporary_memory


initialize_database()


result = save_temporary_memory(

    user_id="user123",

    session_id="session001",

    mode="classroom",

    memory_type="exam",

    content="DSA exam is on Monday at 10 AM.",

    summary="DSA exam rescheduled.",

    date="2026-09-07",

    time="10:00",

    importance="high"
)


print(result)