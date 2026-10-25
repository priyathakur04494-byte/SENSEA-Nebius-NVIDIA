from app.memory_service import delete_expired_memories

deleted = delete_expired_memories()

print("Deleted:", deleted)