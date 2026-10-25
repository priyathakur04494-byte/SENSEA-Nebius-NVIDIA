from datetime import datetime, timedelta

from app.database import get_connection


def save_temporary_memory(
    user_id,
    session_id,
    mode,
    memory_type,
    content,
    summary,
    date,
    time,
    importance
):
    now = datetime.utcnow()

    expires_at = now + timedelta(hours=24)

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO temporary_memory (
            user_id,
            session_id,
            mode,
            memory_type,
            content,
            summary,
            date,
            time,
            importance,
            created_at,
            expires_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        session_id,
        mode,
        memory_type,
        content,
        summary,
        date,
        time,
        importance,
        now.isoformat(),
        expires_at.isoformat()
    ))

    connection.commit()

    memory_id = cursor.lastrowid

    connection.close()

    return {
        "memory_id": memory_id,
        "storage": "temporary",
        "expires_at": expires_at.isoformat()
    }


def delete_expired_memories():

    now = datetime.utcnow().isoformat()

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM temporary_memory
        WHERE expires_at <= ?
    """, (now,))

    deleted = cursor.rowcount

    connection.commit()
    connection.close()

    return deleted