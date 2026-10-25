import sqlite3
import os


DATABASE_PATH = "data/temporary_memory.db"


def get_connection():

    os.makedirs("data", exist_ok=True)

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


def initialize_database():

    connection = get_connection()

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS temporary_memory (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id TEXT NOT NULL,

            session_id TEXT,

            mode TEXT NOT NULL,

            memory_type TEXT NOT NULL,

            content TEXT NOT NULL,

            summary TEXT,

            date TEXT,

            time TEXT,

            importance TEXT,

            created_at TEXT NOT NULL,

            expires_at TEXT NOT NULL

        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_temp_user_mode
        ON temporary_memory(user_id, mode)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_temp_user_date
        ON temporary_memory(user_id, date)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS
        idx_temp_expiry
        ON temporary_memory(expires_at)
    """)

    connection.commit()

    connection.close()