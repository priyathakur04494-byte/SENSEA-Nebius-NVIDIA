import sys
sys.path.insert(0, '.')

from app.core.config import settings
from app.db.database import engine, SessionLocal, Base
from sqlalchemy import text, inspect

print("DB URL configured:", bool(settings.database_url))

if engine is None:
    print("FAIL: Engine is None")
    sys.exit(1)

try:
    with engine.connect() as conn:
        result = conn.execute(text('SELECT 1'))
        print(f"SELECT 1 result: {result.scalar()}")
        print("Connection succeeded")
        
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        print(f"Tables in database: {tables}")
        
        expected = {'users', 'conversations', 'captions', 'speakers', 'sound_events', 'deadlines', 'preferences'}
        found = set(tables)
        missing = expected - found
        if missing:
            print(f"MISSING tables: {missing}")
        else:
            print("All 7 expected tables exist!")
            
except Exception as e:
    print(f"Connection error: {e}")
    import traceback
    traceback.print_exc()