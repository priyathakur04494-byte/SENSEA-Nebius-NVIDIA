from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    pass


engine = (
    create_engine(
        settings.database_url,
        pool_pre_ping=True,
    )
    if settings.database_url
    else None
)


SessionLocal = (
    sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )
    if engine
    else None
)


def get_db():
    if SessionLocal is None:
        raise RuntimeError("Database is not configured yet.")

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()