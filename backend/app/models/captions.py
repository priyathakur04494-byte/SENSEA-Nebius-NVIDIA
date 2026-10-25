from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class Caption(Base):
    __tablename__ = "captions"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id"),
        nullable=False,
        index=True,
    )

    speaker_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("speakers.id"),
        nullable=True,
        index=True,
    )

    original_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    simplified_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    translated_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    language: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="en",
    )

    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    is_final: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
