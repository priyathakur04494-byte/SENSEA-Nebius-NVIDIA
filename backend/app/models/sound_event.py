from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class SoundEvent(Base):
    __tablename__ = "sound_events"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    conversation_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("conversations.id"),
        nullable=True,
        index=True,
    )

    label: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    alert_level: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    direction: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    confidence: Mapped[float | None] = mapped_column(
        nullable=True,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
