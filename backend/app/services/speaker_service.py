from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select
from uuid import UUID

from app.models.speaker import Speaker
from app.models.user import User
from app.schemas.speaker import SpeakerCreate, SpeakerUpdate


def get_speaker(db: Session, speaker_id: UUID):
    statement = select(Speaker).where(Speaker.id == speaker_id)
    result = db.execute(statement)
    return result.scalar_one_or_none()


def get_all_speakers(db: Session, user_id: UUID | None = None):
    statement = select(Speaker)
    if user_id is not None:
        statement = statement.where(Speaker.user_id == user_id)
    return db.execute(statement).scalars().all()


def create_speaker(db: Session, user_id: UUID, name: str | None, label: str, color_tag: str | None, is_important_person: bool):
    user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
    if not user:
        raise ValueError(f"User with id {user_id} not found")
    speaker = Speaker(
        user_id=user_id,
        name=name,
        label=label,
        color_tag=color_tag,
        is_important_person=is_important_person,
    )
    db.add(speaker)
    db.commit()
    db.refresh(speaker)
    return speaker


def update_speaker(db: Session, db_speaker: Speaker, name: str | None, label: str | None, color_tag: str | None, is_important_person: bool | None):
    if name is not None:
        db_speaker.name = name
    if label is not None:
        db_speaker.label = label
    if color_tag is not None:
        db_speaker.color_tag = color_tag
    if is_important_person is not None:
        db_speaker.is_important_person = is_important_person
    db.add(db_speaker)
    db.commit()
    db.refresh(db_speaker)
    return db_speaker


def remove_speaker(db: Session, db_speaker: Speaker):
    db.delete(db_speaker)
    db.commit()


def get_speaker_or_404(db: Session, speaker_id: UUID):
    speaker = get_speaker(db, speaker_id)
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")
    return speaker