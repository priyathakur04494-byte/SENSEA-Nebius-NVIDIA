from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.captions import Caption
from app.models.conversation import Conversation
from app.models.speaker import Speaker
from app.schemas.caption import CaptionCreate, CaptionUpdate


def get_caption(db: Session, caption_id: UUID):
    statement = select(Caption).where(Caption.id == caption_id)
    result = db.execute(statement)
    return result.scalar_one_or_none()


def get_captions_by_conversation(db: Session, conversation_id: UUID):
    statement = select(Caption).where(Caption.conversation_id == conversation_id)
    result = db.execute(statement)
    return result.scalars().all()


def create_caption(db: Session, caption_in: CaptionCreate):
    conversation = db.execute(
        select(Conversation).where(Conversation.id == caption_in.conversation_id)
    ).scalar_one_or_none()
    if not conversation:
        raise ValueError(f"Conversation with id {caption_in.conversation_id} not found")

    speaker = None
    if caption_in.speaker_id is not None:
        speaker = db.execute(
            select(Speaker).where(Speaker.id == caption_in.speaker_id)
        ).scalar_one_or_none()
        if not speaker:
            raise ValueError(
                f"Speaker with id {caption_in.speaker_id} not found"
            )

    caption = Caption(
        conversation_id=caption_in.conversation_id,
        speaker_id=caption_in.speaker_id,
        original_text=caption_in.original_text,
        simplified_text=caption_in.simplified_text,
        translated_text=caption_in.translated_text,
        language=caption_in.language,
        sequence_number=caption_in.sequence_number,
        is_final=caption_in.is_final,
    )
    db.add(caption)
    db.commit()
    db.refresh(caption)
    return caption


def update_caption(db: Session, db_caption: Caption, caption_in: CaptionUpdate):
    update_data = caption_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_caption, field, value)
    db.add(db_caption)
    db.commit()
    db.refresh(db_caption)
    return db_caption


def remove_caption(db: Session, db_caption: Caption):
    db.delete(db_caption)
    db.commit()