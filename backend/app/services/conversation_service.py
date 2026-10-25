from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models.conversation import Conversation
from app.models.user import User
from app.schemas.conversation import ConversationCreate, ConversationUpdate


def get_conversation(db: Session, conversation_id: UUID):
    statement = select(Conversation).where(Conversation.id == conversation_id)
    result = db.execute(statement)
    return result.scalar_one_or_none()


def get_all_conversations(db: Session, user_id: UUID | None = None):
    statement = select(Conversation)
    if user_id is not None:
        statement = statement.where(Conversation.user_id == user_id)
    return db.execute(statement).scalars().all()


def create_conversation(db: Session, user_id: UUID, title: str | None = None):
    user = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
    if not user:
        raise ValueError(f"User with id {user_id} not found")
    conversation = Conversation(user_id=user_id, title=title)
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    return conversation


def update_conversation(db: Session, db_conversation: Conversation, title: str | None = None):
    if title is not None:
        db_conversation.title = title
    db.add(db_conversation)
    db.commit()
    db.refresh(db_conversation)
    return db_conversation


def remove_conversation(db: Session, db_conversation: Conversation):
    db.delete(db_conversation)
    db.commit()


def get_conversation_or_404(db: Session, conversation_id: UUID):
    conversation = get_conversation(db, conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation