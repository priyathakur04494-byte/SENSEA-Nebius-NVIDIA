from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID

from app.db.database import get_db
from app.schemas.conversation import ConversationCreate, ConversationResponse, ConversationUpdate
from app.services.conversation_service import (
    create_conversation,
    get_all_conversations,
    get_conversation_or_404,
    update_conversation,
    remove_conversation,
)


router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("/", response_model=ConversationResponse, status_code=201)
def create(conversation_in: ConversationCreate, db: Session = Depends(get_db)):
    return create_conversation(db, conversation_in.user_id, conversation_in.title)


@router.get("/", response_model=list[ConversationResponse])
def list_all(user_id: UUID, db: Session = Depends(get_db)):
    return get_all_conversations(db, user_id)


@router.get("/{conversation_id}", response_model=ConversationResponse)
def get(conversation_id: UUID, db: Session = Depends(get_db)):
    return get_conversation_or_404(db, conversation_id)


@router.put("/{conversation_id}", response_model=ConversationResponse)
def update(conversation_id: UUID, conversation_in: ConversationUpdate, db: Session = Depends(get_db)):
    db_conversation = get_conversation_or_404(db, conversation_id)
    return update_conversation(db, db_conversation, conversation_in.title)


@router.delete("/{conversation_id}", status_code=204)
def delete(conversation_id: UUID, db: Session = Depends(get_db)):
    db_conversation = get_conversation_or_404(db, conversation_id)
    remove_conversation(db, db_conversation)