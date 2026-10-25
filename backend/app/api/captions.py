from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID

from app.db.database import get_db
from app.schemas.caption import CaptionCreate, CaptionResponse, CaptionUpdate
from app.services.caption_service import (
    create_caption,
    get_caption,
    get_captions_by_conversation,
    update_caption,
    remove_caption,
)


router = APIRouter(prefix="/captions", tags=["captions"])


@router.post("/", response_model=CaptionResponse, status_code=201)
def create(caption_in: CaptionCreate, db: Session = Depends(get_db)):
    return create_caption(db, caption_in)


@router.get("/", response_model=list[CaptionResponse])
def list_by_conversation(conversation_id: UUID, db: Session = Depends(get_db)):
    return get_captions_by_conversation(db, conversation_id)


@router.get("/{caption_id}", response_model=CaptionResponse)
def get(caption_id: UUID, db: Session = Depends(get_db)):
    caption = get_caption(db, caption_id)
    if not caption:
        raise HTTPException(status_code=404, detail="Caption not found")
    return caption


@router.put("/{caption_id}", response_model=CaptionResponse)
def update(caption_id: UUID, caption_in: CaptionUpdate, db: Session = Depends(get_db)):
    caption = get_caption(db, caption_id)
    if not caption:
        raise HTTPException(status_code=404, detail="Caption not found")
    return update_caption(db, caption, caption_in)


@router.delete("/{caption_id}", status_code=204)
def delete(caption_id: UUID, db: Session = Depends(get_db)):
    caption = get_caption(db, caption_id)
    if not caption:
        raise HTTPException(status_code=404, detail="Caption not found")
    remove_caption(db, caption)