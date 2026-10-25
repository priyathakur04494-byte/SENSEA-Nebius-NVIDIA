from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import UUID

from app.db.database import get_db
from app.schemas.speaker import SpeakerCreate, SpeakerUpdate, SpeakerResponse
from app.services.speaker_service import (
    create_speaker,
    get_all_speakers,
    get_speaker_or_404,
    update_speaker,
    remove_speaker,
)


router = APIRouter(prefix="/speakers", tags=["speakers"])


@router.post("/", response_model=SpeakerResponse, status_code=201)
def create(speaker_in: SpeakerCreate, db: Session = Depends(get_db)):
    return create_speaker(
        db,
        user_id=speaker_in.user_id,
        name=speaker_in.name,
        label=speaker_in.label,
        color_tag=speaker_in.color_tag,
        is_important_person=speaker_in.is_important_person,
    )


@router.get("/", response_model=list[SpeakerResponse])
def list_all(user_id: UUID, db: Session = Depends(get_db)):
    return get_all_speakers(db, user_id=user_id)


@router.get("/{speaker_id}", response_model=SpeakerResponse)
def get(speaker_id: UUID, db: Session = Depends(get_db)):
    return get_speaker_or_404(db, speaker_id)


@router.put("/{speaker_id}", response_model=SpeakerResponse)
def update(speaker_id: UUID, speaker_in: SpeakerUpdate, db: Session = Depends(get_db)):
    db_speaker = get_speaker_or_404(db, speaker_id)
    return update_speaker(
        db,
        db_speaker=db_speaker,
        name=speaker_in.name,
        label=speaker_in.label,
        color_tag=speaker_in.color_tag,
        is_important_person=speaker_in.is_important_person,
    )


@router.delete("/{speaker_id}", status_code=204)
def delete(speaker_id: UUID, db: Session = Depends(get_db)):
    db_speaker = get_speaker_or_404(db, speaker_id)
    remove_speaker(db, db_speaker)