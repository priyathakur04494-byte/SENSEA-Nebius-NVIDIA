from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.user import UserCreate, UserUpdate, UserResponse
from app.services.user_service import (
    create_user,
    get_user,
    get_all_users,
    update_user,
    remove_user,
)
from uuid import UUID


router = APIRouter(prefix="/users", tags=["users"])


@router.post("/", response_model=UserResponse, status_code=201)
def create(user_in: UserCreate, db: Session = Depends(get_db)):
    return create_user(db, user_in)


@router.get("/", response_model=list[UserResponse])
def list_all(db: Session = Depends(get_db)):
    return get_all_users(db)


@router.get("/{user_id}", response_model=UserResponse)
def get(user_id: UUID, db: Session = Depends(get_db)):
    user = get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put("/{user_id}", response_model=UserResponse)
def update(user_id: UUID, user_in: UserUpdate, db: Session = Depends(get_db)):
    user = get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return update_user(db, user, user_in)


@router.delete("/{user_id}", status_code=204)
def delete(user_id: UUID, db: Session = Depends(get_db)):
    user = get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    remove_user(db, user)