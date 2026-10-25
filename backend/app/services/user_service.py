from sqlalchemy.orm import Session
from sqlalchemy import select
from uuid import UUID

from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate


def get_user(db: Session, user_id: UUID):
    statement = select(User).where(User.id == user_id)
    result = db.execute(statement)
    return result.scalar_one_or_none()


get_all_users = lambda db: db.execute(select(User)).scalars().all()


def create_user(db: Session, user_in: UserCreate):
    user = User(
        email=user_in.email,
        display_name=user_in.display_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user(db: Session, db_user: User, user_in: UserUpdate):
    update_data = user_in.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_user, field, value)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def remove_user(db: Session, db_user: User):
    db.delete(db_user)
    db.commit()