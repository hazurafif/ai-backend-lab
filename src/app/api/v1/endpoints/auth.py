"""Auth routes: /register, /login (JWT) and /users/me."""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from ....core.dependencies import get_current_user
from ....core.exceptions import Conflict
from ....core.fake_users import fake_users_db
from ....core.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_password_hash,
    verify_password,
)
from ....schema.auth_schema import Token, User, UserCreate

router = APIRouter(tags=["auth"])


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(payload: UserCreate):
    # Demo user store; replace with a real database in production.
    if payload.username in fake_users_db:
        raise Conflict(detail=f"Username '{payload.username}' is already taken")
    fake_users_db[payload.username] = {
        "username": payload.username,
        "full_name": payload.full_name,
        "email": payload.email,
        "hashed_password": get_password_hash(payload.password),
        "disabled": False,
    }
    return fake_users_db[payload.username]


@router.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # Demo user store; replace with a real database in production.
    user = fake_users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": form_data.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/users/me/", response_model=User)
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"]}
