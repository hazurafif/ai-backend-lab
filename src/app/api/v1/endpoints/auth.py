"""Auth routes: /register, /login, /users/me, admin user management.

Users live in the durable users store (`persistence.users`: Postgres `users`
table, in-memory fallback). On first start with an empty store, a default
admin account is seeded by `Persistence.ensure_default_admin`. Roles:
`user` (default, can chat) and `admin` (manages users and agent resources).
"""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from ....core.database import persistence
from ....core.dependencies import get_admin_user, get_current_user
from ....core.exceptions import BadRequest, Conflict, NotFound
from ....core.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_password_hash,
    verify_password,
)
from ....schema.auth_schema import Token, User, UserCreate, UserUpdate

router = APIRouter(tags=["auth"])


@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(payload: UserCreate):
    # Registration always creates a regular `user`; role changes are admin-only.
    user = await persistence.users.create_user(
        username=payload.username,
        hashed_password=get_password_hash(payload.password),
        email=payload.email,
        full_name=payload.full_name,
        role="user",
    )
    if user is None:
        raise Conflict(detail=f"Username '{payload.username}' is already taken")
    return user


@router.post("/login", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await persistence.users.get_user(form_data.username)
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
    return current_user


@router.get("/users", response_model=list[User])
async def list_users(_: dict = Depends(get_admin_user)):
    """Admin: list all users (newest first, no password hashes)."""
    return await persistence.users.list_users()


@router.patch("/users/{username}", response_model=User)
async def update_user(
    username: str,
    body: UserUpdate,
    current_user: dict = Depends(get_admin_user),
):
    """Admin: change a user's role and/or disabled state.

    Admins cannot demote or disable their own account (lockout guard).
    """
    if username == current_user["username"]:
        demoting = body.role is not None and body.role.value != "admin"
        if demoting or body.disabled is True:
            raise BadRequest(detail="You cannot demote or disable your own account")
    user = await persistence.users.update_user(
        username,
        role=body.role.value if body.role is not None else None,
        disabled=body.disabled,
    )
    if user is None:
        raise NotFound(detail=f"User '{username}' not found")
    return user
