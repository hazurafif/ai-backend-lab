"""Auth routes: /register, /login, /refresh, /users/me, admin user management.

Users live in the durable users store (`persistence.users`: Postgres `users`
table, in-memory fallback). On first start with an empty store, a default
admin account is seeded by `Persistence.ensure_default_admin`. Roles:
`user` (default, can chat) and `admin` (manages users and agent resources).
Login failures are rate-limited per client IP + username (in-memory sliding
window, see `core/rate_limit.py`).
"""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from ....core.database import persistence
from ....core.dependencies import get_admin_user, get_current_user
from ....core.exceptions import BadRequest, Conflict, NotFound, PermissionDenied
from ....core.rate_limit import login_limiter
from ....core.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    get_password_hash,
    verify_password,
)
from ....schema.auth_schema import (
    AdminUserCreate,
    PasswordChange,
    RefreshRequest,
    Token,
    User,
    UserCreate,
    UserUpdate,
)
from ....services.workspace import ensure_user_workspace, remove_user_workspace

router = APIRouter(tags=["auth"])


def _login_key(client_host: str | None, username: str) -> str:
    return f"{client_host or 'unknown'}:{username}"


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
    ensure_user_workspace(payload.username)
    return user


@router.post("/login", response_model=Token)
async def login_for_access_token(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    key = _login_key(request.client.host if request.client else None, form_data.username)
    if not login_limiter.check(key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts; try again later",
        )
    user = await persistence.users.get_user(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        login_limiter.record_failure(key)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    login_limiter.record_success(key)
    access_token = create_access_token(
        data={"sub": form_data.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh_token = create_refresh_token(data={"sub": form_data.username})
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


@router.post("/refresh", response_model=Token, response_model_exclude_none=True)
async def refresh_access_token(body: RefreshRequest):
    """Exchange a valid refresh token for a fresh access token.

    Refresh tokens are stateless JWTs (no revocation store); logout is
    client-side discard of both tokens.
    """
    token_data = decode_refresh_token(body.refresh_token)
    if token_data is None or token_data.username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = await persistence.users.get_user(token_data.username)
    if user is None or user.get("disabled"):
        raise PermissionDenied(detail="Account is disabled or no longer exists")
    access_token = create_access_token(
        data={"sub": token_data.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/users/me/password", response_model=User)
async def change_own_password(
    body: PasswordChange,
    current_user: dict = Depends(get_current_user),
):
    """Change your own password: old password must verify first."""
    if not verify_password(body.old_password, current_user["hashed_password"]):
        raise BadRequest(detail="Old password is incorrect")
    user = await persistence.users.update_user(
        current_user["username"],
        hashed_password=get_password_hash(body.new_password),
    )
    if user is None:
        raise NotFound(detail="User not found")
    return user


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


@router.post("/users", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user_by_admin(
    payload: AdminUserCreate,
    _: dict = Depends(get_admin_user),
):
    """Admin: create a user; the admin role may be granted directly."""
    user = await persistence.users.create_user(
        username=payload.username,
        hashed_password=get_password_hash(payload.password),
        email=payload.email,
        full_name=payload.full_name,
        role=payload.role.value,
    )
    if user is None:
        raise Conflict(detail=f"Username '{payload.username}' is already taken")
    ensure_user_workspace(payload.username)
    return user


@router.delete("/users/{username}", status_code=204)
async def delete_user_by_admin(
    username: str,
    current_user: dict = Depends(get_admin_user),
):
    """Admin: delete a user — their workspace dir, store data and chat
    history rows are purged; the deletion is committed to the workspace git."""
    if username == current_user["username"]:
        raise BadRequest(detail="You cannot delete your own account")
    if not await persistence.users.delete_user(username):
        raise NotFound(detail=f"User '{username}' not found")
    # Chat history rows first (thread ids come from the thread metadata ns).
    thread_ids = [it.key or "" for it in await persistence.store.asearch(("threads", username))]
    for thread_id in thread_ids:
        await persistence.chat_history.delete_thread(thread_id)
    # Purge store namespaces: memories/uploads, thread metadata, user agent
    # configs + their skill snapshots, user skills.
    for ns in (
        (username,),
        ("threads", username),
        ("agents", username),
        ("user", "skills", username),
    ):
        for item in await persistence.store.asearch(ns):
            await persistence.store.adelete(ns, item.key or "")
    for item in await persistence.store.asearch(("agent", "agent_skills", username)):
        await persistence.store.adelete(("agent", "agent_skills", username), item.key or "")
    remove_user_workspace(username)
    return None
