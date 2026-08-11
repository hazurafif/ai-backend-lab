"""Shared API dependencies (auth)."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from .database import persistence
from .exceptions import PermissionDenied
from .security import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def _credentials_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Validate the JWT and load the user from the users store."""
    token_data = decode_access_token(token)
    if token_data is None or token_data.username is None:
        raise _credentials_exception()
    user = await persistence.users.get_user(token_data.username)
    if user is None:
        raise _credentials_exception()
    if user.get("disabled"):
        raise PermissionDenied(detail="Account is disabled")
    return user


async def get_admin_user(current_user: dict = Depends(get_current_user)):
    """Require an authenticated user with the `admin` role."""
    if current_user.get("role") != "admin":
        raise PermissionDenied(detail="Admin role required")
    return current_user
