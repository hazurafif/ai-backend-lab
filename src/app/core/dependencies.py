"""Shared API dependencies (auth)."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from .database import persistence
from .security import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
# auto_error=False: returns None instead of raising when no header is sent.
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)


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
    return user


async def get_optional_current_user(
    token: str | None = Depends(oauth2_scheme_optional),
):
    """Starter-mode auth: validate the token when present, else "guest".

    Mirrors /api/chat, which scopes thread metadata to the Bearer JWT user
    when one is sent and falls back to a guest namespace otherwise (the
    frontend has no login yet). Used for the global agent resource routes
    (skills, MCP tool servers) so the settings page can manage them without
    a token; the namespaces they operate on are not per-user anyway.
    """
    if token is None:
        return {"username": "guest"}
    token_data = decode_access_token(token)
    if token_data is None or token_data.username is None:
        raise _credentials_exception()
    return {"username": token_data.username}
