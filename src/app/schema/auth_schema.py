"""Auth API models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class UserRole(StrEnum):
    """Stored roles; `admin` gates agent resource management."""

    USER = "user"
    ADMIN = "admin"


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[a-zA-Z0-9_]+$")
    password: str = Field(min_length=8, max_length=128)
    email: str | None = Field(default=None, max_length=254)
    full_name: str | None = Field(default=None, max_length=100)


class UserUpdate(BaseModel):
    """Admin-only user management payload: change role and/or disabled state."""

    role: UserRole | None = None
    disabled: bool | None = None

    @model_validator(mode="after")
    def at_least_one_field(self) -> UserUpdate:
        if self.role is None and self.disabled is None:
            raise ValueError("provide at least one of 'role' or 'disabled'")
        return self


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None
    role: UserRole = UserRole.USER


class UserInDB(User):
    hashed_password: str
