from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from jwt.exceptions import InvalidTokenError

from ..core.config import settings
from ..schema.auth_schema import TokenData

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


# Token type claims: access vs refresh (refresh tokens cannot be used as
# access tokens and vice versa). Revocation is client-side discard for now.
_ACCESS_TYPE = "access"
_REFRESH_TYPE = "refresh"


def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())


def get_password_hash(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """JWT access token; defaults to ACCESS_TOKEN_EXPIRE_MINUTES."""
    to_encode = data.copy()
    to_encode.setdefault("type", _ACCESS_TYPE)
    expire = datetime.now(UTC) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict, expires_delta: timedelta | None = None):
    """JWT refresh token (type=refresh), valid REFRESH_TOKEN_EXPIRE_DAYS."""
    to_encode = data.copy()
    to_encode.setdefault("type", _REFRESH_TYPE)
    expire = datetime.now(UTC) + (expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return encoded_jwt


def _decode(token: str, expected_type: str) -> TokenData | None:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    except InvalidTokenError:
        return None
    # Legacy tokens predating the type claim are access tokens.
    if payload.get("type", _ACCESS_TYPE) != expected_type:
        return None
    username: str | None = payload.get("sub")
    if username is None:
        return None
    return TokenData(username=username)


def decode_access_token(token: str) -> TokenData | None:
    """Validate an access token; returns None for refresh tokens / junk."""
    return _decode(token, _ACCESS_TYPE)


def decode_refresh_token(token: str) -> TokenData | None:
    """Validate a refresh token; returns None for access tokens / junk."""
    return _decode(token, _REFRESH_TYPE)
