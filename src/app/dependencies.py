from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from . import auth

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = auth.decode_access_token(token)
    if token_data is None or token_data.username is None:
        raise credentials_exception
    return {"username": token_data.username}
