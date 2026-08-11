from pydantic import BaseModel, Field


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


class UserInDB(User):
    hashed_password: str


# --- Chat / agent API ---


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to the agent")
    thread_id: str | None = Field(
        default=None,
        description="Conversation id. Omit to start a new conversation.",
    )


class ThreadOut(BaseModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str | None = None
