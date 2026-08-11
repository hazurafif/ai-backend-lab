"""Agent resource models (skills, MCP tool servers)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SKILL_NAME_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class SkillIn(BaseModel):
    """Create/update payload for an agent skill (stored as SKILL.md)."""

    name: str = Field(
        ...,
        pattern=SKILL_NAME_PATTERN,
        min_length=1,
        max_length=64,
        description=(
            "Skill identifier: lowercase alphanumeric and hyphens (per the Agent Skills spec)"
        ),
    )
    description: str = Field(..., min_length=1, max_length=1024)
    content: str = Field(..., min_length=1, description="Markdown instructions body")


class SkillOut(BaseModel):
    name: str
    description: str = ""
    content: str
    path: str = ""


class ToolServerIn(BaseModel):
    """Create/update payload for an MCP tool server (gofastmcp)."""

    name: str = Field(
        ...,
        pattern=SKILL_NAME_PATTERN,
        min_length=1,
        max_length=64,
        description="Server name, referenced in tool names and /health",
    )
    transport: Literal["streamable_http", "stdio"] = "streamable_http"
    url: str | None = Field(default=None, description="Required for streamable_http")
    command: str | None = Field(default=None, description="Required for stdio")
    args: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True

    def config(self) -> dict:
        """Shape expected by MultiServerMCPClient (mcp_servers.json format)."""
        cfg: dict = {"transport": self.transport}
        if self.transport == "streamable_http":
            cfg["url"] = self.url
            if self.headers:
                cfg["headers"] = self.headers
        else:
            cfg["command"] = self.command
            if self.args:
                cfg["args"] = self.args
            if self.env:
                cfg["env"] = self.env
        return cfg


class ToolServerOut(BaseModel):
    name: str
    transport: str = ""
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    headers: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
