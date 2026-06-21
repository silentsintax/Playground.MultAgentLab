"""
LAYER 2 — SKILLS (capabilities that USE the tools).

A Skill is the model-facing surface of a capability: a name, a description, a JSON schema for
its arguments, and an async handler. Here every skill wraps an MCP tool from tools.py — the
handler binds an open MCP session and performs a tools/call round-trip. This is the layer
that turns "a tool exists somewhere" into "a thing this agent is allowed to do."

(Agents never call the MCP session directly; they only ever see Skills.)
"""

from dataclasses import dataclass
from typing import Awaitable, Callable

Handler = Callable[[dict], Awaitable[str]]


@dataclass
class Skill:
    name: str
    description: str
    parameters: dict          # JSON Schema for the arguments
    handler: Handler          # async (args: dict) -> str


def build_skills(session, tools) -> dict[str, Skill]:
    """
    Wrap each MCP tool (from session.list_tools()) as a Skill.

    `session` is an open mcp.ClientSession; `tools` is the list of discovered tools.
    Returns a name -> Skill catalog that main.py hands out to the right agents.
    """
    catalog: dict[str, Skill] = {}
    for tool in tools:
        async def handler(args: dict, _name: str = tool.name) -> str:
            result = await session.call_tool(_name, args)
            # MCP tool results are a list of content blocks; collect the text ones.
            return "".join(getattr(block, "text", "") for block in result.content)

        catalog[tool.name] = Skill(
            name=tool.name,
            description=tool.description or "",
            parameters=tool.inputSchema or {"type": "object"},
            handler=handler,
        )
    return catalog
