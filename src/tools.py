"""
LAYER 1 — TOOLS (an MCP server).

These are the raw capabilities, exposed over the Model Context Protocol. Each @mcp.tool()
becomes a tool any MCP client can discover (tools/list) and invoke (tools/call). This file
is a standalone server: `python tools.py` runs it over stdio. In a real system these would
talk to databases, HTTP APIs, internal services — possibly written in another language.
Nothing here knows about skills, agents, or models.
"""

import ast
import operator
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-tools")

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> float:
    """Evaluate arithmetic safely via the AST — never use eval() on model output."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval").body)


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '(3 + 4) * 2'."""
    return str(_safe_eval(expression))


@mcp.tool()
def get_time() -> str:
    """Get the current date and time in UTC (ISO-8601)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@mcp.tool()
def get_weather(city: str) -> str:
    """Get a (pretend) current weather report for a city."""
    seed = sum(ord(c) for c in city)              # stable, deterministic "data"
    temp = 15 + seed % 16                          # 15–30 °C
    conditions = ["sunny", "partly cloudy", "overcast", "light rain"]
    return f"{city}: {temp}°C, {conditions[seed % len(conditions)]}."


@mcp.tool()
def echo(text: str) -> str:
    """Echo back the given text. Handy for testing the connection."""
    return text


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
