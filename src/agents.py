"""
LAYER 3 — AGENTS (the runtime that USES the skills).

This file holds everything needed to *run* an agent:
  - the conversation types (Message, ToolCall),
  - the model behind one tiny interface (ChatModel) with two implementations:
        MockChatModel     — deterministic, offline, no API key (the default),
        AnthropicChatModel — real generative AI (Claude); this is where "gen AI"
                             actually plugs in,
  - the Agent (plain config: a role + skills + a model),
  - the Harness (the loop that drives an agent — the heart of the system),
  - delegate(): wrap a whole sub-agent as a single Skill (this is what makes it MULTI-AGENT).
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field

from skills import Skill


# --------------------------------------------------------------------------- conversation
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: str                                   # "system" | "user" | "assistant" | "tool"
    text: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None             # set on tool messages
    tool_name: str | None = None                # set on tool messages


# --------------------------------------------------------------------------- the agent
@dataclass
class Agent:
    name: str
    system_prompt: str
    skills: list[Skill]
    model: "ChatModel"


# --------------------------------------------------------------------------- the harness
class Harness:
    """
    THE HARNESS — the loop that turns a model's single steps into useful work:
        1. ask the model what to do given the conversation + the agent's skills,
        2. if it asked for tools, RUN each skill and append the results,
        3. loop so the model can react,
        4. if it returned plain text instead, that's the answer — stop,
        5. stop anyway after max_steps.
    The harness has no idea what any skill does — including that a skill might run
    ANOTHER agent. That ignorance is exactly why the design stays small.
    """

    def __init__(self, max_steps: int = 8, log=print):
        self.max_steps = max_steps
        self.log = log

    async def run(self, agent: Agent, goal: str) -> str:
        messages = [Message("system", agent.system_prompt), Message("user", goal)]
        self.log(f"┌─ {agent.name} received goal: {goal}")

        for _ in range(self.max_steps):
            reply = await agent.model.complete(messages, agent.skills)
            messages.append(reply)

            if not reply.tool_calls:                       # plain text => final answer
                self.log(f"└─ {agent.name} answered: {_short(reply.text)}")
                return reply.text or ""

            for call in reply.tool_calls:                  # run each requested skill
                skill = next((s for s in agent.skills if s.name == call.name), None)
                if skill is None:
                    result = f"ERROR: unknown skill '{call.name}'"
                else:
                    try:
                        result = await skill.handler(call.arguments)
                    except Exception as exc:               # let the model see + recover
                        result = f"ERROR running '{call.name}': {exc}"
                self.log(f"│  {agent.name} → {call.name}({call.arguments}) ⇒ {_short(result)}")
                messages.append(Message("tool", result, tool_call_id=call.id, tool_name=call.name))

        self.log(f"└─ {agent.name} hit the {self.max_steps}-step limit.")
        return "(stopped: reached the maximum number of steps)"


def delegate(skill_name: str, description: str, sub_agent: Agent, harness: Harness) -> Skill:
    """
    Wrap a sub-agent as a Skill. When a parent agent 'calls' this skill, the harness simply
    runs the sub-agent on the given task and returns its answer as the tool result. To the
    parent, talking to a specialist agent looks exactly like calling any other tool.
    """
    async def handler(args: dict) -> str:
        return await harness.run(sub_agent, args.get("task", ""))

    return Skill(
        name=skill_name,
        description=description,
        parameters={
            "type": "object",
            "properties": {"task": {"type": "string", "description": "What to ask the specialist."}},
            "required": ["task"],
        },
        handler=handler,
    )


# --------------------------------------------------------------------------- the model seam
class ChatModel:
    """The only thing an agent needs from an LLM: given the conversation and the allowed
    skills, return the next assistant Message (either tool_calls or final text)."""

    async def complete(self, messages: list[Message], skills: list[Skill]) -> Message:
        raise NotImplementedError


class MockChatModel(ChatModel):
    """
    A deterministic stand-in so the sample runs offline with no API key. It cannot reason;
    it follows a tiny hand-written policy keyed on which skills the calling agent has and
    which tool results already exist. Swap in AnthropicChatModel for real decisions.
    """

    def __init__(self):
        self._n = 0

    async def complete(self, messages, skills) -> Message:
        names = {s.name for s in skills}
        if "ask_mathematician" in names:
            return self._orchestrate(messages)
        if "calculate" in names:
            return self._math(messages)
        if "get_time" in names:
            return self._research(messages, names)
        return Message("assistant", "I have no tools, so here is a plain answer.")

    def _orchestrate(self, messages) -> Message:
        if not _has(messages, "ask_mathematician"):
            return self._call("ask_mathematician", {"task": "Calculate 3 * 12.50"})
        if not _has(messages, "ask_researcher"):
            return self._call("ask_researcher",
                              {"task": "Report the current UTC time, then the weather in Sao Paulo."})
        cost = _last(messages, "ask_mathematician")
        research = _last(messages, "ask_researcher")
        return Message("assistant", f"Here is your dinner plan:\n• Cost: {cost}\n• {research}")

    def _math(self, messages) -> Message:
        if not _has(messages, "calculate"):
            match = re.search(r"[\d.]+(?:\s*[-+*/]\s*[\d.]+)+", _first_user(messages))
            return self._call("calculate", {"expression": match.group(0) if match else "0"})
        return Message("assistant", f"The total is {_last(messages, 'calculate')}.")

    def _research(self, messages, names) -> Message:
        if not _has(messages, "get_time"):
            return self._call("get_time", {})
        if "get_weather" in names and not _has(messages, "get_weather"):
            match = re.search(r"weather in ([^.,;]+)", _first_user(messages), re.IGNORECASE)
            return self._call("get_weather", {"city": match.group(1).strip() if match else "Sao Paulo"})
        time = _last(messages, "get_time")
        weather = f" Weather — {_last(messages, 'get_weather')}" if "get_weather" in names else ""
        return Message("assistant", f"Current UTC time is {time}.{weather}")

    def _call(self, name: str, args: dict) -> Message:
        self._n += 1
        return Message("assistant", tool_calls=[ToolCall(f"call_{self._n}", name, args)])


class AnthropicChatModel(ChatModel):
    """
    Real generative AI: Claude via the Anthropic Messages API. Used automatically when
    ANTHROPIC_API_KEY is set. Unlike the mock, THIS model genuinely decides which skills to
    call and when it is done — the agents, skills, harness, tools, and MCP code do not change
    at all. That is the entire point of the ChatModel seam.
    """

    def __init__(self, model: str | None = None):
        import anthropic                                    # lazy: only needed for real runs
        self._client = anthropic.Anthropic()                # reads ANTHROPIC_API_KEY
        self._model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    async def complete(self, messages, skills) -> Message:
        system = "\n".join(m.text for m in messages if m.role == "system" and m.text)
        api_messages = self._to_api(messages)
        tools = [{"name": s.name, "description": s.description, "input_schema": s.parameters}
                 for s in skills]

        # The SDK call is synchronous; run it off the event loop.
        resp = await asyncio.to_thread(
            self._client.messages.create,
            model=self._model, max_tokens=1024, system=system,
            messages=api_messages, tools=tools,
        )

        text_parts, tool_calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(block.id, block.name, block.input))

        if tool_calls:
            return Message("assistant", "".join(text_parts) or None, tool_calls=tool_calls)
        return Message("assistant", "".join(text_parts))

    @staticmethod
    def _to_api(messages) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m.role == "user":
                out.append({"role": "user", "content": m.text or ""})
            elif m.role == "assistant":
                blocks = []
                if m.text:
                    blocks.append({"type": "text", "text": m.text})
                for c in (m.tool_calls or []):
                    blocks.append({"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments})
                out.append({"role": "assistant", "content": blocks})
            elif m.role == "tool":
                out.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.text or ""}
                ]})
            # system messages are passed separately via the top-level `system` field
        return out


# --------------------------------------------------------------------------- helpers
def _has(messages, tool) -> bool:
    return any(m.role == "tool" and m.tool_name == tool for m in messages)


def _last(messages, tool) -> str:
    for m in reversed(messages):
        if m.role == "tool" and m.tool_name == tool:
            return m.text or ""
    return ""


def _first_user(messages) -> str:
    return next((m.text for m in messages if m.role == "user"), "") or ""


def _short(s: str | None, n: int = 120) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + "…"
