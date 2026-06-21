# Multi-agent system in Python — layered by file

Four files, one layer each. Each layer only knows about the one below it.

```
tools.py    LAYER 1  raw capabilities, exposed as an MCP server
   ▲
skills.py   LAYER 2  skills that wrap the tools (the model-facing surface)
   ▲
agents.py   LAYER 3  the runtime: model (mock or real gen AI), Agent, Harness, delegate
   ▲
main.py     LAYER 4  orchestration: connect MCP, build skills, build agents, run
```

## Run it

```bash
# Docker (no config needed — uses the offline mock model):
docker build -t multiagent-py .
docker run --rm multiagent-py

# ...or locally:
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Use real generative AI (Claude) instead of the mock:

```bash
docker run --rm -e ANTHROPIC_API_KEY=sk-ant-... \
  -e ANTHROPIC_MODEL=claude-sonnet-4-6 multiagent-py
```

## The layers

**`tools.py` — tools (MCP).** Each `@mcp.tool()` (`calculate`, `get_time`, `get_weather`,
`echo`) is a raw capability published over the Model Context Protocol. `python tools.py`
runs it as a standalone stdio server. This file knows nothing about skills or agents.

**`skills.py` — skills that use the tools.** `build_skills(session, tools)` wraps each
discovered MCP tool as a `Skill` (name + description + JSON schema + async handler). The
handler performs a `tools/call` over the open MCP session. Agents only ever see Skills —
never the MCP session directly.

**`agents.py` — agents that use the skills.** The runtime:
- `ChatModel` — the one interface a model must satisfy. `MockChatModel` drives the demo
  offline; `AnthropicChatModel` is real generative AI (this is where the "brain" plugs in,
  and swapping it changes nothing else).
- `Agent` — plain config: a name, a system prompt (role), its skills, and its model.
- `Harness` — the loop: ask the model → run any skills it requested → feed results back →
  repeat until it returns plain text. This is the core of the whole system.
- `delegate(...)` — wraps a whole sub-agent as a single Skill. Calling it runs that agent on
  the same harness, so **agents call agents** with no special machinery. This is the
  multi-agent mechanism.

**`main.py` — orchestration.** Opens the MCP connection to `tools.py`, builds the skill
catalog, gives the mathematician the `calculate` skill and the researcher the `get_time` /
`get_weather` skills, then builds an orchestrator whose only skills are "ask a specialist".
One `harness.run(orchestrator, goal)` drives the whole thing.

## What one run does

The orchestrator delegates the cost to the mathematician (which calls `calculate` over MCP),
delegates time + weather to the researcher (which calls `get_time` and `get_weather` over
MCP), then composes the final answer — each sub-agent running its own nested harness loop.

## Where gen AI fits

The generative model is the brain behind `ChatModel`. With `MockChatModel` every decision is
hardcoded so the demo is deterministic and free. Set `ANTHROPIC_API_KEY` and
`AnthropicChatModel` takes over: the model reads each skill's description + schema and decides
for itself which tool or specialist to use and when it's done. The tools, skills, harness, and
MCP code are unchanged — only the brain swaps. To use a different provider (OpenAI, a local
model via Ollama, etc.), add one more `ChatModel` subclass.
