"""
LAYER 4 — MAIN (orchestration / wiring).

Brings the layers together:
  tools.py (MCP server)  ->  skills.py (skills that wrap the tools)  ->
  agents.py (specialists + an orchestrator)  ->  run the harness on one goal.

Runs offline with MockChatModel by default. Set ANTHROPIC_API_KEY to use real Claude.
"""

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import skills as skills_module
from agents import Agent, AnthropicChatModel, Harness, MockChatModel, delegate

HERE = os.path.dirname(os.path.abspath(__file__))
GOAL = ("Plan a tiny dinner: figure out the cost of 3 portions at 12.50 each, "
        "tell me the current UTC time, and the weather in Sao Paulo.")


async def main() -> None:
    # 1. Pick the brain: real generative AI if a key is present, else the offline mock.
    model = AnthropicChatModel() if os.getenv("ANTHROPIC_API_KEY") else MockChatModel()
    print(f"Model: {type(model).__name__}")
    print("─" * 60)

    # 2. Launch the MCP tool server (tools.py) as a child process over stdio.
    server = StdioServerParameters(command=sys.executable, args=[os.path.join(HERE, "tools.py")])
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 3. Discover the tools and wrap them as skills.
            tools = (await session.list_tools()).tools
            catalog = skills_module.build_skills(session, tools)
            print("Skills (wrapping MCP tools):", ", ".join(catalog))
            print("─" * 60)

            harness = Harness(max_steps=8)

            # 4. Two SPECIALIST agents, each given a subset of skills.
            mathematician = Agent(
                name="Mathematician",
                system_prompt="You do arithmetic. Use the calculate skill, then state the result plainly.",
                skills=[catalog["calculate"]],
                model=model,
            )
            researcher = Agent(
                name="Researcher",
                system_prompt="You look things up. Use get_time for the time and get_weather for weather.",
                skills=[catalog["get_time"], catalog["get_weather"]],
                model=model,
            )

            # 5. The ORCHESTRATOR: its only skills are "ask a specialist" (other agents).
            orchestrator = Agent(
                name="Orchestrator",
                system_prompt=("You coordinate specialists to fulfil the user's request. "
                               "Delegate math to the mathematician and lookups to the researcher, "
                               "then summarize."),
                skills=[
                    delegate("ask_mathematician", "Ask the math specialist to compute something.",
                             mathematician, harness),
                    delegate("ask_researcher", "Ask the research specialist to look something up.",
                             researcher, harness),
                ],
                model=model,
            )

            # 6. Run it.
            answer = await harness.run(orchestrator, GOAL)

    print("─" * 60)
    print("FINAL ANSWER:\n" + answer)


if __name__ == "__main__":
    asyncio.run(main())
