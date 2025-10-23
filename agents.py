"""
Crash ADK — Agent Definitions (Gemini, OpenAI, Anthropic)

This module defines six specialized agents plus a Coordinator (Crash Assistant)
using Google ADK + LiteLLM. It assumes your analysis tools are implemented in
`tools/crash_data_tools.py` with the function signatures/documentation we aligned on.

Requirements (install in your environment / notebook once):
  pip install google-adk litellm

Environment:
  - GOOGLE_API_KEY           (for Gemini)
  - OPENAI_API_KEY          (for OpenAI)
  - ANTHROPIC_API_KEY       (for Anthropic)
  - export GOOGLE_GENAI_USE_VERTEXAI=False  # using direct API keys

Project layout assumption:
crash_adk_project/
  ├─ tools/
  │   └─ crash_data_tools.py   # your tool implementations live here
  ├─ agents/
  │   └─ crash_adk_agents.py   # <— this file
  └─ ...
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# If you organize tools as a package, update the import path accordingly.
try:
    from tools.crash_data_tools import (
        get_crash_summary,
        get_external_condition,
        get_influence,
        get_travel_direction,
        get_interaction,
        get_time_based_crash_trends,
        get_location,
        generate_crash_report,
    )
except Exception as e:
    # Fallback for flat-file testing (same dir). Adjust as needed.
    from crash_data_tools import (
        get_crash_summary,
        get_external_condition,
        get_influence,
        get_travel_direction,
        get_interaction,
        get_time_based_crash_trends,
        get_location,
        generate_crash_report,
    )

# ------------------------------
# Model configuration
# ------------------------------
# You can freely swap models per agent. LiteLlm will route to provider
# using the model name prefix and the relevant API key from env.
MODEL_GEMINI_2_0_FLASH = "gemini-2.0-flash"
MODEL_GPT_4O = "openai/gpt-4.1"
MODEL_CLAUDE_SONNET = "anthropic/claude-sonnet-4-20250514"

# A simple helper to create a LiteLlm-backed model for an agent.
# You may create different ones per agent if you want provider diversity.

def make_model(model_name: str) -> LiteLlm:
    """Return a LiteLlm model wrapper for ADK Agents.

    LiteLlm looks for OPENAI_API_KEY / ANTHROPIC_API_KEY / etc. in env.
    For Gemini, set GOOGLE_API_KEY and ensure GOOGLE_GENAI_USE_VERTEXAI=False.
    """
    return LiteLlm(model=model_name)


# ------------------------------
# Agent definitions (specialists)
# ------------------------------

def build_crash_data_analyst_agent() -> Agent:
    """Crash Data Analyst Agent

    Focus: overall numeric summaries and multi-facet breakdowns.
    Tools: summary, time trends, direction, interaction.
    """
    return Agent(
        name="crash_data_analyst_v1",
        model=make_model(MODEL_GEMINI_2_0_FLASH),
        description=(
            "Analyzes crash records end-to-end: summaries, time patterns, "
            "driver/vehicle interactions, and directional effects."
        ),
        instruction=(
            "You are a meticulous crash data analyst. Always use tools to fetch "
            "facts. Prefer concise, quantitative summaries with clear caveats. "
            "When counts are small, note possible noise."
        ),
        tools=[
            get_crash_summary,
            get_time_based_crash_trends,
            get_travel_direction,
            get_interaction,
        ],
    )


def build_condition_analyst_agent() -> Agent:
    """Condition Analyst Agent

    Focus: environmental factors — weather, light, surface, risk level.
    """
    return Agent(
        name="condition_analyst_v1",
        model=make_model(MODEL_GEMINI_2_0_FLASH),
        description=(
            "Specialist in environmental conditions and roadway surface/light effects."
        ),
        instruction=(
            "Analyze how weather, light, surface, and environmental risk relate to "
            "crashes. Use distributions and rate metrics (injuries/fatalities per 100)."
        ),
        tools=[
            get_external_condition,
            get_crash_summary,  # for quick context
        ],
    )


def build_behavior_analyst_agent() -> Agent:
    """Behavior Analyst Agent

    Focus: driver influence (alcohol/drug) and violations/actions.
    """
    return Agent(
        name="behavior_analyst_v1",
        model=make_model(MODEL_CLAUDE_SONNET),
        description=(
            "Specialist in driver behavior: alcohol/drug influence, unit actions, and violations."
        ),
        instruction=(
            "Investigate human factors. Normalize raw flags where needed. "
            "Report joint combinations when relevant."
        ),
        tools=[
            get_influence,
            get_interaction,
        ],
    )


def build_spatial_analyst_agent() -> Agent:
    """Spatial Analyst Agent

    Focus: locations, junctions, hotspots, and travel directions.
    """
    return Agent(
        name="spatial_analyst_v1",
        model=make_model(MODEL_GPT_4O),
        description=(
            "GIS-oriented analyst identifying hotspots and junction/direction patterns."
        ),
        instruction=(
            "Focus on spatial signals: hotspots by street/cross street, junction relations, "
            "and direction pair effects. Return concise ranked lists."
        ),
        tools=[
            get_location,
            get_travel_direction,
        ],
    )


def build_temporal_analyst_agent() -> Agent:
    """Temporal Analyst Agent

    Focus: hourly/daily/monthly/yearly trends and peaks; light overlays.
    """
    return Agent(
        name="temporal_analyst_v1",
        model=make_model(MODEL_GEMINI_2_0_FLASH),
        description=(
            "Time-series specialist analyzing crash timing, periodicity, and peaks."
        ),
        instruction=(
            "Aggregate by requested time unit. Highlight peaks and hypothesize plausible "
            "drivers (e.g., commuting hours, nightlife)."
        ),
        tools=[
            get_time_based_crash_trends,
            get_crash_summary,
        ],
    )


def build_crash_report_agent() -> Agent:
    """Crash Report Agent

    Focus: synthesize quantitative outputs into stakeholder-friendly reports.
    """
    return Agent(
        name="crash_report_agent_v1",
        model=make_model(MODEL_CLAUDE_SONNET),
        description=(
            "Report writer that compiles key metrics and drivers into readable summaries."
        ),
        instruction=(
            "Write concise, decision-focused reports. Include headline KPIs, risk drivers, "
            "temporal peaks, and top locations. Note data limitations where applicable."
        ),
        tools=[
            generate_crash_report,
            # May call other tools directly if you want broader access:
            get_crash_summary,
            get_external_condition,
            get_influence,
            get_time_based_crash_trends,
            get_location,
        ],
    )


# ------------------------------
# Coordinator (Crash Assistant)
# ------------------------------
class CrashCoordinator:
    """Lightweight coordinator that routes queries to a specialized agent.

    This is a pragmatic, explicit router. If you prefer an LLM-based router,
    you can also instantiate an Agent with access to *no* tools and let it
    decide which specialist to call via an external controller.
    """

    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents

    def route(self, user_query: str) -> str:
        q = user_query.lower()
        # Very simple keyword routing. Adjust to your needs or replace with LLM routing.
        if any(k in q for k in ["weather", "light", "surface", "environmental"]):
            return "condition_analyst_v1"
        if any(k in q for k in ["alcohol", "drug", "violation", "influence", "behavior"]):
            return "behavior_analyst_v1"
        if any(k in q for k in ["intersection", "street", "junction", "location", "hotspot", "latitude", "longitude"]):
            return "spatial_analyst_v1"
        if any(k in q for k in ["hour", "day", "month", "year", "time", "trend", "peak"]):
            return "temporal_analyst_v1"
        if any(k in q for k in ["report", "summary", "narrative", "write", "kpi"]):
            return "crash_report_agent_v1"
        # Default generalist
        return "crash_data_analyst_v1"

    def list_agents(self) -> List[str]:
        return list(self.agents.keys())


def build_agents_bundle() -> Dict[str, Agent]:
    """Create all specialized agents and return as a name→Agent dict."""
    return {
        "crash_data_analyst_v1": build_crash_data_analyst_agent(),
        "condition_analyst_v1": build_condition_analyst_agent(),
        "behavior_analyst_v1": build_behavior_analyst_agent(),
        "spatial_analyst_v1": build_spatial_analyst_agent(),
        "temporal_analyst_v1": build_temporal_analyst_agent(),
        "crash_report_agent_v1": build_crash_report_agent(),
    }


# ------------------------------
# Optional: convenience factory for coordinator
# ------------------------------

def build_coordinator() -> CrashCoordinator:
    agents = build_agents_bundle()
    return CrashCoordinator(agents=agents)


# ------------------------------
# Example: minimal usage with ADK Runner (synchronous wrapper)
# ------------------------------
if __name__ == "__main__":
    # This block demonstrates basic routing; integrating with ADK Runner and
    # session service typically happens in your app/notebook. For example:
    #
    # from google.adk.sessions import InMemorySessionService
    # from google.adk.runners import Runner
    # session = InMemorySessionService()
    # runner = Runner(session_service=session, agent=selected_agent)
    # async for event in runner.run_async(...): ...

    bundle = build_agents_bundle()
    coord = build_coordinator()

    demo_queries = [
        "Which weather conditions are riskiest at night?",
        "Top violations leading to injuries?",
        "Where are the worst intersections?",
        "Show hourly crash trends for last year",
        "Generate a two-paragraph crash report",
        "Overall summary of crashes in the dataset",
    ]

    for q in demo_queries:
        routed = coord.route(q)
        print(f"Query: {q}\n → Routed to: {routed}\n")
