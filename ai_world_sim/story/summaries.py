"""Episode summary generation from the event log.

Produces human-readable text summaries of what happened in an episode.
This is the hook for future LLM-based story narration.

TODO: Feed summary + full event log to Claude API to generate narrative.
TODO: Track cross-episode statistics (e.g. average lifespan trend over training).
TODO: Identify recurring patterns in agent behavior for emergent behavior reports.
"""

from __future__ import annotations

from ai_world_sim.story.events import EventLog
from ai_world_sim.world.entities import Agent
from ai_world_sim.world.sim import WorldSim


def episode_summary(
    world: WorldSim,
    agent: Agent,
    log: EventLog,
    max_events: int = 20,
) -> str:
    """Return a short text summary of the episode."""
    lines: list[str] = [
        "=== Episode Summary ===",
        f"Seed:       {world.seed}",
        f"World size: {world.width}×{world.height}",
        f"Survived:   {agent.alive}",
        f"Days lived: {world.day}",
        f"Season:     {world.season.label()}",
        f"Age:        {agent.age} ticks",
        f"HP:         {agent.hp:.1f}",
        f"Hunger:     {agent.hunger:.1f}",
        f"Fatigue:    {agent.fatigue:.1f}",
        f"Inventory:  {dict(agent.inventory)}",
        f"Events:     {len(log)} total",
    ]

    notable = _select_notable_events(log, max_events)
    if notable:
        lines.append("\n--- Notable Events ---")
        lines.extend(f"  {e}" for e in notable)

    return "\n".join(lines)


def _select_notable_events(log: EventLog, n: int) -> list[str]:
    """Return up to *n* noteworthy event strings from the log."""
    keywords = ["died", "season changed", "foraged", "gathered", "moved"]
    seen: list[str] = []
    for event in log.events:
        msg_lower = event.message.lower()
        if any(kw in msg_lower for kw in keywords):
            seen.append(str(event))
    # Sample evenly across the episode rather than just the tail.
    if len(seen) <= n:
        return seen
    step = len(seen) // n
    return [seen[i] for i in range(0, len(seen), step)][:n]


def season_report(log: EventLog) -> str:
    """Summarise season-change events from the log."""
    changes = log.filter_containing("season changed")
    if not changes:
        return "No season changes recorded."
    return "Season changes:\n" + "\n".join(f"  {e}" for e in changes)
