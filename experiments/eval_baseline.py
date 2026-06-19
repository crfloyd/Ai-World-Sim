"""Baseline evaluation: 100 episodes with freshly-initialised AgentBrain.

No training is performed.  The policy is a random initialisation, so actions
are driven by the random weights — but the action mask ensures only valid
actions are sampled.  This run establishes the V0 baseline: where does an
untrained policy stand before learning anything?

Outputs
-------
experiments/
  baseline_results.json          raw per-episode data
  baseline_report.md             written summary
  plots/
    01_survival_duration.png
    02_episode_reward.png
    03_death_causes.png
    04_hunger_distribution.png
    05_thirst_distribution.png
    06_sleep_frequency.png
    07_home_frequency.png
    08_action_distribution.png
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_world_sim.common.config import DEFAULT_WORLD_CONFIG_PATH, load_config
from ai_world_sim.rl.env import WorldEnv
from ai_world_sim.rl.model import AgentBrain
from ai_world_sim.rl.observations import (
    DRINK, EAT, FORAGE, HUNT, MOVE_EAST, MOVE_NORTH, MOVE_SOUTH,
    MOVE_WEST, NUM_ACTIONS, REST, RETRIEVE_FOOD, SLEEP, STORE_FOOD,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_SEEDS      = 100
MAX_STEPS    = 1000
SEED_OFFSET  = 0   # seeds 0..99 (training range)
OUT_DIR      = Path(__file__).parent
PLOTS_DIR    = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ACTION_NAMES = [
    "move_N", "move_S", "move_E", "move_W",
    "forage", "hunt", "drink", "eat",
    "rest", "sleep", "store_food", "retrieve_food",
]

DEATH_PALETTE = {
    "dehydration": "#4393c3",
    "starvation":  "#d6604d",
    "wolf":        "#b35806",
    "survived":    "#4dac26",
}

# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _obs_to_tensors(obs: dict) -> tuple:
    return (
        torch.from_numpy(obs["local_grid"]).unsqueeze(0),
        torch.from_numpy(obs["self_features"]).unsqueeze(0),
        torch.from_numpy(obs["memory_features"]).unsqueeze(0),
        torch.from_numpy(obs["action_mask"]).unsqueeze(0),
    )


def _death_cause(info: dict, terminated: bool) -> str:
    if not terminated:
        return "survived"
    if info["thirst"] >= 99.9:
        return "dehydration"
    if info["hunger"] >= 99.9:
        return "starvation"
    return "wolf"


def run_episode(brain: AgentBrain, world_cfg: dict, seed: int) -> dict:
    env = WorldEnv(env_config={
        "world_config_override": {
            "world":   world_cfg.get("world", {}),
            "animals": world_cfg.get("animals", {}),
            "memory":  world_cfg.get("memory", {}),
            "agents":  world_cfg.get("agents", {}),
            "rewards": world_cfg.get("rewards", {}),
        },
        "max_steps_per_episode": MAX_STEPS,
        "seed_range": [seed, seed],
    })

    obs, _ = env.reset(seed=seed)

    episode_reward  = 0.0
    action_counts   = Counter()
    hunger_history  = []
    thirst_history  = []
    at_home_ticks   = 0
    sleep_ticks     = 0
    last_info: dict = {}

    terminated = truncated = False

    while not (terminated or truncated):
        grid, self_f, mem_f, mask = _obs_to_tensors(obs)
        with torch.no_grad():
            action_t, _, _ = brain.act(grid, self_f, mem_f, mask, deterministic=False)
        action = int(action_t.item())

        obs, reward, terminated, truncated, last_info = env.step(action)

        episode_reward += reward
        action_counts[action] += 1
        hunger_history.append(last_info["hunger"])
        thirst_history.append(last_info["thirst"])

        if action == SLEEP:
            sleep_ticks += 1
        # is_at_home is self_features[7] in the *returned* obs
        if obs["self_features"][7] > 0.5:
            at_home_ticks += 1

    steps = last_info.get("steps", MAX_STEPS)
    death_cause = _death_cause(last_info, terminated)

    return {
        "seed":             seed,
        "episode_reward":   round(episode_reward, 4),
        "survival_ticks":   steps,
        "death_cause":      death_cause,
        "mean_hunger":      float(np.mean(hunger_history)) if hunger_history else 0.0,
        "mean_thirst":      float(np.mean(thirst_history)) if thirst_history else 0.0,
        "max_hunger":       float(np.max(hunger_history)) if hunger_history else 0.0,
        "max_thirst":       float(np.max(thirst_history)) if thirst_history else 0.0,
        "final_hp":         last_info.get("hp", 0.0),
        "final_hunger":     last_info.get("hunger", 0.0),
        "final_thirst":     last_info.get("thirst", 0.0),
        "sleep_frequency":  sleep_ticks / steps if steps > 0 else 0.0,
        "home_frequency":   at_home_ticks / steps if steps > 0 else 0.0,
        "action_counts":    dict(action_counts),
        "day_reached":      last_info.get("day", 0),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

STYLE = {"color": "#2166ac", "alpha": 0.75, "edgecolor": "white", "linewidth": 0.5}


def _save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(OUT_DIR)}")


def plot_survival_duration(results: list[dict]) -> None:
    ticks = [r["survival_ticks"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ticks, bins=20, **STYLE)
    ax.axvline(np.mean(ticks), color="#d6604d", lw=1.5, linestyle="--",
               label=f"mean {np.mean(ticks):.0f}")
    ax.axvline(np.median(ticks), color="#4dac26", lw=1.5, linestyle=":",
               label=f"median {np.median(ticks):.0f}")
    ax.set_xlabel("Ticks survived")
    ax.set_ylabel("Episodes")
    ax.set_title("Survival Duration (100 seeds, untrained policy)")
    ax.legend()
    _save(fig, "01_survival_duration.png")


def plot_episode_reward(results: list[dict]) -> None:
    rewards = [r["episode_reward"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(rewards, bins=20, **STYLE)
    ax.axvline(np.mean(rewards), color="#d6604d", lw=1.5, linestyle="--",
               label=f"mean {np.mean(rewards):.2f}")
    ax.set_xlabel("Cumulative reward")
    ax.set_ylabel("Episodes")
    ax.set_title("Episode Reward Distribution")
    ax.legend()
    _save(fig, "02_episode_reward.png")


def plot_death_causes(results: list[dict]) -> None:
    counts = Counter(r["death_cause"] for r in results)
    labels  = list(counts.keys())
    values  = list(counts.values())
    colours = [DEATH_PALETTE.get(l, "#888888") for l in labels]
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.0f%%",
        colors=colours, startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(12)
    ax.set_title("Death Cause Breakdown (100 episodes)")
    _save(fig, "03_death_causes.png")


def plot_hunger_distribution(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist([r["mean_hunger"] for r in results], bins=20, **STYLE)
    axes[0].set_xlabel("Mean hunger / episode")
    axes[0].set_ylabel("Episodes")
    axes[0].set_title("Mean Hunger (0=full, 100=starving)")
    axes[1].hist([r["max_hunger"] for r in results], bins=20, **STYLE)
    axes[1].set_xlabel("Peak hunger / episode")
    axes[1].set_title("Peak Hunger Reached")
    fig.suptitle("Hunger Distribution")
    fig.tight_layout()
    _save(fig, "04_hunger_distribution.png")


def plot_thirst_distribution(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist([r["mean_thirst"] for r in results], bins=20, **STYLE)
    axes[0].set_xlabel("Mean thirst / episode")
    axes[0].set_ylabel("Episodes")
    axes[0].set_title("Mean Thirst (0=hydrated, 100=dehydrated)")
    axes[1].hist([r["max_thirst"] for r in results], bins=20, **STYLE)
    axes[1].set_xlabel("Peak thirst / episode")
    axes[1].set_title("Peak Thirst Reached")
    fig.suptitle("Thirst Distribution")
    fig.tight_layout()
    _save(fig, "05_thirst_distribution.png")


def plot_sleep_frequency(results: list[dict]) -> None:
    freqs = [r["sleep_frequency"] * 100 for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(freqs, bins=20, **STYLE)
    ax.axvline(np.mean(freqs), color="#d6604d", lw=1.5, linestyle="--",
               label=f"mean {np.mean(freqs):.1f}%")
    ax.set_xlabel("Sleep action frequency (%)")
    ax.set_ylabel("Episodes")
    ax.set_title("Sleep Frequency (% of ticks using SLEEP action)")
    ax.legend()
    _save(fig, "06_sleep_frequency.png")


def plot_home_frequency(results: list[dict]) -> None:
    freqs = [r["home_frequency"] * 100 for r in results]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(freqs, bins=20, **STYLE)
    ax.axvline(np.mean(freqs), color="#d6604d", lw=1.5, linestyle="--",
               label=f"mean {np.mean(freqs):.1f}%")
    ax.set_xlabel("Ticks at home (%)")
    ax.set_ylabel("Episodes")
    ax.set_title("Home Return Frequency (% of ticks at home position)")
    ax.legend()
    _save(fig, "07_home_frequency.png")


def plot_action_distribution(results: list[dict]) -> None:
    totals = Counter()
    for r in results:
        for k, v in r["action_counts"].items():
            totals[int(k)] += v

    total = sum(totals.values())
    actions = list(range(NUM_ACTIONS))
    pcts = [totals.get(a, 0) / total * 100 for a in actions]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(ACTION_NAMES, pcts, color="#2166ac", alpha=0.8, edgecolor="white")
    ax.set_ylabel("Action share (%)")
    ax.set_xlabel("Action")
    ax.set_title("Action Distribution Across All Episodes")
    ax.tick_params(axis="x", rotation=30)
    for bar, pct in zip(bars, pcts):
        if pct > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{pct:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save(fig, "08_action_distribution.png")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(results: list[dict], elapsed_s: float) -> Path:
    rewards  = [r["episode_reward"]  for r in results]
    ticks    = [r["survival_ticks"]  for r in results]
    m_hunger = [r["mean_hunger"]     for r in results]
    m_thirst = [r["mean_thirst"]     for r in results]
    sleep_f  = [r["sleep_frequency"] * 100 for r in results]
    home_f   = [r["home_frequency"]  * 100 for r in results]
    deaths   = Counter(r["death_cause"] for r in results)
    survived = deaths.get("survived", 0)

    # Action totals
    action_totals = Counter()
    for r in results:
        for k, v in r["action_counts"].items():
            action_totals[int(k)] += v
    total_actions = sum(action_totals.values())
    top_actions = sorted(action_totals.items(), key=lambda x: -x[1])[:5]

    def pct(n): return f"{n}/{len(results)} ({n/len(results)*100:.0f}%)"
    def stat(arr): return f"{np.mean(arr):.2f} ± {np.std(arr):.2f}  [min {np.min(arr):.2f}, max {np.max(arr):.2f}]"

    lines = [
        f"# V0 Baseline Evaluation Report",
        f"",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Policy:** untrained AgentBrain (random initialisation, action mask active)  ",
        f"**Episodes:** {len(results)} seeds (0–{len(results)-1})  ",
        f"**Max steps / episode:** {MAX_STEPS}  ",
        f"**Evaluation time:** {elapsed_s:.0f}s",
        f"",
        f"---",
        f"",
        f"## Survival Summary",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Episodes survived to max_steps | {pct(survived)} |",
        f"| Mean survival ticks | {np.mean(ticks):.1f} ± {np.std(ticks):.1f} |",
        f"| Median survival ticks | {np.median(ticks):.0f} |",
        f"| Min / Max survival | {np.min(ticks):.0f} / {np.max(ticks):.0f} |",
        f"",
        f"## Episode Reward",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Mean reward | {np.mean(rewards):.3f} |",
        f"| Std reward | {np.std(rewards):.3f} |",
        f"| Min / Max reward | {np.min(rewards):.3f} / {np.max(rewards):.3f} |",
        f"",
        f"## Deaths by Cause",
        f"",
        f"| Cause | Count | % |",
        f"|---|---|---|",
    ]
    for cause in ["dehydration", "starvation", "wolf", "survived"]:
        n = deaths.get(cause, 0)
        lines.append(f"| {cause} | {n} | {n/len(results)*100:.0f}% |")

    lines += [
        f"",
        f"## Hunger Distribution",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Mean hunger (per episode mean) | {stat(m_hunger)} |",
        f"| % episodes reaching max hunger | {sum(1 for r in results if r['max_hunger'] >= 99.9)}/{len(results)} |",
        f"",
        f"## Thirst Distribution",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Mean thirst (per episode mean) | {stat(m_thirst)} |",
        f"| % episodes reaching max thirst | {sum(1 for r in results if r['max_thirst'] >= 99.9)}/{len(results)} |",
        f"",
        f"## Behaviour Frequency",
        f"",
        f"| Behaviour | Mean % of ticks | Std |",
        f"|---|---|---|",
        f"| Sleep action used | {np.mean(sleep_f):.1f}% | ±{np.std(sleep_f):.1f}% |",
        f"| Ticks spent at home | {np.mean(home_f):.1f}% | ±{np.std(home_f):.1f}% |",
        f"",
        f"## Action Distribution",
        f"",
        f"| Rank | Action | Share |",
        f"|---|---|---|",
    ]
    for rank, (a_idx, cnt) in enumerate(top_actions, 1):
        lines.append(f"| {rank} | {ACTION_NAMES[a_idx]} | {cnt/total_actions*100:.1f}% |")

    lines += [
        f"",
        f"## Plots",
        f"",
        f"![Survival Duration](plots/01_survival_duration.png)",
        f"![Episode Reward](plots/02_episode_reward.png)",
        f"![Death Causes](plots/03_death_causes.png)",
        f"![Hunger Distribution](plots/04_hunger_distribution.png)",
        f"![Thirst Distribution](plots/05_thirst_distribution.png)",
        f"![Sleep Frequency](plots/06_sleep_frequency.png)",
        f"![Home Frequency](plots/07_home_frequency.png)",
        f"![Action Distribution](plots/08_action_distribution.png)",
        f"",
        f"---",
        f"",
        f"## Interpretation",
        f"",
        f"The untrained policy's action distribution reflects **random exploration constrained",
        f"by the action mask** rather than any learned survival strategy.  Key observations:",
        f"",
        f"- **Survival rate** of {survived}/{len(results)} ({survived/len(results)*100:.0f}%) at {MAX_STEPS} ticks indicates",
        f"  whether random valid actions alone are sufficient for near-term survival.",
        f"- **Dominant death cause** ({deaths.most_common(1)[0][0] if deaths else 'N/A'}) tells us which threat the policy",
        f"  must first learn to manage.",
        f"- **Mean thirst** and **mean hunger** at episode end show how quickly stats",
        f"  saturate under a random policy.",
        f"- **Sleep frequency** and **home frequency** near zero would confirm that",
        f"  safe-rest and home-return behaviours are not yet emergent.",
        f"",
        f"A trained policy should show: reduced deaths by dehydration/starvation,",
        f"higher survival ticks, non-random action distribution (DRINK, EAT, FORAGE",
        f"over-represented relative to their mask availability), and measurable home",
        f"frequency indicating learned navigation.",
    ]

    report_path = OUT_DIR / "baseline_report.md"
    report_path.write_text("\n".join(lines))
    print(f"  saved {report_path.relative_to(OUT_DIR)}")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    world_cfg = load_config(DEFAULT_WORLD_CONFIG_PATH)

    print(f"\nV0 Baseline Evaluation — {N_SEEDS} seeds, untrained policy")
    print(f"  max_steps={MAX_STEPS}  seeds={SEED_OFFSET}..{SEED_OFFSET + N_SEEDS - 1}")
    print()

    torch.manual_seed(0)
    brain = AgentBrain()
    brain.eval()

    results: list[dict] = []
    t_start = time.perf_counter()

    for i in range(N_SEEDS):
        seed = SEED_OFFSET + i
        r = run_episode(brain, world_cfg, seed)
        results.append(r)
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_start
            avg_ticks = np.mean([x["survival_ticks"] for x in results])
            print(f"  {i+1:3d}/{N_SEEDS}  "
                  f"mean_survival={avg_ticks:.0f}t  "
                  f"elapsed={elapsed:.0f}s")

    elapsed = time.perf_counter() - t_start
    print(f"\n  Done in {elapsed:.0f}s")
    print()

    # Save raw results
    json_path = OUT_DIR / "baseline_results.json"
    with json_path.open("w") as fh:
        json.dump({
            "n_seeds": N_SEEDS,
            "max_steps": MAX_STEPS,
            "seed_offset": SEED_OFFSET,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": results,
        }, fh, indent=2)
    print(f"  saved baseline_results.json")

    # Plots
    print()
    plot_survival_duration(results)
    plot_episode_reward(results)
    plot_death_causes(results)
    plot_hunger_distribution(results)
    plot_thirst_distribution(results)
    plot_sleep_frequency(results)
    plot_home_frequency(results)
    plot_action_distribution(results)

    # Report
    print()
    write_report(results, elapsed)

    print("\n  All done.")


if __name__ == "__main__":
    main()
