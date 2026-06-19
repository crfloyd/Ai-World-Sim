"""Baseline evaluation: 100 episodes per predator profile with untrained AgentBrain.

No training is performed.  The policy uses random initialisation, so actions
are driven by random weights constrained by the action mask.  Running across
multiple predator profiles reveals which threats dominate deaths and what the
trained policy will need to learn first.

Profiles
--------
  none    — no wolves; death should be driven by thirst / starvation
  light   — 1 wolf, short range, reduced damage; wolves are a non-dominant threat
  normal  — current world defaults (3 wolves, range 10, 15 dmg)

Usage
-----
    # Run all three profiles and generate a comparison report (default)
    python experiments/eval_baseline.py

    # Run a single profile
    python experiments/eval_baseline.py --predator-profile none
    python experiments/eval_baseline.py --predator-profile light
    python experiments/eval_baseline.py --predator-profile normal

    # Override episode count / length
    python experiments/eval_baseline.py --n-seeds 50 --max-steps 500

Outputs (gitignored, written to experiments/results/)
------------------------------------------------------
experiments/results/
  baseline_results_<profile>.json
  baseline_report.md
  plots/
    01_survival_duration.png
    02_episode_reward.png
    03_death_causes_comparison.png     (all three profiles side-by-side)
    04_hunger_distribution.png
    05_thirst_distribution.png
    06_sleep_frequency.png
    07_home_frequency.png
    08_action_distribution.png
"""

from __future__ import annotations

import argparse
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
# Defaults
# ---------------------------------------------------------------------------

N_SEEDS     = 100
MAX_STEPS   = 1000
SEED_OFFSET = 0

OUT_DIR   = Path(__file__).parent / "results"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ACTION_NAMES = [
    "move_N", "move_S", "move_E", "move_W",
    "forage", "hunt", "drink", "eat",
    "rest", "sleep", "store_food", "retrieve_food",
]

DEATH_ORDER   = ["dehydration", "starvation", "wolf", "survived"]
DEATH_PALETTE = {
    "dehydration": "#4393c3",
    "starvation":  "#d6604d",
    "wolf":        "#b35806",
    "survived":    "#4dac26",
}

PROFILES = ["none", "light", "normal"]

PROFILE_LABEL = {
    "none":   "None  (0 wolves)",
    "light":  "Light (1 wolf, range 5, 8 dmg)",
    "normal": "Normal (3 wolves, range 10, 15 dmg)",
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _build_env_config(world_cfg: dict, profile: str, seed: int) -> dict:
    """Return WorldEnv env_config with the named predator profile active."""
    import copy
    cfg = copy.deepcopy(world_cfg)
    cfg["animals"]["predator_curriculum_phase"] = profile
    return {
        "world_config_override": {
            "world":   cfg.get("world", {}),
            "animals": cfg.get("animals", {}),
            "memory":  cfg.get("memory", {}),
            "agents":  cfg.get("agents", {}),
            "rewards": cfg.get("rewards", {}),
        },
        "max_steps_per_episode": MAX_STEPS,
        "seed_range": [seed, seed],
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


def run_episode(brain: AgentBrain, world_cfg: dict, profile: str, seed: int) -> dict:
    env = WorldEnv(env_config=_build_env_config(world_cfg, profile, seed))
    obs, _ = env.reset(seed=seed)

    episode_reward  = 0.0
    action_counts: Counter = Counter()
    hunger_history: list[float] = []
    thirst_history: list[float] = []
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
        if obs["self_features"][7] > 0.5:
            at_home_ticks += 1

    steps = last_info.get("steps", MAX_STEPS)
    death_cause = _death_cause(last_info, terminated)

    return {
        "seed":            seed,
        "profile":         profile,
        "episode_reward":  round(episode_reward, 4),
        "survival_ticks":  steps,
        "death_cause":     death_cause,
        "mean_hunger":     float(np.mean(hunger_history)) if hunger_history else 0.0,
        "mean_thirst":     float(np.mean(thirst_history)) if thirst_history else 0.0,
        "max_hunger":      float(np.max(hunger_history)) if hunger_history else 0.0,
        "max_thirst":      float(np.max(thirst_history)) if thirst_history else 0.0,
        "final_hp":        last_info.get("hp", 0.0),
        "final_hunger":    last_info.get("hunger", 0.0),
        "final_thirst":    last_info.get("thirst", 0.0),
        "sleep_frequency": sleep_ticks / steps if steps > 0 else 0.0,
        "home_frequency":  at_home_ticks / steps if steps > 0 else 0.0,
        "action_counts":   dict(action_counts),
        "day_reached":     last_info.get("day", 0),
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

STYLE = {"color": "#2166ac", "alpha": 0.75, "edgecolor": "white", "linewidth": 0.5}


def _save(fig: plt.Figure, name: str) -> None:
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved results/plots/{name}")


# ---------------------------------------------------------------------------
# Per-profile plots (run on whichever profiles were evaluated)
# ---------------------------------------------------------------------------

def plot_survival_duration(all_results: dict[str, list[dict]], profiles: list[str]) -> None:
    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4), sharey=True)
    if len(profiles) == 1:
        axes = [axes]
    for ax, p in zip(axes, profiles):
        ticks = [r["survival_ticks"] for r in all_results[p]]
        ax.hist(ticks, bins=20, **STYLE)
        ax.axvline(np.mean(ticks), color="#d6604d", lw=1.5, linestyle="--",
                   label=f"μ={np.mean(ticks):.0f}")
        ax.axvline(np.median(ticks), color="#4dac26", lw=1.5, linestyle=":",
                   label=f"med={np.median(ticks):.0f}")
        ax.set_title(f"Predator: {p}")
        ax.set_xlabel("Ticks survived")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Episodes")
    fig.suptitle("Survival Duration by Predator Profile")
    fig.tight_layout()
    _save(fig, "01_survival_duration.png")


def plot_episode_reward(all_results: dict[str, list[dict]], profiles: list[str]) -> None:
    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4), sharey=True)
    if len(profiles) == 1:
        axes = [axes]
    for ax, p in zip(axes, profiles):
        rewards = [r["episode_reward"] for r in all_results[p]]
        ax.hist(rewards, bins=20, **STYLE)
        ax.axvline(np.mean(rewards), color="#d6604d", lw=1.5, linestyle="--",
                   label=f"μ={np.mean(rewards):.2f}")
        ax.set_title(f"Predator: {p}")
        ax.set_xlabel("Cumulative reward")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Episodes")
    fig.suptitle("Episode Reward by Predator Profile")
    fig.tight_layout()
    _save(fig, "02_episode_reward.png")


def plot_death_causes_comparison(all_results: dict[str, list[dict]], profiles: list[str]) -> None:
    """Grouped bar chart: death causes for each profile side-by-side."""
    causes = DEATH_ORDER
    x = np.arange(len(causes))
    width = 0.8 / len(profiles)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, p in enumerate(profiles):
        counts = Counter(r["death_cause"] for r in all_results[p])
        n = len(all_results[p])
        pcts = [counts.get(c, 0) / n * 100 for c in causes]
        offset = (i - len(profiles) / 2 + 0.5) * width
        bars = ax.bar(x + offset, pcts, width, label=PROFILE_LABEL[p],
                      color=[DEATH_PALETTE[c] for c in causes],
                      alpha=0.85, edgecolor="white")
        for bar, pct in zip(bars, pcts):
            if pct >= 5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(causes)
    ax.set_ylabel("% of episodes")
    ax.set_title("Death Cause Breakdown by Predator Profile")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    _save(fig, "03_death_causes_comparison.png")


def plot_hunger_distribution(all_results: dict[str, list[dict]], profiles: list[str]) -> None:
    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4), sharey=True)
    if len(profiles) == 1:
        axes = [axes]
    for ax, p in zip(axes, profiles):
        ax.hist([r["mean_hunger"] for r in all_results[p]], bins=20, **STYLE)
        ax.set_title(f"Predator: {p}")
        ax.set_xlabel("Mean hunger / episode")
    axes[0].set_ylabel("Episodes")
    fig.suptitle("Mean Hunger Distribution (0=full, 100=starving)")
    fig.tight_layout()
    _save(fig, "04_hunger_distribution.png")


def plot_thirst_distribution(all_results: dict[str, list[dict]], profiles: list[str]) -> None:
    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4), sharey=True)
    if len(profiles) == 1:
        axes = [axes]
    for ax, p in zip(axes, profiles):
        ax.hist([r["mean_thirst"] for r in all_results[p]], bins=20, **STYLE)
        ax.set_title(f"Predator: {p}")
        ax.set_xlabel("Mean thirst / episode")
    axes[0].set_ylabel("Episodes")
    fig.suptitle("Mean Thirst Distribution (0=hydrated, 100=dehydrated)")
    fig.tight_layout()
    _save(fig, "05_thirst_distribution.png")


def plot_sleep_frequency(all_results: dict[str, list[dict]], profiles: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [[r["sleep_frequency"] * 100 for r in all_results[p]] for p in profiles]
    bp = ax.boxplot(data, tick_labels=[PROFILE_LABEL[p] for p in profiles], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#2166ac")
        patch.set_alpha(0.6)
    ax.set_ylabel("Sleep action frequency (%)")
    ax.set_title("Sleep Frequency by Predator Profile")
    ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    _save(fig, "06_sleep_frequency.png")


def plot_home_frequency(all_results: dict[str, list[dict]], profiles: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [[r["home_frequency"] * 100 for r in all_results[p]] for p in profiles]
    bp = ax.boxplot(data, tick_labels=[PROFILE_LABEL[p] for p in profiles], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4dac26")
        patch.set_alpha(0.6)
    ax.set_ylabel("Ticks at home (%)")
    ax.set_title("Home Return Frequency by Predator Profile")
    ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    _save(fig, "07_home_frequency.png")


def plot_action_distribution(all_results: dict[str, list[dict]], profiles: list[str]) -> None:
    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4), sharey=True)
    if len(profiles) == 1:
        axes = [axes]
    for ax, p in zip(axes, profiles):
        totals: Counter = Counter()
        for r in all_results[p]:
            for k, v in r["action_counts"].items():
                totals[int(k)] += v
        total = sum(totals.values())
        pcts = [totals.get(a, 0) / total * 100 for a in range(NUM_ACTIONS)]
        ax.bar(ACTION_NAMES, pcts, color="#2166ac", alpha=0.8, edgecolor="white")
        ax.set_title(f"Predator: {p}")
        ax.set_xlabel("Action")
        ax.tick_params(axis="x", rotation=40)
    axes[0].set_ylabel("Action share (%)")
    fig.suptitle("Action Distribution (untrained policy, by profile)")
    fig.tight_layout()
    _save(fig, "08_action_distribution.png")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_report(
    all_results: dict[str, list[dict]],
    profiles: list[str],
    elapsed_s: float,
    n_seeds: int,
    max_steps: int,
) -> Path:
    def _stat(arr: list[float]) -> str:
        return (
            f"{np.mean(arr):.2f} ± {np.std(arr):.2f}"
            f"  [min {np.min(arr):.2f}, max {np.max(arr):.2f}]"
        )

    lines: list[str] = [
        "# V0 Baseline Evaluation Report",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Policy:** untrained AgentBrain (random initialisation, action mask active)  ",
        f"**Episodes per profile:** {n_seeds}  ",
        f"**Max steps / episode:** {max_steps}  ",
        f"**Profiles evaluated:** {', '.join(profiles)}  ",
        f"**Evaluation time:** {elapsed_s:.0f}s",
        "",
        "---",
        "",
        "## Survival Summary",
        "",
        "| Profile | Mean ticks | Median | Min | Max | Survived (%) |",
        "|---|---|---|---|---|---|",
    ]
    for p in profiles:
        res = all_results[p]
        ticks = [r["survival_ticks"] for r in res]
        surv = sum(1 for r in res if r["death_cause"] == "survived")
        lines.append(
            f"| {p} | {np.mean(ticks):.1f} ± {np.std(ticks):.1f}"
            f" | {np.median(ticks):.0f}"
            f" | {np.min(ticks):.0f} | {np.max(ticks):.0f}"
            f" | {surv}/{n_seeds} ({surv/n_seeds*100:.0f}%) |"
        )

    lines += [
        "",
        "## Episode Reward",
        "",
        "| Profile | Mean | Std | Min | Max |",
        "|---|---|---|---|---|",
    ]
    for p in profiles:
        rewards = [r["episode_reward"] for r in all_results[p]]
        lines.append(
            f"| {p} | {np.mean(rewards):.3f} | {np.std(rewards):.3f}"
            f" | {np.min(rewards):.3f} | {np.max(rewards):.3f} |"
        )

    lines += [
        "",
        "## Deaths by Cause",
        "",
        "| Profile | Dehydration | Starvation | Wolf | Survived |",
        "|---|---|---|---|---|",
    ]
    for p in profiles:
        res = all_results[p]
        counts = Counter(r["death_cause"] for r in res)
        n = len(res)
        row = " | ".join(
            f"{counts.get(c, 0)} ({counts.get(c, 0)/n*100:.0f}%)"
            for c in DEATH_ORDER
        )
        lines.append(f"| {p} | {row} |")

    lines += [
        "",
        "## Hunger Distribution",
        "",
        "| Profile | Mean hunger | Episodes hitting max |",
        "|---|---|---|",
    ]
    for p in profiles:
        res = all_results[p]
        mh = [r["mean_hunger"] for r in res]
        at_max = sum(1 for r in res if r["max_hunger"] >= 99.9)
        lines.append(f"| {p} | {_stat(mh)} | {at_max}/{n_seeds} |")

    lines += [
        "",
        "## Thirst Distribution",
        "",
        "| Profile | Mean thirst | Episodes hitting max |",
        "|---|---|---|",
    ]
    for p in profiles:
        res = all_results[p]
        mt = [r["mean_thirst"] for r in res]
        at_max = sum(1 for r in res if r["max_thirst"] >= 99.9)
        lines.append(f"| {p} | {_stat(mt)} | {at_max}/{n_seeds} |")

    lines += [
        "",
        "## Behaviour Frequencies",
        "",
        "| Profile | Sleep (%) | At home (%) |",
        "|---|---|---|",
    ]
    for p in profiles:
        res = all_results[p]
        sleep_f = [r["sleep_frequency"] * 100 for r in res]
        home_f  = [r["home_frequency"]  * 100 for r in res]
        lines.append(
            f"| {p} | {np.mean(sleep_f):.1f} ± {np.std(sleep_f):.1f}"
            f" | {np.mean(home_f):.1f} ± {np.std(home_f):.1f} |"
        )

    lines += [
        "",
        "## Plots",
        "",
        "![Survival Duration](plots/01_survival_duration.png)",
        "![Episode Reward](plots/02_episode_reward.png)",
        "![Death Causes](plots/03_death_causes_comparison.png)",
        "![Hunger Distribution](plots/04_hunger_distribution.png)",
        "![Thirst Distribution](plots/05_thirst_distribution.png)",
        "![Sleep Frequency](plots/06_sleep_frequency.png)",
        "![Home Frequency](plots/07_home_frequency.png)",
        "![Action Distribution](plots/08_action_distribution.png)",
        "",
        "---",
        "",
        "## Interpretation",
        "",
    ]

    # Build interpretation based on actual data
    if "none" in all_results and "normal" in all_results:
        none_wolf_pct = Counter(r["death_cause"] for r in all_results["none"])
        norm_wolf_pct = Counter(r["death_cause"] for r in all_results["normal"])
        n = n_seeds
        lines += [
            f"**`none` profile**: {none_wolf_pct.get('wolf', 0)}/{n} wolf kills, "
            f"{none_wolf_pct.get('dehydration', 0)}/{n} dehydration, "
            f"{none_wolf_pct.get('starvation', 0)}/{n} starvation. "
            f"Without wolves, the agent dies on its own schedule — "
            f"confirming that thirst (200-tick TTD) is the primary environmental threat.",
            "",
            f"**`normal` profile**: {norm_wolf_pct.get('wolf', 0)}/{n} wolf kills confirm "
            f"wolves dominate episode outcomes, masking thirst/hunger signals completely.  "
            f"The random policy never develops avoidance so mean survival stays near "
            f"{np.mean([r['survival_ticks'] for r in all_results['normal']]):.0f} ticks.",
            "",
        ]

    if "light" in all_results:
        light_counts = Counter(r["death_cause"] for r in all_results["light"])
        n = n_seeds
        light_ticks = np.mean([r["survival_ticks"] for r in all_results["light"]])
        lines += [
            f"**`light` profile**: {light_counts.get('wolf', 0)}/{n} wolf kills, "
            f"{light_counts.get('dehydration', 0)}/{n} dehydration. "
            f"Mean survival {light_ticks:.0f} ticks — wolves are present but "
            f"{'dominant' if light_counts.get('wolf', 0) > n // 2 else 'no longer dominant'}, "
            f"allowing thirst and starvation signals to appear in training data.",
            "",
        ]

    lines += [
        "**Training recommendation**: Start with `predator_curriculum_phase: light` so the",
        "policy sees wolf threat without being overwhelmed.  Once mean survival exceeds",
        "~300 ticks on `light`, advance to `normal` to introduce full ecological pressure.",
        "The `none` profile is useful for isolating hunger/thirst learning without predator",
        "interference during early curriculum phases.",
        "",
        "**What the trained policy must learn (priority order):**",
        "1. `light` / `none`: drink water when thirsty — prevents dehydration deaths",
        "2. `light` / `none`: forage and eat — prevents starvation",
        "3. `light`: avoid or flee single wolves — extends survival significantly",
        "4. `normal`: navigate multi-wolf pressure, use home safety, sleep carefully",
    ]

    report_path = OUT_DIR / "baseline_report.md"
    report_path.write_text("\n".join(lines))
    print(f"  saved results/baseline_report.md")
    return report_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="V0 baseline evaluation across predator profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predator-profile", choices=["none", "light", "normal"], default=None,
        help="Run a single profile. Omit to run all three and generate a comparison report.",
    )
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS,
                        help="Episodes per profile.")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS,
                        help="Max ticks per episode.")
    parser.add_argument("--seed-offset", type=int, default=SEED_OFFSET,
                        help="First seed index.")
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip evaluation; load saved JSON files and regenerate plots/report.")
    args = parser.parse_args()

    run_profiles = [args.predator_profile] if args.predator_profile else PROFILES

    if args.plots_only:
        all_results: dict[str, list[dict]] = {}
        t_total = 0.0
        for profile in run_profiles:
            json_path = OUT_DIR / f"baseline_results_{profile}.json"
            if not json_path.exists():
                print(f"  [ERROR] missing {json_path}; run without --plots-only first")
                raise SystemExit(1)
            with json_path.open() as fh:
                saved = json.load(fh)
            all_results[profile] = saved["results"]
            print(f"  loaded {json_path} ({len(saved['results'])} episodes)")
        print()
        print("  Generating plots …")
        plot_survival_duration(all_results, run_profiles)
        plot_episode_reward(all_results, run_profiles)
        plot_death_causes_comparison(all_results, run_profiles)
        plot_hunger_distribution(all_results, run_profiles)
        plot_thirst_distribution(all_results, run_profiles)
        plot_sleep_frequency(all_results, run_profiles)
        plot_home_frequency(all_results, run_profiles)
        plot_action_distribution(all_results, run_profiles)
        print()
        write_report(all_results, run_profiles, t_total, args.n_seeds, args.max_steps)
        print("\n  All done.")
        return

    world_cfg = load_config(DEFAULT_WORLD_CONFIG_PATH)

    torch.manual_seed(0)
    brain = AgentBrain()
    brain.eval()

    print(f"\nV0 Baseline Evaluation — untrained policy")
    print(f"  profiles: {run_profiles}  n_seeds: {args.n_seeds}  max_steps: {args.max_steps}")
    print()

    all_results: dict[str, list[dict]] = {}
    t_total = time.perf_counter()

    for profile in run_profiles:
        print(f"  Profile: {profile}")
        results: list[dict] = []
        t0 = time.perf_counter()

        for i in range(args.n_seeds):
            seed = args.seed_offset + i
            r = run_episode(brain, world_cfg, profile, seed)
            results.append(r)
            if (i + 1) % 20 == 0:
                avg = np.mean([x["survival_ticks"] for x in results])
                deaths = Counter(x["death_cause"] for x in results)
                print(f"    {i+1:3d}/{args.n_seeds}  "
                      f"mean_survival={avg:.0f}t  "
                      f"wolf={deaths.get('wolf',0)}  "
                      f"dehyd={deaths.get('dehydration',0)}  "
                      f"starv={deaths.get('starvation',0)}  "
                      f"surv={deaths.get('survived',0)}")

        elapsed = time.perf_counter() - t0
        all_results[profile] = results
        deaths = Counter(r["death_cause"] for r in results)
        ticks = [r["survival_ticks"] for r in results]
        print(f"    → mean {np.mean(ticks):.0f}t  "
              f"wolf={deaths.get('wolf',0)}%  "
              f"dehyd={deaths.get('dehydration',0)}%  "
              f"surv={deaths.get('survived',0)}%  "
              f"({elapsed:.0f}s)\n")

        # Save per-profile raw results
        json_path = OUT_DIR / f"baseline_results_{profile}.json"
        with json_path.open("w") as fh:
            json.dump({
                "profile": profile,
                "n_seeds": args.n_seeds,
                "max_steps": args.max_steps,
                "seed_offset": args.seed_offset,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "results": results,
            }, fh, indent=2)
        print(f"  saved results/baseline_results_{profile}.json")
        print()

    elapsed_total = time.perf_counter() - t_total
    print(f"  All profiles done in {elapsed_total:.0f}s total")
    print()

    # Generate plots and report
    print("  Generating plots …")
    plot_survival_duration(all_results, run_profiles)
    plot_episode_reward(all_results, run_profiles)
    plot_death_causes_comparison(all_results, run_profiles)
    plot_hunger_distribution(all_results, run_profiles)
    plot_thirst_distribution(all_results, run_profiles)
    plot_sleep_frequency(all_results, run_profiles)
    plot_home_frequency(all_results, run_profiles)
    plot_action_distribution(all_results, run_profiles)

    print()
    write_report(all_results, run_profiles, elapsed_total, args.n_seeds, args.max_steps)
    print("\n  All done.")


if __name__ == "__main__":
    main()
