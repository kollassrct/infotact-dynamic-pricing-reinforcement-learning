"""
compare_week2.py
----------------
Loads baseline and Q-Learning results and prints
a final Week 2 comparison table.

Run: python evaluation/compare_week2.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle

from env.dynamic_pricing_env import DynamicPricingEnv
from env.demand_simulator import Industry, INDUSTRY_NAMES
from agents.baseline_fixed import get_fixed_agents
from agents.baseline_discount import DiscountAgent, DiscountAgentV2
from training.train_qlearning import (
    discretize_obs, select_action, TOTAL_DAYS, SEASON, EVAL_EPISODES
)


def load_q_table(industry_name: str) -> dict:
    path = f"results/models/qtable_{industry_name.lower()}.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["q_table"]


def eval_q_learning(industry: Industry) -> dict:
    """Evaluate saved Q-Learning model."""
    name     = INDUSTRY_NAMES[industry]
    q_table  = load_q_table(name)

    env = DynamicPricingEnv(
        industry   = industry,
        season     = SEASON,
        total_days = TOTAL_DAYS,
    )
    n_actions   = env.n_actions
    revenues    = []
    occupancies = []

    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset(seed=ep + 9999)
        state  = discretize_obs(obs, env)
        done   = False
        while not done:
            action = select_action(q_table, state, epsilon=0.0, n_actions=n_actions)
            obs, _, terminated, truncated, _ = env.step(action)
            state  = discretize_obs(obs, env)
            done   = terminated or truncated

        s = env.summary()
        revenues.append(s["total_revenue"])
        occupancies.append(s["occupancy_pct"])

    return {
        "Agent"        : "Q-Learning",
        "Avg Revenue"  : round(np.mean(revenues),    2),
        "Avg Occupancy": round(np.mean(occupancies), 2),
    }


def eval_baselines(industry: Industry) -> list:
    """Evaluate all baseline agents."""
    env_temp  = DynamicPricingEnv(industry=industry)
    n_actions = env_temp.n_actions

    agents = get_fixed_agents(n_actions) + [
        DiscountAgent(n_actions,  name="Discount-Basic"),
        DiscountAgentV2(n_actions, name="Discount-V2"),
    ]

    results = []
    for agent in agents:
        env = DynamicPricingEnv(
            industry   = industry,
            season     = SEASON,
            total_days = TOTAL_DAYS,
            seed       = 42,
        )
        revenues    = []
        occupancies = []

        for ep in range(EVAL_EPISODES):
            obs, _ = env.reset(seed=ep)
            if hasattr(agent, "reset"):
                try:
                    agent.reset(total_days=TOTAL_DAYS)
                except TypeError:
                    agent.reset()
            done = False
            while not done:
                action = agent.select_action(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            s = env.summary()
            revenues.append(s["total_revenue"])
            occupancies.append(s["occupancy_pct"])

        results.append({
            "Agent"        : agent.name,
            "Avg Revenue"  : round(np.mean(revenues),    2),
            "Avg Occupancy": round(np.mean(occupancies), 2),
        })

    return results


def main():
    print("\n" + "="*65)
    print("   DynamicRL — Week 2 Final Comparison")
    print("="*65)

    all_rows = []

    for industry in Industry:
        name     = INDUSTRY_NAMES[industry]
        baselines = eval_baselines(industry)
        ql        = eval_q_learning(industry)

        print(f"\n📊 {name}")
        print(f"  {'Agent':<20} {'Avg Revenue':>14} {'Occupancy':>12}")
        print("  " + "-"*48)

        best_baseline_rev = 0
        for row in baselines:
            marker = ""
            if row["Avg Revenue"] > best_baseline_rev:
                best_baseline_rev = row["Avg Revenue"]
            print(
                f"  {row['Agent']:<20} "
                f"${row['Avg Revenue']:>12,.0f}   "
                f"{row['Avg Occupancy']:>8.1f}%"
            )
            all_rows.append({"Industry": name, **row})

        # Q-Learning row with comparison marker
        beat = "✅ BEAT" if ql["Avg Revenue"] > best_baseline_rev else "❌ Below"
        print(
            f"  {'Q-Learning':<20} "
            f"${ql['Avg Revenue']:>12,.0f}   "
            f"{ql['Avg Occupancy']:>8.1f}%   {beat} baseline"
        )
        all_rows.append({"Industry": name, **ql})

    # Save full comparison
    df = pd.DataFrame(all_rows)
    df.to_csv("results/week2_comparison.csv", index=False)

    print("\n" + "="*65)
    print("✅ Week 2 comparison complete!")
    print("📁 Saved to results/week2_comparison.csv")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()