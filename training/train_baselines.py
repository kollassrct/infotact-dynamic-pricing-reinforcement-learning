"""
train_baselines.py
------------------
Runs Fixed and Discount baseline agents across all 4 industries.
Prints a revenue comparison table.

Run: python training/train_baselines.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from env.dynamic_pricing_env import DynamicPricingEnv
from env.demand_simulator import Industry, INDUSTRY_NAMES
from agents.baseline_fixed import get_fixed_agents
from agents.baseline_discount import DiscountAgent, DiscountAgentV2


EPISODES   = 100
TOTAL_DAYS = 30
SEASON     = "normal"


def run_agent(agent, industry: Industry, episodes: int = EPISODES) -> dict:
    """
    Runs one agent for N episodes on one industry.
    Returns averaged metrics.
    """
    env = DynamicPricingEnv(
        industry   = industry,
        season     = SEASON,
        total_days = TOTAL_DAYS,
        seed       = 42,
    )

    revenues    = []
    occupancies = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)

        # Reset agent if it supports it
        if hasattr(agent, "reset"):
            try:
                agent.reset(total_days=TOTAL_DAYS)
            except TypeError:
                agent.reset()

        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        summary = env.summary()
        revenues.append(summary["total_revenue"])
        occupancies.append(summary["occupancy_pct"])

    return {
        "avg_revenue"   : round(np.mean(revenues),    2),
        "std_revenue"   : round(np.std(revenues),     2),
        "avg_occupancy" : round(np.mean(occupancies), 2),
        "min_revenue"   : round(np.min(revenues),     2),
        "max_revenue"   : round(np.max(revenues),     2),
    }


def main():
    print("\n" + "="*65)
    print("   DynamicRL — Baseline Evaluation")
    print(f"   Episodes: {EPISODES} | Days: {TOTAL_DAYS} | Season: {SEASON}")
    print("="*65)

    results = []

    for industry in Industry:
        industry_name = INDUSTRY_NAMES[industry]

        # Build agents for this industry
        env_temp = DynamicPricingEnv(industry=industry)
        n_actions = env_temp.n_actions

        agents = get_fixed_agents(n_actions) + [
            DiscountAgent(n_actions, name="Discount-Basic"),
            DiscountAgentV2(n_actions, name="Discount-V2"),
        ]

        print(f"\n📊 Industry: {industry_name}")
        print("-" * 55)

        for agent in agents:
            metrics = run_agent(agent, industry)
            results.append({
                "Industry"     : industry_name,
                "Agent"        : agent.name,
                "Avg Revenue"  : f"${metrics['avg_revenue']:,.0f}",
                "Std Dev"      : f"${metrics['std_revenue']:,.0f}",
                "Avg Occupancy": f"{metrics['avg_occupancy']:.1f}%",
                "Min Revenue"  : f"${metrics['min_revenue']:,.0f}",
                "Max Revenue"  : f"${metrics['max_revenue']:,.0f}",
            })
            print(
                f"  {agent.name:<20} | "
                f"Revenue: ${metrics['avg_revenue']:>10,.0f} | "
                f"Occupancy: {metrics['avg_occupancy']:>5.1f}%"
            )

    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/baseline_results.csv", index=False)

    print("\n" + "="*65)
    print("✅ Baseline evaluation complete!")
    print("📁 Results saved to results/baseline_results.csv")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()