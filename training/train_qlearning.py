"""
train_qlearning.py
------------------
Trains a Q-Learning agent on DynamicPricingEnv.

Run: python training/train_qlearning.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle

from env.dynamic_pricing_env import DynamicPricingEnv
from env.demand_simulator import Industry, INDUSTRY_NAMES


# ── Hyperparameters ──────────────────────────────────────────
EPISODES      = 3000
TOTAL_DAYS    = 30
SEASON        = "normal"
ALPHA         = 0.1       # Learning rate
GAMMA         = 0.95      # Discount factor
EPSILON_START = 1.0       # Start: 100% random exploration
EPSILON_END   = 0.05      # End:   5% random exploration
EPSILON_DECAY = 0.995     # Decay per episode
EVAL_EPISODES = 100       # Episodes for final evaluation
# ─────────────────────────────────────────────────────────────


def discretize_obs(obs, env) -> tuple:
    """
    Convert continuous observation into a discrete Q-table key.

    Bins each dimension into a small number of buckets.
    """
    inventory    = int(obs[0])
    days_left    = int(obs[1])
    demand_level = int(obs[2])   # already 0/1/2
    comp_bucket  = int(obs[3])   # already 0-4
    industry     = int(obs[4])   # already 0-3
    season       = int(obs[5])   # already 0-2

    max_inv = env.cfg["max_inventory"]

    # Bin inventory into 5 levels: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
    inv_bin = min(4, int(inventory / max_inv * 5))

    # Bin days into 4 levels: last week, 2nd week, 3rd week, beyond
    if days_left <= 7:
        day_bin = 0
    elif days_left <= 14:
        day_bin = 1
    elif days_left <= 21:
        day_bin = 2
    else:
        day_bin = 3

    return (inv_bin, day_bin, demand_level, comp_bucket, industry, season)


def make_q_table(env) -> dict:
    """Initialize Q-table as a dictionary with zeros."""
    return {}


def get_q_value(q_table: dict, state: tuple, action: int, n_actions: int) -> float:
    """Get Q-value, defaulting to 0 for unseen states."""
    return q_table.get((state, action), 0.0)


def update_q_table(
    q_table:  dict,
    state:    tuple,
    action:   int,
    reward:   float,
    next_state: tuple,
    done:     bool,
    n_actions: int,
) -> dict:
    """Bellman equation update."""
    current_q = get_q_value(q_table, state, action, n_actions)

    if done:
        target = reward
    else:
        next_q_values = [get_q_value(q_table, next_state, a, n_actions) for a in range(n_actions)]
        target = reward + GAMMA * max(next_q_values)

    new_q = current_q + ALPHA * (target - current_q)
    q_table[(state, action)] = new_q
    return q_table


def select_action(q_table: dict, state: tuple, epsilon: float, n_actions: int) -> int:
    """ε-greedy action selection."""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)   # Explore
    else:
        q_values = [get_q_value(q_table, state, a, n_actions) for a in range(n_actions)]
        return int(np.argmax(q_values))       # Exploit


def train(industry: Industry) -> tuple:
    """
    Train Q-Learning agent on one industry.
    Returns (q_table, training_rewards, eval_metrics)
    """
    env = DynamicPricingEnv(
        industry   = industry,
        season     = SEASON,
        total_days = TOTAL_DAYS,
    )
    n_actions       = env.n_actions
    q_table         = make_q_table(env)
    epsilon         = EPSILON_START
    training_rewards= []

    print(f"\n🚀 Training Q-Learning on {INDUSTRY_NAMES[industry]}...")
    print(f"   Episodes: {EPISODES} | α={ALPHA} | γ={GAMMA} | ε: {EPSILON_START}→{EPSILON_END}")
    print("   " + "-"*45)

    for ep in range(EPISODES):
        obs, _  = env.reset(seed=ep)
        state   = discretize_obs(obs, env)
        ep_reward = 0.0
        done    = False

        while not done:
            action          = select_action(q_table, state, epsilon, n_actions)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_state      = discretize_obs(next_obs, env)
            done            = terminated or truncated

            q_table = update_q_table(
                q_table, state, action, reward, next_state, done, n_actions
            )

            state      = next_state
            ep_reward += reward

        training_rewards.append(ep_reward)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Print progress every 500 episodes
        if (ep + 1) % 500 == 0:
            avg_r = np.mean(training_rewards[-500:])
            print(f"   Episode {ep+1:>4}/{EPISODES} | Avg Reward: ${avg_r:>10,.0f} | ε: {epsilon:.3f}")

    # ── Evaluation ───────────────────────────────────────────
    print(f"\n📊 Evaluating Q-Learning on {INDUSTRY_NAMES[industry]}...")
    revenues    = []
    occupancies = []

    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset(seed=ep + 10000)
        state  = discretize_obs(obs, env)
        done   = False

        while not done:
            action = select_action(q_table, state, epsilon=0.0, n_actions=n_actions)
            obs, _, terminated, truncated, _ = env.step(action)
            state  = discretize_obs(obs, env)
            done   = terminated or truncated

        summary = env.summary()
        revenues.append(summary["total_revenue"])
        occupancies.append(summary["occupancy_pct"])

    eval_metrics = {
        "agent"         : "Q-Learning",
        "industry"      : INDUSTRY_NAMES[industry],
        "avg_revenue"   : round(np.mean(revenues),    2),
        "std_revenue"   : round(np.std(revenues),     2),
        "avg_occupancy" : round(np.mean(occupancies), 2),
        "min_revenue"   : round(np.min(revenues),     2),
        "max_revenue"   : round(np.max(revenues),     2),
        "q_table_size"  : len(q_table),
    }

    print(
        f"   Avg Revenue : ${eval_metrics['avg_revenue']:>10,.0f}"
        f" ± ${eval_metrics['std_revenue']:,.0f}"
    )
    print(f"   Avg Occupancy: {eval_metrics['avg_occupancy']:.1f}%")
    print(f"   Q-table size : {eval_metrics['q_table_size']:,} entries")

    return q_table, training_rewards, eval_metrics


def main():
    print("\n" + "="*60)
    print("   DynamicRL — Q-Learning Training")
    print("="*60)

    os.makedirs("results/models", exist_ok=True)

    all_metrics = []

    for industry in Industry:
        q_table, rewards, metrics = train(industry)
        all_metrics.append(metrics)

        # Save Q-table
        model_path = f"results/models/qtable_{INDUSTRY_NAMES[industry].lower()}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"q_table": q_table, "rewards": rewards}, f)

        # Save reward curve
        rewards_path = f"results/qtrain_rewards_{INDUSTRY_NAMES[industry].lower()}.csv"
        pd.DataFrame({"episode": range(len(rewards)), "reward": rewards}).to_csv(
            rewards_path, index=False
        )

    # Summary table
    print("\n" + "="*60)
    print("   Q-Learning Results Summary")
    print("="*60)
    print(f"  {'Industry':<12} {'Avg Revenue':>14} {'Occupancy':>12}")
    print("  " + "-"*42)
    for m in all_metrics:
        print(f"  {m['industry']:<12} ${m['avg_revenue']:>12,.0f}   {m['avg_occupancy']:>8.1f}%")

    print("\n✅ Q-Learning training complete!")
    print("📁 Models saved to results/models/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()