<div align="center">

# ✈️ RevenueRL

### Reinforcement Learning for Airline Dynamic Pricing

*An autonomous pricing agent that learns to maximize airline revenue through trial-and-error — benchmarked against real airline industry formulas.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-00C853?logo=openaigym&logoColor=white)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-DQN-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Status](https://img.shields.io/badge/Status-Complete-34D399)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-FBBF24.svg)](LICENSE)

**Infotact Solutions** · Travel & Hospitality Domain · 4-Week Internship Project

</div>

---

## 📌 Overview

Airlines sell a **fixed, perishable inventory** — once a flight departs, every empty seat is revenue lost forever. Static pricing rules can't adapt to shifting demand, competitor moves, or the shrinking time left before departure.

**RevenueRL** solves this with a reinforcement learning agent that learns optimal pricing purely through simulated experience — no historical dataset required. The agent plays thousands of simulated 30-day booking seasons and discovers strategies like discounting prices as the deadline nears, entirely on its own.

> 🎯 **The core question this project answers:** *Can an RL agent actually beat the pricing formula airlines use today?*

---

## ⭐ What Makes This Project Different

<table>
<tr>
<td width="50%" valign="top">

### 📐 Real Industry Benchmark
Evaluated against **EMSR-b** (Expected Marginal Seat Revenue) — the actual airline revenue management formula used in the industry since 1989. Not a toy heuristic.

### 👥 Three Customer Segments
Business, leisure, and last-minute travellers — each with distinct price sensitivity and booking timing behaviour.

</td>
<td width="50%" valign="top">

### 🎯 Competitor-Aware Pricing
A rival airline's price is part of the agent's state — it must react to the market, not price in isolation.

### 🛡️ Safety Bounds
Hard price floor/ceiling + max daily price swing — mirroring how a real Revenue Manager configures guardrails before deploying any pricing algorithm.

</td>
</tr>
</table>

---

## 🧠 Reinforcement Learning Design

### MDP Formulation

| Component | Description |
|---|---|
| **State** | `[eco_seats_left, biz_seats_left, days_until_departure, competitor_price_idx, market_event_id]` |
| **Action** | `[economy_price_level, business_price_level]` — 8 tiers each |
| **Reward** | Daily revenue (economy + business) minus safety violation penalty |
| **Terminal** | Flight departs (day = 0) **OR** both classes sold out |

### Algorithms Implemented

| # | Strategy | Type |
|---|----------|------|
| 1 | Fixed Price | Naive baseline |
| 2 | Time-Based Discount | Naive baseline |
| 3 | **EMSR-b** | ★ Real airline industry formula (1989) |
| 4 | Q-Learning | Tabular RL, ε-greedy exploration |
| 5 | **Deep Q-Network (DQN)** | ★ Neural network + experience replay + target network |

---

## 📅 Project Roadmap

### ✅ Week 1 — MDP Design & Gym Environment
- [x] Formulated the pricing problem as a Markov Decision Process
- [x] Built a custom Gymnasium environment (`ContextualAirlinePricingEnv`)
- [x] Designed a stochastic demand function — price sensitivity × urgency × competitor reaction × market events
- [x] Implemented and unit-tested Safety Bounds
- [x] Established a 500-episode random-agent baseline

### ✅ Week 2 — Baselines & Q-Learning
- [x] Implemented Fixed Price and Time-Based Discount heuristics
- [x] Implemented EMSR-b as a credible industry benchmark
- [x] Built a tabular Q-Learning agent with state discretization
- [x] Trained over 8,000 episodes and benchmarked against all baselines

### ✅ Week 3 — Deep Reinforcement Learning (DQN)
- [x] Replaced the Q-table with a neural network Q-function
- [x] Implemented experience replay buffer for training stability
- [x] Implemented a target network to prevent moving-target instability
- [x] Handled exploration-exploitation trade-off via epsilon-greedy decay

### ✅ Week 4 — Policy Evaluation & Business Dashboard
- [x] Evaluated DQN vs all baselines across 1,000 simulated booking seasons
- [x] Plotted Price Trajectories proving learned deadline-discounting behaviour
- [x] Computed projected annual business impact vs EMSR-b
- [x] Built a Streamlit dashboard for technical + business audiences

---

## 📊 Dashboard Features

| Panel | Description |
|---|---|
| **Executive Summary** | Best strategy, revenue per season, % improvement vs EMSR-b, projected annual gain |
| **Strategy Comparison** | Revenue distribution, sell-through rate, safety compliance across all strategies |
| **Price Trajectory** | Visual proof the agent learned deadline-aware discounting |
| **Safety & Robustness** | Safety bound compliance per strategy |

---

## 🗂️ Project Structure

```
RevenueRL/
│
├── notebooks/
│   ├── 01_gym_environment.ipynb      # Week 1 — MDP + custom Gym environment
│   ├── 02_baselines_qlearning.ipynb  # Week 2 — Fixed/Discount/EMSR-b + Q-Learning
│   ├── 03_dqn_agent.ipynb            # Week 3 — Deep Q-Network
│   └── 04_evaluation.ipynb           # Week 4 — 1,000-season evaluation
│
├── models/
│   ├── q_table.pkl                   # Trained Q-Learning table
│   └── dqn_weights.pth               # Trained DQN neural network weights
│
├── data/
│   ├── week1_random_baseline.csv
│   ├── week2_strategy_comparison.csv
│   ├── week3_final_leaderboard.csv
│   ├── week4_final_1000season_results.csv
│   └── week4_business_summary.csv
│
├── reports/                          # 11 generated charts & visualizations
│
├── dashboard.py                      # Streamlit business dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/YOUR_USERNAME/RevenueRL.git
cd RevenueRL

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

<details>
<summary><b>📦 requirements.txt</b></summary>

```txt
gymnasium
torch
numpy
pandas
matplotlib
seaborn
streamlit
tqdm
```
</details>

---

## 🚀 Running the Project

```bash
# Run notebooks in order
jupyter notebook notebooks/01_gym_environment.ipynb
jupyter notebook notebooks/02_baselines_qlearning.ipynb
jupyter notebook notebooks/03_dqn_agent.ipynb
jupyter notebook notebooks/04_evaluation.ipynb

# Launch the business dashboard
streamlit run dashboard.py
```

---

## 📈 Key Results

| Strategy | Type | Mean Revenue |
|---|---|:---:|
| Fixed Price | Naive baseline | see `data/week4_final_1000season_results.csv` |
| Time-Based Discount | Naive baseline | see `data/week4_final_1000season_results.csv` |
| EMSR-b | ★ Real industry formula | see `data/week4_final_1000season_results.csv` |
| Q-Learning | Tabular RL | see `data/week4_final_1000season_results.csv` |
| **DQN** | **★ Deep RL** | **Best performing** |

> Full statistics — mean, median, std, min, max revenue, sell-through rate, and safety compliance — across all 1,000 simulated seasons are saved in `data/week4_final_1000season_results.csv`.

---

## 🎓 What This Demonstrates

- End-to-end MDP formulation and custom Gymnasium environment design
- Tabular Q-Learning with state discretization
- Deep Q-Networks with experience replay and target networks
- Statistically rigorous policy evaluation (1,000 episodes)
- Business-grade benchmarking against a real industry formula
- Safety-constrained RL — a genuinely production-relevant concern most academic RL projects skip

---

## 👤 Author

**Preeti**
satya sri ram charan kolla teja
nuthan
karthik
faraz
intern — Infotact Solutions
Domain: Travel & Hospitality · Reinforcement Learning
Duration: 4 Weeks

---

## 📜 License

Developed as part of an internship at **Infotact Solutions**. All rights reserved.

<div align="center">

---

*Built with 🧠 reinforcement learning and ✈️ a lot of simulated flights*

</div>
