# DynamicRL 🎯
### A Multi-Domain Reinforcement Learning Framework for Revenue Optimization of Finite Inventory

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29%2B-green)](https://gymnasium.farama.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Overview

Many industries share a fundamental constraint: **finite, perishable inventory**. An unsold airline seat, hotel room, or concert ticket on the day of the event is revenue lost forever. Static pricing strategies fail to adapt to shifting demand, competitor moves, and time pressure.

**DynamicRL** solves this with a universal reinforcement learning engine that learns optimal dynamic pricing policies across four industries — maximizing cumulative revenue while balancing sell-through rates and pricing integrity.

---

## 🏭 Supported Industries

| Industry | Inventory Unit | Expiry Type |
|---|---|---|
| ✈️ Airlines | Seats per flight | Flight departure |
| 🏨 Hotels | Rooms per night | Check-in date |
| 🎵 Concerts | Tickets per show | Event day |
| 🏟️ Stadium Events | Seats per game | Game day |

---

## 🧠 Reinforcement Learning Design

### MDP Formulation

| Component | Description |
|---|---|
| **State** | Remaining inventory, days left, demand level, competitor price, event type, seasonality |
| **Action** | Select a price level for the current booking day |
| **Reward** | Revenue earned minus penalties for unsold inventory and undesirable pricing behavior |
| **Goal** | Maximize cumulative revenue across the booking horizon |

### Algorithms

- **Baseline 1** — Fixed pricing strategy
- **Baseline 2** — Discount/rule-based pricing
- **Q-Learning** — Tabular Q-table with ε-greedy exploration
- **DQN** — Deep Q-Network with experience replay and target network

---

## 📅 Project Roadmap

### Week 1 — Environment & Simulation
<<<<<<< HEAD
- [x] Design the MDP (states, actions, rewards)
- [x] Build custom Gymnasium environment
- [x] Implement multi-industry demand simulator
- [x] Support all four industry types
- [x] Unit testing & environment validation

### Week 2 — Baselines & Q-Learning
- [x] Implement fixed-price and discount baselines
- [x] Build Q-Learning agent with Q-table
- [x] Add ε-greedy exploration strategy
- [x] Compare revenues across baselines vs Q-Learning

### Week 3 — Deep Q-Network (DQN)
- [x] Implement DQN with neural network Q-function
- [x] Add experience replay buffer
- [x] Add target network for stable training
- [x] Hyperparameter tuning
- [x] Cross-industry evaluation

### Week 4 — Evaluation & Dashboard
- [x] Evaluate over 1,000 simulations
- [x] Build Streamlit dashboard with full KPI suite
- [x] Business insights automation
- [x] Final report
=======
- [ ] Design the MDP (states, actions, rewards)
- [ ] Build custom Gymnasium environment
- [ ] Implement multi-industry demand simulator
- [ ] Support all four industry types
- [ ] Unit testing & environment validation

### Week 2 — Baselines & Q-Learning
- [ ] Implement fixed-price and discount baselines
- [ ] Build Q-Learning agent with Q-table
- [ ] Add ε-greedy exploration strategy
- [ ] Compare revenues across baselines vs Q-Learning

### Week 3 — Deep Q-Network (DQN)
- [ ] Implement DQN with neural network Q-function
- [ ] Add experience replay buffer
- [ ] Add target network for stable training
- [ ] Hyperparameter tuning
- [ ] Cross-industry evaluation

### Week 4 — Evaluation & Dashboard
- [ ] Evaluate over 1,000 simulations
- [ ] Build Streamlit dashboard with full KPI suite
- [ ] Business insights automation
- [ ] Final report
>>>>>>> 0814427e644ae0a52b36c8bc38e4adf5ba83d757

---

## 📊 Streamlit Dashboard Features

The dashboard is designed for both business executives and ML practitioners.

| Panel | Description |
|---|---|
| **Executive Overview** | High-level KPIs: total revenue, occupancy rate, avg. price |
| **Industry Selector** | Switch between Airlines, Hotels, Concerts, Stadium |
| **Live Booking Simulator** | Simulate bookings in real time |
| **Dynamic Pricing Timeline** | Price trajectory across the booking horizon |
| **Revenue Comparison** | Fixed vs. discount vs. Q-Learning vs. DQN |
| **Occupancy Gauge** | Inventory utilization at a glance |
| **Demand Curve** | Demand level vs. pricing response |
| **RL Learning Metrics** | Training reward curves, loss, epsilon decay |
| **Policy Visualization** | Q-value heatmaps and optimal policy grids |
| **What-If Analysis** | Adjust inventory/demand/days and see projected revenue |
| **Business Insights** | Automated recommendations generated from results |

---

## 🗂️ Project Structure

```
DynamicRL/
│
├── env/
│   ├── dynamic_pricing_env.py       # Custom Gymnasium environment
│   └── demand_simulator.py          # Multi-industry demand model
│
├── agents/
│   ├── baseline_fixed.py            # Fixed price baseline
│   ├── baseline_discount.py         # Discount/rule-based baseline
│   ├── q_learning_agent.py          # Tabular Q-Learning agent
│   └── dqn_agent.py                 # Deep Q-Network agent
│
├── training/
│   ├── train_qlearning.py           # Q-Learning training loop
│   └── train_dqn.py                 # DQN training loop
│
├── evaluation/
│   ├── simulate.py                  # 1000-episode evaluation runner
│   └── metrics.py                   # Revenue, occupancy, pricing metrics
│
├── dashboard/
│   └── app.py                       # Streamlit dashboard
│
├── results/
│   ├── models/                      # Saved Q-tables and DQN weights
│   └── plots/                       # Generated figures
│
├── report/
│   └── final_report.pdf             # Week 4 final report
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10+
- pip

### Install Dependencies

```bash
git clone https://github.com/<your-username>/DynamicRL.git
cd DynamicRL
pip install -r requirements.txt
```

### Key Dependencies

```txt
gymnasium
numpy
pandas
torch
matplotlib
streamlit
plotly
scikit-learn
```

---

## 🚀 Running the Project

### Train Q-Learning Agent

```bash
python training/train_qlearning.py --industry airline --episodes 5000
```

### Train DQN Agent

```bash
python training/train_dqn.py --industry hotel --episodes 10000
```

### Run Evaluation (1000 simulations)

```bash
python evaluation/simulate.py --agent dqn --industry all
```

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📈 Sample Results (Placeholder)

| Agent | Industry | Avg. Revenue | Occupancy % |
|---|---|---|---|
| Fixed Price | Airline | $42,300 | 74% |
| Discount | Airline | $48,100 | 83% |
| Q-Learning | Airline | $54,600 | 89% |
| **DQN** | **Airline** | **$61,200** | **94%** |

> Results will be updated after Week 4 evaluation across all 1,000 simulations.

---

## 📄 Report

The final project report (Week 4) covers:
- MDP design decisions
- Demand simulation methodology
- Agent architecture & hyperparameters
- Cross-industry performance comparison
- Business recommendations

Available in `report/final_report.pdf` after Week 4 completion.

---

## 👤 Author

**K Satya Sri Ram Charan Teja Kolla**  
<<<<<<< HEAD
B.Tech CSE (Data Science) — Malla Reddy College of Engineering, Hyderabad  
GitHub: [kollassrct](https://github.com/kollassrct)
=======
**Preeti Auditto**
**K Nuthan Sai**
**Karthik Chadda**
**Faraz Khan**

>>>>>>> 0814427e644ae0a52b36c8bc38e4adf5ba83d757

---

## 📜 License

<<<<<<< HEAD
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
=======
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
>>>>>>> 0814427e644ae0a52b36c8bc38e4adf5ba83d757
