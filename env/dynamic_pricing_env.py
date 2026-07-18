"""
dynamic_pricing_env.py
----------------------
Custom Gymnasium environment for DynamicRL.

State  : [inventory, days_left, demand_level, competitor_price_bucket, industry, season]
Action : Price level index (0=cheapest ... 5=most expensive)
Reward : Revenue today − penalties for unsold inventory / bad pricing
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .demand_simulator import DemandSimulator, Industry, INDUSTRY_CONFIG, INDUSTRY_NAMES


UNSOLD_PENALTY_PER_UNIT = 5.0
UNDERCUT_PENALTY        = 2.0
OVERPRICE_PENALTY       = 1.0


class DynamicPricingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        industry:    Industry = Industry.AIRLINE,
        season:      str      = "normal",
        total_days:  int      = 30,
        seed:        int      = None,
        render_mode: str      = None,
    ):
        super().__init__()

        self.industry    = Industry(industry)
        self.season      = season
        self.total_days  = total_days
        self.render_mode = render_mode
        self.cfg         = INDUSTRY_CONFIG[self.industry]
        self.demand_sim  = DemandSimulator(self.industry, season=self.season, seed=seed)
        self._seed       = seed

        # Action space — pick a price level
        n_prices = len(self.cfg["price_levels"])
        self.action_space = spaces.Discrete(n_prices)

        # Observation space
        self.observation_space = spaces.Box(
            low  = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high = np.array([
                self.cfg["max_inventory"],
                total_days,
                2,   # demand level
                4,   # competitor price bucket
                3,   # industry
                2,   # season
            ], dtype=np.float32),
            dtype=np.float32,
        )

        self._season_idx = {"off": 0, "normal": 1, "peak": 2}[season]

        # Episode state
        self.inventory       = 0
        self.days_left       = 0
        self.day_of_week     = 0
        self.total_revenue   = 0.0
        self.units_sold      = 0
        self.price_history   = []
        self.revenue_history = []
        self.demand_history  = []

    # ----------------------------------------------------------------
    # Core Gym Methods
    # ----------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.demand_sim = DemandSimulator(self.industry, season=self.season, seed=seed)

        self.inventory       = self.cfg["max_inventory"]
        self.days_left       = self.total_days
        self.day_of_week     = int(np.random.randint(0, 7))
        self.total_revenue   = 0.0
        self.units_sold      = 0
        self.price_history   = []
        self.revenue_history = []
        self.demand_history  = []

        return self._get_obs(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        price_mult  = self.cfg["price_levels"][action]
        price       = self.cfg["base_price"] * price_mult
        comp_price  = self.demand_sim.get_competitor_price(self.cfg["base_price"], self.days_left)

        demand      = self.demand_sim.get_demand(
            days_remaining  = self.days_left,
            price_multiplier= price_mult,
            day_of_week     = self.day_of_week,
        )

        units_today   = min(demand, self.inventory)
        revenue_today = units_today * price

        # Update state
        self.inventory     -= units_today
        self.units_sold    += units_today
        self.days_left     -= 1
        self.day_of_week    = (self.day_of_week + 1) % 7
        self.total_revenue += revenue_today

        self.price_history.append(price)
        self.revenue_history.append(revenue_today)
        self.demand_history.append(demand)

        # Reward shaping
        reward = revenue_today

        if price < comp_price * 0.7 and units_today > 0:
            reward -= UNDERCUT_PENALTY * units_today

        if self.days_left <= 3 and self.inventory > 0 and price_mult >= 1.5:
            reward -= OVERPRICE_PENALTY * self.inventory * 0.1

        terminated = self.days_left <= 0

        if terminated and self.inventory > 0:
            reward -= UNSOLD_PENALTY_PER_UNIT * self.inventory

        info = {
            "price"         : price,
            "price_mult"    : price_mult,
            "demand"        : demand,
            "units_sold"    : units_today,
            "revenue_today" : revenue_today,
            "total_revenue" : self.total_revenue,
            "inventory_left": self.inventory,
            "comp_price"    : comp_price,
        }

        return self._get_obs(), reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            inv_pct = (1 - self.inventory / self.cfg["max_inventory"]) * 100
            print(
                f"[{INDUSTRY_NAMES[self.industry]}] "
                f"Day {self.total_days - self.days_left}/{self.total_days} | "
                f"Inventory: {self.inventory}/{self.cfg['max_inventory']} "
                f"({inv_pct:.1f}% sold) | "
                f"Revenue: ${self.total_revenue:,.0f}"
            )

    def close(self):
        pass

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        demand_level = self.demand_sim.get_demand_level(
            days_remaining  = max(1, self.days_left),
            price_multiplier= 1.0,
            day_of_week     = self.day_of_week,
        )
        comp_price  = self.demand_sim.get_competitor_price(
            self.cfg["base_price"], max(1, self.days_left)
        )
        comp_bucket = self._price_to_bucket(comp_price, self.cfg["base_price"])

        return np.array([
            self.inventory,
            self.days_left,
            demand_level,
            comp_bucket,
            int(self.industry),
            self._season_idx,
        ], dtype=np.float32)

    @staticmethod
    def _price_to_bucket(price: float, base_price: float) -> int:
        ratio = price / base_price
        if ratio < 0.75:   return 0
        elif ratio < 0.9:  return 1
        elif ratio < 1.1:  return 2
        elif ratio < 1.35: return 3
        else:              return 4

    def get_normalized_obs(self) -> np.ndarray:
        obs  = self._get_obs()
        high = self.observation_space.high
        return obs / np.where(high == 0, 1, high)

    @property
    def n_actions(self) -> int:
        return self.action_space.n

    @property
    def obs_dim(self) -> int:
        return self.observation_space.shape[0]

    def summary(self) -> dict:
        max_inv = self.cfg["max_inventory"]
        return {
            "industry"         : INDUSTRY_NAMES[self.industry],
            "season"           : self.season,
            "total_revenue"    : round(self.total_revenue, 2),
            "units_sold"       : self.units_sold,
            "units_unsold"     : self.inventory,
            "occupancy_pct"    : round(self.units_sold / max_inv * 100, 2),
            "avg_price"        : round(np.mean(self.price_history), 2) if self.price_history else 0,
            "avg_daily_revenue": round(np.mean(self.revenue_history), 2) if self.revenue_history else 0,
        }