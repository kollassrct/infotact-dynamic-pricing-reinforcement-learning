"""
demand_simulator.py
-------------------
Multi-industry demand simulator for DynamicRL.
Generates realistic demand levels based on industry type,
days remaining, seasonality, and price sensitivity.
"""

import numpy as np
from enum import IntEnum


class Industry(IntEnum):
    AIRLINE = 0
    HOTEL = 1
    CONCERT = 2
    STADIUM = 3


INDUSTRY_NAMES = {
    Industry.AIRLINE: "Airline",
    Industry.HOTEL:   "Hotel",
    Industry.CONCERT: "Concert",
    Industry.STADIUM: "Stadium",
}

INDUSTRY_CONFIG = {
    Industry.AIRLINE: {
        "base_demand":        50,
        "price_sensitivity":  0.8,
        "peak_days":          [0, 6],
        "urgency_boost_days": 3,
        "max_inventory":      180,
        "price_levels":       [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
        "base_price":         200,
    },
    Industry.HOTEL: {
        "base_demand":        40,
        "price_sensitivity":  0.6,
        "peak_days":          [4, 5, 6],
        "urgency_boost_days": 2,
        "max_inventory":      100,
        "price_levels":       [0.7, 0.85, 1.0, 1.15, 1.35, 1.6],
        "base_price":         150,
    },
    Industry.CONCERT: {
        "base_demand":        60,
        "price_sensitivity":  0.5,
        "peak_days":          [4, 5],
        "urgency_boost_days": 5,
        "max_inventory":      5000,
        "price_levels":       [0.5, 0.75, 1.0, 1.25, 1.75, 2.5],
        "base_price":         80,
    },
    Industry.STADIUM: {
        "base_demand":        80,
        "price_sensitivity":  0.4,
        "peak_days":          [5, 6],
        "urgency_boost_days": 7,
        "max_inventory":      50000,
        "price_levels":       [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
        "base_price":         60,
    },
}


class DemandSimulator:
    """
    Simulates daily booking demand for a given industry.
    """

    def __init__(self, industry: Industry, season: str = "normal", seed: int = None):
        self.industry = industry
        self.config   = INDUSTRY_CONFIG[industry]
        self.season   = season
        self.rng      = np.random.default_rng(seed)

    def get_demand(self, days_remaining: int, price_multiplier: float, day_of_week: int = None) -> int:
        cfg = self.config
        if day_of_week is None:
            day_of_week = self.rng.integers(0, 7)

        demand = cfg["base_demand"]

        # Seasonality
        season_multiplier = {"peak": 1.4, "normal": 1.0, "off": 0.65}[self.season]
        demand *= season_multiplier

        # Day-of-week boost
        if day_of_week in cfg["peak_days"]:
            demand *= 1.3

        # Urgency curve
        urgency_days = cfg["urgency_boost_days"]
        if days_remaining <= urgency_days:
            urgency_factor = 1.0 + (urgency_days - days_remaining) / urgency_days * 0.8
            demand *= urgency_factor
        elif days_remaining > 60:
            demand *= 0.6

        # Price elasticity
        elasticity   = cfg["price_sensitivity"]
        price_effect = max(0.1, 1.0 - elasticity * (price_multiplier - 1.0))
        demand      *= price_effect

        # Noise
        noise   = self.rng.normal(1.0, 0.15)
        demand *= max(0.1, noise)

        return max(0, int(round(demand)))

    def get_competitor_price(self, base_price: float, days_remaining: int) -> float:
        urgency = max(0, 1.0 - days_remaining / 30) * 0.5
        noise   = self.rng.normal(1.0, 0.1)
        return round(base_price * (1.0 + urgency) * noise, 2)

    def get_demand_level(self, days_remaining: int, price_multiplier: float, day_of_week: int = None) -> int:
        demand = self.get_demand(days_remaining, price_multiplier, day_of_week)
        base   = self.config["base_demand"]
        if demand < base * 0.7:
            return 0   # Low
        elif demand < base * 1.3:
            return 1   # Medium
        else:
            return 2   # High