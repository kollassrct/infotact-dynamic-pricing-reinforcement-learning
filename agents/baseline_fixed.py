"""
baseline_fixed.py
-----------------
Fixed Price Baseline Agent.
Always selects the same price level throughout the episode.
Used as the simplest benchmark — RL agents must beat this.
"""


class FixedPriceAgent:
    """
    Selects a fixed price action for every step.

    Args:
        action_index : index into the industry's price_levels list.
                       0 = cheapest, -1 = most expensive.
        name         : label used in result tables and charts.
    """

    def __init__(self, action_index: int = 2, name: str = "Fixed-Mid"):
        self.action_index = action_index
        self.name         = name

    def select_action(self, obs=None) -> int:
        """Always returns the same action. Observation is ignored."""
        return self.action_index

    def reset(self):
        """No internal state to reset."""
        pass


def get_fixed_agents(n_actions: int) -> list:
    """
    Returns 3 fixed agents: low, mid, high price.

    Args:
        n_actions : number of price levels for the industry

    Returns:
        list of FixedPriceAgent
    """
    mid = n_actions // 2
    return [
        FixedPriceAgent(action_index=0,             name="Fixed-Low"),
        FixedPriceAgent(action_index=mid,           name="Fixed-Mid"),
        FixedPriceAgent(action_index=n_actions - 1, name="Fixed-High"),
    ]