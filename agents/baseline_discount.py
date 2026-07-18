"""
baseline_discount.py
--------------------
Discount Baseline Agent.

A rule-based agent that adjusts price based on how much time
is left in the booking window. Starts high, discounts near the end
to avoid unsold inventory.

Strategy:
  - Early  (>60% days left) : high price
  - Middle (30–60% left)    : mid price
  - Late   (<30% days left) : low price (clearance)
"""


class DiscountAgent:
    """
    Rule-based discount agent.

    Args:
        n_actions   : total number of price levels for the industry
        early_idx   : action index to use when plenty of time remains
        mid_idx     : action index to use in the middle period
        late_idx    : action index to use when days are running out
        name        : label for results tables
    """

    def __init__(
        self,
        n_actions:  int,
        early_idx:  int = None,
        mid_idx:    int = None,
        late_idx:   int = None,
        name:       str = "Discount",
    ):
        self.n_actions = n_actions
        self.name      = name

        # Default to high → mid → low if not specified
        self.early_idx = early_idx if early_idx is not None else n_actions - 1
        self.mid_idx   = mid_idx   if mid_idx   is not None else n_actions // 2
        self.late_idx  = late_idx  if late_idx  is not None else 0

        # Set during reset
        self.total_days = None

    def reset(self, total_days: int = 30):
        """Call at the start of each episode to set the time horizon."""
        self.total_days = total_days

    def select_action(self, obs) -> int:
        """
        Pick price based on how much time is left.

        Args:
            obs : observation array
                  obs[1] = days_left  (index 1 in our state space)

        Returns:
            action index
        """
        if self.total_days is None:
            return self.mid_idx

        days_left  = int(obs[1])
        pct_left   = days_left / self.total_days

        if pct_left > 0.6:
            return self.early_idx   # Lots of time → charge high
        elif pct_left > 0.3:
            return self.mid_idx     # Middle period → mid price
        else:
            return self.late_idx    # Running out of time → discount


class DiscountAgentV2(DiscountAgent):
    """
    Enhanced discount agent that also considers inventory level.

    If inventory is low (< 20% remaining), hold price high
    regardless of time — scarcity pricing.
    """

    def select_action(self, obs) -> int:
        from env.demand_simulator import INDUSTRY_CONFIG, Industry

        days_left     = int(obs[1])
        industry_idx  = int(obs[4])
        industry      = Industry(industry_idx)
        max_inv       = INDUSTRY_CONFIG[industry]["max_inventory"]
        inventory     = int(obs[0])
        inv_pct       = inventory / max_inv

        # Scarcity: less than 20% left → hold high price
        if inv_pct < 0.2:
            return self.early_idx

        # Otherwise fall back to time-based logic
        return super().select_action(obs)