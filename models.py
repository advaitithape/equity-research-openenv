"""
Data models for the Equity Research Environment.

5-step equity research workflow:
  Step 1 - compute_metrics:    Agent computes 8 financial metrics from raw data
  Step 2 - analyze_trend:      Agent identifies company trajectory
  Step 3 - select_labels:      Agent selects relevant labels from predefined list
  Step 4 - choose_thesis:      Agent chooses final investment thesis
  Step 5 - allocate_portfolio: Agent recommends portfolio allocation % (0-100)
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ── Predefined sets ───────────────────────────────────────────────────────────

ALL_LABELS = [
    "high_debt",
    "declining_revenue",
    "margin_pressure",
    "weak_cashflow",
    "regulatory_risk",
    "strong_growth",
    "high_profitability",
    "business_expansion",
    "renewable_transition",
    "strong_market_position",
]

VALID_THESIS = ["bullish", "neutral", "bearish"]
VALID_TRENDS = ["improving", "stable", "deteriorating"]


# ── Action model ──────────────────────────────────────────────────────────────

class EquityAction(Action):
    """
    Action for the Equity Research environment.

    Step 1 - compute_metrics:
        {"type": "compute_metrics", "data": {"pe_ratio": 26.11, ...}}

    Step 2 - analyze_trend:
        {"type": "analyze_trend", "data": "improving"}

    Step 3 - select_labels:
        {"type": "select_labels", "data": ["business_expansion", "margin_pressure"]}

    Step 4 - choose_thesis:
        {"type": "choose_thesis", "data": "neutral"}

    Step 5 - allocate_portfolio:
        {"type": "allocate_portfolio", "data": 65}
    """

    type: str = Field(
        ...,
        description=(
            "Action type: 'compute_metrics' | 'analyze_trend' | "
            "'select_labels' | 'choose_thesis' | 'allocate_portfolio'"
        )
    )
    data: Any = Field(
        ...,
        description=(
            "Payload: dict of 8 metrics | trend string | "
            "list of labels | thesis string | integer 0-100"
        )
    )


# ── Observation model ─────────────────────────────────────────────────────────

class EquityObservation(Observation):
    """Observation returned by the Equity Research environment."""

    company: str = Field(default="", description="Company name")
    ticker: str  = Field(default="", description="Ticker key")

    financials: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "3 years of raw financial data (FY2023, FY2024, FY2025). "
            "Values in Crores except shares (units) and price (Rs)."
        )
    )
    news: List[str] = Field(
        default_factory=list,
        description="4 recent news headlines about the company"
    )

    current_step: int = Field(
        default=1,
        description=(
            "Current step: 1=compute_metrics, 2=analyze_trend, "
            "3=select_labels, 4=choose_thesis, 5=allocate_portfolio"
        )
    )
    task_description: str = Field(
        default="",
        description="Plain English instructions for the current step"
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="Valid action types or values for this step"
    )
    last_action_result: str = Field(
        default="",
        description="Feedback on the previous action"
    )

    # Accumulated context across steps
    computed_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Metrics from Step 1, available from Step 2 onwards"
    )
    trend: Optional[str] = Field(
        default=None,
        description="Trend from Step 2, available from Step 3 onwards"
    )
    selected_labels: Optional[List[str]] = Field(
        default=None,
        description="Labels from Step 3, available from Step 4 onwards"
    )
    chosen_thesis: Optional[str] = Field(
        default=None,
        description="Thesis from Step 4, available in Step 5"
    )

    cumulative_reward: float = Field(
        default=0.0,
        description="Total reward accumulated so far"
    )