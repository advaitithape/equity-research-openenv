"""
Equity Research Workflow Environment — 5-Step Version.

Step 1 — compute_metrics:    Compute 8 financial ratios     (max reward 0.25)
Step 2 — analyze_trend:      Identify company trajectory    (max reward 0.10)
Step 3 — select_labels:      Select risk/strength labels    (max reward 0.25)
Step 4 — choose_thesis:      Choose investment thesis       (max reward 0.25)
Step 5 — allocate_portfolio: Recommend allocation %         (max reward 0.15)

Total possible reward: 1.0
Penalties (-0.05) for wrong action types or empty submissions.
"""

import json
import os
import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        EquityAction, EquityObservation,
        ALL_LABELS, VALID_THESIS, VALID_TRENDS
    )
except ImportError:
    from models import (
        EquityAction, EquityObservation,
        ALL_LABELS, VALID_THESIS, VALID_TRENDS
    )


# ── Constants ─────────────────────────────────────────────────────────────────

WRONG_ACTION_PENALTY = -0.05
EMPTY_ACTION_PENALTY = -0.05


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_json(filename: str) -> dict:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "data", filename)
    with open(path, "r") as f:
        return json.load(f)


def _build_agent_financials(raw: dict) -> dict:
    """
    Extract clean FY2023/FY2024/FY2025 data for the agent.
    All values in Crores except shares (units) and price (Rs).
    None values are safely defaulted to 0 for computation fields.
    """
    pl = raw["profit_and_loss"]
    bs = raw["balance_sheet"]
    cf = raw["cash_flow"]

    def fy(section, field, year):
        val = section.get(field, {}).get(year)
        return val  # keep None for non-computation fields

    def fy_num(section, field, year):
        val = section.get(field, {}).get(year)
        return val if val is not None else 0  # safe for arithmetic

    result = {}
    for yr in ["FY2023", "FY2024", "FY2025"]:
        pbt      = fy_num(pl, "profit_before_tax", yr)
        interest = fy_num(pl, "interest", yr)
        dep      = fy_num(pl, "depreciation", yr)
        oth_inc  = fy_num(pl, "other_income", yr)
        op_profit = round(pbt + interest + dep - oth_inc, 2)

        result[yr] = {
            "sales":                fy(pl, "sales", yr),
            "net_profit":           fy(pl, "net_profit", yr),
            "operating_profit":     op_profit,
            "interest":             interest,
            "depreciation":         dep,
            "other_income":         oth_inc,
            "profit_before_tax":    pbt,
            "equity_capital":       fy(bs, "equity_share_capital", yr),
            "reserves":             fy(bs, "reserves", yr),
            "borrowings":           fy(bs, "borrowings", yr),  # may be None for insurers
            "total_assets":         fy(bs, "total_assets", yr),
            "cash_and_bank":        fy(bs, "cash_and_bank", yr),
            "shares_outstanding":   fy(bs, "shares_outstanding", yr),
            "cash_from_operations": fy(cf, "cash_from_operations", yr),
        }

    result["current_price"] = raw["meta"]["current_price"]
    result["market_cap_cr"] = raw["meta"]["market_cap_cr"]
    return result


# ── Task descriptions ─────────────────────────────────────────────────────────

TASK_DESCRIPTIONS = {
    1: (
        "STEP 1 — METRIC COMPUTATION\n"
        "Using your financial knowledge, compute these 8 metrics from the raw data.\n"
        "All financial values are in Crores (Cr). Shares in units. Price in Rs.\n"
        "Use FY2025 as primary year. Use FY2024 only where a second year is needed.\n"
        "Note: For insurance/financial companies, some fields like borrowings may be 0 or null.\n\n"
        "Metrics to compute: pe_ratio, pb_ratio, operating_margin, net_profit_margin, "
        "roe, debt_to_equity, interest_coverage, revenue_growth\n\n"
        "Action format:\n"
        "  {\"type\": \"compute_metrics\", \"data\": {\"pe_ratio\": <float>, "
        "\"pb_ratio\": <float>, \"operating_margin\": <float>, "
        "\"net_profit_margin\": <float>, \"roe\": <float>, "
        "\"debt_to_equity\": <float>, \"interest_coverage\": <float>, "
        "\"revenue_growth\": <float>}}"
    ),
    2: (
        "STEP 2 — TREND ANALYSIS\n"
        "Based on the metrics you computed and the 3-year financial data, "
        "identify the company's current trajectory.\n\n"
        "Valid trends: improving | stable | deteriorating\n\n"
        "Consider: revenue growth direction, margin trends, profit trajectory over 3 years.\n\n"
        "Action format:\n"
        "  {\"type\": \"analyze_trend\", \"data\": \"improving\"}"
    ),

    3: (
        "STEP 3 — LABEL SELECTION\n"
        "Using financial metrics, trend, AND news headlines, select 2 to 5 labels "
        "that best describe this company's current situation.\n\n"

        "Available labels:\n"
        "  high_debt, declining_revenue, margin_pressure, weak_cashflow,\n"
        "  regulatory_risk, strong_growth, high_profitability,\n"
        "  business_expansion, renewable_transition, strong_market_position,\n"
        "  management_risk, valuation_concern, sector_tailwind, capex_heavy\n\n"

        "Guidelines:\n"
        "- Use BOTH financial data and news — not just one\n"
        "- Some labels come mainly from metrics (e.g., high_debt, strong_growth)\n"
        "- Some labels come mainly from news (e.g., regulatory_risk, management_risk)\n"
        "- Some require combining both (e.g., valuation_concern, capex_heavy)\n\n"

        "Examples:\n"
        "- High debt + low interest coverage → high_debt\n"
        "- Strong revenue growth + high margins → strong_growth, high_profitability\n"
        "- Regulatory investigation or governance issue → regulatory_risk / management_risk\n"
        "- Expansion projects or acquisitions → business_expansion / capex_heavy\n"
        "- Renewable energy focus → renewable_transition\n"
        "- Industry tailwinds (AI, infra boom) → sector_tailwind\n\n"

        "Important:\n"
        "- Do NOT select labels blindly from metrics\n"
        "- Ensure each label is supported by either financial data OR news\n"
        "- Avoid over-selecting labels — choose only the most relevant 2–5\n\n"

        "Action format:\n"
        "  {\"type\": \"select_labels\", \"data\": [\"label_1\", \"label_2\", ...]}"
    ),

    4: (
        "STEP 4 — Choose investment thesis (bullish / neutral / bearish)\n\n"
    
        "Guidelines:\n"
        "- Positive signals (e.g., strong_growth, high_profitability, business_expansion) generally support a BULLISH thesis\n"
        "- Negative signals (e.g., high_debt, regulatory_risk, weak_cashflow, management_risk) generally support a BEARISH thesis\n"
        "- If both positive and negative signals are present in similar strength, the thesis should be NEUTRAL\n\n"

        "Additional Insight:\n"
        "- Financial metrics represent long-term strength\n"
        "- News headlines represent short-term sentiment\n"
        "- Weak fundamentals + negative news → Bearish\n"
        "- Strong fundamentals + positive news → Bullish\n"
        "- Strong fundamentals + negative recent news → Neutral or short-term Bearish\n\n"

        "Important:\n"
        "- Avoid defaulting to neutral\n"
        "- Use labels + trend together to decide\n"
        "- Do NOT rely on a single label\n"
        
    )
}


# ── Grading functions ─────────────────────────────────────────────────────────

def _grade_metrics(predicted: dict, ground_truth: dict) -> tuple:
    """
    Max reward: 0.25
    Each of 8 metrics worth (1/8) * 0.25
    Tolerance: 5% relative error
    Penalty: -0.05 for empty submission
    """
    if not isinstance(predicted, dict) or len(predicted) == 0:
        return EMPTY_ACTION_PENALTY, (
            f"PENALTY: Empty metrics submission. reward={EMPTY_ACTION_PENALTY}"
        )

    required = [
        "pe_ratio", "pb_ratio", "operating_margin", "net_profit_margin",
        "roe", "debt_to_equity", "interest_coverage", "revenue_growth"
    ]

    correct = 0
    lines   = []

    for metric in required:
        if metric not in predicted:
            lines.append(f"  x {metric}: missing")
            continue
        try:
            pred_val = float(predicted[metric])
        except (TypeError, ValueError):
            lines.append(f"  x {metric}: invalid value")
            continue

        gt_val = float(ground_truth[metric])
        error  = abs(pred_val - gt_val) / abs(gt_val) if gt_val != 0 else abs(pred_val)

        if error <= 0.05:
            correct += 1
            lines.append(f"  ok {metric}: {pred_val:.2f} (~{gt_val:.2f})")
        else:
            lines.append(
                f"  x {metric}: {pred_val:.2f} (~{gt_val:.2f}, err={error*100:.1f}%)"
            )

    reward   = round((correct / len(required)) * 0.25, 4)
    feedback = (
        f"Metrics: {correct}/{len(required)} correct (reward={reward:.4f}/0.25)\n"
        + "\n".join(lines)
    )
    return reward, feedback


def _grade_trend(predicted: str, ground_truth: str) -> tuple:
    """
    Max reward: 0.10
    Exact match: 0.10
    Off by one level: 0.05 (e.g. improving vs stable)
    Wrong direction: 0.0
    Penalty: -0.05 for invalid trend value
    """
    if not isinstance(predicted, str) or predicted.strip() == "":
        return EMPTY_ACTION_PENALTY, (
            f"PENALTY: Empty trend submission. reward={EMPTY_ACTION_PENALTY}"
        )

    predicted = predicted.strip().lower()

    if predicted not in VALID_TRENDS:
        return WRONG_ACTION_PENALTY, (
            f"PENALTY: Invalid trend '{predicted}'. "
            f"Must be one of: {VALID_TRENDS}. reward={WRONG_ACTION_PENALTY}"
        )

    if predicted == ground_truth:
        reward   = 0.10
        feedback = f"Trend correct: '{predicted}' (reward=0.10/0.10)"
    else:
        # Adjacent levels get partial credit
        order = ["deteriorating", "stable", "improving"]
        pred_idx = order.index(predicted)
        gt_idx   = order.index(ground_truth)
        if abs(pred_idx - gt_idx) == 1:
            reward   = 0.05
            feedback = (
                f"Trend adjacent: got '{predicted}', expected '{ground_truth}' "
                f"(reward=0.05/0.10)"
            )
        else:
            reward   = 0.0
            feedback = (
                f"Trend wrong: got '{predicted}', expected '{ground_truth}' "
                f"(reward=0.00/0.10)"
            )

    return round(reward, 4), feedback


def _grade_labels(predicted: list, ground_truth: list) -> tuple:
    """
    Max reward: 0.25
    F1 score against ground truth labels * 0.25
    Penalty: -0.05 for empty or all-invalid labels
    """
    if not isinstance(predicted, list) or len(predicted) == 0:
        return EMPTY_ACTION_PENALTY, (
            f"PENALTY: Empty label list. reward={EMPTY_ACTION_PENALTY}"
        )

    pred_set = set(p for p in predicted if p in ALL_LABELS)
    gt_set   = set(ground_truth)

    if not pred_set:
        return EMPTY_ACTION_PENALTY, (
            f"PENALTY: No valid labels — all outside allowed list. "
            f"reward={EMPTY_ACTION_PENALTY}"
        )

    tp        = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall    = tp / len(gt_set)   if gt_set   else 0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0
    )

    reward   = round(f1 * 0.25, 4)
    feedback = (
        f"Labels: F1={f1:.2f} (reward={reward:.4f}/0.25)\n"
        f"  Selected: {sorted(pred_set)}\n"
        f"  Expected: {sorted(gt_set)}\n"
        f"  Correct:  {sorted(pred_set & gt_set)}\n"
        f"  Missed:   {sorted(gt_set - pred_set)}\n"
        f"  Extra:    {sorted(pred_set - gt_set)}"
    )
    return reward, feedback


def _grade_thesis(
    predicted: str, gt_thesis: str,
    pred_labels: list, gt_labels: list,
    pred_trend: str, gt_trend: str
) -> tuple:
    """
    Max reward: 0.25
    +0.15 correct thesis label
    +0.10 thesis consistent with labels AND trend
    Penalty: -0.05 for invalid thesis
    """
    if not isinstance(predicted, str) or predicted.strip() == "":
        return EMPTY_ACTION_PENALTY, (
            f"PENALTY: Empty thesis. reward={EMPTY_ACTION_PENALTY}"
        )

    predicted = predicted.strip().lower()

    if predicted not in VALID_THESIS:
        return WRONG_ACTION_PENALTY, (
            f"PENALTY: Invalid thesis '{predicted}'. "
            f"Must be one of: {VALID_THESIS}. reward={WRONG_ACTION_PENALTY}"
        )

    reward = 0.0
    lines  = []

    # Part 1: correct thesis (0.15)
    if predicted == gt_thesis:
        reward += 0.15
        lines.append(f"  ok Thesis correct: '{predicted}' (+0.15)")
    else:
        lines.append(
            f"  x Thesis wrong: '{predicted}' vs '{gt_thesis}' (+0.0)"
        )

    # Part 2: internal consistency with labels + trend (0.10)
    positive_labels = {
        "strong_growth", "high_profitability", "business_expansion",
        "renewable_transition", "strong_market_position"
    }
    negative_labels = {
        "high_debt", "declining_revenue", "margin_pressure",
        "weak_cashflow", "regulatory_risk"
    }

    pred_set  = set(pred_labels) if pred_labels else set()
    positives = pred_set & positive_labels
    negatives = pred_set & negative_labels

    label_consistent = (
        (predicted == "bullish"  and len(positives) > len(negatives)) or
        (predicted == "bearish"  and len(negatives) > len(positives)) or
        (predicted == "neutral")
    )

    trend_consistent = (
        (predicted == "bullish"  and pred_trend in ["improving", "stable"]) or
        (predicted == "bearish"  and pred_trend in ["deteriorating", "stable"]) or
        (predicted == "neutral")
    )

    gt_overlap = len(pred_set & set(gt_labels)) / max(len(set(gt_labels)), 1)

    if label_consistent and trend_consistent and gt_overlap >= 0.4:
        reward += 0.10
        lines.append(
            f"  ok Thesis consistent with labels+trend "
            f"(overlap={gt_overlap:.0%}) (+0.10)"
        )
    elif label_consistent or trend_consistent:
        reward += 0.05
        lines.append(
            f"  ~ Thesis partially consistent (+0.05)"
        )
    else:
        lines.append(f"  x Thesis inconsistent with labels+trend (+0.0)")

    reward   = round(reward, 4)
    feedback = (
        f"Thesis graded (reward={reward:.4f}/0.25)\n" + "\n".join(lines)
    )
    return reward, feedback


# ── Environment ───────────────────────────────────────────────────────────────

class MyEnvironment(Environment):
    """
    Equity Research Workflow Environment — 5 Steps.

    Reward structure:
      Step 1 compute_metrics:    max 0.25
      Step 2 analyze_trend:      max 0.10
      Step 3 select_labels:      max 0.25
      Step 4 choose_thesis:      max 0.25
      Step 5 allocate_portfolio: max 0.15
      Total:                     max 1.00

    Penalties: -0.05 for wrong action type or empty/invalid submission.
    12 companies: diverse sectors, balanced thesis distribution.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state   = State(episode_id=str(uuid4()), step_count=0)
        self._raw     = _load_json("financials.json")
        self._news    = _load_json("news.json")
        self._gt      = _load_json("ground_truth.json")
        self._tickers = list(self._raw.keys())

        # Episode state
        self._ticker:  str   = ""
        self._step:    int   = 0
        self._reward:  float = 0.0
        self._metrics: dict  = {}
        self._trend:   str   = ""
        self._labels:  list  = []
        self._thesis:  str   = ""
        self._fin:     dict  = {}

    def reset(self) -> EquityObservation:
        """Start a new episode with a randomly selected company."""
        self._state   = State(episode_id=str(uuid4()), step_count=0)
        self._ticker  = random.choice(self._tickers)
        self._step    = 1
        self._reward  = 0.0
        self._metrics = {}
        self._trend   = ""
        self._labels  = []
        self._thesis  = ""
        self._fin     = _build_agent_financials(self._raw[self._ticker])

        return EquityObservation(
            company=self._raw[self._ticker]["company"],
            ticker=self._ticker,
            financials=self._fin,
            news=self._news[self._ticker],
            current_step=1,
            task_description=TASK_DESCRIPTIONS[1],
            available_actions=["compute_metrics"],
            last_action_result="",
            computed_metrics=None,
            trend=None,
            selected_labels=None,
            chosen_thesis=None,
            cumulative_reward=0.0,
            done=False,
            reward=0.0,
        )

    def step(self, action: EquityAction) -> EquityObservation:  # type: ignore[override]
        """Execute one step and return graded observation."""
        self._state.step_count += 1

        atype = getattr(action, "type", "")
        adata = getattr(action, "data", None)
        gt    = self._gt[self._ticker]
        news  = self._news[self._ticker]

        # ── Step 1: Metric computation ────────────────────────────────────
        if self._step == 1:
            if atype != "compute_metrics":
                reward, feedback = WRONG_ACTION_PENALTY, (
                    f"PENALTY: Expected 'compute_metrics', got '{atype}'. "
                    f"reward={WRONG_ACTION_PENALTY}"
                )
            else:
                reward, feedback = _grade_metrics(adata, gt["metrics"])
            self._metrics  = adata if isinstance(adata, dict) else {}
            self._reward  += reward
            self._step     = 2

            return EquityObservation(
                company=self._raw[self._ticker]["company"],
                ticker=self._ticker,
                financials=self._fin,
                news=news,
                current_step=2,
                task_description=TASK_DESCRIPTIONS[2],
                available_actions=VALID_TRENDS,
                last_action_result=feedback,
                computed_metrics=self._metrics,
                trend=None,
                selected_labels=None,
                chosen_thesis=None,
                cumulative_reward=round(self._reward, 4),
                done=False,
                reward=reward,
            )

        # ── Step 2: Trend analysis ────────────────────────────────────────
        elif self._step == 2:
            if atype != "analyze_trend":
                reward, feedback = WRONG_ACTION_PENALTY, (
                    f"PENALTY: Expected 'analyze_trend', got '{atype}'. "
                    f"reward={WRONG_ACTION_PENALTY}"
                )
            else:
                reward, feedback = _grade_trend(adata, gt["trend"])
            self._trend    = adata if isinstance(adata, str) else ""
            self._reward  += reward
            self._step     = 3

            return EquityObservation(
                company=self._raw[self._ticker]["company"],
                ticker=self._ticker,
                financials=self._fin,
                news=news,
                current_step=3,
                task_description=TASK_DESCRIPTIONS[3],
                available_actions=ALL_LABELS,
                last_action_result=feedback,
                computed_metrics=self._metrics,
                trend=self._trend,
                selected_labels=None,
                chosen_thesis=None,
                cumulative_reward=round(self._reward, 4),
                done=False,
                reward=reward,
            )

        # ── Step 3: Label selection ───────────────────────────────────────
        elif self._step == 3:
            if atype != "select_labels":
                reward, feedback = WRONG_ACTION_PENALTY, (
                    f"PENALTY: Expected 'select_labels', got '{atype}'. "
                    f"reward={WRONG_ACTION_PENALTY}"
                )
            else:
                reward, feedback = _grade_labels(adata, gt["labels"])
            self._labels   = adata if isinstance(adata, list) else []
            self._reward  += reward
            self._step     = 4

            return EquityObservation(
                company=self._raw[self._ticker]["company"],
                ticker=self._ticker,
                financials=self._fin,
                news=news,
                current_step=4,
                task_description=TASK_DESCRIPTIONS[4],
                available_actions=VALID_THESIS,
                last_action_result=feedback,
                computed_metrics=self._metrics,
                trend=self._trend,
                selected_labels=self._labels,
                chosen_thesis=None,
                cumulative_reward=round(self._reward, 4),
                done=False,
                reward=reward,
            )

        # ── Step 4: Thesis selection ──────────────────────────────────────
        elif self._step == 4:
            if atype != "choose_thesis":
                reward, feedback = WRONG_ACTION_PENALTY, (
                    f"PENALTY: Expected 'choose_thesis', got '{atype}'. "
                    f"reward={WRONG_ACTION_PENALTY}"
                )
            else:
                reward, feedback = _grade_thesis(
                    adata, gt["thesis"],
                    self._labels, gt["labels"],
                    self._trend, gt["trend"]
                )
            self._thesis   = adata if isinstance(adata, str) else ""
            self._reward  += reward
            self._step     = 5

            return EquityObservation(
                company=self._raw[self._ticker]["company"], 
                ticker=self._ticker, 
                financials=self._fin, 
                news=news, 
                current_step=5, 
                task_description=( 
                    f"Episode complete. Final reward: {round(self._reward, 4):.4f} / 1.0"
                    ), 
                available_actions=[], 
                last_action_result=feedback, 
                computed_metrics=self._metrics, 
                trend=self._trend, 
                selected_labels=self._labels, 
                chosen_thesis=self._thesis, 
                cumulative_reward=round(self._reward, 4), 
                done=True, 
                reward=reward, 
            )

        # ── Already complete ──────────────────────────────────────────────
        else:
            return EquityObservation(
                company=self._raw[self._ticker]["company"],
                ticker=self._ticker,
                financials=self._fin,
                news=news,
                current_step=6,
                task_description="Episode complete. Call reset() to start a new episode.",
                available_actions=[],
                last_action_result="ERROR: Episode already complete.",
                computed_metrics=self._metrics,
                trend=self._trend,
                selected_labels=self._labels,
                chosen_thesis=self._thesis,
                cumulative_reward=round(self._reward, 4),
                done=True,
                reward=0.0,
            )

    @property
    def state(self) -> State:
        return self._state