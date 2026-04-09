"""
Inference Script — Equity Research Environment v2
==================================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Usage:
    python inference.py
"""

import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from my_env.models import EquityAction, EquityObservation
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "my_env"))
    from models import EquityAction, EquityObservation


# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

TASK_NAME               = "equity_research"
BENCHMARK               = "equity_research_env"
MAX_STEPS               = 4
SUCCESS_SCORE_THRESHOLD = 0.5

ALL_TICKERS = [
    "reliance", "hdfc", "infosys", "lnt", "tata_power",
    "tcs", "itc", "hul", "sjvn", "adani_enterprises", "std_capital"
]

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN or OPENAI_API_KEY must be set.", flush=True)
    sys.exit(1)

openai_client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


# ── Stdout loggers ────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.3f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.3f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Environment client ────────────────────────────────────────────────────────

class MyEnv(EnvClient[EquityAction, EquityObservation, State]):

    def _step_payload(self, action: EquityAction) -> Dict:
        return {"type": action.type, "data": action.data}

    def _parse_result(self, payload: Dict) -> StepResult[EquityObservation]:
        obs_data = payload.get("observation", {})
        obs = EquityObservation(
            company=obs_data.get("company", ""),
            ticker=obs_data.get("ticker", ""),
            financials=obs_data.get("financials", {}),
            news=obs_data.get("news", []),
            current_step=obs_data.get("current_step", 1),
            task_description=obs_data.get("task_description", ""),
            available_actions=obs_data.get("available_actions", []),
            last_action_result=obs_data.get("last_action_result", ""),
            computed_metrics=obs_data.get("computed_metrics"),
            trend=obs_data.get("trend"),
            selected_labels=obs_data.get("selected_labels"),
            chosen_thesis=obs_data.get("chosen_thesis"),
            cumulative_reward=obs_data.get("cumulative_reward", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


# ── LLM helpers ───────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    completion = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return (completion.choices[0].message.content or "").strip()


def parse_json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_step1_prompt(obs: EquityObservation) -> str:
    return f"""You are an equity research analyst.

Company: {obs.company} ({obs.ticker})

{obs.task_description}

Raw Financial Data:
{json.dumps(obs.financials, indent=2)}

Respond with ONLY a JSON object (no explanation, no markdown):
{{
  "type": "compute_metrics",
  "data": {{
    "pe_ratio": <float>,
    "pb_ratio": <float>,
    "operating_margin": <float>,
    "net_profit_margin": <float>,
    "roe": <float>,
    "debt_to_equity": <float>,
    "interest_coverage": <float>,
    "revenue_growth": <float>
  }}
}}"""


def build_step2_prompt(obs: EquityObservation) -> str:
    return f"""You are an equity research analyst.

Company: {obs.company} ({obs.ticker})

{obs.task_description}

Financial Data (3 years):
{json.dumps(obs.financials, indent=2)}

Metrics computed:
{json.dumps(obs.computed_metrics, indent=2)}

Respond with ONLY a JSON object (no explanation, no markdown):
{{
  "type": "analyze_trend",
  "data": "improving"
}}

Valid values: "improving", "stable", "deteriorating" """


def build_step3_prompt(obs: EquityObservation) -> str:
    return f"""You are an equity research analyst.

Company: {obs.company} ({obs.ticker})

{obs.task_description}

Metrics: {json.dumps(obs.computed_metrics, indent=2)}
Trend identified: {obs.trend}
News Headlines: {json.dumps(obs.news, indent=2)}
Step 1 feedback: {obs.last_action_result}

Respond with ONLY a JSON object (no explanation, no markdown):
{{
  "type": "select_labels",
  "data": ["label_1", "label_2", ...]
}}

Only use labels from:
high_debt, declining_revenue, margin_pressure, weak_cashflow, regulatory_risk,
strong_growth, high_profitability, business_expansion, renewable_transition, strong_market_position,
management_risk, valuation_concern, sector_tailwind, capex_heavy"""


def build_step4_prompt(obs: EquityObservation) -> str:
    return f"""You are an equity research analyst.

Company: {obs.company} ({obs.ticker})

{obs.task_description}

Metrics: {json.dumps(obs.computed_metrics, indent=2)}
Trend: {obs.trend}
Labels: {json.dumps(obs.selected_labels, indent=2)}
Step 3 feedback: {obs.last_action_result}

Respond with ONLY a JSON object (no explanation, no markdown):
{{
  "type": "choose_thesis",
  "data": "neutral"
}}

Valid values: "bullish", "neutral", "bearish" """


# ── Run one episode ───────────────────────────────────────────────────────────

PROMPT_BUILDERS = {
    1: build_step1_prompt,
    2: build_step2_prompt,
    3: build_step3_prompt,
    4: build_step4_prompt,
}

FALLBACK_ACTIONS = {
    1: EquityAction(type="compute_metrics", data={}),
    2: EquityAction(type="analyze_trend", data="stable"),
    3: EquityAction(type="select_labels", data=["strong_growth"]),
    4: EquityAction(type="choose_thesis", data="neutral"),
}


def run_episode(env: MyEnv, ticker: str) -> Dict:
    rewards:     List[float] = []
    steps_taken: int         = 0
    success:     bool        = False
    obs                      = None

    log_start(task=f"{TASK_NAME}_{ticker}", env=BENCHMARK, model=MODEL_NAME)

    try:
        for attempt in range(50):
            try:
                reset_result = env.reset(ticker=ticker)
                obs = reset_result.observation
                if obs.ticker == ticker:
                    break
            except Exception as reset_err:
                if attempt >= 49:
                    raise
                time.sleep(0.3)
                continue
        else:
            raise RuntimeError(f"Could not get ticker '{ticker}' after 50 resets")

        done = reset_result.done

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            prompt = PROMPT_BUILDERS[step](obs)
            raw    = call_llm(prompt)

            try:
                action_dict = parse_json(raw)
                action = EquityAction(type=action_dict["type"], data=action_dict["data"])
            except Exception:
                action = FALLBACK_ACTIONS[step]

            step_result = env.step(action)
            obs         = step_result.observation
            reward      = step_result.reward or 0.0
            done        = step_result.done
            error       = obs.last_action_result if obs.last_action_result.startswith("ERROR") else None

            rewards.append(reward)
            steps_taken = step

            action_str = f"{action.type}({json.dumps(action.data, separators=(',', ':'))})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

            time.sleep(0.5)

        score   = round(sum(rewards), 4)
        success = score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    except Exception as e:
        score = round(sum(rewards), 4)
        log_end(success=False, steps=steps_taken, score=score, rewards=rewards)
        print(f"[DEBUG] Episode failed for {ticker}: {type(e).__name__}: {e}", flush=True)
        raise

    return {
        "ticker":  ticker,
        "company": obs.company if obs else ticker,
        "steps":   steps_taken,
        "rewards": rewards,
        "score":   round(sum(rewards), 4),
        "success": success,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    all_results = []

    with MyEnv(base_url=ENV_BASE_URL).sync() as env:
        for ticker in ALL_TICKERS:
            try:
                result = run_episode(env, ticker=ticker)
                all_results.append(result)
            except Exception as e:
                all_results.append({
                    "ticker":  ticker,
                    "company": ticker,
                    "steps":   0,
                    "rewards": [],
                    "score":   0.0,
                    "success": False,
                    "error":   str(e),
                })
            time.sleep(1.0)

    avg = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0

    with open("inference_results.json", "w") as f:
        json.dump({"model": MODEL_NAME, "results": all_results, "average": avg}, f, indent=2)

    print(f"\nBaseline average score: {avg:.4f} / 1.0", flush=True)


if __name__ == "__main__":
    main()
