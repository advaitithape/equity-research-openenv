"""
Equity Research Environment — Baseline Inference Script.
"""

import json
import os
import re
import time
from typing import Dict

from openai import OpenAI
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import EquityAction, EquityObservation
except ImportError:
    from models import EquityAction, EquityObservation


MODEL = "gpt-4o-mini"


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set")
    return OpenAI(api_key=api_key)


class MyEnv(EnvClient[EquityAction, EquityObservation, State]):

    def _step_payload(self, action: EquityAction) -> Dict:
        return {"type": action.type, "data": action.data}

    def _parse_result(self, payload: Dict) -> StepResult[EquityObservation]:
        obs_data = payload.get("observation", {})

        observation = EquityObservation(
            company=obs_data.get("company", ""),
            ticker=obs_data.get("ticker", ""),
            financials=obs_data.get("financials", {}),
            news=obs_data.get("news", []),
            current_step=obs_data.get("current_step", 1),
            task_description=obs_data.get("task_description", ""),
            available_actions=obs_data.get("available_actions", []),
            last_action_result=obs_data.get("last_action_result", ""),
            computed_metrics=obs_data.get("computed_metrics"),
            selected_labels=obs_data.get("selected_labels"),
            cumulative_reward=obs_data.get("cumulative_reward", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


def call_llm(prompt: str) -> str:
    response = _get_openai_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def parse_json(text: str):
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
    return json.loads(text)


# ── Prompts ─────────────────────────────────────────

def step1_prompt(obs):
    return f"""
Compute metrics.

Financials:
{json.dumps(obs.financials)}

Return JSON:
{{
"type":"compute_metrics",
"data":{{"pe_ratio":0,"pb_ratio":0,"operating_margin":0,"net_profit_margin":0,
"roe":0,"debt_to_equity":0,"interest_coverage":0,"revenue_growth":0}}
}}
"""


def step2_prompt(obs):
    return f"""
Analyze trend from financials.

Return JSON:
{{"type":"analyze_trend","data":"improving"}}
"""


def step3_prompt(obs):
    return f"""
Select labels.

Metrics:
{json.dumps(obs.computed_metrics)}

News:
{json.dumps(obs.news)}

Return JSON:
{{"type":"select_labels","data":["label"]}}
"""


def step4_prompt(obs):
    return f"""
Choose thesis.

Return JSON:
{{"type":"choose_thesis","data":"neutral"}}
"""


def step5_prompt(obs):
    return f"""
Allocate capital (0-100).

Return JSON:
{{"type":"allocate_portfolio","data":60}}
"""


# ── Run Episode ─────────────────────────────────────

def run_episode(env):

    obs = env.reset().observation

    total = 0
    rewards = []

    for step in range(1, 6):

        if step == 1:
            prompt = step1_prompt(obs)
        elif step == 2:
            prompt = step2_prompt(obs)
        elif step == 3:
            prompt = step3_prompt(obs)
        elif step == 4:
            prompt = step4_prompt(obs)
        else:
            prompt = step5_prompt(obs)

        try:
            action_dict = parse_json(call_llm(prompt))
            action = EquityAction(**action_dict)
        except:
            action = EquityAction(type="compute_metrics", data={})

        res = env.step(action)
        obs = res.observation

        reward = res.reward or 0
        rewards.append(reward)
        total += reward

        if obs.done:
            break

        time.sleep(0.3)

    return total, rewards


# ── Main ────────────────────────────────────────────

def main():

    base_url = "http://localhost:8000"
    scores = []

    with MyEnv(base_url=base_url).sync() as env:

        for _ in range(5):
            total, rewards = run_episode(env)
            scores.append(total)
            print(f"Score: {total:.4f} | {rewards}")

    avg = sum(scores) / len(scores)
    print(f"\nAverage: {avg:.4f}")


if __name__ == "__main__":
    main()