---
title: Equity Research Environment
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Equity Research Environment

A structured 4-step equity research workflow where an AI agent analyzes real financial data and recent news to perform multi-step reasoning and generate an investment thesis.

The environment simulates how real-world financial analysts combine quantitative fundamentals with qualitative signals (news and sentiment) to make investment decisions.

Based on 11 real Indian companies across sectors.

---

## Motivation

Equity research is a high-value real-world task performed by financial analysts daily. This environment simulates the structured reasoning process an analyst follows:

1. Compute quantitative metrics from raw financial data
2. Identify key signals (risks and strengths) from metrics and news
3. Form a logically consistent investment thesis

It provides a challenging, grounded benchmark for evaluating LLM agents on multi-step financial reasoning. Unlike toy environments, every task here mirrors what a real analyst does — and the agent must use its own financial knowledge to succeed.

---

Each episode corresponds to one company. The agent receives raw financial data (3 years of P&L, balance sheet, and cash flow statements) along with 4 recent news headlines, then completes 4 sequential tasks:

| Step | Task | Difficulty | Max Reward |
|------|------|------------|------------|
| 1 | Compute 8 financial metrics from raw data | Easy | 0.20 |
| 2 | Analyze trend (improving / stable / deteriorating) | Medium | 0.10 |
| 3 | Select 2–5 labels using metrics and news | Medium | 0.20 |
| 4 | Choose investment thesis (bullish / neutral / bearish) | Hard | 0.25 |

**Total possible reward per episode: 0.75**

The agent must combine:
- financial metrics (long-term fundamentals)
- trend (direction)
- news (short-term sentiment)

to produce a coherent investment decision.

---

## Action Space

The agent sends exactly one action per step.

**Step 1 — compute_metrics**
```json
{
  "type": "compute_metrics",
  "data": {
    "pe_ratio": 26.11,
    "pb_ratio": 2.16,
    "operating_margin": 17.37,
    "net_profit_margin": 7.04,
    "roe": 8.26,
    "debt_to_equity": 0.44,
    "interest_coverage": 6.96,
    "revenue_growth": 7.09
  }
}
```

**Step 2 — analyze_trend**

```json
{
  "type": "analyze_trend",
  "data": "improving"
}
```

**Step 3 — select_labels**
```json
{
  "type": "select_labels",
  "data": ["business_expansion", "margin_pressure", "strong_market_position"]
}
```

Available labels:

high_debt, declining_revenue, margin_pressure, weak_cashflow, regulatory_risk,
strong_growth, high_profitability, business_expansion, renewable_transition,
strong_market_position, management_risk, valuation_concern, sector_tailwind, capex_heavy

**Step 4 — choose_thesis**
```json
{
  "type": "choose_thesis",
  "data": "neutral"
}
```

Valid values: `bullish`, `neutral`, `bearish`

---

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `company` | string | Company name |
| `ticker` | string | Short ticker key (e.g. reliance, hdfc) |
| `financials` | object | 3 years of raw financial data (FY2023, FY2024, FY2025) |
| `news` | list[string] | 4 recent news headlines about the company |
| `current_step` | int | Current step number (1, 2, 3 or 4) |
| `task_description` | string | Plain English description of the current task |
| `available_actions` | list[string] | Valid action types for this step |
| `last_action_result` | string | Feedback on the previous action taken |
| `computed_metrics` | object | Metrics from Step 1 (available from Step 2 onwards) |
| `selected_labels` | list[string] | Labels from Step 3 (available in Step 4) |
| `trend` | string | Trend identified in Step 2 (available from Step 3 onwards) |
| `chosen_thesis` | string | Thesis selected in Step 4 |
| `cumulative_reward` | float | Total reward accumulated so far in this episode |
| `done` | bool | Whether the episode is complete |
| `reward` | float | Reward from the last action |

---

## Grading Logic

### Step 1 — compute_metrics (max 0.20)
The agent computes 8 financial metrics from raw data.

Each metric is considered correct if within **5% relative error** of ground truth.

```
reward = (correct_metrics / 8) × 0.20
```

Partial credit is awarded proportionally.

---

### Step 2 — analyze_trend (max 0.10)
The agent identifies company trajectory: `improving`, `stable`, or `deteriorating`.

Reward is assigned based on correctness of the predicted trend.

---

### Step 3 — select_labels (max 0.20)
The agent selects 2–5 labels using financial data and news.

Graded using F1 score against ground truth:


```
reward = F1(predicted_labels, ground_truth_labels) × 0.20
```

Partial overlap receives partial credit.

---

### Step 4 — choose_thesis (max 0.25)
The agent selects a final investment thesis.

Reward is based on:

- correctness of thesis
- consistency with selected labels
- alignment with trend
- penalties if strong signals are ignored

A weak agent defaulting to "neutral" without proper reasoning will receive low reward.

---

## Reward Function

Rewards are distributed across all steps:

- **Step 1 (Metrics)**: Partial credit for each correctly computed metric
- **Step 2 (Trend)**: Reward for correctly identifying company trajectory
- **Step 3 (Labels)**: F1-score based reward comparing selected labels to ground truth
- **Step 4 (Thesis)**:
  - Reward for correct thesis (bullish / neutral / bearish)
  - Additional reward for consistency with selected labels and trend
  - Penalties applied when strong signals (positive/negative) are ignored

This ensures:
- continuous learning signal
- meaningful differentiation between strong and weak agents

---

## Companies

The environment includes 11 Indian companies across sectors, including:

Reliance Industries, HDFC Bank, Infosys, Larsen & Toubro, Tata Power, TCS, ITC, Hindustan Unilever, SJVN, Adani Enterprises, and Standard Capital Markets.

> Note: Companies with incomplete financial data were excluded to ensure stable evaluation.

> Data snapshot: March 2026. Source: Screener.in

---

## Baseline Scores

Baseline agent: `gpt-4o-mini` at temperature 0.0

**Average Score: ~0.52**

This reflects a non-trivial environment where:
- weak agents underperform
- strong agents can achieve significantly higher scores
- meaningful score variance exists across companies

The baseline demonstrates that the environment rewards structured reasoning rather than simple heuristics.

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (for containerized deployment)
- OpenAI API key or compatible LLM endpoint

### Install uv (if not installed)
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Running Locally

**1. Clone the repo and navigate to the environment:**
```bash
git [text](https://github.com/advaitithape/equity-research-openenv)
cd equity-research-openenv
```

**2. Start the environment server:**
```bash
uv run server
# Server starts at http://localhost:8000
# Interactive UI at http://localhost:8000/web
```

**3. Run the inference script (in a separate terminal):**
```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=<your-openai-api-key>
export ENV_BASE_URL=http://localhost:8000

python inference.py
```

**Windows PowerShell:**
```powershell
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:HF_TOKEN="<your-openai-api-key>"
$env:ENV_BASE_URL="http://localhost:8000"

python inference.py
```

### Running with Docker

```bash
# Build
docker build -t equity-research-env:latest -f my_env/server/Dockerfile my_env/

# Run
docker run -p 8000:8000 equity-research-env:latest
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | Execute an action, returns observation + reward |
| `/state` | GET | Get current environment state |
| `/schema` | GET | Get action and observation schemas |
| `/ws` | WebSocket | Persistent WebSocket session |
| `/health` | GET | Health check |
| `/web` | GET | Interactive web UI |
| `/docs` | GET | OpenAPI documentation |

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API endpoint | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier | `gpt-4o-mini` |
| `HF_TOKEN` | Hugging Face / OpenAI API key | required |
| `ENV_BASE_URL` | Environment server URL | `http://localhost:8000` |

---

## Project Structure

```
RL hackathon/
├── inference.py
├── openenv.yaml
├── README.md
├── my_env/
│   ├── __init__.py
│   ├── client.py
│   ├── models.py
│   ├── pyproject.toml
│   ├── data/
│   │   ├── financials.json
│   │   ├── ground_truth.json
│   │   └── news.json
│   └── server/
│       ├── __init__.py
│       ├── app.py
│       ├── my_env_environment.py
│       ├── Dockerfile
│       └── requirements.txt
```
