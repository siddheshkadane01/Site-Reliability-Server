---
title: Site Reliability Server
emoji: "☁️"
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
  - sre
  - reinforcement-learning
  - agent-evaluation
  - infrastructure
  - operations
---

# Site Reliability Server

Site Reliability Server is a deterministic OpenEnv environment for
training and evaluating agents on production-style incident response.
The agent acts as an on-call SRE operating a six-service microservice
stack, diagnosing failures from live telemetry, deploy history, and
incident tickets, then applying safe remediations through the standard
`reset() -> step() -> state()` loop.

The environment is built for hackathon validation first:

- Deterministic evaluation by default
- Typed Pydantic request/response models
- Four graded tasks with increasing operational difficulty
- Dense per-step reward shaping
- Root-level `inference.py` that emits the exact required `[START]`,
  `[STEP]`, and `[END]` records
- Hugging Face Space and Docker-compatible runtime

## Why This Environment Is Useful

This is not a toy game. It models a real workflow used by platform
and reliability teams:

- triaging alert floods
- identifying the true root cause in a dependency graph
- distinguishing CPU pressure from configuration regressions
- applying remediations with trade-offs
- recovering service health without destructive guesswork

That makes it suitable for both RL-style environment interaction and
agentic evaluation of reliability-oriented reasoning. The benchmark
also borrows from chaos-engineering practice: several tasks present
plausible but harmful interventions, and the agent is rewarded for
stabilizing the system without creating collateral damage.

## Architecture

The project follows the same high-signal design pattern used by the
strongest OpenEnv references: a thin validator-facing server, a
dedicated environment controller, a pure simulator, and deterministic
graders.

- `server/app.py` is the minimal OpenEnv entrypoint.
- `main.py` exposes the FastAPI routes and typed API contract.
- `env/environment.py` owns episode lifecycle, observation assembly,
  reward shaping, and grader integration.
- `env/simulator.py` models the six-service system and incident
  mechanics.
- `env/graders.py` scores operator behavior deterministically using
  task-specific rubrics.

This separation keeps the HTTP surface small, the simulation testable,
and the benchmark easy to extend with new SRE scenarios.

## Tasks

| Task | Difficulty | Objective | Max Steps |
|---|---|---|---:|
| `easy` | Easy | Identify and recover the primary root-cause service in a cascading incident | 15 |
| `medium` | Medium | Restore all production health signals during a multi-metric service degradation without oscillating between temporary and durable fixes | 15 |
| `hard` | Hard | Diagnose one of multiple plausible db-proxy config regressions and recover the data path | 20 |
| `expert` | Expert | Resolve a dual-service cascade in the correct causal order | 25 |

### Difficulty Progression

- `easy` focuses on root-cause identification and clean recovery.
- `medium` requires full-system remediation instead of fixing just
  one metric, and rewards capacity-aware recovery over restart churn.
- `hard` requires configuration reasoning using deploy history,
  runtime evidence, and multiple plausible suspects while resisting
  misleading rollback signals.
- `expert` requires ordered intervention across coupled failures where
  the wrong restart order causes relapse and post-recovery stalling
  is penalized.

## API Contract

### Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `GET /health`

### Reset Request

`POST /reset` accepts:

- `task_id`
- `scenario_id` optional
- `seed` optional
- `deterministic` optional
- `evaluation_mode` optional

Default behavior is submission-safe:

- `deterministic=true`
- `evaluation_mode=true`
- canonical scenario selection per task
- no stochastic drift

### Observation Space

Each observation includes:

- `step`
- `max_steps`
- `task_id`
- `metrics`
- `logs`
- `deploy_history`
- `current_config`
- `service_graph`
- `active_alerts`
- `health_summary`
- `incident_context`

### Action Space

Supported actions:

- `CHECK_LOGS`
- `INSPECT_SERVICE`
- `DRAIN_TRAFFIC`
- `RESTART_SERVICE`
- `SCALE_UP`
- `SCALE_DOWN`
- `ROLLBACK`
- `UPDATE_CONFIG`
- `SILENCE_ALERT`

Target services:

- `api-gateway`
- `auth-service`
- `user-service`
- `order-service`
- `db-proxy`
- `cache-service`

Optional action fields:

- `config_key`
- `config_value`
- `reason`

### Reward Design

Reward is dense and shaped per step:

- positive reward for overall health improvement
- positive reward for latency reduction
- positive reward for progress on the critical incident services
- small penalty for costly `SCALE_UP`
- penalties for invalid or repeated low-value actions
- explicit penalties for disruptive actions such as restarting healthy
  services or hiding active alerts
- cleanup bonus for silencing alerts only after a service is actually
  healthy

This creates useful intermediate learning signal instead of a sparse
binary end state.

`DRAIN_TRAFFIC` is intentionally modeled as a real SRE trade-off: it
can reduce live pressure quickly, but it is penalized as a temporary
brownout tactic if the agent overuses it instead of completing the
durable fix.

The per-step reward and the final grader score are intentionally
decoupled. An agent can accumulate high cumulative reward through
health improvements while still receiving a lower grader score if it
skips diagnosis, applies fixes in the wrong order, or causes
collateral damage. This prevents reward hacking from inflating
benchmark scores.

## Determinism and Reproducibility

Evaluation runs are deterministic by default.

- `reset()` selects canonical scenarios unless a specific
  `scenario_id` is supplied.
- simulator timestamps come from a deterministic simulation clock,
  not wall-clock time
- stochastic drift is disabled in evaluation mode
- if drift is enabled for experimentation, it is driven by a seeded
  RNG

Result: the same task, scenario, action sequence, and seed produce
the same trajectory and grader score.

## Graders

All graders are deterministic and return scores in `[0.0, 1.0]`.
Final grader scores are computed independently from the per-step
reward signal — the grader evaluates the full episode trajectory
against a task-specific rubric, not cumulative reward.

Grader design is sequence-aware and exploit-resistant:

- `easy`: requires a diagnostic action targeting the root-cause
  service before any remediation. Root identification is not awarded
  for correct action type alone — sequence is enforced by step index.
  An agent that restarts without first inspecting receives partial
  credit only.

- `medium`: scores metric restoration quality, load-shedding
  discipline, and dependency follow-up. Secondary service credit
  requires both a log check and a meaningful remediation action on
  the secondary service, in that order. Oscillation between
  contradictory fixes zeroes the discipline component entirely.

- `hard`: requires evidence gathering before config modification,
  with sequence enforced by step index. Awards credit for the exact
  config key and value, and requires a service restart after the
  fix — not before. Partial credit for near-correct config values is
  awarded only when the submitted value is not exact, eliminating
  double-counting between correctness components.

- `expert`: scores causal ordering of restarts across coupled
  services, per-service recovery quality, collateral damage
  avoidance, alert hygiene, and finish quality after recovery.
  Excess restarts of critical services beyond the allowed count are
  penalized within the collateral component. Post-recovery stalling
  is penalized with increasing severity per unnecessary action.

## Baseline Inference

The submission baseline lives in root-level `inference.py`.

Properties:

- uses `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME`
- does not rely on `OPENAI_API_KEY`
- starts the local env automatically if it is not already running
- emits only the required stdout record types
- writes machine-readable results to `baseline_scores.json`
- runs canonical deterministic scenarios for all four tasks
- keeps runtime bounded and reproducible for repeated evaluator
  re-runs

### Required Environment Variables
```bash
export HF_TOKEN=<your_hf_router_token>
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
```

Optional:
```bash
export LOCAL_IMAGE_NAME=<image_name_if_you_use_local_images>
export OPENENV_BASE_URL=http://127.0.0.1:7860
```

### How Authentication Works

The environment server itself requires no API keys — it is a pure
simulation with no external dependencies. Keys are only needed by
`inference.py` to make LLM calls to the Hugging Face router.

Before running `inference.py`, export your token in the shell:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxx
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

The script reads `HF_TOKEN` via `os.getenv("HF_TOKEN")` at runtime.
The token is never written to disk or committed to the repository.

For automated evaluation, set these variables in the execution
environment before invoking the script. The script will also accept
`OPENAI_API_KEY` as a fallback if `HF_TOKEN` is not set.

The Hugging Face Space serving the environment API requires no
authentication — all endpoints (`/reset`, `/step`, `/state`,
`/grader`, `/health`) are publicly accessible.

### Required Stdout Format

The script emits exactly these record types:
```text
[START] task=<task_name> env=site-reliability-server model=<model_name>
[STEP] step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```

No banners, headings, or extra logging are written to stdout.

### Reference Baseline Scores

Deterministic baseline scores on canonical evaluation scenarios using
Qwen/Qwen2.5-72B-Instruct via Hugging Face router:

| Task     | Scenario      | Score  |
|----------|---------------|-------:|
| `easy`   | `easy-001`    | 0.999  |
| `medium` | `medium-001`  | 0.906  |
| `hard`   | `hard-001`    | 0.897  |
| `expert` | `expert-001`  | 0.9745 |

Mean score: **0.9441**

Scores reflect a calibrated rubric that rewards correct reasoning
over brute-force recovery. A high score requires the right actions
in the right order, not just a healthy final state. Full per-task
breakdowns including grader component scores are written to
`baseline_scores.json` on each run.

## Local Reproduction

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Run the API
```bash
uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1
```

### 3. Sanity Check the Service
```bash
curl http://127.0.0.1:7860/health
curl -X POST http://127.0.0.1:7860/reset \
  -H 'Content-Type: application/json' \
  -d '{}'
```

Expected output:

- `/health` returns status `ok`
- `/reset` returns a typed observation for the `easy` task

### 4. Run Baseline Evaluation
```bash
python inference.py
```

Expected stdout shape:
```text
[START] task=easy env=site-reliability-server model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"action_type":"CHECK_LOGS",...} reward=0.00 done=false error=null
[END] success=true steps=2 rewards=0.00,0.42
```

Expected artifact:
```bash
cat baseline_scores.json
```

`baseline_scores.json` contains:

- benchmark metadata
- model and router configuration
- per-task scenario id
- per-task score
- per-task rewards
- grader breakdowns
- mean score
- total runtime

### 5. Run Tests
```bash
pytest -q
```

### 6. Run OpenEnv Validation
```bash
openenv validate
```

## Docker
```bash
docker build -t site-reliability-server .
docker run --rm -p 7860:7860 site-reliability-server
```

Expected result:

- container starts without manual steps
- `/health` returns `200`
- `/reset` returns a valid observation payload

## Hugging Face Space

The deployed Space is available at the link in the Space header above.

Submission expectations satisfied by the Space setup:

- `sdk: docker`
- OpenEnv tags present
- reset/step/state/grader routes exposed by FastAPI
- deterministic evaluation mode enabled by default

## Repository Layout
```text
.
├── Dockerfile
├── inference.py
├── main.py
├── openenv.yaml
├── requirements.txt
├── pyproject.toml
├── test_env.py
├── env
│   ├── environment.py
│   ├── graders.py
│   ├── models.py
│   ├── simulator.py
│   └── tasks.py
├── scenarios
│   ├── easy
│   ├── medium
│   ├── hard
│   └── expert
└── server
    └── app.py
```

## Validation Checklist

- Hugging Face Space responds to `/health` and `/reset`
- OpenEnv metadata is present in `openenv.yaml`
- Typed models back the API contract
- deterministic evaluation mode is enabled by default
- `inference.py` is root-level and validator-safe
- Docker build is single-command and self-contained
- tests cover deterministic reset and trajectory stability
- incident tickets, deployment history, and ordered cascades make
  the benchmark useful beyond toy evaluation