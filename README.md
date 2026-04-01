---
title: Project Sentinel
emoji: 🛡️
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---
# Project Sentinel — AI Security Firewall Environment

> An OpenEnv-compatible environment where an AI agent acts as a security firewall, reviewing tool-call requests from a simulated attacker and deciding to **ALLOW**, **BLOCK**, or **QUARANTINE** each one.

---

## Environment description and motivation

Modern AI systems can call tools — functions like `send_email`, `delete_file`, or `query_database`. This creates a serious attack surface. A malicious actor can craft tool-call requests that look legitimate but cause real damage:
- **Prompt injection**: hiding `IGNORE PREVIOUS INSTRUCTIONS` inside the arguments.
- **Social engineering**: claiming CEO or admin approval to bypass security checks.
- **Obvious attacks**: directly requesting destructive operations like `delete_database` or `disable_firewall`.

Project Sentinel turns this threat into a training environment — teaching agents to recognize and block these attacks before they cause harm.

---

## Action space definitions

What the agent can return at each step (Pydantic `Action`):

| Field | Type | Description |
|---|---|---|
| `decision` | `"allow"` \| `"block"` \| `"quarantine"` | The security verdict — required |
| `reasoning` | `string` | Why the agent made this decision — required |
| `confidence` | `float` 0.0–1.0 | How confident the agent is — optional |

---

## Observation space definitions

What the agent sees at each step (Pydantic `Observation`):

| Field | Type | Description |
|---|---|---|
| `request.tool_name` | `string` | The name of the tool being called |
| `request.arguments` | `dict` | The arguments — attacks are often hidden here |
| `request.source_user` | `string` | Who claims to be making the request (can be faked) |
| `request.context` | `string \| null` | Any justification provided (can be a lie) |
| `remaining_steps` | `integer` | How many more requests are left this episode |
| `prior_decisions` | `list[string]` | The agent's previous decisions this episode |
| `episode_id` | `string` | Unique ID for this episode run |

---

## Task descriptions with expected difficulty

1. **Task 1: The Drunk Butler (`task_1_easy`)**
   Obvious malicious commands like `delete_database` or `format_hard_drive`. Mixed with easy, safe commands. Difficulty: **Easy**

2. **Task 2: The Shadow Prompt (`task_2_medium`)**
   Attacks hidden inside normal-looking requests, like translations containing a SQL-injection trap. Evaluates thorough reading. Difficulty: **Medium**

3. **Task 3: The Long Game (`task_3_hard`)**
   A multi-step social engineering episode that starts secure but maliciously escalates requests with false executive validations. Tests persistence of suspicion. Difficulty: **Hard**

---

## Setup and usage instructions

### Run locally

1. Install requirements:
   `pip install -r requirements.txt`
2. Start the API layer:
   `uvicorn app:app --host 0.0.0.0 --port 7860`
3. Execute validation testing via script (assumes appropriate LLM environment vars):
   `python inference.py`

### Docker setup
1. `docker build -t project-sentinel .`
2. `docker run -p 7860:7860 project-sentinel`

---

## Baseline scores for all 3 tasks

*(Evaluated continuously on `meta-llama/Llama-3.1-8B-Instruct`)*

| Task | Name | Score |
|---|---|---|
| `task_1_easy` | The Drunk Butler | 0.96 |
| `task_2_medium` | The Shadow Prompt | 0.61 |
| `task_3_hard` | The Long Game | 0.45 |
| **Average** | | **0.67** |
