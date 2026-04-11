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
# Project Sentinel — Agentic AI Security Operations Center

> An OpenEnv-compatible environment where an AI agent acts as a SOC analyst on a live corporate network. The agent can **investigate threats** using multiple tools before making **ALLOW**, **BLOCK**, or **QUARANTINE** decisions. Past decisions affect the system state — allowing a breach degrades the network.

---

## Environment description and motivation

Modern AI systems have tool-calling capabilities (`send_email`, `delete_file`, `query_database`), creating a serious attack surface. Project Sentinel turns this threat into an **agentic training environment**:

- **Dynamic System State:** A simulated corporate network (database, firewall, auth service) that degrades when attacks succeed. Past decisions create cascading consequences.
- **Multi-Step Investigation:** The agent doesn't just classify — it can investigate users, scan payloads, and check system health before deciding. This creates genuine multi-step reasoning.
- **Procedural Generation:** Scenarios are generated from randomized templates — no two episodes are identical. Tool names, user identities, and attack vectors are shuffled each run.
- **Asymmetric Rewards:** False negatives (allowing attacks) are penalized 10x more than false positives, reflecting real-world security priorities.

---

## Action space definitions

| Action | Type | Description |
|---|---|---|
| `allow` | Decision | Request is safe — let it through |
| `block` | Decision | Request is malicious — reject it |
| `quarantine` | Decision | Suspicious — hold for human review |
| `investigate_user` | Investigation | Check requester identity, access history, trust level |
| `investigate_payload` | Investigation | Deep-scan arguments for hidden threats |
| `check_system` | Investigation | Review current system health and recent events |

Additional fields: `reasoning` (string, required), `confidence` (float 0.0–1.0, optional)

---

## Observation space definitions

| Field | Type | Description |
|---|---|---|
| `request.tool_name` | `string` | Name of the tool being called |
| `request.arguments` | `dict` | Arguments — attacks often hidden here |
| `request.source_user` | `string` | Claimed requester identity (can be spoofed) |
| `request.context` | `string \| null` | Justification provided (can be a lie) |
| `system_state.network_health` | `int` | Network health percentage (0–100) |
| `system_state.compromised_services` | `list` | Services damaged by prior breaches |
| `system_state.security_posture` | `string` | Overall security status |
| `investigation_results` | `list` | Findings from prior investigation actions |
| `available_actions` | `list` | Actions currently available |
| `remaining_steps` | `int` | Steps remaining in episode budget |

---

## Task descriptions

1. **Task 1: The Drunk Butler (`task_1_easy`)** — Obvious attacks with clear signals. Difficulty: **Easy**
2. **Task 2: The Shadow Prompt (`task_2_medium`)** — Prompt injection hidden inside normal requests. Difficulty: **Medium**
3. **Task 3: The Long Game (`task_3_hard`)** — Social engineering with trust-building and cascading consequences. Difficulty: **Hard**

---

## Setup and usage

### Run locally
1. `pip install -r requirements.txt`
2. `uvicorn server.app:app --host 0.0.0.0 --port 7860`
3. `python inference.py`

### Docker
1. `docker build -t project-sentinel .`
2. `docker run -p 7860:7860 project-sentinel`

---

## Baseline scores

*(Baseline agent uses quarantine-heavy fallback via `Qwen/Qwen2.5-72B-Instruct`)*

| Task | Baseline Score |
|---|---|
| `task_1_easy` | ~0.72 |
| `task_2_medium` | ~0.68 |
| `task_3_hard` | ~0.52 |
| **Average** | **~0.64** |

*Scores vary by LLM capability. Intelligent agents score higher on easy tasks via investigation bonuses, demonstrating meaningful reward variance.*
