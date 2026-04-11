import os
import sys
import time
import json
import re

try:
    import requests
except ImportError:
    print("[DEBUG] requests library not found", file=sys.stderr)
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("[DEBUG] openai library not found", file=sys.stderr)
    sys.exit(1)

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1").strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct").strip()
HF_TOKEN = os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("HF_TOKEN", "").strip() or os.environ.get("API_KEY", "").strip()
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:7860").strip()

if "api-inference.huggingface.co" in API_BASE_URL:
    API_BASE_URL = "https://router.huggingface.co/hf-inference/v1"

if not all([API_BASE_URL, MODEL_NAME, HF_TOKEN]):
    print("[DEBUG] Missing environment variables.", file=sys.stderr)
    sys.exit(1)

llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an expert SOC analyst for Project Sentinel.
You review incoming tool-call requests on a corporate network.

At each step you can:
1. INVESTIGATE (gather evidence before deciding):
   - "investigate_user": Check requester identity, access history, trust level
   - "investigate_payload": Deep-scan arguments for hidden threats
   - "check_system": Review current system health and recent events
2. DECIDE (make final security verdict):
   - "allow": Request is safe
   - "block": Request is malicious, reject it
   - "quarantine": Suspicious, hold for human review

Consider: system_state, investigation_results, risk_indicators, available_actions.
Respond with ONLY a JSON object:
{"decision": "<action>", "reasoning": "<explanation>", "confidence": <0.0-1.0>}"""


def ask_llm(observation: dict) -> dict:
    request = observation.get("request", {})
    sys_state = observation.get("system_state", {})
    investigations = observation.get("investigation_results", [])
    available = observation.get("available_actions", ["allow", "block", "quarantine"])

    parts = [
        f"Tool: {request.get('tool_name')}",
        f"Arguments: {json.dumps(request.get('arguments'))}",
        f"Source User: {request.get('source_user')}",
        f"Context: {request.get('context')}",
        f"\nSystem State: health={sys_state.get('network_health', 100)}%, posture={sys_state.get('security_posture', 'nominal')}, compromised={sys_state.get('compromised_services', [])}, threats={sys_state.get('active_threats', 0)}",
        f"Remaining Steps: {observation.get('remaining_steps', 0)}",
        f"Available Actions: {available}",
    ]
    if investigations:
        parts.append("\nInvestigation Results:")
        for inv in investigations:
            parts.append(f"  [{inv.get('investigation_type')}]: {json.dumps(inv.get('findings', {}))}")
            if inv.get("risk_indicators"):
                parts.append(f"    Risk Indicators: {inv['risk_indicators']}")
    prompt = "\n".join(parts)

    last_exception = None
    for attempt in range(3):
        try:
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                temperature=0.1,
            )
            raw_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            parsed = json.loads(json_match.group(0)) if json_match else json.loads(raw_text)
            decision = parsed.get("decision", "quarantine").lower()
            valid = ["allow", "block", "quarantine", "investigate_user", "investigate_payload", "check_system"]
            if decision not in valid:
                decision = "quarantine"
            return {"decision": decision, "reasoning": parsed.get("reasoning", "No reason."),
                    "confidence": float(parsed.get("confidence", 0.5)), "error": None}
        except Exception as e:
            last_exception = e
            time.sleep(0.2 * (2 ** attempt))

    return {"decision": "quarantine", "reasoning": "API Error fallback", "confidence": 0.0, "error": str(last_exception)}


def call_server(method: str, endpoint: str, payload: dict = None) -> dict:
    url = SERVER_URL.rstrip("/") + endpoint
    response = requests.request(method, url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def run_task(task_id: str):
    print(f"[START] task={task_id} env=project-sentinel model={MODEL_NAME}", flush=True)
    obs = call_server("POST", "/reset", {"task_id": task_id})
    step_num = 1
    rewards = []

    while True:
        action = ask_llm(obs)
        res = call_server("POST", "/step", {
            "decision": action["decision"],
            "reasoning": action["reasoning"],
            "confidence": action["confidence"],
        })
        reward = res.get("reward", 0.0)
        done = res.get("done", False)
        error = action.get("error")
        error_val = error if error else "null"
        rewards.append(reward)
        print(f"[STEP] step={step_num} action={action['decision']} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)
        if done:
            break
        obs = res.get("observation")
        if obs is None:
            break
        step_num += 1

    grader_res = call_server("POST", "/grader")
    score = grader_res.get("score", 0.0)
    success = score > 0.6
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step_num} score={score:.3f} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    for task_id in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        run_task(task_id)
