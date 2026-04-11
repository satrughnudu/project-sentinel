"""
Baseline inference script for Project Sentinel.

Runs each task sequentially, calling the LLM for decisions. Falls back
to quarantine if the API is down (which happens a lot with free-tier HF).

Expected env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN (or OPENAI_API_KEY)
"""
import os, sys, time, json, re

try:
    import requests
except ImportError:
    sys.exit("requests not installed")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai not installed")


API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1").strip()
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct").strip()
HF_TOKEN     = (os.environ.get("HF_TOKEN", "").strip()
                or os.environ.get("OPENAI_API_KEY", "").strip()
                or os.environ.get("API_KEY", "").strip())
SERVER_URL   = os.environ.get("SERVER_URL", "http://localhost:7860").strip()

# old HF endpoint was deprecated, auto-correct it
if "api-inference.huggingface.co" in API_BASE_URL:
    API_BASE_URL = "https://router.huggingface.co/hf-inference/v1"

if not all([API_BASE_URL, MODEL_NAME, HF_TOKEN]):
    sys.exit("Missing one of: API_BASE_URL, MODEL_NAME, HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_MSG = """You are a SOC analyst triaging tool-call requests on a corporate network.

You can INVESTIGATE before deciding:
- "investigate_user": check who's making the request
- "investigate_payload": scan the arguments for hidden threats
- "check_system": check network health and recent incidents

Or DECIDE:
- "allow": safe, let it through
- "block": malicious, reject it
- "quarantine": suspicious, hold for review

Reply with JSON only: {"decision": "...", "reasoning": "...", "confidence": 0.0-1.0}"""


def ask_llm(obs):
    req = obs.get("request", {})
    ss = obs.get("system_state", {})
    invs = obs.get("investigation_results", [])
    avail = obs.get("available_actions", ["allow", "block", "quarantine"])

    prompt = f"""Tool: {req.get('tool_name')}
Args: {json.dumps(req.get('arguments'))}
User: {req.get('source_user')}
Context: {req.get('context')}

System: health={ss.get('network_health', 100)}%, posture={ss.get('security_posture', 'nominal')}, compromised={ss.get('compromised_services', [])}, threats={ss.get('active_threats', 0)}
Remaining steps: {obs.get('remaining_steps', 0)}
Available: {avail}"""

    if invs:
        prompt += "\n\nInvestigation results:"
        for inv in invs:
            prompt += f"\n  [{inv.get('investigation_type')}] {json.dumps(inv.get('findings', {}))}"
            ri = inv.get("risk_indicators", [])
            if ri:
                prompt += f"\n    Indicators: {ri}"

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_MSG},
                          {"role": "user", "content": prompt}],
                temperature=0.1)
            text = resp.choices[0].message.content.strip()
            m = re.search(r'\{.*\}', text, re.DOTALL)
            parsed = json.loads(m.group(0) if m else text)
            d = parsed.get("decision", "quarantine").lower()
            valid = {"allow", "block", "quarantine", "investigate_user", "investigate_payload", "check_system"}
            if d not in valid:
                d = "quarantine"
            return {"decision": d, "reasoning": parsed.get("reasoning", ""),
                    "confidence": float(parsed.get("confidence", 0.5)), "error": None}
        except Exception as e:
            time.sleep(0.2 * (2 ** attempt))
            last_err = e

    # API is down, just quarantine everything
    return {"decision": "quarantine", "reasoning": "API unavailable, defaulting to safe option",
            "confidence": 0.0, "error": str(last_err)}


def call(method, endpoint, payload=None):
    url = SERVER_URL.rstrip("/") + endpoint
    r = requests.request(method, url, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def run_task(task_id):
    print(f"[START] task={task_id} env=project-sentinel model={MODEL_NAME}", flush=True)

    obs = call("POST", "/reset", {"task_id": task_id})
    step = 1
    rewards = []

    while True:
        action = ask_llm(obs)
        res = call("POST", "/step", {
            "decision": action["decision"],
            "reasoning": action["reasoning"],
            "confidence": action["confidence"]})

        r = res.get("reward", 0.0)
        done = res.get("done", False)
        err = action.get("error") or "null"
        rewards.append(r)

        print(f"[STEP] step={step} action={action['decision']} reward={r:.2f} done={str(done).lower()} error={err}", flush=True)

        if done:
            break
        obs = res.get("observation")
        if not obs:
            break
        step += 1

    g = call("POST", "/grader")
    score = g.get("score", 0.0)
    rstr = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(score > 0.6).lower()} steps={step} score={score:.3f} rewards={rstr}", flush=True)


if __name__ == "__main__":
    for tid in ["task_1_easy", "task_2_medium", "task_3_hard"]:
        run_task(tid)
