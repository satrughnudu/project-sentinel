import json
import logging
import os
import random
import subprocess
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from server.env import SentinelEnv, TASK_DEFS
from server.models import Action, Observation, StepResult

log = logging.getLogger(__name__)

app = FastAPI(title="Project Sentinel", version="2.0.0")
env = SentinelEnv()


@app.get("/")
def root():
    return {"status": "running", "project": "Project Sentinel",
            "version": "2.0.0", "tasks": list(TASK_DEFS.keys())}


@app.post("/reset", response_model=Observation)
async def reset_env(request: Request):
    # handle bare POST with no body (scaler's pre-validator does this)
    task_id = None
    try:
        raw = await request.body()
        if raw:
            body = json.loads(raw)
            task_id = body.get("task_id") if isinstance(body, dict) else None
    except Exception:
        pass

    if not task_id:
        task_id = random.choice(list(TASK_DEFS.keys()))

    try:
        return env.reset(task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
def step_env(action: Action):
    try:
        return env.step(action=action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def get_state():
    return env.state()


@app.get("/tasks")
def list_tasks():
    schema = {
        "decision": {"type": "string", "enum": ["allow", "block", "quarantine",
                      "investigate_user", "investigate_payload", "check_system"]},
        "reasoning": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    return [
        {"id": "task_1_easy", "name": "The Drunk Butler",
         "description": "Obvious destructive attacks mixed with routine ops.",
         "difficulty": "easy", "num_scenarios": 8, "action_schema": schema},
        {"id": "task_2_medium", "name": "The Shadow Prompt",
         "description": "Prompt injection and social engineering hidden in normal requests.",
         "difficulty": "medium", "num_scenarios": 8, "action_schema": schema},
        {"id": "task_3_hard", "name": "The Long Game",
         "description": "Trust-building followed by escalating social engineering attacks.",
         "difficulty": "hard", "num_scenarios": 8, "action_schema": schema},
    ]


@app.post("/grader")
def grader():
    try:
        score = env.grader() if env.done else 0.0
        return {
            "score": round(float(score), 4),
            "task_id": env.current_task_id,
            "total_reward": round(float(env.total_reward), 4),
            "steps_taken": len(env.episode_history),
            "done": env.done,
            "system_health": env.system_state.get("network_health", 100),
        }
    except Exception as e:
        # never crash the grader - return 0 if something weird happens
        return JSONResponse(status_code=200, content={
            "score": 0.0, "task_id": None, "total_reward": 0.0,
            "steps_taken": 0, "done": False, "error": str(e)})


@app.get("/baseline")
def baseline():
    missing = [v for v in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"] if not os.environ.get(v)]
    if missing:
        raise HTTPException(400, detail=f"Missing env vars: {missing}")

    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True, text=True, timeout=1200, env=os.environ.copy())

        scores = {}
        cur_task = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("[START]"):
                parts = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
                cur_task = parts.get("task")
            elif line.startswith("[END]") and cur_task:
                parts = dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
                try:
                    scores[cur_task] = float(parts.get("score", "0"))
                except ValueError:
                    pass
                cur_task = None

        avg = sum(scores.values()) / len(scores) if scores else 0.0
        return {"task_1_easy": scores.get("task_1_easy", 0.0),
                "task_2_medium": scores.get("task_2_medium", 0.0),
                "task_3_hard": scores.get("task_3_hard", 0.0),
                "average": round(avg, 4),
                "stderr": result.stderr[-500:] if result.stderr else None}
    except subprocess.TimeoutExpired:
        raise HTTPException(504, detail="Inference timed out")
    except Exception as e:
        raise HTTPException(500, detail=str(e))


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
