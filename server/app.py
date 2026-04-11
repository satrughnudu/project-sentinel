import os
import random
import subprocess
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from server.env import SentinelEnv, TASK_DEFINITIONS
from server.models import Action, Observation, StepResult

app = FastAPI(title="Project Sentinel — AI Security Firewall", version="2.0.0")
env = SentinelEnv()

@app.get("/")
def root():
    return {
        "status": "running",
        "project": "Project Sentinel",
        "version": "2.0.0",
        "tasks": list(TASK_DEFINITIONS.keys()),
    }

@app.post("/reset", response_model=Observation)
async def reset_environment(request: Request):
    task_id = None
    try:
        body_bytes = await request.body()
        if body_bytes:
            import json
            body = json.loads(body_bytes)
            task_id = body.get("task_id") if isinstance(body, dict) else None
    except Exception:
        task_id = None
    if not task_id:
        task_id = random.choice(list(TASK_DEFINITIONS.keys()))
    try:
        return env.reset(task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResult)
def step_environment(action: Action):
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
    action_schema = {
        "decision": {
            "type": "string",
            "enum": ["allow", "block", "quarantine", "investigate_user", "investigate_payload", "check_system"],
            "description": "Security verdict or investigation action",
        },
        "reasoning": {"type": "string", "description": "Explanation of the decision"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    return [
        {
            "id": "task_1_easy", "name": "The Drunk Butler",
            "description": "Obvious attacks with clear signals mixed with routine operations. Agent can investigate before deciding.",
            "difficulty": "easy", "num_scenarios": 8, "action_schema": action_schema,
        },
        {
            "id": "task_2_medium", "name": "The Shadow Prompt",
            "description": "Subtle prompt injection and data exfiltration hidden inside normal-looking requests.",
            "difficulty": "medium", "num_scenarios": 8, "action_schema": action_schema,
        },
        {
            "id": "task_3_hard", "name": "The Long Game",
            "description": "Multi-step social engineering with trust-building, authority spoofing, and cascading consequences.",
            "difficulty": "hard", "num_scenarios": 8, "action_schema": action_schema,
        },
    ]

@app.post("/grader")
def run_grader():
    try:
        score = env.grader() if env.done else 0.0
        return {
            "score": round(float(score), 4),
            "task_id": env.current_task_id,
            "total_reward": round(float(env.total_reward), 4),
            "steps_taken": int(len(env.episode_history)),
            "done": bool(env.done),
            "system_health": env.system_state.get("network_health", 100),
        }
    except Exception as e:
        return JSONResponse(status_code=200, content={
            "score": 0.0, "task_id": None, "total_reward": 0.0,
            "steps_taken": 0, "done": False, "error": str(e),
        })

@app.get("/baseline")
def run_baseline():
    missing = [v for v in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"] if not os.environ.get(v)]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing environment variables: {missing}")
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True, text=True, timeout=1200, env=os.environ.copy(),
        )
        scores: dict = {}
        current_task = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("[START]"):
                parts = dict(kv.split("=", 1) for kv in line.split(" ")[1:] if "=" in kv)
                current_task = parts.get("task")
            elif line.startswith("[END]") and current_task:
                parts = dict(kv.split("=", 1) for kv in line.split(" ")[1:] if "=" in kv)
                try:
                    scores[current_task] = float(parts.get("score", "0.0"))
                except ValueError:
                    pass
                current_task = None
        avg = sum(scores.values()) / len(scores) if scores else 0.0
        return {
            "task_1_easy": scores.get("task_1_easy", 0.0),
            "task_2_medium": scores.get("task_2_medium", 0.0),
            "task_3_hard": scores.get("task_3_hard", 0.0),
            "average": round(avg, 4),
            "stderr": result.stderr[-500:] if result.stderr else None,
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Inference script timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run inference: {str(e)}")


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
