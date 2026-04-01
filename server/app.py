import os
import random
import subprocess
import sys
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from server.env import SentinelEnv, TASK_SCENARIOS
from server.models import Action, Observation, StepResult

app = FastAPI(title="Project Sentinel — AI Security Firewall", version="1.0.0")
env = SentinelEnv()

# ── ROOT ─────────────────────────────────────────────────────
@app.get("/")
def root():
    """Health check — returns running status. Grader pings this first."""
    return {
        "status": "running",
        "project": "Project Sentinel",
        "version": "1.0.0",
        "tasks": list(TASK_SCENARIOS.keys()),
    }

# ── RESET ─────────────────────────────────────────────────────
@app.post("/reset", response_model=Observation)
async def reset_environment(request: Request):
    """
    Starts a new episode and returns the first Observation.

    Accepts an optional JSON body: {"task_id": "task_1_easy"}
    If no body / no task_id is provided → a random task is selected.
    This makes the endpoint robust against bare POST /reset pings from
    Scaler's automated pre-validation script.
    """
    task_id = None
    try:
        # Read body safely — works even when Content-Type is missing or body is empty
        body_bytes = await request.body()
        if body_bytes:
            import json
            body = json.loads(body_bytes)
            task_id = body.get("task_id") if isinstance(body, dict) else None
    except Exception:
        task_id = None

    # Fall back to a random task if none specified
    if not task_id:
        task_id = random.choice(list(TASK_SCENARIOS.keys()))

    try:
        return env.reset(task_id=task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ── STEP ──────────────────────────────────────────────────────
@app.post("/step", response_model=StepResult)
def step_environment(action: Action):
    """
    Submits an Action (decision + reasoning) and advances the episode.
    Returns a StepResult with reward, next observation, and done flag.
    """
    try:
        return env.step(action=action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── STATE ─────────────────────────────────────────────────────
@app.get("/state")
def get_state():
    """Returns full environment state for debugging and transparency."""
    return env.state()

# ── TASKS ─────────────────────────────────────────────────────
@app.get("/tasks")
def list_tasks():
    """Lists all available tasks with their schemas."""
    action_schema = {
        "decision": {
            "type": "string",
            "enum": ["allow", "block", "quarantine"],
            "description": "The security verdict",
        },
        "reasoning": {"type": "string", "description": "Why did you make this decision?"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    return [
        {
            "id": "task_1_easy",
            "name": "The Drunk Butler",
            "description": "Obvious malicious commands — destructive tool names and clear intent.",
            "difficulty": "easy",
            "num_scenarios": 8,
            "action_schema": action_schema,
        },
        {
            "id": "task_2_medium",
            "name": "The Shadow Prompt",
            "description": "Attacks hidden inside normal-looking requests via prompt injection.",
            "difficulty": "medium",
            "num_scenarios": 8,
            "action_schema": action_schema,
        },
        {
            "id": "task_3_hard",
            "name": "The Long Game",
            "description": "Multi-step social engineering with escalating false authorizations.",
            "difficulty": "hard",
            "num_scenarios": 8,
            "action_schema": action_schema,
        },
    ]

# ── GRADER ────────────────────────────────────────────────────
@app.post("/grader")
def run_grader():
    """
    Returns the normalized [0.0, 1.0] score for the completed episode.
    If no episode has been run yet, returns score=0.0 gracefully.
    Never raises 4xx/5xx — always returns a valid JSON response.
    """
    try:
        score = env.grader() if env.done else 0.0
        return {
            "score": round(float(score), 4),
            "task_id": env.current_task_id,
            "total_reward": round(float(env.total_reward), 4),
            "steps_taken": int(len(env.episode_history)),
            "done": bool(env.done),
        }
    except Exception as e:
        # Always return a valid response — never let grader crash the server
        return JSONResponse(
            status_code=200,
            content={"score": 0.0, "task_id": None, "total_reward": 0.0, "steps_taken": 0, "done": False, "error": str(e)},
        )

# ── BASELINE ──────────────────────────────────────────────────
@app.get("/baseline")
def run_baseline():
    """Runs the inference.py script as a subprocess and returns collected scores."""
    missing = [v for v in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"] if not os.environ.get(v)]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing environment variables: {missing}")

    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=1200,
            env=os.environ.copy(),
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
