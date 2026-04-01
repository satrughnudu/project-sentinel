import os
import subprocess
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env import SentinelEnv, TASK_SCENARIOS
from models import Action, Observation, StepResult

class ResetRequest(BaseModel):
    task_id: str

app = FastAPI(title="Project Sentinel — AI Security Firewall", version="1.0.0")
env = SentinelEnv()

@app.get("/")
def root():
    return {"status": "running", "project": "Project Sentinel", "version": "1.0.0", "tasks": 3}

@app.post("/reset", response_model=Observation)
def reset_environment(body: ResetRequest):
    """Starts a new episode returning an Observation."""
    try:
        return env.reset(task_id=body.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", response_model=StepResult)
def step_environment(action: Action):
    """Takes an agent Action and advances correctly returning StepResult."""
    try:
        return env.step(action=action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def get_state():
    """State endpoint exporting serialized env state dict."""
    return env.state()

@app.get("/tasks")
def list_tasks():
    """Generates task configurations dynamically"""
    action_schema = {
        "decision": {
            "type": "string",
            "enum": ["allow", "block", "quarantine"],
            "description": "The security verdict"
        },
        "reasoning": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    }
    tasks = [
        {
            "id": "task_1_easy",
            "name": "The Drunk Butler",
            "description": "Obvious malicious commands.",
            "difficulty": "easy",
            "action_schema": action_schema
        },
        {
            "id": "task_2_medium",
            "name": "The Shadow Prompt",
            "description": "Attacks hidden inside normal-looking requests.",
            "difficulty": "medium",
            "action_schema": action_schema
        },
        {
            "id": "task_3_hard",
            "name": "The Long Game",
            "description": "Multi-step social engineering.",
            "difficulty": "hard",
            "action_schema": action_schema
        }
    ]
    return tasks

@app.post("/grader")
def run_grader():
    """Generates the [0,1] normalized episodic score upon completion."""
    try:
        if env.current_task_id is None or not env.done:
            raise HTTPException(status_code=400, detail="Episode not started or not finished.")
        return {
            "score": env.grader(),
            "task_id": env.current_task_id,
            "total_reward": env.total_reward,
            "steps_taken": len(env.episode_history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/baseline")
def run_baseline():
    """Server hook running synchronous inference subprocess"""
    missing_vars = [v for v in ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"] if not os.environ.get(v)]
    if missing_vars:
        raise HTTPException(status_code=400, detail=f"Missing required environment variables: {missing_vars}")

    try:
        result = subprocess.run([sys.executable, "inference.py"], capture_output=True, text=True, timeout=1200, env=os.environ.copy())
        scores = {}
        current_task = None
        
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("[START]"):
                parts = dict(kv.split("=", 1) for kv in line.split(" ")[1:] if "=" in kv)
                current_task = parts.get("task", "")
            elif line.startswith("[END]") and current_task:
                parts = dict(kv.split("=", 1) for kv in line.split(" ")[1:] if "=" in kv)
                try:
                    scores[current_task] = float(parts.get("score", "0.0"))
                except ValueError:
                    pass
                current_task = None
                
        # Calculate Average since inference.py no longer prints it explicitly
        avg = sum(scores.values()) / len(scores) if scores else 0.0
        scores["average"] = avg
        
        return {
            "task_1_easy": scores.get("task_1_easy", 0.0),
            "task_2_medium": scores.get("task_2_medium", 0.0),
            "task_3_hard": scores.get("task_3_hard", 0.0),
            "average": scores.get("average", 0.0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run inference: {str(e)}")
