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
        for line in result.stdout.splitlines():
            if "score:" in line.lower():
                parts = line.split(":")
                try:
                    val = float(parts[-1].strip())
                    if "task_1" in line.lower() or "easy" in line.lower(): scores["task_1_easy"] = val
                    elif "task_2" in line.lower() or "medium" in line.lower(): scores["task_2_medium"] = val
                    elif "task_3" in line.lower() or "hard" in line.lower(): scores["task_3_hard"] = val
                    elif "average" in line.lower(): scores["average"] = val
                except ValueError:
                    pass
        
        return {
            "task_1_easy": scores.get("task_1_easy", 0.0),
            "task_2_medium": scores.get("task_2_medium", 0.0),
            "task_3_hard": scores.get("task_3_hard", 0.0),
            "average": scores.get("average", 0.0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to run inference: {str(e)}")
