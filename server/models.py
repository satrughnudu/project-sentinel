# models.py — Project Sentinel
# Typed Pydantic v2 models for the OpenEnv environment interface.

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class ToolCallRequest(BaseModel):
    """Incoming tool-call request from a simulated actor.
    The agent must inspect this and decide if it is safe or dangerous."""

    tool_name: str = Field(..., description="Name of the tool/function being called.")
    arguments: Dict[str, Any] = Field(..., description="Arguments passed to the tool.")
    source_user: str = Field(..., description="Identity of the requester (can be spoofed).")
    context: Optional[str] = Field(default=None, description="Optional justification or context.")


class Observation(BaseModel):
    """Complete observation given to the agent at each step."""

    request: ToolCallRequest = Field(..., description="The tool-call request to evaluate.")
    remaining_steps: int = Field(..., description="Steps remaining in this episode.")
    prior_decisions: List[str] = Field(default_factory=list, description="Previous decisions this episode.")
    episode_id: str = Field(..., description="Unique identifier for this episode.")


class Action(BaseModel):
    """Agent's decision after evaluating a request."""

    decision: Literal["allow", "block", "quarantine"] = Field(
        ..., description="Security verdict: allow, block, or quarantine."
    )

    @field_validator("decision", mode="before")
    @classmethod
    def lowercase_decision(cls, v):
        """Normalize casing to prevent validation errors from grading agents."""
        return v.lower() if isinstance(v, str) else v

    reasoning: str = Field(..., description="Explanation of the security decision.")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score.")


class StepResult(BaseModel):
    """Result returned by env.step() after an agent action."""

    observation: Optional[Observation] = Field(default=None, description="Next observation (None if done).")
    reward: float = Field(..., description="Reward for the last action.")
    done: bool = Field(..., description="True if episode is finished.")
    info: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic metadata.")
