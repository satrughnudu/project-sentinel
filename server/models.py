# models.py — Project Sentinel v2
# Typed Pydantic v2 models for the agentic security environment.

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class ToolCallRequest(BaseModel):
    """Incoming tool-call request from a simulated actor."""
    tool_name: str = Field(..., description="Name of the tool/function being called.")
    arguments: Dict[str, Any] = Field(..., description="Arguments passed to the tool.")
    source_user: str = Field(..., description="Identity of the requester (can be spoofed).")
    context: Optional[str] = Field(default=None, description="Optional justification or context.")


class SystemState(BaseModel):
    """Current state of the simulated corporate network."""
    network_health: int = Field(default=100, description="Network health 0-100.")
    compromised_services: List[str] = Field(default_factory=list, description="Compromised service names.")
    active_threats: int = Field(default=0, description="Unresolved active threats.")
    security_posture: str = Field(default="nominal", description="Overall security status.")
    incident_count: int = Field(default=0, description="Total incidents processed.")


class InvestigationResult(BaseModel):
    """Results from an investigation action."""
    investigation_type: str = Field(..., description="Type of investigation performed.")
    findings: Dict[str, Any] = Field(..., description="Investigation findings.")
    risk_indicators: List[str] = Field(default_factory=list, description="Identified risk indicators.")


class Observation(BaseModel):
    """Complete observation given to the agent at each step."""
    request: ToolCallRequest = Field(..., description="The tool-call request to evaluate.")
    system_state: SystemState = Field(..., description="Current network state.")
    remaining_steps: int = Field(..., description="Steps remaining in this episode.")
    prior_decisions: List[str] = Field(default_factory=list, description="Previous decisions.")
    episode_id: str = Field(..., description="Unique identifier for this episode.")
    investigation_results: List[InvestigationResult] = Field(default_factory=list, description="Prior investigations on current request.")
    available_actions: List[str] = Field(default_factory=list, description="Actions available to the agent.")


class Action(BaseModel):
    """Agent's decision or investigation action."""
    decision: Literal["allow", "block", "quarantine", "investigate_user", "investigate_payload", "check_system"] = Field(
        ..., description="Security verdict or investigation action."
    )

    @field_validator("decision", mode="before")
    @classmethod
    def lowercase_decision(cls, v):
        return v.lower() if isinstance(v, str) else v

    reasoning: str = Field(..., description="Explanation of the decision.")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score.")


class StepResult(BaseModel):
    """Result returned by env.step() after an agent action."""
    observation: Optional[Observation] = Field(default=None, description="Next observation (None if done).")
    reward: float = Field(..., description="Reward for the last action.")
    done: bool = Field(..., description="True if episode is finished.")
    info: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic metadata.")
