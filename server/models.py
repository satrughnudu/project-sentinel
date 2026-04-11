# Pydantic models for the Sentinel environment API.
# These map 1:1 to the OpenEnv observation/action/step protocol.

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator


class ToolCallRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    source_user: str
    context: Optional[str] = None


class SystemState(BaseModel):
    network_health: int = 100
    compromised_services: List[str] = Field(default_factory=list)
    active_threats: int = 0
    security_posture: str = "nominal"
    incident_count: int = 0


class InvestigationResult(BaseModel):
    investigation_type: str
    findings: Dict[str, Any]
    risk_indicators: List[str] = Field(default_factory=list)


class Observation(BaseModel):
    request: ToolCallRequest
    system_state: SystemState
    remaining_steps: int
    prior_decisions: List[str] = Field(default_factory=list)
    episode_id: str
    investigation_results: List[InvestigationResult] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)


class Action(BaseModel):
    decision: Literal[
        "allow", "block", "quarantine",
        "investigate_user", "investigate_payload", "check_system",
    ]
    reasoning: str
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @field_validator("decision", mode="before")
    @classmethod
    def _normalize(cls, v):
        # agents sometimes send "BLOCK" or "Block" - just lowercase it
        if isinstance(v, str):
            return v.lower()
        return v


class StepResult(BaseModel):
    observation: Optional[Observation] = None
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
