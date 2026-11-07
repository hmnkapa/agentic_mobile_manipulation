"""Client-independent contracts. Authentication is inherited, never copied into runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict


class CanonicalEvent(TypedDict, total=False):
    """Public stream data with original line provenance; missing timestamps stay null.

    kind: assistant_text, reasoning_summary, tool_call, tool_result, usage,
    session_started, run_finished, client_error. No inferred/private reasoning.
    """
    at: str | int | float | None
    kind: str
    client: str
    turn: int | None
    call_id: str
    server: str
    tool: str
    arguments: Any
    content: Any
    result: Any
    usage: dict
    reasoning_summary: str
    source_line: int
    source: str
    error: Any


@dataclass(frozen=True)
class ClientConfig:
    client: str
    model: str
    instruction: str
    mcp_command: str
    mcp_args: list[str]
    output: Path
    timeout: float | None = 900
    agent_profile: str = "skills"
    isolate_client_storage: bool = False
    reasoning_effort: str | None = None
    model_provider_profile: str | None = None
    model_provider_gateway: str | None = None
    qwen_config: Path | None = None

    def __post_init__(self):
        if self.client not in {"codex", "opencode", "kimi", "qwen"}:
            raise ValueError(f"Unsupported client: {self.client}")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if not isinstance(self.mcp_args, list) or any(not isinstance(x, str) for x in self.mcp_args):
            raise ValueError("mcp_args must be an argv string array")
        if not self.model or not self.mcp_command:
            raise ValueError("model and mcp_command must be nonempty")
        if self.reasoning_effort is not None and self.reasoning_effort not in {'low','medium','high','xhigh'}:
            raise ValueError('Unsupported reasoning effort')
        if self.client != 'codex' and (self.reasoning_effort is not None or self.model_provider_profile is not None):
            raise ValueError('Model provider and reasoning effort overrides are supported only for Codex')
        if self.model_provider_gateway is not None and (self.client != 'codex' or not self.model_provider_profile):
            raise ValueError('A provider gateway requires an explicit Codex provider profile')
        if self.client == 'qwen' and (self.qwen_config is None or self.timeout is not None):
            raise ValueError('Qwen requires qwen_config and an unbounded episode')
        if self.client != 'qwen' and self.qwen_config is not None:
            raise ValueError('qwen_config applies only to Qwen')

    @property
    def prompt(self) -> str:
        return "Use only the manipulation MCP tools to complete this simulation task.\n" + self.instruction


@dataclass(frozen=True)
class ClientCapability:
    client: str
    status: Literal["available", "unavailable", "unsupported"]
    version: str | None = None
    executable: str | None = None
    reason: str | None = None
    # CLI detection does not establish model login, RGB delivery or simulator success.
    validation_level: str = "cli_capability_probe"


@dataclass
class PreparedProject:
    argv: list[str]
    cwd: Path | None
    stdin: str | None
    env_overlay: dict[str, str] = field(default_factory=dict)
    native_sessions: Path | None = None
    client_storage: dict | None = None
    mcp_prefix: list[str] = field(default_factory=list)


@dataclass
class ClientRunResult:
    returncode: int
    timed_out: bool
    policy_finished_at_unix: float
    duration_seconds: float
    pid: int
    finish_received: bool = False
    cleanup_reason: str | None = None


@dataclass
class ParsedEvents:
    raw_events: list[dict]
    canonical: list[CanonicalEvent]
    compat: list[dict]
    invalid_lines: list[int]
    tools_called: list[str]
    formal_finish_observed: bool
