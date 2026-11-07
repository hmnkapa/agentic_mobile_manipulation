"""Shared process runner; deadline starts upstream after simulator/MCP initialization."""
from __future__ import annotations

import os
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Callable

from .types import ClientCapability, ClientRunResult, PreparedProject
from .events import parse_events, successful_finish_event


def run_process(project: PreparedProject, output: Path, timeout: float | None,
                on_start: Callable[[int], None] | None = None, *,
                event_client: str = 'codex', finish_exit_grace: float = 5.0,
                model_startup_watchdog: float | None = None) -> ClientRunResult:
    started = time.monotonic()
    timed_out = False
    finish_at = finish_monotonic = None
    cleanup_reason = None
    model_ready = False
    kwargs = {"start_new_session": True, "text": True}
    if project.cwd is not None:
        kwargs["cwd"] = str(project.cwd)
    if project.env_overlay:
        kwargs["env"] = {**os.environ, **project.env_overlay}
    with (output / "model_events.jsonl").open("w") as stdout, (output / "client.stderr.log").open("w") as stderr:
        process = subprocess.Popen(project.argv, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, **kwargs)
        if on_start:
            on_start(process.pid)
        # stdout remains the unmodified evidence file. Tail complete lines once,
        # without reparsing all previous RGB payloads on each poll.
        with (output / 'model_events.jsonl').open('rb') as stream:
            first = True
            while True:
                try:
                    process.communicate(project.stdin if first else None, timeout=0.1)
                except subprocess.TimeoutExpired:
                    pass
                first = False
                if event_client in {'codex', 'qwen'} and finish_at is None:
                    while True:
                        offset = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith(b'\n'):
                            stream.seek(offset)
                            break
                        if b'"finish"' not in line or b'"mcp_tool_call"' not in line:
                            continue
                        try:
                            receipt = successful_finish_event(json.loads(line))
                        except (ValueError, UnicodeError):
                            receipt = False
                        if receipt:
                            finish_at, finish_monotonic = time.time(), time.monotonic()
                            break
                if process.poll() is not None:
                    finished = finish_at or time.time()
                    break
                if model_startup_watchdog is not None and not model_ready:
                    path=output/'qwen_runtime.json'
                    try:
                        state=json.loads(path.read_text())
                        model_ready=bool(state.get('model_ready_at'))
                    except (OSError,ValueError):
                        pass
                    if not model_ready and time.monotonic()-started>=model_startup_watchdog:
                        finished=time.time()
                        cleanup_reason='model_initialization_watchdog'
                        break
                if finish_monotonic is not None:
                    if time.monotonic() - finish_monotonic >= finish_exit_grace:
                        finished = finish_at
                        cleanup_reason = 'client_linger_after_successful_finish'
                        break
                elif timeout is not None and time.monotonic() - started >= timeout:
                    finished = time.time()
                    timed_out = True
                    break
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=20)
    return ClientRunResult(process.returncode, timed_out, finished,
                           time.monotonic() - started, process.pid,
                           finish_received=finish_at is not None, cleanup_reason=cleanup_reason)


class ClientAdapter:
    name: str
    required_flags: tuple[str, ...] = ()
    help_args: tuple[str, ...] = ("--help",)

    def probe(self) -> ClientCapability:
        executable = shutil.which(self.name)
        if executable is None:
            return ClientCapability(self.name, "unavailable", reason="Executable is not installed on PATH")
        try:
            version = subprocess.check_output([self.name, "--version"], text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
            if self.required_flags:
                help_text = subprocess.check_output([self.name, *self.help_args], text=True, stderr=subprocess.DEVNULL, timeout=10)
                missing = [flag for flag in self.required_flags if flag not in help_text]
                if missing:
                    return ClientCapability(self.name, "unsupported", version, executable,
                                            "Required CLI flags missing: " + ", ".join(missing))
            return ClientCapability(self.name, "available", version, executable)
        except (OSError, subprocess.SubprocessError) as exc:
            return ClientCapability(self.name, "unsupported", executable=executable,
                                    reason="CLI capability check failed: " + type(exc).__name__)

    def prepare_project(self, config, instructions: str, tool_names: list[str]) -> PreparedProject:
        raise NotImplementedError

    def run(self, project, output, timeout, on_start=None):
        return run_process(project, output, timeout, on_start, event_client=self.name)

    def parse(self, path: Path):
        return parse_events(path, self.name)
