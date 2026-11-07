"""Normalize public client streams without rewriting the raw stdout evidence.

Adapters preserve call IDs, MCP content blocks (including images), and source lines.
Unknown records remain in model_events.jsonl. Missing clocks are not manufactured.
"""
from __future__ import annotations

import json
from pathlib import Path
from .types import ParsedEvents


def event_path(directory: Path) -> Path:
    compat = directory / "model_events.compat.jsonl"
    return compat if compat.exists() else directory / "model_events.jsonl"


def _object(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _tool(name, server=None):
    name = str(name or "")
    for prefix in ("mcp__manipulation__", "manipulation_", "manipulation."):
        if name.startswith(prefix):
            return "manipulation", name[len(prefix):]
    return server or "unknown", name


def _result(value):
    value = _object(value)
    if isinstance(value, dict) and isinstance(value.get("content"), list):
        return value
    if isinstance(value, list):
        return {"content": value}
    if isinstance(value, dict) and "output" in value:
        return _result(value["output"])
    return {"content": [{"type": "text", "text": value if isinstance(value, str) else json.dumps(value)}]}


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(x.get("text", "") for x in content if isinstance(x, dict) and x.get("type") == "text")
    return ""


def successful_finish_event(event):
    """Only a completed successful receipt closes the policy, never its request."""
    if not isinstance(event, dict) or event.get('type') != 'item.completed':
        return False
    item = event.get('item') or {}
    if not isinstance(item, dict) or item.get('type') != 'mcp_tool_call':
        return False
    if (item.get('server'), item.get('tool')) != ('manipulation', 'finish') or item.get('error'):
        return False
    result = item.get('result') or {}
    if not isinstance(result, dict) or result.get('isError'):
        return False
    for block in result.get('content') or []:
        if isinstance(block, dict) and block.get('type') == 'text':
            value = _object(block.get('text'))
            if isinstance(value, dict) and value.get('ok') is True and value.get('closed') is True:
                return True
    return False


def parse_events(path: Path, client: str) -> ParsedEvents:
    rows = []; invalid = []; canonical = []; compat = []
    calls = {}; emitted = set(); completed = set(); turn = 0

    def emit(kind, line, at, **values):
        event = {"at": at, "kind": kind, "client": client, "turn": turn or None,
                 "source": "model_events.jsonl", "source_line": line, **values}
        canonical.append(event)
        return event

    def message(content, line, at):
        text = _text(content)
        if text:
            emit("assistant_text", line, at, content=text)
            if client != "codex":
                compat.append({"type": "item.completed", "at": at, "source": client + "_adapter",
                    "source_line": line, "item": {"type": "agent_message", "id": f"message-{line}", "text": text}})

    def tool_call(call_id, name, arguments, line, at, server=None):
        server, name = _tool(name, server)
        call_id = str(call_id or f"call-{line}")
        item = {"id": call_id, "type": "mcp_tool_call", "server": server, "tool": name,
                "arguments": _object(arguments) if arguments is not None else {}}
        calls[call_id] = item
        if call_id not in emitted:
            emit("tool_call", line, at, call_id=call_id, server=server, tool=name, arguments=item["arguments"])
            if client != "codex":
                compat.append({"type": "item.started", "at": at, "source": client + "_adapter",
                               "source_line": line, "item": item.copy()})
            emitted.add(call_id)
        return call_id

    def tool_result(call_id, value, line, at, error=None):
        call_id = str(call_id or f"call-{line}")
        if call_id in completed:
            return
        item = calls.get(call_id, {"id": call_id, "type": "mcp_tool_call", "server": "unknown",
                                   "tool": "unknown", "arguments": {}})
        result = _result(value) if value is not None else None
        emit("tool_result", line, at, call_id=call_id, server=item["server"], tool=item["tool"],
             result=result, error=error)
        if client != "codex":
            compat.append({"type": "item.completed", "at": at, "source": client + "_adapter",
                           "source_line": line, "item": {**item, "result": result, "error": error}})
        completed.add(call_id)

    if not path.exists():
        return ParsedEvents([], [], [], [], [], False)
    for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not line.lstrip().startswith("{"):
            continue
        try:
            data = json.loads(line)
        except ValueError:
            invalid.append(line_no); continue
        if not isinstance(data, dict):
            continue
        rows.append(data)
        at = data.get("at", data.get("timestamp", data.get("time")))
        typ = data.get("type")
        if client in {"codex", "qwen"}:
            if client == "codex":
                compat.append(data)
            if typ == "turn.started":
                turn += 1
            item = data.get("item") or {}
            if typ == "item.completed" and item.get("type") == "agent_message":
                message(item.get("text"), line_no, at)
            if item.get("type") == "mcp_tool_call" and typ in {"item.started", "item.completed"}:
                call_id = tool_call(item.get("id"), item.get("tool"), item.get("arguments"), line_no, at, item.get("server"))
                if typ == "item.completed":
                    tool_result(call_id, item.get("result"), line_no, at, item.get("error"))
            if typ == "thread.started":
                emit("session_started", line_no, at, content={"thread_id": data.get("thread_id")})
        elif client == "opencode":
            part = data.get("part") or {}; state = part.get("state") or {}
            if typ == "step_start":
                turn += 1
            if typ == "text" or part.get("type") == "text":
                message(part.get("text", data.get("text")), line_no, at)
            if typ == "tool_use" or part.get("type") == "tool":
                call_id = tool_call(part.get("callID", part.get("id")), part.get("tool", data.get("name")),
                                    state.get("input", data.get("input")), line_no, at)
                if state.get("status") in {"completed", "error"} or "output" in state:
                    value = state.get("output")
                    if state.get("attachments"):
                        # Preserve attachments as recorded; do not invent delivered image bytes.
                        value = {**_result(value), "attachments": state["attachments"]}
                    tool_result(call_id, value, line_no, at, state.get("error"))
        elif client == "kimi":
            if typ == "context.append_loop_event":
                data = data.get("event") or {}; typ = data.get("type"); at = at or data.get("time")
            elif isinstance(data.get("payload"), dict):
                data = data["payload"]; typ = data.get("type", typ)
            role = data.get("role")
            if role == "assistant" or typ in {"assistant_message", "assistant", "message"}:
                turn += 1
                message(data.get("content", data.get("text")), line_no, at)
                for call in data.get("tool_calls") or []:
                    function = call.get("function") or call
                    tool_call(call.get("id"), function.get("name"), function.get("arguments", function.get("args")), line_no, at)
            if typ in {"tool.call", "tool_call", "function_call"}:
                tool_call(data.get("toolCallId", data.get("id", data.get("call_id"))), data.get("name", data.get("tool")),
                          data.get("args", data.get("arguments", data.get("input"))), line_no, at)
            if role == "tool" or typ in {"tool.result", "tool_result", "tool_call_result", "function_call_output"}:
                call_id = data.get("tool_call_id", data.get("toolCallId", data.get("call_id", data.get("id"))))
                tool_result(call_id, data.get("result", data.get("content", data.get("output"))), line_no, at, data.get("error"))
            if typ == "content.part":
                part = data.get("part") or {}
                if part.get("type") == "text":
                    message(part.get("text"), line_no, at)
        # Only an explicitly identified summary is exported as a summary. Reasoning
        # token counters, encrypted content and think/reasoning blocks are not summaries.
        summary = data.get("reasoning_summary")
        if isinstance(summary, str):
            emit("reasoning_summary", line_no, at, reasoning_summary=summary)
        usage = data.get("usage") or (data.get("part") or {}).get("tokens")
        if isinstance(usage, dict):
            emit("usage", line_no, at, usage=usage)
            if client != "codex":
                compat.append({"type": "turn.completed", "at": at, "source": client + "_adapter", "usage": usage})
        if typ in {"error", "turn.failed"}:
            emit("client_error", line_no, at, error=data.get("error", data.get("message")))
        if typ in {"turn.completed", "step_finish", "step.end", "done", "result"}:
            emit("run_finished" if typ in {"done", "result"} else "turn_finished", line_no, at)

    tools = []; closed = False
    for event in compat:
        item = event.get("item") or {}
        if event.get("type") != "item.completed" or item.get("type") != "mcp_tool_call":
            continue
        tools.append(item.get("tool"))
        closed = closed or successful_finish_event(event)
    return ParsedEvents(rows, canonical, compat, invalid, tools, closed)


def write_derived_events(output: Path, parsed: ParsedEvents, client: str):
    def write(path, events):
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in events))
    write(output / "canonical_events.jsonl", parsed.canonical)
    if client != "codex":
        write(output / "model_events.compat.jsonl", parsed.compat)
