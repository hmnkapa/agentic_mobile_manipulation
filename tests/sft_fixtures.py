"""Shared synthetic CPU fixtures; never autonomous task evidence."""
from copy import deepcopy
import base64
import hashlib
import json
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
import zlib
from manipulation_agent.sft.source import EpisodeSource
from manipulation_agent.tools import tool_specs
from manipulation_agent.vision_policy import system_prompt

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value))


def save_lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def png(r, g):
    def chunk(kind, data):
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack("!2I5B", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes([0, r, g, 0]))) + chunk(b"IEND", b""))


class Fixture:
    def __init__(self, root, *, scene="scene_a"):
        self.root = root
        self.run = root / "episode"
        self.ctrl = root / "controller"
        self.run.mkdir(parents=True)
        self.ctrl.mkdir()
        (self.run / "frames").mkdir()
        shutil.copytree(ROOT / "src", self.ctrl / "source_snapshot", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        self.events, self.raw, self.captures = [], [{"type": "thread.started", "thread_id": "fixture-thread"}], []
        specs = tool_specs("skills")
        digest = hashlib.sha256()
        for path in sorted((self.ctrl / "source_snapshot").rglob("*.py")):
            digest.update(str(Path("src") / path.relative_to(self.ctrl / "source_snapshot")).encode())
            digest.update(path.read_bytes())
        schemas = {s["name"]: s["inputSchema"] for s in specs}
        self.controller = {"controller": "codex_cli", "agent_profile": "skills", "goal_skill": "hierarchical",
            "instruction": "Synthetic CPU fixture instruction", "model": "synthetic-fixture", "model_reasoning_effort": "low",
            "status": "passed", "exit_code": 0, "formal_finish_observed": True,
            "command": ["codex", "exec", "-c", "developer_instructions=" + json.dumps(system_prompt("skills"))],
            "source": {"source_sha256": digest.hexdigest()},
            "mcp_preflight": {"tool_names": sorted(schemas),
                "schema_sha256": hashlib.sha256(json.dumps(schemas, sort_keys=True).encode()).hexdigest()}}
        self.run_record = {"run_id": root.name, "task_success": True,
            "config": {"backend": "omnigibson", "policy": "serve", "observation_mode": "rgb_only",
                       "instruction": self.controller["instruction"], "validation_level": "synthetic_cpu_fixture"},
            "evaluation": {"official_task_success": True, "task_success": True,
                           "task_instance": {"scene": scene, "task": "fixture_task", "instance": 301, "seed": 0},
                           "private_secret": "NEVER_INCLUDE_PRIVATE_EVALUATOR"}}
        self.events.append({"id": "private", "kind": "private_executor_result", "details": {"object": "NEVER_INCLUDE_PRIVATE_OBJECT"}})
        self.observation = None

    def capture(self, *, color_offset=0):
        number = len(self.captures)
        images, archived = [], []
        for i, view in enumerate(("front", "back", "left", "right")):
            ref = f"rgb-{number:05d}-{view}"
            data = png(number + color_offset, i)
            relative = f"frames/{ref}.png"
            (self.run / relative).write_bytes(data)
            sha = hashlib.sha256(data).hexdigest()
            images.append({"image_ref": ref, "view": view, "sha256": sha, "mime_type": "image/png", "width": 1, "height": 1})
            archived.append({"image_ref": ref, "file": relative, "sha256": sha})
        self.captures.append({"images": archived})
        self.observation = {"images": images, "revision": number, "observation_mode": "rgb_only",
                            "capture": {"capture_id": f"capture-{number}", "sim_step": number,
                                        "captured_at": "2026-11-02T00:00:00+00:00", "sim_time_seconds": float(number)}}
        return deepcopy(self.observation)

    def args(self):
        return {"primitive": "toggle_on", "revision": self.observation["revision"],
                "target": {"image_ref": self.observation["images"][2]["image_ref"], "point": [0.1234, 0.9876]},
                "placement_yaw_degrees": None, "wait_seconds": None}

    def add(self, tool, args, result, *, text=""):
        number = sum(e.get("kind") == "tool_call" for e in self.events)
        call_id, request, evidence = f"item_{number}", f"req-{number}", f"result-{number}"
        item = {"id": call_id, "type": "mcp_tool_call", "server": "manipulation", "tool": tool, "arguments": deepcopy(args)}
        if text:
            self.raw.append({"type": "item.completed", "item": {"type": "agent_message", "id": f"text-{number}", "text": text}})
        self.raw.append({"type": "item.started", "item": deepcopy(item)})
        self.events.extend([
            {"id": f"call-{number}", "kind": "tool_call", "request_id": request, "origin": "controller", "name": tool, "arguments": deepcopy(args)},
            {"id": evidence, "kind": "tool_result", "request_id": request, "origin": "controller", "name": tool, "result": deepcopy(result)}])
        payload = {**result, "evidence_id": evidence}
        blocks = [{"type": "text", "text": json.dumps(payload)}]
        for image in result.get("observation", {}).get("images", []):
            data = (self.run / "frames" / (image["image_ref"] + ".png")).read_bytes()
            blocks.extend([{"type": "text", "text": f"RGB view={image['view']}, image_ref={image['image_ref']}"},
                           {"type": "image", "mimeType": "image/png", "data": base64.b64encode(data).decode()}])
        item.update(result={"content": blocks}, error=None)
        self.raw.append({"type": "item.completed", "item": item})

    def initialize(self):
        self.add("initialize", {}, {"ok": True, "initialized": True, "observation": self.capture(),
                  "notes": {"version": 0, "content": ""}, "control": {"manipulation_protocol": "target_locked_approach_v2"}}, text="Public plan")

    def act(self):
        args = self.args()
        self.add("act", args, {"ok": True, "effect": {"status": "completed", "primitive": "toggle_on"},
                              "observation": self.capture()}, text="Public action explanation")

    def finish(self):
        self.add("finish", {"outcome": "achieved", "reason": "Visible completion"}, {"ok": True, "closed": True})

    def simple(self, *, finish=True):
        self.initialize()
        self.act()
        if finish:
            self.finish()
        return self.flush()

    def interleave(self, call_id, texts):
        start = next(i for i, row in enumerate(self.raw)
                     if row["type"] == "item.started" and row["item"]["id"] == call_id)
        for offset, text in enumerate(texts, 1):
            self.raw.insert(start + offset, {"type": "item.completed", "item": {
                "type": "agent_message", "id": f"{call_id}-progress-{offset}", "text": text}})

    def legacy_catalog(self, *, recorded_executor="official_symbolic", recorded_actions="retain"):
        # A frozen CPU-only catalog whose schema depends on historical kwargs.
        path = self.ctrl / "source_snapshot/manipulation_agent/tools/__init__.py"
        path.write_text(path.read_text() + '''
_base_tool_specs = tool_specs
def tool_specs(profile='skills', *, executor='snapshot-custom', nonofficial_actions='snapshot-retain'):
    specs = _base_tool_specs(profile)
    for spec in specs:
        spec['inputSchema']['title'] = executor + '/' + nonofficial_actions
    return specs
''')
        digest = hashlib.sha256()
        for source in sorted((self.ctrl / "source_snapshot").rglob("*.py")):
            digest.update(str(Path("src") / source.relative_to(self.ctrl / "source_snapshot")).encode())
            digest.update(source.read_bytes())
        self.controller["source"]["source_sha256"] = digest.hexdigest()
        schemas = {s["name"]: s["inputSchema"] for s in tool_specs("skills")}
        for schema in schemas.values():
            schema["title"] = recorded_executor + "/" + recorded_actions
        self.controller["mcp_preflight"]["schema_sha256"] = hashlib.sha256(
            json.dumps(schemas, sort_keys=True).encode()).hexdigest()

    def flush(self):
        save(self.run / "run.json", self.run_record)
        save(self.ctrl / "controller.json", self.controller)
        save_lines(self.run / "events.jsonl", self.events)
        save_lines(self.ctrl / "model_events.jsonl", self.raw)
        save_lines(self.run / "captures.jsonl", self.captures)
        save(self.run / "task_notes.json", {"content": "NEVER_READ_FINAL_NOTES"})
        return EpisodeSource(self.run, self.ctrl)


class SftFixtureTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, name="one", **kwargs):
        return Fixture(self.root / name, **kwargs)


def training_package(root, *, rich=False):
    from manipulation_agent.sft.dataset import prepare_dataset, write_package
    sources = []
    for name, scene in (("alpha", "scene_a"), ("beta", "scene_b")):
        f = Fixture(Path(root) / name, scene=scene)
        f.initialize()
        if rich:
            f.add("list_skills", {}, {"ok": True, "skills": []})
            f.add("read_skill", {"name": "visual-exploration", "resource": "SKILL.md"}, {"ok": True, "text": "Skill text"})
            f.add("task_notes", {"operation": "read", "content": None, "expected_version": None},
                  {"ok": True, "notes": {"version": 0, "content": ""}})
            f.add("look", {"yaw_degrees": 0, "revision": f.observation["revision"]},
                  {"ok": True, "observation": f.capture()})
            f.add("act", f.args(), {"ok": False, "error": {"code": "ACTION_FAILED"}, "observation": f.observation})
        for _ in range(18 if rich else 1):
            f.act()
        f.finish()
        sources.append(f.flush())
    return write_package(prepare_dataset(sources, batch=True), Path(root) / "package")
