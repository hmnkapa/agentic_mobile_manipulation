"""Synthetic CPU fixtures only; these are not autonomous simulator experiments."""
from copy import deepcopy
import base64
import hashlib
import io
from itertools import combinations
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
import zlib

from manipulation_agent.sft.cli import main
from manipulation_agent.sft.common import ConversionError
from manipulation_agent.sft.dataset import (choose_validation_scenes, episode_samples,
    manifest_sources, prepare_dataset, write_package)
from manipulation_agent.sft.history import compress_history, validate_record
from manipulation_agent.sft.protocol import (build_student_context, from_student_arguments,
    rewrite_interface_text, to_student_arguments)
from manipulation_agent.sft.source import EpisodeSource, load_episode
from manipulation_agent.sft.verify_encode import check_encoding
from manipulation_agent.tools import tool_specs
from manipulation_agent.vision_policy import system_prompt



from sft_fixtures import ROOT, Fixture, save, save_lines, png


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def fixture(self, name="one", **kwargs):
        return Fixture(self.root / name, **kwargs)
































    def test_packaging_is_self_contained_and_preserves_sources(self):
        fixture = self.fixture()
        source = fixture.simple()
        originals = {p: p.read_bytes() for p in (fixture.run / "run.json", fixture.ctrl / "model_events.jsonl")}
        data = prepare_dataset([source])
        output = write_package(data, self.root / "package")
        rows = [json.loads(line) for line in (output / "dataset.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(json.loads((output / "stats.json").read_text())["schema_version"], 1)
        self.assertEqual(json.loads((output / "config.json").read_text())["converter_version"], "manipulation_sft_v1.3.0")
        self.assertEqual(set(rows[0]), {"messages", "images", "tools"})
        for row in rows:
            for image in row["images"]:
                self.assertFalse(Path(image).is_absolute())
                self.assertTrue((output / image).is_file())
        self.assertEqual(len(list((output / "images").iterdir())), 8)
        for p, before in originals.items():
            self.assertEqual(p.read_bytes(), before)
        with self.assertRaisesRegex(ConversionError, "new output"):
            write_package(data, output)
        moved = self.root / "moved_package"
        output.rename(moved)
        shutil.rmtree(fixture.root)
        self.assertTrue(all((moved / path).is_file() for row in rows for path in row["images"]))

    def test_source_changes_between_check_and_write_are_rejected(self):
        fixture = self.fixture()
        source = fixture.simple()
        data = prepare_dataset([source])
        with (fixture.ctrl / "model_events.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ConversionError, "input changed"):
            write_package(data, self.root / "package")
        self.assertFalse((self.root / "package").exists())

    def test_output_inside_source_is_rejected(self):
        fixture = self.fixture()
        data = prepare_dataset([fixture.simple()])
        with self.assertRaisesRegex(ConversionError, "must be separate"):
            write_package(data, fixture.run / "sft")
        self.assertFalse((fixture.run / "sft").exists())

    def test_batch_isolates_errors_and_keeps_scenes_disjoint_deterministically(self):
        a, b, broken = self.fixture("a", scene="scene_a"), self.fixture("b", scene="scene_b"), self.fixture("bad", scene="scene_c")
        sources = [f.simple() for f in (a, b, broken)]
        (broken.run / broken.captures[0]["images"][0]["file"]).write_bytes(b"bad")
        data = prepare_dataset(sources, batch=True)
        self.assertEqual(data.stats["error_episodes"], 1)
        self.assertTrue(data.errors[0]["task_success"])
        second = prepare_dataset(list(reversed(sources)), batch=True)
        self.assertEqual(data.validation_scenes, second.validation_scenes)
        self.assertEqual(list(data.records()), list(second.records()))
        by_split = {split: {m["scene"] for _, m in data.records() if m["split"] == split} for split in ("train", "val")}
        self.assertFalse(by_split["train"] & by_split["val"])
        output = write_package(data, self.root / "batch")
        self.assertEqual(len((output / "errors.jsonl").read_text().splitlines()), 1)
        self.assertTrue((output / "train.jsonl").stat().st_size)
        self.assertTrue((output / "val.jsonl").stat().st_size)

    def test_single_scene_batch_does_not_fabricate_validation_split(self):
        fixture = self.fixture()
        with self.assertRaisesRegex(ConversionError, "at least two"):
            prepare_dataset([fixture.simple()], batch=True)
        counts = {f"scene_{i}": i + 1 for i in range(20)}
        self.assertEqual(choose_validation_scenes(counts), choose_validation_scenes(dict(reversed(list(counts.items())))))
        self.assertLess(len(choose_validation_scenes(counts)), len(counts))

    def test_scene_split_finds_closest_full_group_not_only_seeded_prefix(self):
        counts = {"house_double_floor_upper": 208, "house_single_floor": 1379,
                  "house_double_floor_lower": 1030, "office_cubicles_right": 111,
                  "restaurant_diner": 256, "hotel_suite_large": 140,
                  "Rs_int": 116, "school_geography": 83}
        selected = choose_validation_scenes(counts)
        self.assertEqual(selected, ["hotel_suite_large", "office_cubicles_right", "school_geography"])
        self.assertEqual(sum(counts[s] for s in selected), 334)
        self.assertEqual(selected, choose_validation_scenes(dict(reversed(list(counts.items())))))

    def test_scene_split_optimum_and_seeded_ties_match_exhaustive_search(self):
        for values in ((1, 1, 18), (3, 3, 24), (5, 10), (1, 3, 9, 10, 12)):
            counts = {f"scene_{i}": value for i, value in enumerate(values)}
            for seed in (42, 17):
                with self.subTest(values=values, seed=seed):
                    ordered = sorted(counts, key=lambda s: (hashlib.sha256(f"{seed}:{s}".encode()).hexdigest(), s))
                    groups = [group for size in range(1, len(ordered))
                              for group in combinations(range(len(ordered)), size)]
                    expected = min(groups, key=lambda group: (
                        abs(10 * sum(counts[ordered[i]] for i in group) - sum(counts.values())), group))
                    self.assertEqual(choose_validation_scenes(counts, seed=seed), sorted(ordered[i] for i in expected))

    def test_manifest_paths_resolve_from_manifest_and_cli_defaults_to_dry_run(self):
        fixture = self.fixture()
        source = fixture.simple()
        manifest = self.root / "sources.jsonl"
        save_lines(manifest, [{"run_dir": "one/episode", "controller_dir": "one/controller"}])
        self.assertEqual(manifest_sources(manifest), [source])
        before = set(self.root.iterdir())
        output = io.StringIO()
        with redirect_stdout(output):
            result = main(["--run-dir", str(source.run_dir), "--controller-dir", str(source.controller_dir)])
        self.assertEqual(result, 0)
        self.assertTrue(json.loads(output.getvalue())["dry_run"])
        self.assertEqual(set(self.root.iterdir()), before)

    def test_cpu_package_import_has_no_simulator_dependencies(self):
        result = subprocess.run([sys.executable, "-B", "-c",
            "import sys; import manipulation_agent.sft.dataset; "
            "assert not any(k.startswith(('omnigibson', 'omni.', 'habitat', 'swift', 'transformers')) for k in sys.modules)"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)




if __name__ == "__main__":
    unittest.main()
