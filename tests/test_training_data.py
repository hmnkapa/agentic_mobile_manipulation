"""Synthetic package corruption, isolation and selection checks."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

from sft_fixtures import SftFixtureTestCase, training_package
from manipulation_agent.sft.common import ConversionError, file_hash
from manipulation_agent.training.artifacts import canonical
from manipulation_agent.training.config import encoding_contract, load_config
from manipulation_agent.training.data import index_package, materialize, select_smoke
from manipulation_agent.sft.processor_budget import PROCESSOR_CONTEXT_POLICY


class TrainingDataTests(SftFixtureTestCase):
    def test_real_token_policy_binds_processor_evidence_and_preserves_rows(self):
        package = training_package(self.root)
        before = index_package(package)
        original_rows = [before.record(sample) for sample in before.samples]
        config_path = package / "config.json"
        config = json.loads(config_path.read_text())
        # Synthetic sidecars exercise the integrity contract, not real encoding.
        contract_path, tokens_path = package / "length_contract.json", package / "token_lengths.jsonl"
        contract_path.write_text(json.dumps({"synthetic_unit_test": True}))
        tokens_path.write_text("".join(json.dumps({"synthetic_sample": s.sample_id}) + "\n" for s in before.samples))
        config.update(context_policy=PROCESSOR_CONTEXT_POLICY, max_len=17408,
                      length_measurement="qwen_processor_tokens", processor_model_fingerprint="a" * 64,
                      processor_encoding_contract=encoding_contract(load_config()),
                      length_contract_sha256=file_hash(contract_path), token_lengths_sha256=file_hash(tokens_path))
        config_path.write_text(json.dumps(config))
        after = index_package(package)
        self.assertEqual([after.record(sample) for sample in after.samples], original_rows)
        self.assertNotEqual(before.fingerprint, after.fingerprint)
        self.assertIn("token_lengths.jsonl", after.files)
        tokens_path.write_text(tokens_path.read_text() + "{}\n")
        with self.assertRaisesRegex(ConversionError, "evidence is missing or changed"):
            index_package(package)

    def test_budget_disabled_and_legacy_full_image_packages_remain_readable(self):
        package = training_package(self.root)
        original = index_package(package)
        records = [original.record(s) for s in original.samples]
        config_path, stats_path = package / "config.json", package / "stats.json"
        config, stats = json.loads(config_path.read_text()), json.loads(stats_path.read_text())
        config.update(max_len=0, keep_all_images=True)
        stats["length_control"]["max_len"] = 0
        config_path.write_text(json.dumps(config))
        stats_path.write_text(json.dumps(stats))
        disabled = index_package(package)
        self.assertEqual([disabled.record(s) for s in disabled.samples], records)

        # Recreate the accepted v1.2.1 metadata contract on these unchanged,
        # under-budget rows, including absence of all new per-sample fields.
        config["converter_version"] = stats["converter_version"] = "manipulation_sft_v1.2.1"
        for key in ("context_policy", "max_len"):
            config.pop(key)
        for key in ("length_control", "approx_tokens_before"):
            stats.pop(key)
        config_path.write_text(json.dumps(config))
        stats_path.write_text(json.dumps(stats))
        (package / "build.json").write_text(json.dumps({"status": "complete", "converter_version": config["converter_version"]}))
        metadata_path = package / "sample_sources.jsonl"
        metadata = [json.loads(line) for line in metadata_path.read_text().splitlines()]
        for row in metadata:
            for key in ("approx_tokens_before", "image_count_before", "slim_triggered", "slimmed_images",
                        "dropped_history_pairs", "images_removed", "over_budget"):
                row.pop(key)
        metadata_path.write_text("".join(json.dumps(row) + "\n" for row in metadata))
        legacy = index_package(package)
        self.assertEqual([legacy.record(s) for s in legacy.samples], records)

    def test_unknown_or_inconsistent_context_contract_is_rejected(self):
        package = training_package(self.root)
        path = package / "config.json"
        original = json.loads(path.read_text())
        variants = [{"context_policy": None}, {"context_policy": "unknown_policy"},
                    *({"max_len": value} for value in (-1, True, 24.5, "24576", None)),
                    {"keep_all_images": True}, {"max_len": 0, "keep_all_images": False},
                    {"converter_version": "unknown"},
                    {"converter_version": "manipulation_sft_v1.2.1"}, {"k": 99}]
        for changed in variants:
            with self.subTest(changed=changed):
                path.write_text(json.dumps({**original, **changed}))
                with self.assertRaisesRegex(ConversionError, "contract is unsupported"):
                    index_package(package)

    def test_movable_package_does_not_need_original_sources(self):
        package = training_package(self.root)
        before = {p.name: file_hash(p) for p in package.iterdir() if p.is_file()}
        moved = self.root / "moved"
        shutil.move(package, moved)
        shutil.rmtree(self.root / "alpha")
        shutil.rmtree(self.root / "beta")
        index = index_package(moved)
        prepared = materialize(index, self.root / "prepared")
        self.assertEqual(sum(prepared["splits"].values()), 6)
        for sample in index.samples:
            self.assertTrue(all(Path(p).is_file() for p in index.record(sample, absolute_images=True)["images"]))
        for split in ("train", "val"):
            original = [json.loads(x) for x in (moved / (split + ".jsonl")).read_text().splitlines()]
            actual = [json.loads(x) for x in (self.root / "prepared" / (split + ".jsonl")).read_text().splitlines()]
            self.assertEqual([json.dumps(r["tools"]) for r in original], [json.dumps(r["tools"]) for r in actual])
        self.assertEqual(before, {p.name: file_hash(p) for p in moved.iterdir() if p.is_file()})

    def test_metadata_uses_separate_split_cursors(self):
        package = training_package(self.root)
        index = index_package(package)
        for split in ("train", "val"):
            self.assertEqual([s.line for s in index.samples if s.split == split], [1, 2, 3])
        for sample in index.samples:
            self.assertEqual(index.record(sample)["messages"][-1]["tool_calls"][0]["id"], sample.call_id)

    def test_missing_extra_or_misaligned_rows_rejected(self):
        for case in ("missing", "extra", "metadata"):
            with self.subTest(case=case):
                root = self.root / case
                root.mkdir()
                package = training_package(root)
                p = package / "train.jsonl"
                rows = p.read_text().splitlines()
                if case == "missing":
                    p.write_text("\n".join(rows[:-1]) + "\n")
                elif case == "extra":
                    p.write_text("\n".join(rows + [rows[0]]) + "\n")
                else:
                    p = package / "sample_sources.jsonl"
                    values = [json.loads(x) for x in p.read_text().splitlines()]
                    values[0]["call_id"] = "incorrect"
                    p.write_text("".join(canonical(x) + "\n" for x in values))
                with self.assertRaises(ConversionError):
                    index_package(package)

    def test_image_corruption_escape_or_failed_build_rejected(self):
        for case in ("image", "escape", "build"):
            with self.subTest(case=case):
                root = self.root / case
                root.mkdir()
                package = training_package(root)
                if case == "build":
                    (package / "build.json").write_text('{"status":"failed"}')
                else:
                    p = package / "images.jsonl"
                    rows = [json.loads(x) for x in p.read_text().splitlines()]
                    if case == "escape":
                        rows[0]["file"] = "../escape.png"
                        p.write_text("".join(canonical(x) + "\n" for x in rows))
                    else:
                        (package / rows[0]["file"]).write_bytes(b"damaged")
                with self.assertRaises(ConversionError):
                    index_package(package)

    def test_duplicate_samples_and_scene_leakage_rejected(self):
        package = training_package(self.root)
        p = package / "sample_sources.jsonl"
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        split = rows[0]["split"]
        second = next(i for i, m in enumerate(rows[1:], 1) if m["split"] == split)
        rows[second] = deepcopy(rows[0])
        dataset = package / (split + ".jsonl")
        records = dataset.read_text().splitlines()
        records[1] = records[0]
        dataset.write_text("\n".join(records) + "\n")
        p.write_text("".join(canonical(x) + "\n" for x in rows))
        with self.assertRaisesRegex(ConversionError, "duplicated"):
            index_package(package)

    def test_whole_scene_cannot_cross_split(self):
        package = training_package(self.root)
        path = package / "sources.jsonl"
        sources = [json.loads(x) for x in path.read_text().splitlines()]
        scene = sources[0]["scene"]
        for s in sources:
            s["scene"] = s["task_instance"]["scene"] = scene
        path.write_text("".join(canonical(x) + "\n" for x in sources))
        path = package / "sample_sources.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        for row in rows:
            row["scene"] = scene
        path.write_text("".join(canonical(x) + "\n" for x in rows))
        with self.assertRaisesRegex(ConversionError, "both splits"):
            index_package(package)

    def test_immutable_row_index_detects_change(self):
        package = training_package(self.root)
        index = index_package(package)
        sample = index.samples[0]
        p = package / (sample.split + ".jsonl")
        p.write_text(p.read_text().replace("Public plan", "Changed plan"))
        with self.assertRaises(ConversionError):
            index.record(sample)

    def test_smoke_is_deterministic_unique_and_retains_required_coverage(self):
        index = index_package(training_package(self.root, rich=True))
        config = load_config(smoke=True)
        reports = [{"sample_id": s.sample_id, "status": "passed", "tokens": 1000 + i,
                    "model_max_length": 262144, "within_budget": True} for i, s in enumerate(index.samples)]
        chosen, selection = select_smoke(index, reports, config)
        again, _ = select_smoke(index, reports, config)
        self.assertEqual(chosen, again)
        self.assertEqual(len({s.sample_id for s in chosen}), 24)
        self.assertEqual(sum(s.split == "train" for s in chosen), 16)
        self.assertEqual(sum(s.split == "val" for s in chosen), 8)
        self.assertEqual({s.target_tool for s in chosen if s.split == "train"},
                         {"initialize", "list_skills", "read_skill", "task_notes", "act", "look", "finish"})
        self.assertTrue(any(s.has_failed_history for s in chosen if s.split == "train"))
        self.assertTrue(any(s.image_count == 0 for s in chosen))
        self.assertTrue(any(s.image_count >= 8 for s in chosen))
        broken = deepcopy(reports)
        for r in broken:
            r["tokens"] = 999999
        with self.assertRaises(ConversionError):
            select_smoke(index, broken, config)
