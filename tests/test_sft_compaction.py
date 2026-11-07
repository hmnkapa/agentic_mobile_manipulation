"""Semantic preservation checks for opt-in historical reply summaries."""
from copy import deepcopy
import json
import unittest

from sft_fixtures import SftFixtureTestCase, training_package
from manipulation_agent.sft.common import ConversionError, file_hash
from manipulation_agent.sft.compact import compact_package
from manipulation_agent.sft.dataset import episode_samples, prepare_dataset, write_package
from manipulation_agent.sft.history import COMPACT_CONTEXT_POLICY
from manipulation_agent.sft.history_compaction import COMPACT_HISTORY, compact_history, compact_record
from manipulation_agent.sft.processor_budget import PROCESSOR_CONTEXT_POLICY, pack_record, unpack_record
from manipulation_agent.sft.source import load_episode
from manipulation_agent.training.config import encoding_contract, load_config
from manipulation_agent.training.data import index_package


def result(message):
    return json.JSONDecoder().raw_decode(message['content'].removeprefix('Result: '))[0]


def row_fixture():
    effect = {'primitive': 'open', 'status': 'completed', 'verification': 'executor_operation_only',
              'local_adjustment': {'repositions': 1, 'primitive_attempts': 1, 'departure': 'checked',
                                   'operation_committed': True, 'recovery_required': False, 'message': 'preparation'}}
    reply = {'ok': True, 'effect': effect, 'observation': {'revision': 1, 'capture': {'captured_at': 'old'}},
             'evidence_id': 'evidence', 'execution_budget': {'scope': 'unbounded'}, 'actions_used': 1,
             'unknown_effect_data': {'must_survive': True}}
    messages = [{'role': 'system', 'content': 'SYSTEM'}, {'role': 'user', 'content': 'USER'}]
    images = []
    calls = [('act', reply, True), ('read_skill', {'text': 'FULL_SKILL'}, False),
             ('initialize', {'ok': True, 'observation': {'revision': 0}}, True),
             ('act', {'ok': False, 'error': {'code': 'STALE_IMAGE', 'message': 'failure evidence'}}, True),
             ('act', reply, True), ('task_notes', {'notes': {'version': 2, 'content': 'FULL_NOTES'}}, False),
             ('act', reply, True)]
    for n, (name, obj, image) in enumerate(calls):
        messages.append({'role': 'assistant', 'content': '', 'loss': False,
                         'tool_calls': [{'id': str(n), 'type': 'function', 'function': {
                             'name': name, 'arguments': '{"revision": 1, "point_2d": [100, 200]}'}}]})
        content = json.dumps(obj)
        if image:
            content += f'\nRGB view=front image_ref=ref-{n}\n<image>'
            images.append(f'images/{n}.png')
        messages.append({'role': 'tool', 'name': name, 'tool_call_id': str(n), 'content': content})
    messages.append({'role': 'assistant', 'content': '<think>\n\n</think>', 'tool_calls': [{
        'id': 'target', 'type': 'function', 'function': {'name': 'act', 'arguments': '{"primitive":"grasp"}'}}]})
    return {'messages': messages, 'images': images, 'tools': []}


class CompactionTests(unittest.TestCase):
    def test_only_old_success_replies_change_and_every_protected_message_survives(self):
        row = row_fixture()
        original = deepcopy(row)
        out, stats = compact_record(row)
        self.assertEqual(row, original)
        self.assertEqual(stats['changed_indices'], [3])
        self.assertEqual(stats['protected_cutoff'], 11)
        self.assertEqual(out['messages'][11:], row['messages'][11:])
        self.assertEqual(out['images'], row['images'])
        self.assertEqual(out['tools'], row['tools'])
        after = result(out['messages'][3])
        self.assertNotIn('observation', after)
        self.assertNotIn('execution_budget', after)
        self.assertEqual(after['effect']['verification'], 'executor_operation_only')
        self.assertEqual(after['effect']['local_adjustment'], {'operation_committed': True, 'recovery_required': False})
        self.assertEqual(after['unknown_effect_data'], {'must_survive': True})
        self.assertTrue(out['messages'][3]['content'].endswith('\nRGB view=front image_ref=ref-0\n<image>'))
        for i in range(len(row['messages'])):
            if i != 3:
                self.assertEqual(out['messages'][i], row['messages'][i])
        self.assertEqual(compact_record(out)[0], out)

    def test_error_flags_non_boolean_success_and_transport_errors_are_preserved(self):
        for patch in ({'ok': False}, {'ok': 1}, {'error': {'code': 'FAIL'}}, {'isError': True}, {'ok': None}):
            with self.subTest(patch=patch):
                row = row_fixture()
                obj = result(row['messages'][3])
                obj.update(patch)
                row['messages'][3]['content'] = json.dumps(obj) + '\n<image>'
                self.assertEqual(compact_record(row)[0]['messages'][3], row['messages'][3])
        row = row_fixture()
        row['messages'][3]['content'] = 'Recorded MCP transport error: TIMEOUT\n<image>'
        self.assertEqual(compact_record(row)[0]['messages'][3], row['messages'][3])

    def test_unknown_recovery_fields_and_non_template_messages_survive(self):
        for patch in ({'operation_committed': False}, {'recovery_required': True}, {'unexpected_warning': 'keep'}):
            with self.subTest(patch=patch):
                row = row_fixture()
                obj = result(row['messages'][3])
                obj['effect']['local_adjustment'].update(patch)
                obj['effect']['message'] = 'Unexpected semantic instruction'
                obj['effect']['transfer'] = {'held_object_retained': True}
                row['messages'][3]['content'] = json.dumps(obj) + '\n<image>'
                after = result(compact_record(row)[0]['messages'][3])
                self.assertEqual(after['effect'], obj['effect'])

    def test_block_history_and_flat_records_use_identical_transform(self):
        row = row_fixture()
        blocks = unpack_record(row)
        history, _ = compact_history(blocks[:-1])
        self.assertEqual(pack_record(history + [blocks[-1]], row['tools']), compact_record(row)[0])
        for k in (0, -1, True):
            with self.subTest(k=k), self.assertRaises(ConversionError):
                compact_record(row, k=k)
        self.assertEqual(compact_record(row, k=10)[0], row)


class CompactionPackageTests(SftFixtureTestCase):
    def test_raw_export_declares_the_selected_policy_and_preserves_scene_split(self):
        sources = []
        for name in ('alpha', 'beta'):
            fixture = self.fixture(name, scene=name)
            fixture.initialize()
            for _ in range(3):
                fixture.act()
            fixture.finish()
            sources.append(fixture.flush())
        legacy = prepare_dataset(sources, batch=True, max_len=0)
        compact = prepare_dataset(sources, batch=True, max_len=0, history_compression=COMPACT_HISTORY)
        self.assertEqual(legacy.validation_scenes, compact.validation_scenes)
        package = write_package(compact, self.root / 'raw_compact')
        index = index_package(package, expected_history_compression=COMPACT_HISTORY)
        self.assertEqual(index.config['context_policy'], COMPACT_CONTEXT_POLICY)
        self.assertEqual(index.stats['splits'], legacy.stats['splits'])

    def test_processor_budget_policy_requires_matching_compact_text_contract(self):
        package = training_package(self.root)
        output = self.root / 'compact'
        compact_package(package, output)
        configuration = json.loads((output / 'config.json').read_text())
        recipe = load_config()
        recipe['data']['history_compression'] = COMPACT_HISTORY
        (output / 'length_contract.json').write_text('{"synthetic_unit_test":true}')
        (output / 'token_lengths.jsonl').write_text('{}\n')
        configuration.update(context_policy=PROCESSOR_CONTEXT_POLICY, max_len=17408, keep_all_images=False,
                             length_measurement='qwen_processor_tokens', processor_model_fingerprint='a'*64,
                             processor_encoding_contract=encoding_contract(recipe),
                             length_contract_sha256=file_hash(output/'length_contract.json'),
                             token_lengths_sha256=file_hash(output/'token_lengths.jsonl'))
        (output / 'config.json').write_text(json.dumps(configuration))
        self.assertEqual(index_package(output).config['history_compression'], COMPACT_HISTORY)
        configuration['processor_encoding_contract'].pop('history_compression')
        (output / 'config.json').write_text(json.dumps(configuration))
        with self.assertRaisesRegex(ConversionError, 'unsupported'):
            index_package(output)

    def test_new_package_preserves_identities_and_binds_policy_and_evidence(self):
        package = training_package(self.root)
        before = index_package(package)
        output = self.root / 'compact'
        report = compact_package(package, output)
        self.assertFalse(report['qwen_encoding_verified'])
        after = index_package(output, expected_history_compression=COMPACT_HISTORY)
        self.assertEqual(after.manifest()['splits'], before.manifest()['splits'])
        self.assertEqual([s.sample_id for s in after.samples], [s.sample_id for s in before.samples])
        self.assertEqual(after.images, before.images)
        for old, new in zip(before.samples, after.samples):
            self.assertEqual(before.record(old)['messages'][-1], after.record(new)['messages'][-1])
        with self.assertRaisesRegex(ConversionError, 'policies differ'):
            index_package(output, expected_history_compression='legacy')
        with self.assertRaisesRegex(ConversionError, 'legacy package'):
            compact_package(output, self.root / 'double_compact')
        (output / 'compression_report.json').write_text('{}')
        with self.assertRaisesRegex(ConversionError, 'evidence is missing or changed'):
            index_package(output)

    def test_raw_converter_and_packaged_transform_agree(self):
        fixture = self.fixture()
        fixture.initialize()
        for _ in range(4):
            fixture.act()
        fixture.finish()
        episode = load_episode(fixture.flush())
        legacy = list(episode_samples(episode, max_len=0))
        compact = list(episode_samples(episode, max_len=0, history_compression=COMPACT_HISTORY))
        for (old, _), (new, _) in zip(legacy, compact):
            self.assertEqual(new, compact_record(old)[0])

    def test_recipe_contract_distinguishes_legacy_and_compact_text_policy(self):
        legacy = load_config()
        compact = deepcopy(legacy)
        compact['data']['history_compression'] = COMPACT_HISTORY
        self.assertNotIn('history_compression', encoding_contract(legacy))
        self.assertEqual(encoding_contract(compact)['history_compression'], COMPACT_HISTORY)
        self.assertNotEqual(encoding_contract(legacy), encoding_contract(compact))
