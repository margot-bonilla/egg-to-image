"""CPU regression tests for the portable handoff; all datasets are synthetic."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from handoff import evaluate, models, protocol, run, smoke


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source, cls.reve_module = smoke.load_fixture_sources()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='eeg-handoff-test-')
        self.directory = Path(self.temp.name).resolve()
        self.spec, self.cache, self.fake = smoke.make_fixture(self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def prepared(self):
        return run.prepare(self.spec, self.cache, fixture=True)

    def test_committed_freeze_is_intact(self):
        frozen = protocol.load_frozen()
        self.assertEqual(len(frozen['development']), 1500)
        self.assertEqual(set(frozen['folds']), {'1', '2'})

    def test_production_rejects_fixture(self):
        with self.assertRaises(ValueError):
            protocol.validate_spec(self.spec)

    def test_development_confirmation_overlap_rejected(self):
        self.spec['confirmation_exclusions'] = ['dev0']
        with self.assertRaises(ValueError):
            self.prepared()

    def test_duplicate_query_rejected(self):
        self.spec['folds']['1']['groups'][0]['rows'] = [0, 0]
        with self.assertRaises(ValueError):
            self.prepared()

    def test_confirmation_fold_rejected(self):
        self.spec['folds']['3'] = self.spec['folds']['1']
        with self.assertRaises(ValueError):
            self.prepared()

    def test_changed_mixture_rejected(self):
        self.spec['mixture'] = {'encoder': .5, 'reve': .5}
        with self.assertRaises(ValueError):
            self.prepared()

    def test_main_and_extra_sources_exclude_images_and_people(self):
        prepared = self.prepared()
        for fold, data in prepared[3].items():
            rows = data['raw'].fields(('subject', 'stem'))
            self.assertTrue(all(r['subject'] not in self.spec['folds'][fold]['subjects'] for r in rows))
            self.assertEqual({r['stem'] for r in rows}, {'train0', 'train1'})
            extra = data['extra'].fields(('stem',))
            # The differently named exact-DINO alias is excluded as well.
            self.assertEqual({r['stem'] for r in extra}, {'train0', 'train1'})
            self.assertEqual(len(data['raw']), 8)
            self.assertEqual(len(data['extra']), 4)

    def test_input_metadata_tamper_rejected(self):
        (self.cache / 'windows/things_eeg2/meta.json').write_text('{}')
        with self.assertRaises(ValueError):
            self.prepared()

    def test_truncated_eeg_header_rejected(self):
        path = self.cache / 'windows/xu2025alljoined/train/X.npy'
        with path.open('r+b') as f:
            f.truncate(path.stat().st_size - 4)
        with self.assertRaises(ValueError):
            self.prepared()

    def test_query_canonical_identity_checked(self):
        self.spec['folds']['1']['rows'][0] = 0  # training image, not its declared dev image
        with self.assertRaises(ValueError):
            self.prepared()

    def test_same_record_cap2_rejected(self):
        aj = self.cache / 'windows/xu2025alljoined'
        records = np.load(aj / 'val/record_id.npy')
        records -= 1000
        np.save(aj / 'val/record_id.npy', records)
        self.spec['inputs']['alljoined'] = smoke.cache_identity(aj)
        with self.assertRaises(ValueError):
            self.prepared()

    def test_token_row_alignment_checked(self):
        path = self.cache / 'reve_tokens/xu2025alljoined/train/image.npy'
        np.save(path, np.zeros(20, dtype=np.int64))
        with self.assertRaises(ValueError):
            self.prepared()

    def test_prefix_layer_checked(self):
        path = self.cache / 'reve_tokens/xu2025alljoined_L18/meta.json'
        meta = json.loads(path.read_text())
        meta['layer'] = 17
        path.write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            self.prepared()

    def test_public_token_sample_and_raw_prefix_equivalence(self):
        prepared = self.prepared()
        models.verify_token_sample(self.fake, prepared[0], prepared[1], prepared[2], 'cpu')
        head = models.make_head(self.reve_module).eval()
        top = models.make_top(self.reve_module, self.fake, copy.deepcopy(head)).eval()
        x, _, _ = prepared[3]['1']['dev'].__getitems__(list(range(4)))
        a = self.reve_module.REVEImageModel(self.fake, [head]).predict(x)
        b = self.reve_module.REVEImageModel(self.fake, [top], 18).predict(x)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(all(not p.requires_grad for p in self.fake.model.parameters()))
        self.assertTrue(all(p.requires_grad for p in top.parameters()))

    def test_output_input_overlap_rejected(self):
        with self.assertRaises(ValueError):
            run.safe_output(self.cache / 'windows/xu2025alljoined/output', [self.cache / 'windows/xu2025alljoined'])
        with self.assertRaises(ValueError):
            run.safe_output(self.cache, [self.cache / 'windows/xu2025alljoined'])

    def test_output_leaf_symlink_rejected(self):
        destination = self.directory / 'link'
        destination.symlink_to(self.cache, target_is_directory=True)
        with self.assertRaises(ValueError):
            run.safe_output(destination, [])

    def test_fixed_mixture_normalises_members_per_trial(self):
        a = np.eye(2, 1536, dtype=np.float32)
        b = a[::-1].copy()
        mixture = evaluate.fixed_mixture(a * 17, b * .01)
        np.testing.assert_allclose(mixture, .75 * a + .25 * b, atol=1e-7)
        with self.assertRaises(ValueError):
            evaluate.fixed_mixture(a, np.zeros_like(a))

    def test_official_rank_and_group_mean_semantics(self):
        rng = np.random.default_rng(91)
        gallery = rng.standard_normal((8, 1536)).astype(np.float32)
        gallery[1] = gallery[0]  # exact tied scores must have average rank
        predictions = np.stack([gallery[0], gallery[2], gallery[3], gallery[4]]).astype(np.float32)
        groups = [{'subject': 's', 'image': 'im0', 'rows': [0, 1]},
                  {'subject': 's', 'image': 'im3', 'rows': [2, 3]}]
        stems = ['im' + str(i) for i in range(8)]
        normal_gallery = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
        for cap in ('1', '2', 'all'):
            vectors = np.stack([predictions[g['rows'] if cap == 'all' else g['rows'][:int(cap)]].astype(np.float64).mean(0)
                                for g in groups])
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            scores = vectors @ normal_gallery.T
            true = scores[np.arange(2), [0, 3]][:, None]
            expected = ((scores > true).sum(1) + (scores >= true).sum(1) - 1) / 2
            report = evaluate.score(predictions, groups, gallery, stems, cap, chunk=1)
            np.testing.assert_array_equal([r['rank'] for r in report['queries']], expected)
            self.assertEqual(report['metrics']['top5'], float((expected < 5).mean()))

    def test_unknown_cap_and_zero_vectors_rejected(self):
        vector = np.eye(2, 1536, dtype=np.float32)
        groups = [{'subject': 's', 'image': 'a', 'rows': [0, 1]}]
        for cap, p in [('3', vector), ('1', np.zeros_like(vector))]:
            with self.assertRaises(ValueError):
                evaluate.score(p, groups, vector, ['a', 'b'], cap)

    def _assert_checkpoint_resume(self, workers):
        prepared = self.prepared()
        data, frozen = prepared[3]['1'], self.spec['folds']['1']
        recipe = self.spec['recipes']['encoder_joint']
        folders = [self.directory / 'full', self.directory / 'resumed']
        full = models.make_encoder(self.source, prepared[0].meta)
        run.fit_stage(full, self.source, data['raw'], 'encoder_joint', recipe, folders[0], data, frozen,
                      'cpu', 8, workers, lambda: None)
        resumed = models.make_encoder(self.source, prepared[0].meta)
        calls = [0]
        def interrupt_second_epoch_evaluation():
            calls[0] += 1
            if calls[0] == 5:
                raise RuntimeError('Synthetic interruption after one complete epoch')
        with self.assertRaises(RuntimeError):
            run.fit_stage(resumed, self.source, data['raw'], 'encoder_joint', recipe, folders[1], data, frozen,
                          'cpu', 8, workers, interrupt_second_epoch_evaluation)
        saved = torch.load(folders[1] / 'checkpoint.pt', map_location='cpu', weights_only=True)
        self.assertEqual(saved['epoch'], 1)
        resumed = models.make_encoder(self.source, prepared[0].meta)
        run.fit_stage(resumed, self.source, data['raw'], 'encoder_joint', recipe, folders[1], data, frozen,
                      'cpu', 8, workers, lambda: None)
        a = torch.load(folders[0] / 'checkpoint.pt', map_location='cpu', weights_only=True)
        b = torch.load(folders[1] / 'checkpoint.pt', map_location='cpu', weights_only=True)
        self.assertEqual(a['history'], b['history'])
        self.assertTrue(all(torch.equal(a['model'][key], b['model'][key]) for key in a['model']))
        with self.assertRaises(ValueError):
            run.fit_stage(resumed, self.source, data['raw'], 'encoder_joint', recipe, folders[1], data, frozen,
                          'cpu', 8, workers, lambda: None, parent_sha='wrong-parent')

    def test_epoch_checkpoint_exact_resume_and_parent_identity(self):
        self._assert_checkpoint_resume(0)

    def test_persistent_worker_checkpoint_exact_resume(self):
        self._assert_checkpoint_resume(2)

    def test_metadata_never_moved_to_accelerator(self):
        from unittest.mock import patch
        prepared = self.prepared()
        real_to = torch.Tensor.to
        def guarded_to(tensor, *args, **kwargs):
            if tensor.dtype == torch.float64:
                raise AssertionError('Informational onset moved to an accelerator')
            return real_to(tensor, *args, **kwargs)
        solver = self.source.Solver.get_instance()
        data = run.loader(prepared[3]['1']['raw'], 'cpu', 8, 0, solver)
        with patch.object(torch.Tensor, 'to', guarded_to):
            batch = next(iter(data))
        self.assertEqual(batch[2]['onset'].device.type, 'cpu')
        self.assertEqual(batch[2]['onset'].dtype, torch.float64)
        data.close()

    def test_new_loader_matches_published_shuffle_rng(self):
        from src.data.window_cache import CachedLoader, _batch
        prepared = self.prepared()
        dataset = prepared[3]['1']['raw']
        for workers in (0, 2):
            torch.manual_seed(97)
            original = (CachedLoader if workers == 0 else torch.utils.data.DataLoader)(
                dataset, batch_size=8, shuffle=True, num_workers=workers,
                collate_fn=_batch, persistent_workers=workers > 0)
            before = [[batch[2]['record_id'].tolist() for batch in original] for _ in range(3)]
            original_rng = torch.get_rng_state().clone()
            if workers:
                original._iterator._shutdown_workers()
            del original
            torch.manual_seed(97)
            solver = self.source.Solver.get_instance()
            replacement = run.loader(dataset, 'cpu', 8, workers, solver)
            after = [[batch[2]['record_id'].tolist() for batch in replacement] for _ in range(3)]
            replacement.close()
            self.assertEqual(before, after)
            self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))

    def test_wrong_raw_reve_layer_is_detectable(self):
        prepared = self.prepared()
        head = models.make_head(self.reve_module).eval()
        top = models.make_top(self.reve_module, self.fake, head).eval()
        x, _, _ = prepared[3]['1']['dev'].__getitems__(list(range(4)))
        correct = self.reve_module.REVEImageModel(self.fake, [top], 18).predict(x)
        wrong = self.reve_module.REVEImageModel(self.fake, [top], None).predict(x)
        self.assertFalse(torch.allclose(correct, wrong, atol=1e-6, rtol=1e-6))

    def test_rejected_output_is_unchanged(self):
        prepared = self.prepared()
        out = self.directory / 'unrelated'
        out.mkdir()
        (out / 'keep.txt').write_text('human artifact')
        with self.assertRaises(ValueError):
            run.execute(self.spec, prepared, out, 'cpu', 0, self.source, self.reve_module, self.fake,
                        fixture=True, max_stage_epochs=2)
        self.assertEqual([p.name for p in out.iterdir()], ['keep.txt'])

    def test_another_input_checkout_is_protected(self):
        (self.directory / '.git').mkdir()
        with self.assertRaises(ValueError):
            run.safe_output(self.directory / 'new-output', [self.cache / 'windows/xu2025alljoined'])
        self.assertFalse((self.directory / 'new-output').exists())

    def test_token_helper_protects_other_checkout_and_dangling_symlink(self):
        from handoff.cache_tokens import validate_destination
        (self.directory / '.git').mkdir()
        with self.assertRaises(ValueError):
            validate_destination(self.cache / 'reve_tokens/new-cache', self.cache / 'windows/xu2025alljoined')
        dangling = self.directory / 'dangling'
        dangling.symlink_to(self.directory / 'missing-target', target_is_directory=True)
        with self.assertRaises(ValueError):
            validate_destination(dangling, self.cache / 'windows/xu2025alljoined')
        self.assertFalse((self.cache / 'reve_tokens/new-cache').exists())

    def test_pinned_token_extraction_fixture(self):
        from handoff.cache_tokens import extract
        prepared = self.prepared()
        for layer in (18, 22):
            destination = self.directory / ('tokens-' + str(layer))
            extract(self.fake, prepared[0], destination, layer, 'cpu', batch_size=8)
            protocol.validate_tokens(destination, prepared[0], layer)
            expected = np.load(prepared[1 if layer == 22 else 2] / 'train/X.npy')
            np.testing.assert_array_equal(np.load(destination / 'train/X.npy'), expected)
            with self.assertRaises(ValueError):
                extract(self.fake, prepared[0], destination, layer, 'cpu')

    def test_returned_gallery_is_ordered_and_reusable(self):
        prepared = self.prepared()
        data, frozen = prepared[3]['1'], self.spec['folds']['1']
        folder = self.directory / 'returned'
        folder.mkdir()
        run.save_gallery_once(data['gallery'], frozen['gallery_stems'], folder)
        saved = np.load(folder / 'gallery.npy', allow_pickle=False)
        np.testing.assert_array_equal(saved, data['gallery'])
        run.save_gallery_once(data['gallery'], frozen['gallery_stems'], folder)
        with self.assertRaises(ValueError):
            run.save_gallery_once(data['gallery'], list(reversed(frozen['gallery_stems'])), folder)

    def test_saved_prediction_tamper_rejected(self):
        prepared = self.prepared()
        model = models.make_encoder(self.source, prepared[0].meta).eval()
        path = self.directory / 'vectors.npy'
        run.predict_once(model, prepared[3]['1']['dev'], path, 'cpu', lambda: None)
        np.save(path, np.zeros((8, 1536), dtype=np.float32))
        with self.assertRaises(ValueError):
            run.predict_once(model, prepared[3]['1']['dev'], path, 'cpu', lambda: None)


if __name__ == '__main__':
    unittest.main()
