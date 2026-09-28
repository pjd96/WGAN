"""Offline regression tests: python -m unittest discover -s tests -v."""
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import lmdb
import numpy as np
from PIL import Image
import torch

from models import dcgan

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def run_script(self, script, *args, success=True):
        result = subprocess.run(
            [sys.executable, str(ROOT / script), *map(str, args)],
            cwd=ROOT, env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
            text=True, capture_output=True, timeout=90)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_training_checkpoint_and_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / 'images' / 'unlabelled'
            images.mkdir(parents=True)
            rng = np.random.default_rng(123)
            # Five images exercise a final training batch of size one.
            for i in range(5):
                Image.fromarray(rng.integers(0, 256, (20, 24, 3), dtype=np.uint8)).save(images / f'{i}.png')
            for name, flags in [('dcgan', []), ('no_bn', ['--noBN']),
                                ('mlp', ['--mlp_G', '--mlp_D']),
                                ('extra_adam', ['--n_extra_layers', '1', '--adam'])]:
                with self.subTest(name=name):
                    run = root / name / 'run with spaces'
                    args = ['--dataset', 'folder', '--dataroot', images.parent,
                            '--imageSize', '16', '--ngf', '4', '--ndf', '4', '--nz', '8',
                            '--batchSize', '2', '--workers', '0', '--niter', '1',
                            '--manualSeed', '123', *flags]
                    result = self.run_script('main.py', *args, '--experiment', run)
                    self.assertIn('Loss_D:', result.stdout)
                    self.assertNotIn('nan', result.stdout.lower())
                    for network in ('G', 'D'):
                        weights = torch.load(run / f'net{network}_epoch_0.pth', weights_only=True)
                        self.assertTrue(all(torch.isfinite(v).all() for v in weights.values()))
                    # Continue from both weights in a separate output directory.
                    resumed = root / name / 'continued'
                    self.run_script('main.py', *args, '--experiment', resumed,
                                    '--netG', run / 'netG_epoch_0.pth',
                                    '--netD', run / 'netD_epoch_0.pth')
                    before = torch.load(run / 'netG_epoch_0.pth', weights_only=True)
                    after = torch.load(resumed / 'netG_epoch_0.pth', weights_only=True)
                    self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before if 'weight' in k))
                    # A single image and a short last generation batch must both work.
                    for count in (1, 3):
                        out = root / name / f'generated{count}'
                        self.run_script('generate.py', '-c', run / 'generator_config.json',
                                        '-w', run / 'netG_epoch_0.pth', '-o', out,
                                        '-n', count, '--batchSize', '2')
                        files = list(out.glob('*.png'))
                        self.assertEqual(len(files), count)
                        for path in files:
                            with Image.open(path) as image:
                                self.assertEqual(image.size, (16, 16))
                                self.assertEqual(image.mode, 'RGB')

    def test_lsun_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = root / 'bedroom_train_lmdb'
            env = lmdb.open(str(database), map_size=1024 * 1024)
            with env.begin(write=True) as txn:
                for i in range(3):
                    stream = io.BytesIO()
                    Image.new('RGB', (20, 20), (i * 50, 80, 120)).save(stream, format='JPEG')
                    txn.put(f'{i:08d}'.encode(), stream.getvalue())
            env.close()
            self.run_script('main.py', '--dataset', 'lsun', '--dataroot', root,
                            '--experiment', root / 'run', '--imageSize', '16',
                            '--ngf', '4', '--ndf', '4', '--batchSize', '2',
                            '--niter', '1', '--workers', '0')

    def test_invalid_image_size(self):
        for cls in (dcgan.DCGAN_G, dcgan.DCGAN_G_nobn, dcgan.DCGAN_D, dcgan.DCGAN_D_nobn):
            with self.assertRaisesRegex(AssertionError, 'power of two'):
                cls(48, 8, 3, 4, 1)

    def test_missing_dataset_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_script('download_data.py', '--dataroot', tmp, '--check-only', success=False)
            self.assertIn('Dataset preparation failed', result.stderr)


if __name__ == '__main__':
    unittest.main()
