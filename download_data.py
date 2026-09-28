"""Download and verify CIFAR-10 before starting an offline training job."""
import argparse
from pathlib import Path
import sys

from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import download_and_extract_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataroot', type=Path, default=Path('data/cifar10'))
    parser.add_argument('--check-only', action='store_true',
                        help='verify existing data without network access')
    opt = parser.parse_args()
    opt.dataroot.mkdir(parents=True, exist_ok=True)
    try:
        try:
            train = CIFAR10(str(opt.dataroot), train=True, download=False)
        except RuntimeError:
            if opt.check_only:
                raise
            # The original Toronto address now redirects to this HTTPS host.
            # Retain torchvision's published archive MD5 and per-file checks.
            download_and_extract_archive(
                'https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz',
                str(opt.dataroot), filename=CIFAR10.filename, md5=CIFAR10.tgz_md5)
            train = CIFAR10(str(opt.dataroot), train=True, download=False)
        test = CIFAR10(str(opt.dataroot), train=False, download=False)
        for dataset in (train, test):
            image, _ = dataset[0]
            image.load()
        print(f'Ready: {len(train)} training and {len(test)} test images in {opt.dataroot.resolve()}')
        print('Training uses only the training split; class labels are ignored.')
    except (OSError, RuntimeError) as exc:
        print(f'Dataset preparation failed: {exc}\n'
              'Check network access and free disk space, then rerun this command.\n'
              'For offline use, place cifar-10-python.tar.gz in the dataroot and rerun\n'
              'without --check-only to verify and extract the local archive.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
