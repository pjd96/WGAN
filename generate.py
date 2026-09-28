"""Generate images from a training checkpoint and its generator_config.json."""
import argparse
import json
from pathlib import Path

import torch
import torchvision.utils as vutils

import models.dcgan as dcgan
import models.mlp as mlp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('-w', '--weights', required=True)
    parser.add_argument('-o', '--output_dir', required=True)
    parser.add_argument('-n', '--nimages', type=int, default=1)
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--manualSeed', type=int, default=1)
    parser.add_argument('--cuda', action='store_true')
    opt = parser.parse_args()
    if opt.nimages < 1 or opt.batchSize < 1:
        parser.error('nimages and batchSize must be positive')
    if opt.cuda and not torch.cuda.is_available():
        parser.error('--cuda requested but CUDA is unavailable')
    device = torch.device('cuda' if opt.cuda else 'cpu')
    torch.manual_seed(opt.manualSeed)
    with open(opt.config) as stream:
        cfg = json.load(stream)
    # Inference uses one device regardless of the training GPU count.
    args = (cfg['imageSize'], cfg['nz'], cfg['nc'], cfg['ngf'], 1)
    if cfg['noBN']:
        netG = dcgan.DCGAN_G_nobn(*args, cfg['n_extra_layers'])
    elif cfg['mlp_G']:
        netG = mlp.MLP_G(*args)
    else:
        netG = dcgan.DCGAN_G(*args, cfg['n_extra_layers'])
    netG.load_state_dict(torch.load(opt.weights, map_location='cpu', weights_only=True))
    netG.to(device).eval()
    output = Path(opt.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for start in range(0, opt.nimages, opt.batchSize):
            count = min(opt.batchSize, opt.nimages - start)
            noise = torch.randn(count, cfg['nz'], 1, 1, device=device)
            fake = netG(noise).mul(0.5).add(0.5).clamp(0, 1).cpu()
            for offset, sample in enumerate(fake):
                vutils.save_image(sample, output / f'generated_{start + offset:02d}.png')
    print(f'Saved {opt.nimages} images to {output}')


if __name__ == '__main__':
    main()
