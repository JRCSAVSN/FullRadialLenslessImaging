import torch
from torchvision.transforms.functional import to_tensor, to_pil_image, resize, center_crop, crop, pad
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from functools import partial
import argparse
import os
import json

from modules.convex_optimization_utils import FISTA
from modules.forward_models import shift_invariant_model
from modules.forward_models_utils import get_big_conv_k
from modules.utils import get_y, get_x, tv2d

parser = argparse.ArgumentParser()
# Path options
parser.add_argument('--meas_path', type=str, default=None, help='Path to the measurement')
parser.add_argument('--psf_path', type=str, required=True, help='Path to the PSF')
parser.add_argument('--save_path', type=str, required=True, help='Path to save experiment results')
# PSF options
parser.add_argument('--psf_reso', type=int, default=384, help='resolution of the PSF')
parser.add_argument('--psf_center_crop_size', type=int, default=None, help='Area of the PSF to be cropped to limit aperture size')
parser.add_argument('--mask_type', type=str, default=None, help='Type of mask to be used (simulation only)')
# Simulation options (for simulations only)
parser.add_argument('--gt_path', type=str, default=None, help='Path to the image')
parser.add_argument('--gt_reso', type=int, default=128, help='resolution of the ground truth image (ignoring padding)')
parser.add_argument('--sigma', type=float, default=5e-4, help='sigma of the Gaussian noise to be added to measurement')
# Measurement options
parser.add_argument('--meas_reso', type=int, default=256, help='resolution of the measurement')
# Optimization options
parser.add_argument('--lrate', type=float, default=1e-1, help='learning rate')
parser.add_argument('--iters', type=int, default=15000, help='number of iterations')
# Regularization params
parser.add_argument('--tv_alpha', type=float, default=0.0, help='Multiplier for 2D total variation regularizer')
parser.add_argument('--redo', type=bool, default=False, help='Allow experiment to be re-run')
args = parser.parse_args()

assert args.meas_path is None or args.gt_path is None, f'Only meas or gt should be defined (choose one)'
assert not (args.meas_path is None and args.gt_path is None), f'One of meas and gt must be defined (choose one)'

if args.psf_center_crop_size is not None:
    psf_center_crop_size = args.psf_center_crop_size
elif 'restricted' in args.psf_path:
    assert args.psf_reso == args.meas_reso, f'PSF ({args.psf_reso}) and measurement ({args.meas_reso}) resolution must be equal for restricted mask'
    psf_center_crop_size = 1200
else:
    psf_center_crop_size = 1464

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)
if os.path.exists(args.save_path + '/recon_final.png') and not args.redo:
    print('Experiment already exists!' )
    exit()

# Save arguments as json
with open(os.path.join(args.save_path, 'args.json'), 'w') as f:
    json.dump(vars(args), f, indent=4)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    k = get_big_conv_k(
        psf_path=args.psf_path,
        psf_reso=args.psf_reso,
        center_crop_size=psf_center_crop_size,
        mask_type=args.mask_type,
    )
    k = k.to(device)

    if args.meas_path is not None:
        y = get_y(
            meas_path=args.meas_path,
            downsample_size=args.meas_reso,
        )
        y = y.to(device)

    if args.gt_path is not None:
        x = get_x(
            gt_path = args.gt_path,
            downsample_size=args.gt_reso
        )
        x = x.to(device)
        y = shift_invariant_model(k=k, x=x)
        y = ((y / y.max()) + torch.randn_like(y) * args.sigma) * y.max() # Normalize it
        y = y.clip(0, 1)

    to_pil_image((k/k.max()).cpu()).save(args.save_path + '/k.png')
    to_pil_image((y/y.max()).cpu()).save(args.save_path + '/y.png')
    if args.gt_path is not None:
        to_pil_image((x/x.max()).cpu()).save(args.save_path + '/x.png')

    fmodel = partial(shift_invariant_model, k=k)
    tv_reg = partial(tv2d, alpha=args.tv_alpha)

    recon, loss = FISTA(y_target=y, forward_model=fmodel, iters=args.iters, lrate=args.lrate, save_path=args.save_path, reg_fn=tv_reg)
    to_pil_image((recon / recon.max()).detach().squeeze().cpu()).save(args.save_path + f'/recon_final.png')
