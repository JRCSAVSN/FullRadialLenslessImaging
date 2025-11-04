import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import resize, pad, center_crop, to_tensor, to_pil_image, crop
from torchvision.transforms import InterpolationMode
from math import ceil, sqrt
import matplotlib.pyplot as plt

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
############################################################ Shift-Invariant Model Utils ###########################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
def get_big_conv_k(psf_path, center_crop_size, psf_reso, mask_type):
    if mask_type == 'restricted':
        k = torch.from_numpy(np.load(psf_path))
        k = center_crop(k, center_crop_size)
        k = pad(k, psf_reso)

    elif mask_type == 'opt_radial':
        k = torch.from_numpy(np.load(psf_path)).unsqueeze(0)
        k = resize(k, psf_reso, InterpolationMode.BILINEAR)

    elif 'tiff' in psf_path:
        k = to_tensor(Image.open(psf_path))
        k = center_crop(k, center_crop_size)
        k = resize(k, (psf_reso, psf_reso), interpolation=InterpolationMode.BILINEAR)

    else:
        raise ValueError('Unknown mask type')

    k /= k.sum()
    return k

####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
############################################################# Shift-Variant Model Utils ############################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################


####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
############################################################## Local Convolution Utils #############################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
####################################################################################################################################################
def get_intensity_maps(center_crop_size):
    xs = [0, 20, 42, 63, 84]
    ys = [42, 63, 85, 106, 127]

    intensity_maps = torch.zeros((25, 1, center_crop_size, center_crop_size))
    idx = 0
    for y in ys:
        for x in xs[::-1]:
            intensity_map = to_tensor(Image.open(f'data/prototype/intensity_maps2/x{x}_y{y}.tiff').convert('L'))
            intensity_map = center_crop(intensity_map, center_crop_size)
            intensity_maps[idx, :] = intensity_map.clone()
            idx += 1
    return intensity_maps

def get_local_conv_k(psf_path, n_patches, psf_reso=732, center_crop_size=1464, **kwargs):
    if 'npy' in psf_path:
        k = torch.from_numpy(np.load(psf_path))
        k = resize(k.unsqueeze(0), center_crop_size)
    elif 'tiff' in psf_path:
        k = to_tensor(Image.open(psf_path))
        k = center_crop(k.unsqueeze(0), center_crop_size)
    intensity_maps = get_intensity_maps(center_crop_size)
    # intensity_maps = intensity_maps / intensity_maps.sum(dim=0, keepdim=True)
    k = k * intensity_maps
    k = resize(k, psf_reso, InterpolationMode.BILINEAR)
    k /= k.sum(dim=(-1, -2), keepdim=True)
    if k.shape[0] != n_patches ** 2:
        k = k.repeat(n_patches * n_patches, 1, 1, 1)
    return k

def get_local_conv_w(centers, meas_reso, center_crop_size=1200, **kwargs):
    """
    Returns weights of shape [P, H, W], one map per center.
    Replace the 'weights' formula with your Eq. (6).
    """
    # Phocolens-like approach
    # H, W = yy.shape
    # P = centers.shape[0]
    # cx = centers[:, 0].view(P, 1, 1)
    # cy = centers[:, 1].view(P, 1, 1)

    intensity_maps = get_intensity_maps(center_crop_size=center_crop_size)
    intensity_maps = resize(intensity_maps, meas_reso, InterpolationMode.BILINEAR)
    weights = intensity_maps / intensity_maps.sum(dim=0, keepdim=True)

    # weights = d / d.max(dim=0).values
    # weights = 1 - weights
    # weights = weights / weights.sum(dim=0, keepdim=True)
    # weights = torch.ones_like(weights)
    # weights /= weights.sum(dim=0, keepdim=True)
    return weights#.flip((-2, -1))#.unsqueeze(1) # return tensor of shape (P, 1, H, W)