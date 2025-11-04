import torch
import numpy as np
from torchvision.transforms.functional import resize, to_tensor, center_crop
from torchvision.transforms import InterpolationMode
from PIL import Image

def tv2d(x, alpha):
    """
    Compute the 2D Total Variation (TV) loss for a batch of images.

    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W)

    Returns:
        torch.Tensor: Scalar TV loss
    """
    diff_x = torch.abs(x[..., :, :-1] - x[..., :, 1:])
    diff_y = torch.abs(x[..., :-1, :] - x[..., 1:, :])
    tv_loss = torch.sum(diff_x) + torch.sum(diff_y)
    return tv_loss * alpha

###################################################################################
###################################################################################
################################ data loading utils ###############################
###################################################################################
###################################################################################
def get_y(meas_path, center_crop_size=1200, downsample_size=256):
    y = to_tensor(Image.open(meas_path))
    y = center_crop(y, center_crop_size)
    y = resize(y, (downsample_size, downsample_size), InterpolationMode.BILINEAR)
    return y

def get_x(gt_path, downsample_size):
    x = to_tensor(Image.open(gt_path))
    x = resize(x, downsample_size, InterpolationMode.BILINEAR)
    return x

###################################################################################
###################################################################################
#################################### Viz utils ####################################
###################################################################################
###################################################################################
import cv2
def adjust_brightness_contrast(image, brightness=0, contrast=30):
    """
    Adjust brightness and increase contrast of an image.

    Parameters:
        image (np.ndarray): Input image (BGR format).
        brightness (int): Value to add to pixels (-255 to 255).
        contrast (int): Contrast factor (-127 to 127). Positive increases contrast.

    Returns:
        np.ndarray: Adjusted image.
    """
    if brightness != 0:
        shadow = max(0, brightness)
        highlight = min(255, 255 + brightness)
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    return image