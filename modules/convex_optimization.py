import torch
from torch import nn
from torch import Tensor
from pathlib import Path
from typing import Dict, Callable
from tqdm import tqdm
from math import sqrt
from functools import partial

from .forward_models import shift_invariant_model
from .convex_optimization_utils import *

class Wiener(nn.Module):
    def __init__(self, psf, w=1, lambda_=1e-4, **kw):
        super(Wiener, self).__init__()
        p_f = torch.fft.fft2(torch.fft.fftshift(psf, dim=(-2, -1)))
        self.register_buffer('PSF', (w * torch.conj(p_f)) / ((torch.abs(p_f) * w) ** 2 + lambda_))
        self.w = w
        self.lambda_ = lambda_

    def forward(self, x):
        x_f = torch.fft.fft2(torch.fft.fftshift(x, dim=(-2, -1)))
        v_f = x_f * self.PSF
        v = torch.fft.fftshift(torch.fft.ifft2(v_f), dim=(-2, -1))
        return torch.real(v)

class Admm(nn.Module):
    def __init__(self, psf: Tensor, iters: int, hyper_parameter: Dict[str, float], **kw):
        super(Admm, self).__init__()
        self.register_buffer('psf', psf)
        self.model = UnrolledADMM(psf=psf, iters=iters, hyper_parameter=hyper_parameter)

    def forward(self, y):
        return self.model.run(y), None

class Fista(nn.Module):
    def __init__(self, psf: torch.Tensor, iters: int, lrate: float, v_init: torch.Tensor = None, forward_model: Callable[[torch.Tensor], torch.Tensor] = None, reg_fn: Callable[[torch.Tensor], torch.Tensor] = None):
        super(Fista, self).__init__()
        if forward_model is None:
            self.f_model = partial(shift_invariant_model, k=psf)
        else:
            self.f_model = forward_model
        self.psf = psf
        self.iters = iters
        self.lrate = lrate
        self.fista = partial(FISTA, forward_model=self.f_model, iters=self.iters, lrate=self.lrate, v_init=v_init, reg_fn=reg_fn)

    def forward(self, y):
        return self.fista(y)