# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import math
import numpy as np
import torch as th
from torch.nn import functional as F
import pandas as pd

def sinc(t):
    """sinc.

    :param t: the input tensor
    """
    return th.where(t == 0, th.tensor(1., device=t.device, dtype=t.dtype), th.sin(t) / t)


def kernel_upsample2(zeros=56):
    """kernel_upsample2.

    """
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t *= math.pi
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def upsample2(x, zeros=56):
    """
    Upsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    *other, time = x.shape
    kernel = kernel_upsample2(zeros).to(x)
    out = F.conv1d(x.view(-1, 1, time), kernel, padding=zeros)[..., 1:].view(*other, time)
    y = th.stack([x, out], dim=-1)
    return y.view(*other, -1)


def kernel_downsample2(zeros=56):
    """kernel_downsample2.

    """
    win = th.hann_window(4 * zeros + 1, periodic=False)
    winodd = win[1::2]
    t = th.linspace(-zeros + 0.5, zeros - 0.5, 2 * zeros)
    t.mul_(math.pi)
    kernel = (sinc(t) * winodd).view(1, 1, -1)
    return kernel


def downsample2(x, zeros=56):
    """
    Downsampling the input by 2 using sinc interpolation.
    Smith, Julius, and Phil Gossett. "A flexible sampling-rate conversion method."
    ICASSP'84. IEEE International Conference on Acoustics, Speech, and Signal Processing.
    Vol. 9. IEEE, 1984.
    """
    if x.shape[-1] % 2 != 0:
        x = F.pad(x, (0, 1))
    xeven = x[..., ::2]
    xodd = x[..., 1::2]
    *other, time = xodd.shape
    kernel = kernel_downsample2(zeros).to(x)
    out = xeven + F.conv1d(xodd.view(-1, 1, time), kernel, padding=zeros)[..., :-1].view(
        *other, time)
    return out.view(*other, -1).mul(0.5)

def resample_labels(x, target_len, discrete = False):
    length = len(x)
    if length == target_len:
        return x
    else:
        if length > target_len:
            n = length - target_len
            i = 0
            interval = length // n
            mask = np.array([True]*length) 
            while i < n:
                mask[i*interval]  = False
                i+=1
            return x[mask]
        else:
            n =  target_len - length
            interval = length // n
            if not discrete:
                output = []
                for j in range(x.shape[-1]):
                    finished = False
                    i = 0
                    data = x[:, j]
                    new_data = []
                    for i_t, t in enumerate(data):
                        if i_t % interval ==0 and not finished:
                            new_data.extend([t, np.nan])
                            i +=1
                        else:
                            new_data.append(t)
                        if i == n:
                            finished = True
                    assert len(new_data) == target_len
                    df = pd.Series(new_data)
                    df = df.interpolate(method='linear', limit=4)
                    assert not df.isnull().values.any()
                    output.append(df.values)
                output = np.array(output).reshape((-1, x.shape[-1]))
            else:
                output = []
                finished = False
                data = x
                new_data = []
                i = 0
                for i_t, t in enumerate(data):
                    if i_t % interval ==0 and not finished:
                        new_data.extend([t, np.nan])
                        i +=1
                    else:
                        new_data.append(t)
                    if i == n:
                        finished = True
                assert len(new_data) == target_len
                df = pd.Series(new_data)
                df = df.interpolate(method='pad', limit=4)
                output = df.values.astype(np.int32)
                assert not df.isnull().values.any()
            
            return output





