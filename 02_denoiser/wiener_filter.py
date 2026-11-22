import numpy as np
import pywt
from scipy.ndimage import uniform_filter


def _local_variance(coef, sigma0_sq, windows=(3, 5, 7, 9)):  # local signal variance, smallest over 4 windows
    est = None
    for w in windows:
        v = np.maximum(uniform_filter(coef ** 2, size=w, mode="reflect") - sigma0_sq, 0.0)
        est = v if est is None else np.minimum(est, v)
    return est


def wavelet_wiener_denoise(channel, sigma=2.0, wavelet="db8", levels=4):  # wavelet-domain Wiener filter
    sigma0_sq = float(sigma) ** 2
    h, w = channel.shape
    lev = int(min(levels, pywt.dwtn_max_level((h, w), wavelet)))  # 3 levels fit in 128 px
    coeffs = pywt.wavedec2(channel, wavelet, level=lev, mode="symmetric")  # Daubechies-8 wavelet
    out = [coeffs[0]]  # coarse part kept: the picture
    for detail in coeffs[1:]:
        shrunk = []
        for band in detail:
            var = _local_variance(band, sigma0_sq)
            shrunk.append(band * (var / (var + sigma0_sq)))  # Wiener shrinkage
        out.append(tuple(shrunk))
    return pywt.waverec2(out, wavelet, mode="symmetric")[:h, :w]


def denoised(patch, sigma=2.0):  # smoothed patch (picture), channel by channel
    x = np.asarray(patch, dtype=np.float64)
    return np.stack([wavelet_wiener_denoise(x[..., c], sigma=sigma) for c in range(3)], axis=-1)


def residual(patch, sigma=2.0):  # removed noise = patch - smoothed patch
    x = np.asarray(patch, dtype=np.float64)
    return x - denoised(x, sigma)
