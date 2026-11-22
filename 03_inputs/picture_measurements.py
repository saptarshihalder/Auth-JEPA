import numpy as np
from scipy.ndimage import laplace


def content(den):  # 13 picture measurements, on the smoothed patch
    px = den.reshape(-1, 3)
    lum = den.mean(2)
    gy, gx = np.gradient(lum)
    return np.array([
        *np.log(px.mean(0) + 1.0),  # brightness (R, G, B)
        *np.log(px.std(0) + 1e-3),  # contrast (R, G, B)
        np.log(np.hypot(gx, gy).mean() + 1e-3),  # amount of edges
        np.log(np.abs(laplace(lum)).mean() + 1e-3),  # fine texture
        float((lum > 245).mean()),  # share of pure-white pixels
        float((lum < 10).mean()),  # share of pure-black pixels
        *np.log(np.quantile(lum, [0.1, 0.5, 0.9]) + 1.0),  # dark, middle, bright levels
    ])
