import numpy as np
from scipy import stats
from scipy.ndimage import laplace
from wiener_filter import denoised  # 02_denoiser/wiener_filter.py

EPS = 1e-8


def phi45(x):  # original 45 noise measurements
    den = denoised(x)
    r = x - den  # removed noise
    lum = r.mean(2)
    f = []
    for c in range(3):  # noise strength and shape (9)
        rc = r[..., c].ravel()
        f += [np.log(rc.var() + EPS), stats.skew(rc), stats.kurtosis(rc)]
    for c in range(3):  # noise similarity to neighbours (9)
        rc = r[..., c] - r[..., c].mean(); d = (rc ** 2).sum() + EPS
        f += [0.5 * ((rc[:, :-k] * rc[:, k:]).sum() + (rc[:-k] * rc[k:]).sum()) / d for k in (1, 2, 3)]
    for i, j in ((0, 1), (1, 2), (0, 2)):  # noise shared between colours (3)
        a, b = r[..., i].ravel(), r[..., j].ravel()
        f.append(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + EPS))
    f += noise_vs_brightness(r, den)  # noise vs brightness (2)
    v = [lum[0::2, 0::2].var(), lum[0::2, 1::2].var(), lum[1::2, 0::2].var(), lum[1::2, 1::2].var()]
    f += [v[i] / (np.mean(v) + EPS) for i in range(3)]  # colour-filter pattern (3)
    f += wavelet_energies(lum)  # noise at different scales (9)
    f += radial_spectrum(lum)  # noise frequency content (8)
    f += blockiness(lum)  # JPEG block edges (2)
    return np.array(f, float)


def noise_vs_brightness(r, den, nbins=8, min_count=64):  # noise grows with brightness (shot noise)
    inten, var_px = den.mean(2).ravel(), (r ** 2).mean(2).ravel()
    qs = np.unique(np.quantile(inten, np.linspace(0, 1, nbins + 1)))  # 8 brightness bins
    xs, ys = [], []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (inten >= lo) & (inten < hi)
        if m.sum() >= min_count:
            xs.append(inten[m].mean()); ys.append(var_px[m].mean())
    if len(xs) < 3:
        return [0.0, float(np.log(var_px.mean() + EPS))]
    slope, intercept = np.polyfit(xs, ys, 1)  # variance = slope * brightness + intercept
    return [float(slope), float(np.log(max(intercept, 0.0) + EPS))]


def wavelet_energies(lum, wavelet="db4", levels=3):  # share of noise energy per scale and direction
    import pywt
    co = pywt.wavedec2(lum, wavelet, level=levels, mode="symmetric")
    total = float(sum((b ** 2).sum() for d in co[1:] for b in d)) + EPS
    return [float(np.log((b ** 2).sum() / total + EPS)) for L in range(1, levels + 1) for b in co[-L]]


def radial_spectrum(lum, nbins=8):  # Fourier magnitude of the noise in 8 rings
    h, w = lum.shape
    mag = np.abs(np.fft.fftshift(np.fft.fft2((lum - lum.mean()) * np.outer(np.hanning(h), np.hanning(w)))))
    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    e = np.linspace(0, 1, nbins + 1)
    return [float(np.log(mag[(rad >= lo) & (rad < hi)].mean() + EPS)) for lo, hi in zip(e[:-1], e[1:])]


def blockiness(lum, block=8):  # jumps at 8 x 8 JPEG block borders
    out = []
    for axis in (1, 0):
        d = np.abs(np.diff(lum, axis=axis))
        on = np.arange(d.shape[axis]) % block == block - 1
        a, b = (d[:, on], d[:, ~on]) if axis == 1 else (d[on], d[~on])
        out.append(float(a.mean() / (b.mean() + EPS)))
    return out


def _phase8(src, axis):  # JPEG grid at any position
    d = np.abs(np.diff(src, axis=axis))
    ph = np.array([(d[:, p::8] if axis == 1 else d[p::8, :]).mean() for p in range(8)])
    return float(np.log(ph.max() / (np.median(ph) + EPS) + EPS))


def phiplus(x):  # 29 added measurements: any crop position, any camera layout
    den = denoised(x)
    r = x - den
    f = []
    for src in (x.mean(2), r.mean(2)):  # JPEG grid, any position (4)
        f += [_phase8(src, 1), _phase8(src, 0)]
    for c in range(3):  # colour-filter pattern, per colour (9)
        rc = r[..., c]
        v = np.array([rc[0::2, 0::2].var(), rc[0::2, 1::2].var(), rc[1::2, 0::2].var(), rc[1::2, 1::2].var()]) + EPS
        f += [np.log(v.max() / v.min()), np.log(v.max() / np.delete(v, v.argmax()).mean()),
              abs(np.log((v[0] + v[3]) / (v[1] + v[2])))]
    for c in range(3):  # colour fill-in pattern (3)
        L = np.abs(laplace(x[..., c]))
        m = np.array([L[0::2, 0::2].mean(), L[0::2, 1::2].mean(), L[1::2, 0::2].mean(), L[1::2, 1::2].mean()]) + EPS
        f.append(np.log(m.max() / m.min()))
    for c in range(3):  # noise vs brightness, per colour (9)
        inten, vp = den[..., c].ravel(), (r[..., c] ** 2).ravel()
        qs = np.unique(np.quantile(inten, np.linspace(0, 1, 9)))
        xs, ys = [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (inten >= lo) & (inten < hi)
            if m.sum() >= 64:
                xs.append(inten[m].mean()); ys.append(vp[m].mean())
        lx = np.log(np.asarray(xs) + 1.0) if xs else np.zeros(0)
        if len(xs) >= 3 and lx[-1] - lx[0] > 0.2:
            ly = np.log(np.asarray(ys) + EPS)
            b1, b0 = np.polyfit(lx, ly, 1)
            f += [float(np.median(ly)), float(np.clip(b1, -5, 5)), float((ly - b0 - b1 * lx).std())]
        else:
            f += [float(np.log(vp.mean() + EPS)), 0.0, 0.0]
    f.append(float((np.abs(r).max(2) < 0.5).mean()))  # noise-free pixels (1)
    for a, b in ((0, 1), (1, 2), (0, 2)):  # noise shared with neighbouring colours (3)
        u, w = r[:, :-1, a].ravel(), r[:, 1:, b].ravel()
        f.append(float(np.corrcoef(u, w)[0, 1]) if u.std() > EPS and w.std() > EPS else 0.0)
    return np.nan_to_num(np.asarray(f, float))


def noise_measurements(tile):  # all 74 = 45 + 29
    x = np.asarray(tile, dtype=np.float64)
    return np.concatenate([phi45(x), phiplus(x)])
