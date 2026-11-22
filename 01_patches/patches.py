import numpy as np

TILE, WIN, G = 128, 512, 4  # patch size, window size, 4 x 4 grid


def window(img, rng):  # one 512 x 512 window, never resized
    h, w = img.shape[:2]
    if h < WIN or w < WIN:  # smaller images are skipped
        return None
    t0 = int(rng.integers(0, (h - WIN) // 2 + 1)) * 2  # even row
    l0 = int(rng.integers(0, (w - WIN) // 2 + 1)) * 2  # even column
    return img[t0:t0 + WIN, l0:l0 + WIN]


def grid_tiles(img):  # 16 patches of 128 x 128
    return [img[i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE]
            for i in range(G) for j in range(G)]
