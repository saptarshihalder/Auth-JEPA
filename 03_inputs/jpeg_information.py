import numpy as np
from PIL import Image

STD = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)  # standard JPEG table
TABLES = {q: np.sort(np.clip(np.floor((STD * (5000 / q if q < 50 else 200 - 2 * q) + 50) / 100), 1, 255))
          for q in range(1, 101)}  # table for each quality 1 to 100


def file_q(path):  # JPEG quality read from the file's compression table (0 = not a JPEG)
    im = Image.open(path)
    if im.format != "JPEG":
        return 0
    t = np.sort(np.array(list(im.quantization[0]), float))
    return min(TABLES, key=lambda q: np.abs(TABLES[q] - t).sum())  # closest standard quality


def jpeg_information(path):  # the 2 format values: JPEG flag, quality / 100
    q = file_q(path)
    return np.array([float(q > 0), q / 100.0], np.float32)
