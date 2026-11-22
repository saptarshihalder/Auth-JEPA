import io
import numpy as np
from PIL import Image


def load(path, jpeg_quality=0):  # raw = original file; JPEG-90 = jpeg_quality=90
    im = Image.open(path).convert("RGB")
    if jpeg_quality:
        b = io.BytesIO()
        im.save(b, "JPEG", quality=jpeg_quality, subsampling=2)  # re-save as JPEG (real and AI images alike)
        b.seek(0)
        im = Image.open(b).convert("RGB")
    return np.asarray(im)
