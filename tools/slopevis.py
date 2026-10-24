"""Fallback camera base: where the slope stops being visible (Sam, 2026-10-19:
"The scale is the slope"). Row by row up each column, a row counts as covered
when it is brighter than the FAA clear-day picture of the slope by more than
the frame's own lowest rows are (lighting) plus 30. Used only when the
cloud-edge reading fails: it reads the grey falling deck (10-19 15:20-15:55)
that the edge test called sky, ~250-350 ft high there."""
import numpy as np
from PIL import Image

import cambase  # noqa: E402  -- the backup itself now lives there, with the clear-day picture

REF = cambase._ref()
skyline = cambase.clear_skyline


def base_bright(path, heights, thr=30, run=3, base_rows=15):
    return cambase.bright_base(np.asarray(Image.open(path).convert("L"), float), heights, thr, run, base_rows)
