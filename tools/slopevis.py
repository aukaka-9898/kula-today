"""Fallback camera base: where the slope stops being visible (Sam, 2026-10-19:
"The scale is the slope"). Row by row up each column, a row counts as covered
when it is brighter than the FAA clear-day picture of the slope by more than
the frame's own lowest rows are (lighting) plus 30. Used only when the
cloud-edge reading fails: it reads the grey falling deck (10-19 15:20-15:55)
that the edge test called sky, ~250-350 ft high there."""
import sys
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.insert(0, r"E:\dev\WX\kula-today\tools")
import camtest as ct, camgeo as g  # noqa: E402

REF = np.asarray(Image.open("E:/dev/WX/KulaModel/review/camera_archive/clearday_13417.jpg").convert("L"), float)
HALF_W, HALF_H = 8, 3


def skyline(x):
    """First row (from the top of the window) where the clear-day slope begins."""
    col = REF[560:720, x - 4:x + 4].mean(axis=1)
    sky = col[:15].mean()
    for i, v in enumerate(col):
        if v < sky - 25:
            return 560 + i
    return 600


def base_bright(path, heights, thr=30, run=3, base_rows=15):
    """Covered = the frame brighter than the clear-day slope by more than the
    lowest rows' own difference (lighting) plus thr. Scan bottom up."""
    frm = np.asarray(Image.open(path).convert("L"), float)
    res = []
    for x in ct.COLS:
        top, bottom = skyline(x) + 2, ct.ROWS[1]
        ex = {r: frm[r - 1:r + 2, x - 6:x + 7].mean() - REF[r - 1:r + 2, x - 6:x + 7].mean()
              for r in range(top, bottom)}
        rows = list(range(bottom - 1, top - 1, -1))
        lift = float(np.median([ex[r] for r in rows[:base_rows]]))
        cov = {r: ex[r] - lift > thr for r in rows}
        if sum(cov[r] for r in rows[:6]) >= 4:
            res.append("below"); continue
        hit = next((rows[i] for i in range(len(rows) - run) if all(cov[rows[i + k]] for k in range(run))), None)
        res.append("above" if hit is None else (heights[x].get(hit) or "above"))
    nums = [v for v in res if isinstance(v, (int, float))]
    if res.count("below") >= 4:
        return "below", res
    if res.count("above") >= 4:
        return "above", res
    if len(nums) < 3:
        return None, res
    spread = float(np.percentile(nums, 75) - np.percentile(nums, 25))
    return (round(float(np.median(nums)), -1) if spread <= 600 else None), res
