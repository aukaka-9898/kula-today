"""The Kekoa South camera's cloudbase from one frame: the cloud-to-slope edge
read down seven columns across Sam's box. Shared by camtest.py (blind tests)
and build.py (logged beside derived each run). No imports from either.

Heights per pixel come from kekoa_heights.json, written from camtest's
column_heights() (camgeo + lidar slope + Sam's pasture anchors). Rewrite it
whenever the camera geometry changes.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COLS = (870, 890, 910, 930, 950, 970, 990)      # columns read across Sam's box
ROWS = (590, 706)          # read window; extended down (Sam, 2026-10-19) to ~2,950 ft of slope
MAX_SPREAD_FT = 600        # wider = cloud in front of the slope
SKYLINE_PX = 4             # an edge this close under the skyline is the ridge, not a base
OVERCAST_BLUE = 20         # sky blue-minus-red at or below this: a grey, overcast sky


def load_heights(path: Path = HERE / "kekoa_heights.json") -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(x): {int(r): v for r, v in col.items()} for x, col in raw.items()}


def skyline(heights: dict, x: int) -> int:
    return min(r for r, v in heights[x].items() if v is not None)


def sky_blue(rgb) -> float:
    s = rgb[80:300, 200:1700]
    return float((s[..., 2] - s[..., 0]).mean())


def edge_base(rgb, heights: dict):
    """(base ft, spread ft, top_edges). base None when fewer than 4 columns
    found an edge or they disagree by more than MAX_SPREAD_FT; top_edges counts
    columns whose only edge was the skyline (visible slope clear)."""
    a = rgb.mean(axis=2)
    hs, top_edges = [], 0
    for x in COLS:
        col = a[ROWS[0]:ROWS[1], x - 6:x + 6].mean(axis=1)
        sm = np.convolve(col, np.ones(3) / 3, mode="same")
        top, bot = sm[:10].mean(), sm[-20:].mean()       # slope reference: the lowest rows
        if top < 135 or top - bot < 40:          # blue sky above, or no clear edge
            continue
        thr = (top + bot) / 2
        i = next((k for k in range(5, len(sm)) if sm[k] < thr <= sm[k - 1]), None)
        # The skyline (Sam, 2026-10-20: "there is no way it was 6,900" -- the
        # base was well below the ridge; with the visible slope clear the only
        # edge left was bright sky against the dark ridge).
        if i is not None and ROWS[0] + i <= skyline(heights, x) + SKYLINE_PX:
            top_edges += 1
            continue
        if i is not None and heights[x].get(ROWS[0] + i) is not None:
            hs.append(heights[x][ROWS[0] + i])
    if len(hs) >= 4:
        spread = float(np.percentile(hs, 75) - np.percentile(hs, 25))
        if spread <= MAX_SPREAD_FT:
            return round(float(np.median(hs)), -1), round(spread, -1), top_edges
    return None, None, top_edges


# The grey-sky backup (Sam, 2026-10-19: "The scale is the slope"): where the
# slope stops being visible against the FAA clear-day picture
# (kekoa_clearday.jpg, kept in the repo for this). Row by row up each column, a
# row counts as covered when it is brighter than the clear-day slope by more
# than the frame's own lowest rows are (lighting) plus thr. Only under a grey
# sky: on bright days the sunlit slope reads brighter than the reference and
# the backup took it for cloud (4,100-4,700 ft all of 10-20, a high day). It
# reads the grey falling deck (10-19 15:20-15:55) the edge test calls sky,
# ~250-350 ft high there.
CLEARDAY = HERE / "kekoa_clearday.jpg"
_REF = None


def _ref():
    global _REF
    if _REF is None:
        from PIL import Image
        _REF = np.asarray(Image.open(CLEARDAY).convert("L"), float)
    return _REF


def clear_skyline(x: int) -> int:
    """First row (from the top of the window) where the clear-day slope begins."""
    col = _ref()[560:720, x - 4:x + 4].mean(axis=1)
    sky = col[:15].mean()
    for i, v in enumerate(col):
        if v < sky - 25:
            return 560 + i
    return 600


def bright_base(gray, heights: dict, thr=30, run=3, base_rows=15):
    """(base ft, "below" or "above" or None, per-column results) from a
    grayscale frame. Scan bottom up."""
    ref = _ref()
    res = []
    for x in COLS:
        top, bottom = clear_skyline(x) + 2, ROWS[1]
        ex = {r: gray[r - 1:r + 2, x - 6:x + 7].mean() - ref[r - 1:r + 2, x - 6:x + 7].mean()
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


# Hidden (Sam, 2026-10-29: "a two hour gap in camera calls, from 1100"): cloud
# or rain right at the camera washes the whole box out evenly, the backup's
# lighting correction cancels it, and it read "above" -- logged as "no edge".
# Contrast in the box against the clear day's: fogged-in frames 0.02-0.15
# (10-27 to 10-29), clear-air frames 0.45-0.78, edges still found down to ~0.25.
HIDDEN_RATIO = 0.16


def box_contrast(gray) -> float:
    w = gray[ROWS[0]:ROWS[1] + 60, 840:1020]
    return float(np.abs(np.diff(w, axis=0)).mean() + np.abs(np.diff(w, axis=1)).mean())


def camera_base(rgb, heights: dict, gray=None):
    """(base ft or None, spread ft or None, why): the edge reading, then the
    grey-sky backup. why: "" (edge), "grey backup", "slope clear", "no edge",
    "bright" (no edge, blue sky: backup skipped), "hidden" (cloud or rain at
    the camera: the slope is lost)."""
    if gray is None:                     # PIL's "L" weights, as the backup was tuned
        gray = rgb @ np.array([0.299, 0.587, 0.114])
    if box_contrast(gray) < HIDDEN_RATIO * box_contrast(_ref()):
        return None, None, "hidden"
    base, spread, top = edge_base(rgb, heights)
    if base is not None:
        return base, spread, ""
    if top >= 4:
        return None, None, "slope clear"
    if sky_blue(rgb) > OVERCAST_BLUE:
        return None, None, "bright"
    v, _ = bright_base(gray, heights, thr=30)
    return (v, None, "grey backup") if isinstance(v, float) else (None, None, "no edge")
