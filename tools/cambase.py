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
