"""Blind camera cloudbase test (Sam, 2026-10-18: "let's run this a few times
over the next days").

    python tools/camtest.py make [DATE] [N]   # archived Kekoa-S frames -> N random blind images + viewer
    python tools/camtest.py score FILE [DATE] # FILE: lines "C01 5300" (or "?"); unblind, append, summarise

make: takes the day's frames from the camera archive (camcapture.py, run every
30 min by the "Kula camera capture" task; it also tops the archive up from the
live feed now), picks N (default 16) daytime frames at random -- unselected, so the
result says how the derived base does on ordinary frames -- masks the time
stamps, and writes C01.jpg.. plus 3x zooms around Sam's box and test.html to
E:/dev/WX/KulaModel/review/<date>_camtest/. The key stays in key.json there:
don't open it before scoring. Serve with the "camtest" launch entry
(http://localhost:8767/<date>_camtest/test.html).

score: for each frame, the camera-geometry base (camgeo.py: tilt calibrated on
Sam's 4,900 ft line, Open-Meteo terrain; read down column 895-916 where cloud
gives way to slope; None when there is no clear cloud-on-slope edge) and the
derived base (all-station median), against Sam's call. Appends to
docs/camera_tests.csv and prints the running totals.

Known limits (Sam): the camera geometry fails when cloud is not against the
mountain or sits between camera and mountain; Sam's own calls can slip
(10-18 C03: 4,000 typed, 5,300 meant), so look again at any big miss.
"""
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import build as b  # noqa: E402
import camgeo as g  # noqa: E402

REVIEW = Path("E:/dev/WX/KulaModel/review")
RESULTS = HERE.parent / "docs" / "camera_tests.csv"
CAM_ID = 13417                       # Kekoa South
BOX = (853, 621, 1043, 692)          # Sam's area to check
ZOOM = (653, 500, 1253, 800)
COL = (895, 916)                     # columns read for the edge
VIEWER = HERE.parent.parent / "KulaModel" / "review" / "2026-10-18_blind" / "blind_test.html"


def folder(day: str) -> Path:
    return REVIEW / f"{day}_camtest"


def fetch(day: str) -> list[tuple[str, Path]]:
    """[(HH:MM HST, path)] for the day's archived south frames."""
    try:
        import camcapture
        camcapture.capture()                  # top up from the live feed
    except Exception as e:
        print(f"capture: {type(e).__name__}")
    raw = REVIEW / "camera_archive" / day
    return sorted((f"{f.stem[:2]}:{f.stem[2:]}", f) for f in raw.glob("*.jpg"))


def make(day: str | None = None, n: int = 16) -> None:
    day = day or f"{datetime.now(b.HST):%Y-%m-%d}"
    frames = [(hm, f) for hm, f in fetch(day) if "07:30" <= hm <= "17:30"]
    pick = random.sample(frames, min(n, len(frames)))
    d = folder(day)
    d.mkdir(parents=True, exist_ok=True)
    key = {}
    for i, (hm, f) in enumerate(pick, 1):
        name = f"C{i:02d}"
        im = Image.open(f).convert("RGB")
        dr = ImageDraw.Draw(im)
        dr.rectangle([0, 0, 1920, 52], fill=(0, 0, 0))
        dr.rectangle(list(BOX), outline=(60, 60, 255), width=2)
        im.save(d / f"{name}.jpg", quality=92)
        im.crop(ZOOM).resize((1800, 900), Image.LANCZOS).save(d / f"{name}_zoom.jpg", quality=92)
        key[name] = hm
    (d / "key.json").write_text(json.dumps(key), encoding="utf-8")
    html = VIEWER.read_text(encoding="utf-8")
    html = (html.replace('"B" + String', '"C" + String').replace("blind1018", f"camtest{day}")
                .replace("const N = 16", f"const N = {len(key)}").replace('<b id="id">B01</b>', '<b id="id">C01</b>')
                .replace("<title>Blind Base Test</title>", "<title>Camera Base Test</title>"))
    (d / "test.html").write_text(html, encoding="utf-8")
    print(f"{len(key)} frames of {len(frames)} -> http://localhost:8767/{day}_camtest/test.html")


COLS = (870, 890, 910, 930, 950, 970, 990)      # columns read across Sam's box
MAX_SPREAD_FT = 600                               # wider = cloud in front of the slope


ROWS = (590, 706)          # read window; extended down (Sam, 2026-10-19) to ~2,950 ft of slope
# Below the pasture top the geometry reads high: Sam's pasture bottom (3,800 ft,
# pixel ~1005, 677) comes out 4,050. Heights under 4,600 are stretched so the
# two pasture landmarks land right; above 4,600 they already match Sam's calls.
TOP_FT, BOTTOM_FT, BOTTOM_GEOM_FT = 4600, 3800, 4050


def _anchor(h):
    if h is None or h >= TOP_FT:
        return h
    return TOP_FT - (TOP_FT - h) * (TOP_FT - BOTTOM_FT) / (TOP_FT - BOTTOM_GEOM_FT)


def column_heights(tilt: float) -> dict:
    return {x: {y: _anchor(g.terrain_at(x, y, tilt)[0]) for y in range(*ROWS)} for x in COLS}


def camera_base(f: Path, heights: dict):
    """(base ft or None, spread ft). The cloud-to-slope edge read down seven
    columns; a real base meets the slope at one height across them, a cloud in
    front of the slope reads the terrain behind it and wanders. On 10-18 a
    spread over 600 ft caught the two worst frames (08:51, 09:08) and no good
    ones (Sam: "can the auto call on bad frames be filtered?")."""
    a = np.asarray(Image.open(f).convert("L"), dtype=float)
    hs = []
    for x in COLS:
        col = a[ROWS[0]:ROWS[1], x - 6:x + 6].mean(axis=1)
        sm = np.convolve(col, np.ones(3) / 3, mode="same")
        top, bot = sm[:10].mean(), sm[-20:].mean()       # slope reference: the lowest rows
        if top < 135 or top - bot < 40:          # blue sky above, or no clear edge
            continue
        thr = (top + bot) / 2
        i = next((k for k in range(5, len(sm)) if sm[k] < thr <= sm[k - 1]), None)
        if i is not None and heights[x].get(ROWS[0] + i) is not None:
            hs.append(heights[x][ROWS[0] + i])
    if len(hs) < 4:
        return None, None
    spread = float(np.percentile(hs, 75) - np.percentile(hs, 25))
    return (round(float(np.median(hs)), -1) if spread <= MAX_SPREAD_FT else None), round(spread, -1)


def score(path: str, day: str | None = None) -> None:
    calls = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split()
        if len(p) >= 2 and p[0][:1] == "C":
            calls[p[0]] = None if p[1] == "?" else float(p[1])
    day = day or f"{datetime.now(b.HST):%Y-%m-%d}"
    d = folder(day)
    key = json.loads((d / "key.json").read_text(encoding="utf-8"))
    heights = column_heights(g.calibrate())
    when = datetime.now(b.HST)
    if day != f"{when:%Y-%m-%d}":
        when = datetime.fromisoformat(f"{day}T23:00").replace(tzinfo=b.HST)
    obs = b.observations(when)
    core = b._core(obs)
    rows = []
    for name, hm in key.items():
        t = pd.Timestamp(f"{day} {hm}")
        c = core[(core.observed <= t) & (core.observed > t - pd.Timedelta("20min"))]
        L = c.sort_values("observed").groupby("station_id").tail(1)
        cam, spread = camera_base(REVIEW / "camera_archive" / day / f"{hm.replace(':', '')}.jpg", heights)
        rows.append({"date": day, "time": hm, "frame": name, "sam": calls.get(name),
                     "camera": cam, "spread": spread,
                     "derived": round(float(L.lcl_ft.median()), -1) if len(L) >= 4 else None})
    r = pd.DataFrame(rows).sort_values("time")
    r.to_csv(d / "results.csv", index=False)
    all_ = pd.concat([pd.read_csv(RESULTS), r]) if RESULTS.exists() else r
    all_ = all_.drop_duplicates(["date", "time"], keep="last")
    all_.to_csv(RESULTS, index=False)
    print(r.to_string(index=False))
    a = all_.dropna(subset=["sam"])
    a = a.assign(hour=a.time.str[:2], der=a.derived - a.sam, cam=a.camera - a.sam)
    print(f"\nall days: {a.date.nunique()} days, {len(a)} calls")
    print(a.groupby("hour")[["der", "cam"]].agg(["mean", "count"]).round(-1).to_string())
    big = r.dropna(subset=["sam"])
    big = big[((big.derived - big.sam).abs() >= 800) | ((big.camera - big.sam).abs() >= 800)]
    if len(big):
        print("\nlook again (800+ ft from both or either):", ", ".join(f"{x.frame} {x.time}" for x in big.itertuples()))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "make"
    if cmd == "make":
        make(sys.argv[2] if len(sys.argv) > 2 else None, int(sys.argv[3]) if len(sys.argv) > 3 else 16)
    else:
        score(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
