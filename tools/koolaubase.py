"""Cloud on the Koʻolau from the FAA Kaneohe Bay SW camera (Sam, 2026-10-25:
"build SW"). Geometry from koolaucam.py; the crest's place in each column
from the clear-air frame (kaneohe_clearday.jpg), so cloud hiding the crest
does not matter.

For each column across the crest:
  capped -- just under the clear-air skyline the dark ridge has turned to
            cloud: brighter than the clear frame there by more than the
            frame's own lighting change (read lower on the slopes), and grey
            or white rather than coloured. Base at or below the crest there.
  clear  -- the ridge is still there. Looking up from the crest, the first
            cloud pixel's elevation angle at the crest's distance gives the
            base above it, if a cloud bottom sits within REACH_PX.

LIMIT, found on the first morning (10-25): a cloud bottom ABOVE the crest is
placed at the crest's distance, but trade cumulus drift in over Kāneʻohe, in
front of the range, and a nearer cloud at the same angle is lower. So "base"
here is an upper bound; "capped" (cloud on the crest itself) is the solid
reading.

    python tools/koolaubase.py FRAME.jpg [out.jpg]   # one frame, annotated
    python tools/koolaubase.py live                  # the feed's frames now
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import koolaucam as kc  # noqa: E402

# Segments of the crest in this view (x ranges, left to right).
SEGMENTS = (("Greenwall", 170, 470), ("Konahuanui", 470, 640), ("Kāneʻohe crest", 640, 1260),
            ("Kahaluʻu crest", 1260, 1730))
# The other two views from the same site (Sam 10-25: "build SE and NW ridge").
# SE: "Waimānalo" from launch (x ~1097) to Puʻu o Kona (x ~1460), "Greenwall"
# from Puʻu o Kona on (Sam's names); the antenna box at the right edge masked.
# NW: up the coast to Kualoa (Puʻukānehoalani, x ~990); names provisional.
# NW's reference is FAA's sunrise picture, so its capped test compares the
# strip under the crest with the slope lower in the SAME frame ("self").
# SE judged NOT USABLE (Sam, 10-25): the Waimānalo crest is 9 miles off and
# low, and cloud just above it can't be told from cloud beyond it. NW dropped
# too (Sam: "about 50% of NW is usable. You are picking up the finger out to
# Kualoa as the crest"). Both kept for reference only; the page uses SW.
VIEWS = {"sw": {"camera": 12784, "segments": SEGMENTS, "mask": None, "capped": "clear"},
         "se": {"camera": 12783, "segments": (("Waimānalo", 1097, 1460), ("Greenwall", 1460, 1800)),
                "mask": ((0, 1090), (1800, 1920)), "capped": "clear"},
         "nw": {"camera": 12785, "segments": (("Up the coast", 20, 900), ("Kualoa", 900, 1100)),
                "mask": ((955, 1025), (1100, 1920)), "capped": "self"}}
SELF_LIFT = 35             # "self": under-crest strip brighter than the slope below by this
# Columns never read: the cell tower and its antennas standing in front of the
# crest (x ~1262-1345; Sam 10-25: "mask the tower" -- it read as cloud), and
# the antennas at the left edge of the frame, which stand up past the crest to
# x ~125 (Sam: "attenae on the right edge" -- "I meant the other right").
# Past the high point at x ~1720 (2,690 ft) the crest drops to ~2,100 ft at
# the frame edge with farther ridges behind it: not read (Sam: "Omit that
# section").
MASK_X = ((0, 130), (1250, 1355), (1730, 1920))
VIEWS["sw"]["mask"] = MASK_X
BELOW = (3, 16)            # rows under the skyline read for cloud on the crest
LIGHT = (45, 70)           # rows under the skyline read for the frame's lighting
CAP_LIFT = 30              # brighter than clear air by this, beyond the lighting change
CAP_SAT = 45               # grey/white: max - min channel under this
REACH_PX = 220             # how far above the crest a cloud bottom is looked for
# Cloud vs the pale haze right over the crest (10:03 clear frame): haze
# ~150 grey with blue minus red ~9-24; cloud 150-170 with blue minus red
# 4 or less. The first 8 rows over the skyline are haze and skipped; a
# bottom needs 4 cloud rows in a row.
HAZE_PX, RUN = 8, 4
# Dark cloud down on the crest (Sam, 10-25: Konahuanui "is not clear. Cam shows
# this" -- a dark grey base, ~120 brightness, sat on the summit and the bright-
# cloud test looked past it). The frame's own clear sky beside it along the
# ridge is the reference (Sam: "following the ridge a bit to the right shows
# clear sky"): the strip 10-25 rows over the crest darker than the 75th
# percentile of that strip within NEAR_PX either side by DARK_MARGIN = cloud
# bottom at the crest.
NEAR_PX, DARK_MARGIN = 200, 25
CLOUD_BR, CLOUD_BRIGHT = 6, 140
_GEOM: dict = {}


def geometry(view="sw"):
    """(geom, columns, clear-air skyline rows, crest distance m, crest ft) where
    the detected clear-air skyline agrees with the terrain model."""
    if view not in _GEOM:
        from PIL import Image
        kc.use(view)
        MASK = VIEWS[view]["mask"]
        g = json.load(open(kc.GEOM))
        clear = np.asarray(Image.open(kc.CLEARDAY).convert("L"), float)
        cols, rows = kc.image_skyline(clear)
        sky = kc.model_skyline(kc.ELEV_FT / 3.28084 + g["mast_m"], kc.profiles())
        az = np.array(list(sky))
        dense = np.arange(az[0], az[-1], 0.05)
        ang = np.interp(dense, az, [v[0] for v in sky.values()])
        xs, ys = kc.project(dense, ang, g)
        keep = np.abs(dense - g["bearing"]) < 55
        xs, ys, dz = xs[keep], ys[keep], dense[keep]
        o = np.argsort(xs)
        model_y = np.interp(cols, xs[o], ys[o])
        az_col = np.interp(cols, xs[o], dz[o])
        dist = np.array([sky[min(sky, key=lambda a: abs(a - z))][2] for z in az_col])
        crest = np.array([sky[min(sky, key=lambda a: abs(a - z))][1] for z in az_col])
        ok = (np.abs(model_y - rows) < 25) & ~np.any([(cols >= a) & (cols <= b) for a, b in MASK], axis=0)
        _GEOM[view] = (g, cols[ok], rows[ok], dist[ok], crest[ok], az_col[ok], clear)
    kc.use(view)
    return _GEOM[view]


def angle_at(g, az, y):
    """Elevation angle of image row y in the column of bearing az (inverts project)."""
    angs = np.linspace(-10, 30, 4001)
    _, ys = kc.project(np.full_like(angs, az), angs, g)
    o = np.argsort(ys)
    return float(np.interp(y, ys[o], angs[o]))


def height_ft(g, az, y, dist_m):
    a = math.radians(angle_at(g, az, y))
    cam = kc.ELEV_FT / 3.28084 + g["mast_m"]
    return (cam + dist_m * math.tan(a) + dist_m ** 2 / (2 * kc.R) * 0.87) * 3.28084


def read(rgb, view="sw"):
    """{columns: [...], segments: {name: {capped share, base ft}}, base_ft, capped}."""
    g, cols, rows, dist, crest, azs, clear = geometry(view)
    selfref = VIEWS[view]["capped"] == "self"
    gray = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    strip = np.array([gray[int(y0) - 25:int(y0) - 10, x - 3:x + 4].mean() for x, y0 in zip(cols, rows)])
    ref = np.array([np.percentile(strip[np.abs(cols - x) <= NEAR_PX], 75) for x in cols])
    out = []
    for x, y0, d, c, az, st, rf in zip(cols, rows, dist, crest, azs, strip, ref):
        y0 = int(round(y0))
        sl = slice(x - 3, x + 4)
        grey = sat[y0 + BELOW[0]:y0 + BELOW[1], sl].mean() < CAP_SAT
        if selfref:
            under = gray[y0 + BELOW[0]:y0 + BELOW[1], sl].mean() - gray[y0 + LIGHT[0]:y0 + LIGHT[1], sl].mean()
            capped = under > SELF_LIFT and grey
        else:
            lift = gray[y0 + LIGHT[0]:y0 + LIGHT[1], sl].mean() - clear[y0 + LIGHT[0]:y0 + LIGHT[1], sl].mean()
            under = gray[y0 + BELOW[0]:y0 + BELOW[1], sl].mean() - clear[y0 + BELOW[0]:y0 + BELOW[1], sl].mean()
            capped = (under - lift > CAP_LIFT) and grey
        base = None
        if not capped and st < rf - DARK_MARGIN:
            base = height_ft(g, az, y0 - 10, d)          # dark cloud down to the crest
        elif not capped:
            top, bot = max(0, y0 - REACH_PX), y0 - HAZE_PX
            colg = gray[top:bot, sl].mean(axis=1)[::-1]                    # upward from the crest
            br = (rgb[top:bot, sl, 2] - rgb[top:bot, sl, 0]).mean(axis=1)[::-1]
            cloud = (colg > CLOUD_BRIGHT) & (br <= CLOUD_BR)
            hit = next((i for i in range(len(cloud) - RUN) if cloud[i:i + RUN].all()), None)
            if hit is not None:
                base = height_ft(g, az, bot - hit, d)
        out.append({"x": int(x), "crest_ft": round(c), "capped": bool(capped),
                    "base_ft": None if base is None else round(base, -1)})
    segs = {}
    for name, a, b in VIEWS[view]["segments"]:
        part = [o for o in out if a <= o["x"] < b]
        if not part:
            continue
        cap = sum(o["capped"] for o in part) / len(part)
        bases = [o["base_ft"] for o in part if o["base_ft"] is not None]
        segs[name] = {"capped": round(cap, 2), "n": len(part),
                      "base_ft": round(float(np.median(bases)), -1) if len(bases) >= 3 else None,
                      "crest_ft": round(float(np.median([o["crest_ft"] for o in part])), -1),
                      "crest_max_ft": round(float(max(o["crest_ft"] for o in part)), -1)}
    caps = [o for o in out if o["capped"]]
    bases = [o["base_ft"] for o in out if o["base_ft"] is not None]
    return {"columns": out, "segments": segs, "capped": round(len(caps) / max(len(out), 1), 2),
            "base_ft": round(float(np.median(bases)), -1) if len(bases) >= 5 else None}


def annotate(rgb, res, path, view="sw"):
    from PIL import Image, ImageDraw
    g, cols, rows, *_ = geometry(view)
    im = Image.fromarray(rgb.astype("uint8"))
    dr = ImageDraw.Draw(im)
    for o, y0 in zip(res["columns"], rows):
        col = (255, 60, 60) if o["capped"] else (255, 255, 0)
        dr.ellipse([o["x"] - 3, y0 - 3, o["x"] + 3, y0 + 3], fill=col)
    for name, s in res["segments"].items():
        x = next(a for n, a, b in VIEWS[view]["segments"] if n == name)
        dr.text((x + 4, 560), f"{name}: capped {s['capped']:.0%}" + (f", base {s['base_ft']:,.0f}" if s["base_ft"] else ""),
                fill=(255, 255, 255))
    im.save(path)


def live(out_dir, view="sw"):
    import camcapture as cc
    from io import BytesIO
    from PIL import Image
    from datetime import datetime
    j = json.loads(cc.get(cc.URL.format(site=760), cc.PAGE.format(site=760)))
    cam = next(c for c in j["payload"]["site"]["cameras"] if c["cameraId"] == VIEWS[view]["camera"])
    for im in sorted(cam["currentImages"], key=lambda x: x["imageDatetime"]):
        t = datetime.fromisoformat(im["imageDatetime"].replace("Z", "+00:00")).astimezone(cc.HST)
        rgb = np.asarray(Image.open(BytesIO(cc.get(im["imageUri"]))).convert("RGB"), float)
        if rgb.mean() < 40:
            continue                                   # night
        res = read(rgb, view)
        print(f"{t:%H:%M}  capped {res['capped']:.0%}  base {res['base_ft']}  " +
              "  ".join(f"{n}: {s['capped']:.0%}/{s['base_ft']}" for n, s in res["segments"].items()))
        annotate(rgb, res, os.path.join(out_dir, f"{view}_{t:%H%M}.jpg"), view)


if __name__ == "__main__":
    if sys.argv[1] == "live":
        live(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "sw")
    else:
        from PIL import Image
        rgb = np.asarray(Image.open(sys.argv[1]).convert("RGB"), float)
        res = read(rgb)
        print(json.dumps({k: v for k, v in res.items() if k != "columns"}, ensure_ascii=False))
        if len(sys.argv) > 2:
            annotate(rgb, res, sys.argv[2])
