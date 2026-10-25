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
            ("Kahaluʻu crest", 1260, 1880))
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
CLOUD_BR, CLOUD_BRIGHT = 6, 140
_GEOM = None


def geometry():
    """(geom, columns, clear-air skyline rows, crest distance m, crest ft) where
    the detected clear-air skyline agrees with the terrain model."""
    global _GEOM
    if _GEOM is None:
        from PIL import Image
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
        ok = np.abs(model_y - rows) < 25
        _GEOM = (g, cols[ok], rows[ok], dist[ok], crest[ok], az_col[ok], clear)
    return _GEOM


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


def read(rgb):
    """{columns: [...], segments: {name: {capped share, base ft}}, base_ft, capped}."""
    g, cols, rows, dist, crest, azs, clear = geometry()
    gray = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    out = []
    for x, y0, d, c, az in zip(cols, rows, dist, crest, azs):
        y0 = int(round(y0))
        sl = slice(x - 3, x + 4)
        lift = gray[y0 + LIGHT[0]:y0 + LIGHT[1], sl].mean() - clear[y0 + LIGHT[0]:y0 + LIGHT[1], sl].mean()
        under = gray[y0 + BELOW[0]:y0 + BELOW[1], sl].mean() - clear[y0 + BELOW[0]:y0 + BELOW[1], sl].mean()
        capped = (under - lift > CAP_LIFT) and sat[y0 + BELOW[0]:y0 + BELOW[1], sl].mean() < CAP_SAT
        base = None
        if not capped:
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
    for name, a, b in SEGMENTS:
        part = [o for o in out if a <= o["x"] < b]
        if not part:
            continue
        cap = sum(o["capped"] for o in part) / len(part)
        bases = [o["base_ft"] for o in part if o["base_ft"] is not None]
        segs[name] = {"capped": round(cap, 2), "n": len(part),
                      "base_ft": round(float(np.median(bases)), -1) if len(bases) >= 3 else None,
                      "crest_ft": round(float(np.median([o["crest_ft"] for o in part])), -1)}
    caps = [o for o in out if o["capped"]]
    bases = [o["base_ft"] for o in out if o["base_ft"] is not None]
    return {"columns": out, "segments": segs, "capped": round(len(caps) / max(len(out), 1), 2),
            "base_ft": round(float(np.median(bases)), -1) if len(bases) >= 5 else None}


def annotate(rgb, res, path):
    from PIL import Image, ImageDraw
    g, cols, rows, *_ = geometry()
    im = Image.fromarray(rgb.astype("uint8"))
    dr = ImageDraw.Draw(im)
    for o, y0 in zip(res["columns"], rows):
        col = (255, 60, 60) if o["capped"] else (255, 255, 0)
        dr.ellipse([o["x"] - 3, y0 - 3, o["x"] + 3, y0 + 3], fill=col)
    for name, s in res["segments"].items():
        x = next(a for n, a, b in SEGMENTS if n == name)
        dr.text((x + 4, 560), f"{name}: capped {s['capped']:.0%}" + (f", base {s['base_ft']:,.0f}" if s["base_ft"] else ""),
                fill=(255, 255, 255))
    im.save(path)


def live(out_dir):
    import camcapture as cc
    from io import BytesIO
    from PIL import Image
    from datetime import datetime
    j = json.loads(cc.get(cc.URL.format(site=760), cc.PAGE.format(site=760)))
    cam = next(c for c in j["payload"]["site"]["cameras"] if c["cameraId"] == 12784)
    for im in sorted(cam["currentImages"], key=lambda x: x["imageDatetime"]):
        t = datetime.fromisoformat(im["imageDatetime"].replace("Z", "+00:00")).astimezone(cc.HST)
        rgb = np.asarray(Image.open(BytesIO(cc.get(im["imageUri"]))).convert("RGB"), float)
        if rgb.mean() < 40:
            continue                                   # night
        res = read(rgb)
        print(f"{t:%H:%M}  capped {res['capped']:.0%}  base {res['base_ft']}  " +
              "  ".join(f"{n}: {s['capped']:.0%}/{s['base_ft']}" for n, s in res["segments"].items()))
        annotate(rgb, res, os.path.join(out_dir, f"{t:%H%M}.jpg"))


if __name__ == "__main__":
    if sys.argv[1] == "live":
        live(sys.argv[2])
    else:
        from PIL import Image
        rgb = np.asarray(Image.open(sys.argv[1]).convert("RGB"), float)
        res = read(rgb)
        print(json.dumps({k: v for k, v in res.items() if k != "columns"}, ensure_ascii=False))
        if len(sys.argv) > 2:
            annotate(rgb, res, sys.argv[2])
