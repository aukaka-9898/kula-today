"""Pixel <-> height for the FAA Kaneohe Bay camera looking SW at the Koʻolau
(site 760, camera 12784; Sam, 2026-10-25: "Better for this purpose" -- the
crest base downrange of Makapuʻu -- and "clear air gives you ridge landmarks").

Camera at 21.42245 N, -157.75706, 562 ft (FAA). The geometry -- bearing,
field of view, tilt, roll, lens barrel and the camera's height -- is fitted
by matching the skyline in a clear-air frame (kaneohe_clearday.jpg, 10:03 on
2026-10-25, the whole crest sharp against the sky) to the skyline the terrain
model gives: the highest elevation angle along each bearing. Terrain:
USGS 10 m (OpenTopoData ned10m, 100 points a request), cached in koolaucam_dem.json.

    python tools/koolaucam.py fit      # skyline match, writes koolaucam_geom.json
    python tools/koolaucam.py check    # draws the model skyline on the frame
"""
# Landmarks in the frame. Konahuanui: the highest point, 3,015 ft at bearing
# 204, 7.8 km, x ~545. "Greenwall" (Sam's name, 2026-10-25): the crest east of
# Konahuanui -- left of it in this SW view -- bearings ~185-200, ~2,000-2,300 ft,
# ~10 km, x ~170-470 (the fit is weakest toward the left edge).
import json
import math
import os
import sys
import time

import numpy as np
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DEM = os.path.join(HERE, "koolaucam_dem.json")
GEOM = os.path.join(HERE, "koolaucam_geom.json")
CLEARDAY = os.path.join(HERE, "kaneohe_clearday.jpg")
LAT0, LON0, ELEV_FT = 21.42245, -157.75706, 562
W, H = 1920, 1080
R = 6371000.0
AZ = np.arange(160.0, 290.01, 1.0)
DISTS = np.arange(300, 15001, 300)


def dest(lat, lon, az, d):
    la, lo, a = map(math.radians, (lat, lon, az))
    la2 = math.asin(math.sin(la) * math.cos(d / R) + math.cos(la) * math.sin(d / R) * math.cos(a))
    lo2 = lo + math.atan2(math.sin(a) * math.sin(d / R) * math.cos(la), math.cos(d / R) - math.sin(la) * math.sin(la2))
    return math.degrees(la2), math.degrees(lo2)


def profiles() -> dict:
    dem = json.load(open(DEM)) if os.path.exists(DEM) else {}
    todo = [a for a in AZ if f"{a:.1f}" not in dem]
    for i in range(0, len(todo), 2):                     # two bearings (100 points) a request
        pair = todo[i:i + 2]
        pts = [dest(LAT0, LON0, a, d) for a in pair for d in DISTS]
        for attempt in range(5):
            r = requests.get("https://api.opentopodata.org/v1/ned10m", timeout=60,
                             params={"locations": "|".join(f"{p[0]:.5f},{p[1]:.5f}" for p in pts)})
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        el = [x["elevation"] if x["elevation"] is not None else 0.0 for x in r.json()["results"]]
        for j, a in enumerate(pair):
            dem[f"{a:.1f}"] = el[j * len(DISTS):(j + 1) * len(DISTS)]
        json.dump(dem, open(DEM, "w"))                   # saved as it goes
        time.sleep(1.1)
    return dem


def model_skyline(cam_m: float, dem: dict):
    """bearing -> (elevation angle deg, height ft, distance m) of the skyline."""
    out = {}
    for a in AZ:
        e = np.array(dem[f"{a:.1f}"], dtype=float)
        drop = DISTS ** 2 / (2 * R) * 0.87                      # curvature with refraction
        ang = np.degrees(np.arctan2(e - drop - cam_m, DISTS))
        i = int(np.argmax(ang))
        out[float(a)] = (float(ang[i]), float(e[i]) * 3.28084, float(DISTS[i]))
    return out


def image_skyline(gray, top=600, bottom=760):
    """Per column, the row where the bright sky gives way to the dark range:
    the largest downward brightness drop in the band where the crest sits."""
    cols = np.arange(40, W - 40, 8)
    rows = []
    for x in cols:
        col = gray[top:bottom, x - 3:x + 4].mean(axis=1)
        sm = np.convolve(col, np.ones(5) / 5, mode="same")
        d = sm[:-4] - sm[4:]
        rows.append(top + int(np.argmax(d[3:-3])) + 5)
    return cols, np.array(rows, float)


def project(az, ang, g):
    """(bearing, elevation angle) -> image (x, y) with barrel distortion and roll."""
    f = (W / 2) / math.tan(math.radians(g["hfov"] / 2))
    x = f * np.tan(np.radians(az - g["bearing"]))
    y = -f * np.tan(np.radians(ang - g["tilt"])) / np.cos(np.radians(az - g["bearing"]))
    r2 = (x ** 2 + y ** 2) / (W / 2) ** 2
    s = 1 + g["k1"] * r2
    x, y = x * s, y * s
    c, sn = math.cos(math.radians(g["roll"])), math.sin(math.radians(g["roll"]))
    return W / 2 + c * x - sn * y, H / 2 + sn * x + c * y


def nelder_mead(f, x0, step, iters=600):
    n = len(x0)
    pts = [x0] + [x0 + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]; vals = [vals[i] for i in order]
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    return pts[int(np.argmin(vals))]


def fit():
    from PIL import Image
    gray = np.asarray(Image.open(CLEARDAY).convert("L"), float)
    cols, rows = image_skyline(gray)
    dem = profiles()

    def resid(p):
        g = dict(zip(("bearing", "hfov", "tilt", "roll", "k1"), p[:5]))
        sky = model_skyline(ELEV_FT / 3.28084 + p[5], dem)
        az = np.array(list(sky)); ang = np.array([v[0] for v in sky.values()])
        dense = np.arange(az[0], az[-1], 0.05)
        xs, ys = project(dense, np.interp(dense, az, ang), g)
        order = np.argsort(xs)
        pred = np.interp(cols, xs[order], ys[order], left=np.nan, right=np.nan)
        r = pred - rows
        return np.where(np.isnan(r), 80.0, np.clip(r, -300, 300))

    # Robust cost (soft L1, 6 px) minimised by Nelder-Mead from a few starts;
    # no scipy on this PC, so a small simplex of our own.
    cost = lambda p: float(np.sum(2 * 36 * (np.sqrt(1 + (resid(p) / 6) ** 2) - 1)))
    best = None
    for b0 in (222, 225, 228):
        for h0 in (95, 105):
            # tilt start from the row offset at zero tilt (about 14 px a degree)
            off = float(np.nanmedian(resid(np.array([b0, h0, 0.0, 0.0, -0.1, 0.0]))))
            p = nelder_mead(cost, np.array([b0, h0, -off / 14.0, 0.0, -0.1, 0.0]),
                            np.array([3, 8, 2, 1.5, 0.1, 10]), iters=1500)
            c = cost(p)
            if best is None or c < best[0]:
                best = (c, p)
    p = best[1]
    r = resid(p)
    g = {"bearing": p[0], "hfov": p[1], "tilt": p[2], "roll": p[3], "k1": p[4], "mast_m": p[5],
         "rms_px": float(np.sqrt(np.mean(r ** 2))), "median_abs_px": float(np.median(np.abs(r)))}
    json.dump({k: round(float(v), 4) for k, v in g.items()}, open(GEOM, "w"), indent=1)
    print(json.dumps({k: round(float(v), 3) for k, v in g.items()}))
    return g


def check(out_path):
    from PIL import Image, ImageDraw
    g = json.load(open(GEOM))
    sky = model_skyline(ELEV_FT / 3.28084 + g["mast_m"], profiles())
    im = Image.open(CLEARDAY).convert("RGB")
    dr = ImageDraw.Draw(im)
    pts = [tuple(map(float, project(a, v[0], g))) for a, v in sky.items() if abs(a - g["bearing"]) < 60]
    dr.line([p for p in pts if 0 <= p[0] < W], fill=(255, 0, 0), width=2)
    cols, rows = image_skyline(np.asarray(Image.open(CLEARDAY).convert("L"), float))
    for x, y in zip(cols, rows):
        dr.ellipse([x - 2, y - 2, x + 2, y + 2], outline=(255, 255, 0))
    for a, (ang, ft, d) in list(sky.items())[::10]:
        x, y = project(a, ang, g)
        if 0 <= x < W:
            dr.text((float(x), float(y) - 18), f"{ft:,.0f}", fill=(255, 255, 255))
    im.save(out_path)


if __name__ == "__main__":
    if sys.argv[1:2] == ["fit"]:
        fit()
    elif sys.argv[1:2] == ["check"]:
        check(sys.argv[2])
