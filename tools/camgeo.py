"""Pixel -> terrain elevation for the FAA Kekoa south camera.

Camera at 20.79729 N, -156.31142, 2,819 ft, bearing 180, 90 deg wide over 1920
px (rectilinear assumed, square pixels). Terrain from Open-Meteo's elevation
API (Copernicus 90 m) sampled along each azimuth. Tilt calibrated on Sam's
pasture top (4,600 ft); bearing 183.8 from the same landmark.
"""
import json, math, os, sys
import numpy as np, requests

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "camgeo_dem.json")
LAT0, LON0, H0 = 20.79729, -156.31142, 2819 / 3.281   # m
# Bearing 183.8, not the FAA's listed 180: Sam's pasture top (20.727188,
# -156.312052; ~4,600 ft, terrain model 4,633) sits at pixel ~905, 658, which
# 180 put 60 px too far right (2026-10-18). Tilt still from the 4,900 ft line.
BEARING, HFOV, W, H = 183.8, 90.0, 1920, 1080
F = (W / 2) / math.tan(math.radians(HFOV / 2))         # focal length in px
R = 6371000.0
DISTS = np.arange(300, 40000, 200)


def dest(lat, lon, az, d):
    la, lo, a = map(math.radians, (lat, lon, az))
    la2 = math.asin(math.sin(la) * math.cos(d / R) + math.cos(la) * math.sin(d / R) * math.cos(a))
    lo2 = lo + math.atan2(math.sin(a) * math.sin(d / R) * math.cos(la), math.cos(d / R) - math.sin(la) * math.sin(la2))
    return math.degrees(la2), math.degrees(lo2)


def profile(az):
    dem = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    key = f"{az:.1f}"
    if key not in dem:
        pts = [dest(LAT0, LON0, az, d) for d in DISTS]
        el = []
        for i in range(0, len(pts), 100):
            ch = pts[i:i + 100]
            r = requests.get("https://api.open-meteo.com/v1/elevation", timeout=60,
                             params={"latitude": ",".join(f"{p[0]:.5f}" for p in ch),
                                     "longitude": ",".join(f"{p[1]:.5f}" for p in ch)})
            if r.status_code == 429:
                import time; time.sleep(65)
                r = requests.get("https://api.open-meteo.com/v1/elevation", timeout=60, params={"latitude": ",".join(f"{p[0]:.5f}" for p in ch), "longitude": ",".join(f"{p[1]:.5f}" for p in ch)})
            r.raise_for_status()
            import time; time.sleep(1.5)
            el += r.json()["elevation"]
        dem[key] = el
        json.dump(dem, open(CACHE, "w"))
    return np.array(dem[key], dtype=float)


def az_of(x):
    return BEARING + math.degrees(math.atan((x - W / 2) / F))


def angle_of(y, tilt):
    # Elevation angle of image row y; tilt = elevation angle of the centre row.
    return tilt + math.degrees(math.atan((H / 2 - y) / F))


def terrain_at(x, y, tilt):
    """Elevation (ft) and distance (km) of the terrain point seen at pixel x, y."""
    az = az_of(x)
    el = profile(round(az, 1))
    drop = DISTS ** 2 / (2 * R) * (1 - 0.13)            # curvature with refraction
    ang = np.degrees(np.arctan((el - H0 - drop) / DISTS))
    target = angle_of(y, tilt)
    # The first point along the ray whose terrain angle reaches the target
    # angle and is not hidden behind nearer terrain.
    best = -1e9
    for d, a, h in zip(DISTS, ang, el):
        if a >= target and a >= best - 1e-9:
            return h * 3.281, d / 1000
        best = max(best, a)
    return None, None


def calibrate():
    """Tilt from Sam's pasture top: 20.727188, -156.312052, 4,600 ft, seen at
    pixel ~905, 658 (2026-10-18). Replaces the hand-drawn 4,900 ft line, which
    Sam says was about 300 ft off."""
    lat, lon, el_m, y = 20.727188, -156.312052, 4600 / 3.281, 658
    la1, lo1, la2, lo2 = map(math.radians, (LAT0, LON0, lat, lon))
    d = 2 * R * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2
                                    + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))
    ang = math.degrees(math.atan((el_m - H0 - d ** 2 / (2 * R) * (1 - 0.13)) / d))
    return ang - math.degrees(math.atan((H / 2 - y) / F))


if __name__ == "__main__":
    tilt = calibrate()
    print(f"tilt {tilt:.2f} deg; check red line ends:",
          [round(terrain_at(x, 651, tilt)[0] or 0, -1) for x in (872, 905, 937)])
    for x, y in [tuple(map(int, a.split(","))) for a in sys.argv[1:]]:
        h, d = terrain_at(x, y, tilt)
        print(f"pixel {x},{y}: az {az_of(x):.1f}, angle {angle_of(y, tilt):.2f} -> "
              f"{'none' if h is None else f'{h:,.0f} ft at {d:.1f} km'}")
