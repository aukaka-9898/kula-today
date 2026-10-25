"""Makapuʻu Today: the second site page (Sam, 2026-10-25: "start building
Makapu'u"). Written by the Kula build each run into docs/makapuu/, but its own
module: nothing here touches Kula's logic, and the shared parts get pulled out
once a third site shows what is really shared (multi-site plan).

Makapuʻu is a ridge site with thermal elements; flyability is the wind on the
ridge -- direction, speed and gusts against Weather Guesser's table -- and the
cloudbase only matters downrange along the Koʻolau ("base does not define
flyability, just downrange potential", Sam 10-21).

Wind now: Bellows (BELH1). While the NWS mesonet feed is frozen (since
2026-10-22) the NWS forecast grid's current hour stands in, labelled as such
(Sam, 10-25: "Forecast grid, labelled"). The crest base comes from Kaneohe's
METAR air (PHNG, open windward coast: the inflow, not the dead air at the foot
of the pali -- Sam 10-21), with its ceilometer as the upwind reference.
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

HERE = Path(__file__).resolve().parent
OUT = HERE / "docs" / "makapuu"
HST = ZoneInfo("Pacific/Honolulu")
HEAD = {"User-Agent": "KulaToday (Makapuu page)", "Accept": "application/geo+json"}

LAUNCH_FT = 1200
BELLOWS = "BELH1"                    # Bellows AFS (HADS), 4.9 mi NW: "pretty good" for surface wind (Sam)
GRID = "https://api.weather.gov/gridpoints/HFO/162,145"
METAR = "https://aviationweather.gov/api/data/metar?ids=PHNG&format=json&hours=3"
OBS_MAX_AGE = timedelta(minutes=90)
# Waimanalo (NW), the card's view (Sam). The /map/... address opened on North.
WEBCAM = "https://weathercams.faa.gov/cameras/state/HI/cameraSite/588/details/camera/12153/full"
CAMERA = (588, 12153)

# Weather Guesser's Sites.xml for Makapu'u, mph, wind FROM: (from, to, safe
# max, caution max). 66-70 is caution only. 71 and round: not flyable --
# turbulence off Rabbit Island reaches launch (Sam, 10-12).
SECTORS = ((0, 19, 15, 17), (20, 60, 16, 20), (61, 65, 12, 15), (66, 70, None, 10))
GUST_FACTOR_MAX = 8                  # WG Charts.vb: green needs gust - speed <= 8
LIGHT_MPH = 10                       # Sam: "Put it at 10. On that ridge even that provides some lift"
IDEAL_DIR = 50                       # Sam 10-21: "sort of ideal. but there are those who will say different"
DAY_HOURS = range(8, 19)
# The Koʻolau crest downrange behind Waimānalo and on to Kāneʻohe (550 m grid
# and 10 m checks, 10-12): 2,100-2,800 ft, ~1,200-1,500 at the Makapuʻu end.
CREST_LOW_FT, CREST_HIGH_FT = 2100, 2800

RATING_CLASS = {"Flyable": "good", "Caution": "fair", "Light": "fair", "Not flyable": "warn"}


def compass(d: float) -> str:
    pts = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")
    return pts[int((d % 360) / 22.5 + 0.5) % 16]


def rate(dir_deg, mph, gust=None) -> tuple[str, str]:
    """(word, why) by WG's WindCellColor, plus Sam's light-wind rule."""
    if mph is None:
        return "Unknown", "no wind reading"
    avg = (mph + gust) / 2 if gust is not None else mph
    factor = (gust - mph) if gust is not None else 0
    if dir_deg is None:                                  # variable: WG matches any row
        row = SECTORS[0]
    else:
        d = round(dir_deg) % 360
        row = next((r for r in SECTORS if r[0] <= d <= r[1]), None)
        if row is None:
            if 71 <= d <= 180:
                return "Not flyable", f"from {d:03d}°: east of 070°, turbulence off Rabbit Island reaches launch"
            return "Not flyable", f"from {d:03d}°: not onto the ridge (0–70°)"
    if mph < LIGHT_MPH:
        return "Light", "under 10 mph: run-off launch only, not recommended"
    _, _, safe, caution = row
    if safe is not None and avg <= safe and factor <= GUST_FACTOR_MAX:
        return "Flyable", ""
    if avg <= caution:
        why = (f"gusts {factor:.0f} mph over the wind speed" if safe is not None and avg <= safe else
               "this direction is caution only" if safe is None else
               f"wind and gusts average {avg:.0f} mph, over the {safe} mph safe limit for this direction")
        return "Caution", why
    return "Not flyable", f"wind and gusts average {avg:.0f} mph, over the {caution} mph limit for this direction"


def return_note(dir_deg) -> str:
    if dir_deg is None:
        return ""
    d = dir_deg % 360
    if d > 180:
        return ""
    if abs(d - IDEAL_DIR) <= 10:
        return "Near the ideal 050°: a fair balance between going downrange and getting back."
    if d < IDEAL_DIR:
        return "More north than the ideal 050°: easier to get back from downrange, the primary LZ less ideal."
    return "More east than the ideal 050°: easier going downrange, harder getting back."


def _intervals(series: dict, conv) -> dict:
    """NWS grid values ('2026-10-25T10:00:00+00:00/PT2H') to {HST hour: value}."""
    out = {}
    for v in series.get("values", []):
        start, dur = v["validTime"].split("/")
        t = datetime.fromisoformat(start).astimezone(HST)
        hours = int(re.search(r"(\d+)H", dur).group(1)) if "H" in dur else 0
        hours += 24 * int(re.search(r"(\d+)D", dur).group(1)) if "D" in dur else 0
        for k in range(max(hours, 1)):
            out[(t + timedelta(hours=k)).replace(minute=0, second=0, microsecond=0)] = \
                None if v["value"] is None else conv(v["value"])
    return out


def grid_hours() -> dict:
    p = requests.get(GRID, headers=HEAD, timeout=60).json()["properties"]
    kmh = lambda v: v * 0.621371
    d, s, g, sky = (_intervals(p["windDirection"], float), _intervals(p["windSpeed"], kmh),
                    _intervals(p["windGust"], kmh), _intervals(p["skyCover"], float))
    return {t: {"dir": d.get(t), "mph": s.get(t), "gust": g.get(t), "sky": sky.get(t)} for t in sorted(s)}


# Bellows also reaches NOAA's HADS (its GOES relay) directly, and HADS kept
# it when the NWS API froze (Sam noticed Weather Guesser still reading it,
# 2026-10-25). SHEF codes, English units, UTC: US wind mph, UD direction
# degrees, UP peak wind mph, TA air F. No dew point. Posts in GOES batches,
# so the newest value can be most of an hour old.
HADS_BELLOWS = ("https://hads.ncep.noaa.gov/nexhads2/servlet/DecodedData"
                "?sinceday=-1&hsa=nil&state=nil&nesdis_ids=B14036D2&of=1")


def bellows_hads() -> dict | None:
    vals: dict = {}
    # HADS sometimes answers GitHub with an empty page (12:22 10-25): three tries.
    for attempt in range(3):
        r = requests.get(HADS_BELLOWS, timeout=40, headers={"User-Agent": "KulaToday (Makapuu page)"})
        if r.ok and BELLOWS in r.text:
            break
        import time
        time.sleep(5 * (attempt + 1))
    else:
        raise ValueError(f"HADS answered without Bellows: {r.text[:80]!r}")
    for line in r.text.splitlines():
        f = line.split("|")
        if len(f) > 4 and f[1] == BELLOWS and f[2] in ("US", "UD", "UP", "TA"):
            try:
                t = datetime.strptime(f[3], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).astimezone(HST)
                vals.setdefault(t, {})[f[2]] = float(f[4])
            except ValueError:
                continue
    full = [t for t, v in vals.items() if "US" in v]
    if not full:
        return None
    t = max(full)
    v = vals[t]
    return {"dir": v.get("UD"), "mph": v["US"], "gust": v.get("UP"), "temp_f": v.get("TA"), "at": t}


BELLOWS_PROBLEMS: list = []
BELLOWS_LAST = OUT / "bellows_last.json"


def bellows_now(now: datetime):
    """(reading, time of the newest one): NWS API first, then HADS."""
    t = None
    try:
        r = requests.get(f"https://api.weather.gov/stations/{BELLOWS}/observations/latest", headers=HEAD, timeout=30)
        p = r.json()["properties"]
        t = datetime.fromisoformat(p["timestamp"]).astimezone(HST)
        if now - t <= OBS_MAX_AGE and p["windSpeed"]["value"] is not None:
            mph = p["windSpeed"]["value"] * 0.621371
            gust = None if p["windGust"]["value"] is None else p["windGust"]["value"] * 0.621371
            return {"dir": p["windDirection"]["value"], "mph": mph, "gust": gust, "at": t, "via": "NWS"}, t
    except Exception:
        pass
    h = None
    try:
        h = bellows_hads()
        if h:
            BELLOWS_LAST.write_text(json.dumps({**h, "at": h["at"].isoformat()}), encoding="utf-8")
    except Exception as e:
        BELLOWS_PROBLEMS.append(f"HADS Bellows: {type(e).__name__}: {e}")
        try:                                             # the last good HADS reading
            c = json.loads(BELLOWS_LAST.read_text(encoding="utf-8"))
            h = {**c, "at": datetime.fromisoformat(c["at"])}
        except Exception:
            pass
    if h and now - h["at"] <= OBS_MAX_AGE:
        return {**h, "via": "HADS"}, h["at"]
    return None, max(x for x in (t, h["at"] if h else None) if x) if (t or h) else None


def kaneohe() -> dict | None:
    rows = requests.get(METAR, timeout=30).json()
    if not rows:
        return None
    m = max(rows, key=lambda r: r["obsTime"])
    return {"t_c": m.get("temp"), "td_c": m.get("dewp"), "clouds": m.get("clouds") or [], "cover": m.get("cover"),
            "at": datetime.fromtimestamp(m["obsTime"], timezone.utc).astimezone(HST)}


# One more HADS station for the air (Sam, 2026-10-25): Heʻeia NERR on the
# shore of Kāneʻohe Bay -- "useful for dewpoint and temp. useless for wind",
# so its wind is neither used nor shown. The Oʻahu Forest NWR RAWS (OFRH1,
# ~1,550 ft above Kahaluʻu) was tried and dropped the same day: "behind the
# ridge. Not usable" (it read the base 3,500 ft). (nesdis, name, elevation ft)
HADS_AIR = {"HRRH1": ("3B052656", "Heʻeia (Kāneʻohe Bay shore)", 0)}


def hads_air(nwsli: str) -> list:
    """[(time HST, T F, RH %)] for the last day, oldest first."""
    nesdis = HADS_AIR[nwsli][0]
    vals: dict = {}
    for line in requests.get(HADS_BELLOWS.replace("B14036D2", nesdis), timeout=40).text.splitlines():
        f = line.split("|")
        if len(f) > 4 and f[1] == nwsli and f[2] in ("TA", "XR"):
            try:
                t = datetime.strptime(f[3], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).astimezone(HST)
                vals.setdefault(t, {})[f[2]] = float(f[4])
            except ValueError:
                continue
    return [(t, v["TA"], v["XR"]) for t, v in sorted(vals.items()) if "TA" in v and 0 < v.get("XR", 0) <= 100]


def dew_f(t_f: float, rh: float) -> float:
    import math
    tc = (t_f - 32) / 1.8
    g = math.log(rh / 100) + 17.625 * tc / (243.04 + tc)
    return 243.04 * g / (17.625 - g) * 1.8 + 32


def lcl_msl(t_f: float, td_f: float, elev_ft: float) -> int:
    return int(round((t_f - td_f) / 1.8 * 125 * 3.28084 + elev_ft, -2))


# The Koʻolau seen from the Kāneʻohe Bay camera (FAA site 760, SW view;
# tools/koolaubase.py). Cloud ON the crest is a solid reading; a cloud bottom
# above the crest is placed at the crest's distance, and cumulus drift in front
# of the range, so it is an upper limit. Sam's wording, 10-25: "Clouds are
# above the ridge and below about 3,300".
CAP_SHARE = 0.3                # a stretch counts as capped with this share of its columns
CAMERA_MAX_AGE = timedelta(minutes=40)


def crest_camera(now: datetime) -> dict | None:
    import sys
    from io import BytesIO
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(HERE / "tools"))
    import koolaubase
    j = requests.get("https://weathercams.faa.gov/api/summary", params={"siteId": 760}, timeout=30,
                     headers={"Referer": "https://weathercams.faa.gov/cameras/state/HI/cameraSite/760/summary"}).json()
    cam = next(c for c in j["payload"]["site"]["cameras"] if c["cameraId"] == 12784)
    im = max(cam["currentImages"], key=lambda x: x["imageDatetime"])
    t = datetime.fromisoformat(im["imageDatetime"].replace("Z", "+00:00")).astimezone(HST)
    if now - t > CAMERA_MAX_AGE:
        return None
    rgb = np.asarray(Image.open(BytesIO(requests.get(im["imageUri"], timeout=30).content)).convert("RGB"), float)
    if rgb.mean() < 60:
        return None                                    # too dark to read
    return {**koolaubase.read(rgb), "at": t}


def cam_text(cam: dict) -> str:
    capped = [n for n, s in cam["segments"].items() if s["capped"] >= CAP_SHARE]
    if capped:
        return "Cloud on the Koʻolau: " + ", ".join(capped) + "."
    if cam["base_ft"]:
        return f"Clouds are above the ridge and below about {round(cam['base_ft'], -2):,.0f} ft."
    return "The ridge is clear, with no cloud base in view above it."


def crest_base(k: dict) -> int | None:
    """Condensation level of Kaneohe's open-coast air: where the trades
    lifted over the Koʻolau condense. 125 m per degree C of spread."""
    if not k or k["t_c"] is None or k["td_c"] is None:
        return None
    return int(round((k["t_c"] - k["td_c"]) * 125 * 3.28084, -2))


def esc(s) -> str:
    return html.escape(str(s))


def chip(word: str) -> str:
    return f'<span class="chip {RATING_CLASS.get(word, "fair")}">{esc(word)}</span>'


def wind_text(w) -> str:
    if not w or w.get("mph") is None:
        return "–"
    d = "variable" if w.get("dir") is None else f'{compass(w["dir"])} ({w["dir"]:03.0f}°)'
    g = f', gusts {w["gust"]:.0f}' if w.get("gust") is not None else ""
    return f'{d} {w["mph"]:.0f} mph{g}'


def spans(hours: list) -> str:
    """'09:00–15:00' runs of flyable (or caution) hours."""
    runs, start, prev = [], None, None
    for h in hours:
        if start is None:
            start = prev = h
        elif h == prev + 1:
            prev = h
        else:
            runs.append((start, prev)); start = prev = h
    if start is not None:
        runs.append((start, prev))
    return ", ".join(f"{a:02d}:00–{b + 1:02d}:00" for a, b in runs)


def day_line(rows: list) -> tuple[str, str]:
    """(best word, sentence) for a day's daylight hours."""
    fly = [t.hour for t, w in rows if w == "Flyable"]
    cau = [t.hour for t, w in rows if w == "Caution"]
    light = [t.hour for t, w in rows if w == "Light"]
    if fly:
        return "Flyable", f"Flyable {spans(fly)}" + (f"; caution {spans(cau)}" if cau else "") + "."
    if cau:
        return "Caution", f"Caution {spans(cau)}." + (f" Light {spans(light)}." if light else "")
    if light:
        return "Light", f"Light {spans(light)}: run-off launch only."
    return "Not flyable", "Not flyable in daylight."


def short_wind(w) -> str:
    """'NNE 15 g21' for the hour tables, narrow enough for a phone."""
    if not w or w.get("mph") is None:
        return "–"
    d = "var" if w.get("dir") is None else compass(w["dir"])
    return f'{d} {w["mph"]:.0f}' + (f' g{w["gust"]:.0f}' if w.get("gust") is not None else "")


def hour_table(rows) -> str:
    out = ""
    for t, w in rows:
        out += (f'<tr><td>{t:%H}:00</td><td class="n">{esc(short_wind(w))}</td>'
                f'<td>{chip(rate(w["dir"], w["mph"], w["gust"])[0])}</td></tr>')
    return (f'<div class="tablewrap"><table class="hist"><thead><tr><th>Hour</th><th class="n">Wind, mph</th>'
            f'<th>Ridge</th></tr></thead><tbody>{out}</tbody></table></div>')


def save_camera() -> str | None:
    from io import BytesIO
    from PIL import Image
    site, cam_id = CAMERA
    j = requests.get("https://weathercams.faa.gov/api/summary", params={"siteId": site}, timeout=30,
                     headers={"Referer": f"https://weathercams.faa.gov/cameras/state/HI/cameraSite/{site}/summary"}).json()
    cam = next(c for c in j["payload"]["site"]["cameras"] if c["cameraId"] == cam_id)
    im = max(cam["currentImages"], key=lambda x: x["imageDatetime"])
    pic = Image.open(BytesIO(requests.get(im["imageUri"], timeout=30).content)).convert("RGB")
    pic.thumbnail((800, 450))
    pic.save(OUT / "camera.jpg", quality=80)
    when = datetime.fromisoformat(im["imageDatetime"].replace("Z", "+00:00")).astimezone(HST)
    return f"camera.jpg?{when:%Y%m%d%H%M}"


# The air aloft (Sam, 10-25: yes to RRFS): the RRFS BUFKIT point at Kaneohe
# (phng, windward, Penn State, ~3.5 days). Thermals by thermal.py as for Kula:
# a parcel of SURFACE air lifted dry through the sounding, Weather Guesser's
# words from the peak index. The parcel is Open-Meteo's surface forecast at
# Kaneohe, as Kula's is Open-Meteo at Kula Ag -- not the sounding's own lowest
# level, which is the air it is compared with (index 0 by construction; the
# first run read Flat every day). The same surface air gives the forecast
# crest base, the way the measured one uses Kaneohe's METAR. The inversion
# is the sounding's (sounding.inversion), at 14:00.
THERMAL_HOURS = (10, 14)
# A base under launch is not a number to fly by: on rainy days Open-Meteo's
# near-saturated surface air gave 200-400 ft (Sam, 10-25: "Need to gate
# absurd base calls"). Shown as "Low" instead.
BASE_FLOOR_FT = LAUNCH_FT


def base_label(ft) -> str:
    if ft is None:
        return "–"
    return "Low" if ft < BASE_FLOOR_FT else f"{ft:,}"

INVERSION_BAND = (1000, 12000)
SURFACE_POINT = (21.45, -157.763)    # Kaneohe (PHNG): open windward coast, the inflow
RELEASE_FT = 20


# Open-Meteo timed out on the 11:44 build (10-25) and the Details page lost
# every forecast hour (Sam: "Makapu'u soundings are blank"). So: three tries,
# and the last good answer kept beside the page as the fallback.
SURFACE_CACHE = OUT / "surface_cache.json"


def _surface_fetch() -> dict:
    last = None
    for attempt in range(3):
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=30, params={
                "latitude": SURFACE_POINT[0], "longitude": SURFACE_POINT[1],
                "hourly": "temperature_2m,dew_point_2m", "temperature_unit": "fahrenheit",
                "timezone": "Pacific/Honolulu", "forecast_days": 5})
            return r.json()["hourly"]
        except Exception as e:
            last = e
    raise last


def surface_hours() -> dict:
    try:
        h = _surface_fetch()
        SURFACE_CACHE.write_text(json.dumps(h), encoding="utf-8")
    except Exception:
        h = json.loads(SURFACE_CACHE.read_text(encoding="utf-8"))   # the last good forecast
    return {datetime.fromisoformat(t).replace(tzinfo=HST): (a, b)
            for t, a, b in zip(h["time"], h["temperature_2m"], h["dew_point_2m"]) if a is not None and b is not None}


def rrfs() -> dict:
    import sounding as sd
    path = sd.fetch("RRFS", station="phng")
    return {"run": sd.run_time(path), "profiles": sd.profiles(path, "RRFS")}


def _profile(air: dict, when: datetime):
    import pandas as pd
    t = pd.Timestamp(when).tz_convert("UTC")
    near = min(air["profiles"], key=lambda k: abs((k - t).total_seconds()))
    return air["profiles"][near] if abs((near - t).total_seconds()) <= 5400 else None


def air_day(air: dict, surf: dict, day) -> dict | None:
    """{word, ti, usable_ft, base_ft, inv} for a day: the RRFS profiles with
    Open-Meteo's surface air at Kaneohe as the parcel."""
    import numpy as np
    import sounding as sd
    import thermal as th
    at = lambda h: datetime.combine(day, datetime.min.time()).replace(hour=h, tzinfo=HST)
    peaks, usable = [], []
    for h in range(THERMAL_HOURS[0], THERMAL_HOURS[1] + 1):
        prof, sa = _profile(air, at(h)), surf.get(at(h))
        if prof is None or sa is None:
            continue
        c = th.parcel_curve(sa[0], sa[1], RELEASE_FT, prof, top_ft=12000)
        top = th.thermal_top(c, RELEASE_FT)
        work = c[c.height_ft <= top]
        peaks.append(float(work.ti_c.min()))
        soar = work[work.ti_c <= th.TI_BANDS[0][0]]
        usable.append(float(soar.height_ft.iloc[-1]) if len(soar) else RELEASE_FT)
    if not peaks:
        return None
    aft, noon = _profile(air, at(14)), surf.get(at(12))
    ti = float(np.mean(peaks))
    return {"word": th.describe(ti), "ti": ti, "usable_ft": float(np.median(usable)),
            "base_ft": None if noon is None else int(round((noon[0] - noon[1]) / 1.8 * 125 * 3.28084, -2)),
            "inv": None if aft is None else sd.inversion(aft, floor_ft=INVERSION_BAND[0], ceiling_ft=INVERSION_BAND[1])}


def inv_text(inv) -> str:
    if not inv:
        return f"None between {INVERSION_BAND[0]:,} and {INVERSION_BAND[1]:,} ft"
    return f'About {round(inv["base_ft"], -2):,.0f} ft (+{inv["warming_f"]:.1f} °F)'


def _try(fn, *a, problems: list):
    try:
        return fn(*a)
    except Exception as e:
        problems.append(f"{fn.__name__}: {type(e).__name__}: {e}")
        return None


def write(now: datetime) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    problems: list = []
    grid = _try(grid_hours, problems=problems) or {}
    got = _try(bellows_now, now, problems=problems)
    problems += BELLOWS_PROBLEMS
    obs, obs_t = got if got else (None, None)
    k = _try(kaneohe, problems=problems)
    air = _try(rrfs, problems=problems)
    surf = _try(surface_hours, problems=problems) or {}
    hour_now = now.replace(minute=0, second=0, microsecond=0)

    # --- Today: the ridge now
    if obs:
        w, src = obs, f'Bellows, {obs["at"]:%H:%M}' + (" (via NOAA HADS)" if obs.get("via") == "HADS" else "")
    elif grid.get(hour_now):
        w = grid[hour_now]
        # Say what actually failed: Bellows itself, or only the feeds this update.
        src = ("NWS forecast for this hour: no Bellows reading came through this update"
               if obs_t and now - obs_t <= timedelta(hours=6) or BELLOWS_PROBLEMS else
               "NWS forecast for this hour: Bellows not reporting" + (f" since {obs_t:%d %b %H:%M}" if obs_t else ""))
    else:
        w, src = None, "No wind available"
    word, why = rate(w["dir"], w["mph"], w["gust"]) if w else ("Unknown", "")
    night = not (6 <= now.hour < 19)
    today_chip = chip(word) if w and not night else ""
    fly_html = (f"<strong>{esc(word)}</strong>" + (f'<div class="sub">{esc(why)}</div>' if why else "")) if w else "Unavailable"
    wind_html = (f'<span class="num">{esc(wind_text(w))}</span><div class="sub">{esc(src)}</div>') if w else "Unavailable"
    dir_html = esc(return_note(w["dir"])) if w and w.get("dir") is not None else ""

    # The base on the range from the air: Kaneohe's METAR. Heʻeia's shore air
    # read ~2,100 ft while the camera showed the crest clear with bases above
    # 3,000 (10-25; Sam: "Maybe we should drop Heeia") -- dropped here and from
    # the Details surface air; it stays on the station map.
    airs = []
    for code in ():
        got = _try(hads_air, code, problems=problems) or []
        if got and now - got[-1][0] <= OBS_MAX_AGE + timedelta(minutes=30):
            t, tf, rh = got[-1]
            airs.append((HADS_AIR[code][1], lcl_msl(tf, dew_f(tf, rh), HADS_AIR[code][2]), t))
    kb = crest_base(k)
    if kb is not None:
        airs.append(("Kaneohe (PHNG)", kb, k["at"]))
    base = airs[0][1] if airs else None
    gated = base is not None and base < BASE_FLOOR_FT
    if gated:
        base = None                                      # gated: see BASE_FLOOR_FT
    cam = _try(crest_camera, now, problems=problems)
    cam_line = cam_text(cam) if cam else ""
    if cam_line:
        base_html = (f'{esc(cam_line)}'
                     f'<div class="sub">Kāneʻohe Bay camera, looking at the Koʻolau, {cam["at"]:%H:%M}.'
                     + (f" Kaneohe's air would condense at about {base:,} ft." if base is not None else "")
                     + "</div>")
    elif base is not None:
        if base < CREST_LOW_FT:
            cap = f"below the crest behind Waimānalo ({CREST_LOW_FT:,}–{CREST_HIGH_FT:,} ft): cloud on the Koʻolau likely"
        elif base < CREST_HIGH_FT:
            cap = "cloud likely on the higher crest downrange"
        else:
            cap = "above the crest downrange"
        layers = ", ".join(f'{c["cover"]} {c["base"]:,} ft' for c in k["clouds"] if c.get("base")) if k else ""
        ceil = layers or ("clear below 12,000 ft" if k and (k.get("cover") or "").startswith(("CLR", "SKC")) else "")
        each = "; ".join(f"{n} {b:,} ft ({t:%H:%M})" for n, b, t in airs)
        base_html = (f'<span class="num">About {base:,} ft</span>, {esc(cap)}'
                     f'<div class="sub">Condensation level of the air at {esc(each)}.'
                     + (f" Kaneohe's ceilometer: {esc(ceil)}." if ceil else "") + "</div>")
    elif gated:
        base_html = ('<strong>Low</strong><div class="sub">Kaneohe\'s air would condense under launch height: '
                     'wet air at the coast.</div>')
    else:
        base_html = "Unavailable"

    today_air = _try(air_day, air, surf, now.date(), problems=problems) if air else None
    if today_air:
        therm_html = (f'Probably <strong>{esc(today_air["word"].lower())}</strong> (forecast)'
                      f'<div class="sub">RRFS at Kaneohe, run {air["run"]:%d %H}Z: thermal index '
                      f'{today_air["ti"]:.1f} °C, 10:00–14:00 (surface air: Open-Meteo at Kaneohe); '
                      + (f'useable lift to about {round(today_air["usable_ft"], -2):,.0f} ft. '
                         if today_air["usable_ft"] > RELEASE_FT + 300 else "no useable lift. ")
                      + 'Useful, not critical, at Makapuʻu.</div>')
        inv_html = f'{esc(inv_text(today_air["inv"]))}<div class="sub">RRFS at Kaneohe, 14:00: the trade lid</div>'
    else:
        therm_html = inv_html = "Unavailable"

    rest = [(t, grid[t]) for t in sorted(grid) if t.date() == now.date() and t.hour in DAY_HOURS and t > hour_now]
    rest_html = hour_table(rest) if rest else '<span class="sub">Nothing left of today\'s daylight.</span>'

    # --- Forecast: tomorrow, then the week
    tom = now.date() + timedelta(days=1)
    trows = [(t, grid[t]) for t in sorted(grid) if t.date() == tom and t.hour in DAY_HOURS]
    tword, tline = day_line([(t, rate(v["dir"], v["mph"], v["gust"])[0]) for t, v in trows]) if trows else ("", "")
    noon = grid.get(datetime.combine(tom, datetime.min.time()).replace(hour=12, tzinfo=HST))
    ta = _try(air_day, air, surf, tom, problems=problems) if air else None
    aloft = ""
    if ta:
        aloft = ((f' Downrange base low (under launch)' if ta["base_ft"] < BASE_FLOOR_FT else
                  f' Downrange base about {ta["base_ft"]:,} ft') if ta["base_ft"] is not None else " ") +                 f', thermals {ta["word"].lower()}, inversion {inv_text(ta["inv"])[0].lower() + inv_text(ta["inv"])[1:]} (RRFS).'
    tom_html = (f'<p class="summary">{esc(tline)} Midday: {esc(wind_text(noon))}. '
                f'{esc(return_note(noon["dir"]) if noon and noon.get("dir") is not None else "")}'
                f'{esc(aloft)}</p>'
                + hour_table(trows)) if trows else "<p>No forecast available.</p>"

    week = ""
    for k_day in range(1, 7):
        d = now.date() + timedelta(days=k_day)
        rows = [(t, grid[t]) for t in sorted(grid) if t.date() == d and t.hour in DAY_HOURS]
        if not rows:
            continue
        bw, line = day_line([(t, rate(v["dir"], v["mph"], v["gust"])[0]) for t, v in rows])
        mid = grid.get(datetime.combine(d, datetime.min.time()).replace(hour=12, tzinfo=HST))
        da = _try(air_day, air, surf, d, problems=problems) if air else None
        week += (f"<tr><td>{d:%a %d}</td><td>{esc(wind_text(mid))}</td><td>{chip(bw)}</td>"
                 f"<td>{esc(line)}</td>"
                 f'<td class="n">{base_label(da["base_ft"]) if da else "–"}</td>'
                 f'<td>{"–" if not da else esc(da["word"])}</td></tr>')

    summary = (f"Now: {word.lower() if w else 'no wind reading'}"
               + (f", {wind_text(w)}" if w else "") + (" (forecast, Bellows out)." if w and not obs else ".")
               + (f" Tomorrow: {tline}" if tline else ""))

    raw = (HERE / "template.html").read_text(encoding="utf-8")
    style = "".join("<style>" + part.split("</style>")[0] + "</style>" for part in raw.split("<style>")[1:])
    built = f"{now:%Y-%m-%dT%H:%M}"
    page = PAGE.format(style=style, updated=f"{now:%a %d %b %Y · %H:%M} HST", webcam=WEBCAM,
                       kind=f"Now · {now:%H:%M} HST", chip=today_chip, fly=fly_html, wind=wind_html,
                       dirnote="",
                       base=base_html + (f'<div class="downret">{dir_html}</div>' if dir_html else ""), therm=therm_html, inv=inv_html, rest=rest_html, tom_title=f"Tomorrow · {tom:%a %d %b}",
                       tom_chip=chip(tword) if tword else "", tomorrow=tom_html, week=week, built=built,
                       diag=esc("; ".join(problems) or "none").replace("--", "-"))
    # IBM Plex has no ʻokina (U+02BB); its fallback glyph stands apart from the
    # letters, so the visible text uses the left quote instead.
    page = page.replace("ʻ", "&#8216;")
    (OUT / "index.html").write_text(page, encoding="utf-8")
    _try(write_station_map, now, k, problems=problems)
    _try(write_details, now, problems=problems)
    (OUT / "details.html").write_text(details_page(now, built), encoding="utf-8")
    camera = _try(save_camera, problems=problems)
    (OUT / "summary.json").write_text(json.dumps({
        "site": "Makapuu", "title": "Makapu‘u Today", "page": "kula-today/makapuu/", "built": built,
        "updated": f"{now:%a %d %b %Y · %H:%M} HST", "kind": f"Now · {now:%H:%M} HST",
        "chip": {"class": RATING_CLASS.get(word, "fair"), "word": word} if w and not night else None,
        "summary": summary, "limited": not obs, "camera": camera}), encoding="utf-8")


# ---------------------------------------------------------------- Details page
# Like Kula's (Sam, 2026-10-25: "detail page for Makapuu"): the thermal
# conditions section -- the same page code as Kula Details, fed by this site's
# details.json -- and a station map. RRFS and GFS at Kaneohe (phng), and the
# Lihue balloon, which Weather Guesser uses for Oʻahu sites. The viewer picks
# the temperature; the dew point is Kaneohe's: measured for hours gone,
# Open-Meteo's forecast for hours ahead.
DETAILS_HOURS = range(8, 17)
DETAILS_MAX_AGE = timedelta(minutes=60)
LIHUE_RAOB = ("https://weather.uwyo.edu/wsgi/sounding?datetime={t:%Y-%m-%d %H}:00:00"
              "&id=91165&type=TEXT:CSV&src=BUFR")
# The map's stations: (feed, id, name, lat, lon, elev ft). The NWS mesonet
# ones (Bellows and the HECO stations) are frozen since 2026-10-22.
MAP_STATIONS = [("bellows", "BELH1", "Bellows", 21.3667, -157.7167, 39),
                ("nws", "065HE", "HECO Mokuhano", 21.2963, -157.66798, 60),
                ("nws", "067HE", "HECO Nonokio", 21.33861, -157.71167, 42),
                ("nws", "134HE", "HECO Kuluanui Rd", 21.30071, -157.70012, 580),
                ("metar", "PHNG", "Kaneohe (PHNG)", 21.45389, -157.76556, 16),
                ("hads-air", "HRRH1", "Heʻeia NERR", 21.4314, -157.8153, 0),
                ("hcdp", "0531", "Kaluanui Ridge", 21.30491, -157.70302, 784),
                ("hcdp", "0533", "Maunawili", 21.36827, -157.76272, 46)]


def _levels(prof, top_ft=20000) -> list:
    p = prof[prof.height_ft <= top_ft].sort_values("height_ft")
    return [[round(float(r.height_ft)), round(float(r.pres), 1), round(float(r.temp_c), 2), round(float(r.dewpoint_c), 2),
             None if r.wind_dir != r.wind_dir else round(float(r.wind_dir)),
             None if r.wind_kt != r.wind_kt else round(float(r.wind_kt))]
            for r in p.itertuples() if r.dewpoint_c > -100]


def lihue_raob(now: datetime):
    """The newest 00Z Lihue ascent, every 100 ft: (UTC time, levels)."""
    from io import StringIO
    import numpy as np
    import pandas as pd
    day = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    for back in range(3):
        t = day - timedelta(days=back)
        r = requests.get(LIHUE_RAOB.format(t=t), timeout=60)
        if r.status_code != 200 or not r.text.startswith("time,"):
            continue
        d = pd.read_csv(StringIO(r.text), skipinitialspace=True)
        d.columns = ["time", "lon", "lat", "pres", "z_m", "temp_c", "dewpoint_c", "ice", "rh", "rhi",
                     "w", "wind_dir", "wind_ms"][:len(d.columns)]
        d = d.dropna(subset=["z_m", "temp_c", "dewpoint_c"]).sort_values("z_m").drop_duplicates("z_m")
        if len(d) < 50:
            continue
        z = d.z_m.to_numpy() * 3.28084
        hts = np.arange(0, min(20000, z.max()) + 1, 100.0)
        idx = np.clip(np.searchsorted(z, hts), 0, len(z) - 1)
        prof = pd.DataFrame({"height_ft": hts,
                             **{c: np.interp(hts, z, d[c].to_numpy()) for c in ("pres", "temp_c", "dewpoint_c")},
                             "wind_dir": d.wind_dir.to_numpy()[idx], "wind_kt": d.wind_ms.to_numpy()[idx] * 1.94384})
        return t, _levels(prof)
    return None, []


def metar_hours(hours: int = 40) -> dict:
    """Kaneohe's measured air by HST hour: {hour: (T F, Td F)}, hourly means."""
    rows = requests.get(METAR.replace("hours=3", f"hours={hours}"), timeout=30).json()
    out: dict = {}
    for m in rows:
        if m.get("temp") is None or m.get("dewp") is None:
            continue
        t = datetime.fromtimestamp(m["obsTime"], timezone.utc).astimezone(HST)
        out.setdefault(t.replace(minute=0, second=0, microsecond=0), []).append(
            (m["temp"] * 9 / 5 + 32, m["dewp"] * 9 / 5 + 32))
    return {k: (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v)) for k, v in out.items()}


def write_details(now: datetime) -> None:
    import pandas as pd
    import sounding as sd
    import thermal as th
    f = OUT / "details.json"
    try:
        old = json.loads(f.read_text(encoding="utf-8"))
        gaps = any("temp_f" not in v for k, v in old["surface"].items() if k >= f"{now:%Y-%m-%dT%H}")
        if now - datetime.fromisoformat(old["made"]) < DETAILS_MAX_AGE and not gaps:
            return                                       # fresh and complete; a gap refetches next build
    except Exception:
        pass
    days = [now.date(), now.date() + timedelta(days=1)]
    wanted = [datetime.combine(d, datetime.min.time()).replace(hour=h, tzinfo=HST) for d in days for h in DETAILS_HOURS]
    out = {"made": now.isoformat(), "release_ft": RELEASE_FT, "launch_ft": LAUNCH_FT, "top_ft": 18000,
           "chart_top": 10000, "surface_label": "Kaneohe", "raob_label": "Lihue balloon",
           "bands": [[None if lim == float("-inf") else lim, w] for lim, w in th.TI_BANDS],
           "sources": {}, "surface": {}}
    for model in ("RRFS", "GFS"):
        try:
            path = sd.fetch(model, station="phng")
            got = sd.profiles(path, model)
        except Exception:
            continue
        hours = {}
        for w in wanted:
            t = pd.Timestamp(w).tz_convert("UTC")
            near = min(got, key=lambda k: abs((k - t).total_seconds()))
            if abs((near - t).total_seconds()) <= 5400:
                hours[f"{w:%Y-%m-%dT%H}"] = _levels(got[near])
        out["sources"][model] = {"run": f"{sd.run_time(path):%Y-%m-%d %H}Z", "where": "Kaneohe (PHNG) BUFKIT",
                                 "hours": hours}
    try:
        t, lv = lihue_raob(now)
        if lv:
            out["sources"]["RAOB"] = {"run": f"{t:%Y-%m-%d %H}Z", "where": "Lihue (LIH) balloon, 91165", "levels": lv}
    except Exception:
        pass
    try:
        meas = metar_hours()
    except Exception:
        meas = {}
    meas_from = {h: "Kaneohe, measured" for h in meas}
    try:
        fc = surface_hours()
    except Exception:
        fc = {}
    for w in wanted:
        row = {}
        if w + timedelta(hours=1) <= now and w in meas:
            row = {"temp_f": round(meas[w][0], 1), "dew_f": round(meas[w][1], 1), "from": meas_from[w]}
        elif w in fc:
            row = {"temp_f": round(fc[w][0], 1), "dew_f": round(fc[w][1], 1), "from": "Kaneohe forecast (Open-Meteo)"}
        y = w - timedelta(days=1)
        if w.date() == now.date() and y in meas:
            row["yesterday_f"] = round(meas[y][0], 1)
        out["surface"][f"{w:%Y-%m-%dT%H}"] = row
    f.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")


def _lcl_ft(t_f, td_f) -> float:
    return round((t_f - td_f) / 1.8 * 125 * 3.28084, -1)


def _hcdp_latest(now: datetime) -> dict:
    """{station id: (time, {variable: value})} for the map's Mesonet stations."""
    import mesonet
    got = mesonet.get("/mesonet/db/measurements", location="hawaii", station_ids="0531,0533",
                      var_ids="Tair_1_Avg,RH_1_Avg,WS_1_Avg,WS_1_Max,WDrs_1_Avg",
                      start_date=(now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S-10:00"),
                      row_mode="json", limit=5000)
    by: dict = {}
    for r in got:
        if r.get("value") is None:
            continue
        t = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).astimezone(HST)
        by.setdefault(r["station_id"], {}).setdefault(t, {})[r["variable"]] = float(r["value"])
    return {sid: (max(v), v[max(v)]) for sid, v in by.items()}


def write_station_map(now: datetime, k: dict | None) -> None:
    import math
    try:
        hcdp = _hcdp_latest(now)
    except Exception:
        hcdp = {}
    try:
        metar = max(requests.get(METAR, timeout=30).json(), key=lambda r: r["obsTime"])
    except Exception:
        metar = None

    def air(row, tf, df, t):
        row.update(temp_f=round(tf, 1), dew_f=round(df, 1), lcl_ft=_lcl_ft(tf, df) + row["elev_ft"],   # above sea level
                   air_at=f"{t:%H:%M}")

    rows = []
    for feed, sid, name, lat, lon, elev in MAP_STATIONS:
        row = {"name": name, "lat": lat, "lon": lon, "elev_ft": elev, "dir": None, "mph": None, "at": None}
        try:
            if feed == "nws":
                p = requests.get(f"https://api.weather.gov/stations/{sid}/observations/latest",
                                 headers=HEAD, timeout=30).json()["properties"]
                t = datetime.fromisoformat(p["timestamp"]).astimezone(HST)
                v = lambda key: p[key]["value"]
                if now - t > OBS_MAX_AGE:
                    row["last"] = f"{t:%d %b %H:%M}"
                else:
                    if v("windSpeed") is not None:
                        row.update(dir=v("windDirection"), mph=round(v("windSpeed") * 0.621371), at=f"{t:%H:%M}")
                        if v("windGust") is not None:
                            row["gust"] = round(v("windGust") * 0.621371)
                    if v("temperature") is not None and v("dewpoint") is not None:
                        air(row, v("temperature") * 1.8 + 32, v("dewpoint") * 1.8 + 32, t)
            elif feed == "bellows":
                b, last = bellows_now(now)
                if b:
                    row.update(dir=None if b["dir"] is None else round(b["dir"]), mph=round(b["mph"]), at=f'{b["at"]:%H:%M}')
                    if b.get("gust") is not None:
                        row["gust"] = round(b["gust"])
                    if b.get("temp_f") is not None:
                        row["temp_f_only"] = round(b["temp_f"], 1)
                elif last:
                    row["last"] = f"{last:%d %b %H:%M}"
            elif feed == "hads-air":
                got = hads_air(sid)
                if got:
                    t, tf, rh = got[-1]
                    air(row, tf, dew_f(tf, rh), t)
                    row["no_wind"] = True
            elif feed == "metar" and metar:
                t = datetime.fromtimestamp(metar["obsTime"], timezone.utc).astimezone(HST)
                if metar.get("wspd") is not None:
                    row.update(dir=None if metar.get("wdir") in (None, "VRB") else metar["wdir"],
                               mph=round(metar["wspd"] * 1.15078), at=f"{t:%H:%M}")
                if metar.get("wgst"):
                    row["gust"] = round(metar["wgst"] * 1.15078)
                if metar.get("temp") is not None and metar.get("dewp") is not None:
                    air(row, metar["temp"] * 1.8 + 32, metar["dewp"] * 1.8 + 32, t)
            elif feed == "hcdp" and sid in hcdp:
                t, v = hcdp[sid]
                if "WS_1_Avg" in v:
                    row.update(dir=None if "WDrs_1_Avg" not in v else round(v["WDrs_1_Avg"] % 360),
                               mph=round(v["WS_1_Avg"] * 2.23694), at=f"{t:%H:%M}")
                    if "WS_1_Max" in v:
                        row["gust"] = round(v["WS_1_Max"] * 2.23694)
                if "Tair_1_Avg" in v and 0 < v.get("RH_1_Avg", 0) <= 100:
                    tc, rh = v["Tair_1_Avg"], v["RH_1_Avg"]
                    g = math.log(rh / 100) + 17.625 * tc / (243.04 + tc)
                    dc = 243.04 * g / (17.625 - g)
                    air(row, tc * 1.8 + 32, dc * 1.8 + 32, t)
        except Exception:
            pass
        if row.get("temp_f") is not None:
            a, b = (row["temp_f"] - 32) / 1.8, (row["dew_f"] - 32) / 1.8
            row["rh"] = round(100 * math.exp(17.625 * b / (243.04 + b) - 17.625 * a / (243.04 + a)))
        rows.append(row)
    (OUT / "stations.json").write_text(json.dumps({"built": f"{now:%Y-%m-%dT%H:%M}", "stations": rows}),
                                       encoding="utf-8")


def details_page(now: datetime, built: str) -> str:
    raw = (HERE / "template.html").read_text(encoding="utf-8")
    style = "".join("<style>" + part.split("</style>")[0] + "</style>" for part in raw.split("<style>")[1:])
    kd = (HERE / "details_template.html").read_text(encoding="utf-8")
    a = kd.index('  <section class="block" aria-labelledby="t-thermal">')
    b = kd.index('  <section class="block" aria-labelledby="t-ladder">')
    intro = ("<p>The air over Makapuʻu from a forecast sounding at Kaneohe or the Lihue balloon, with a parcel "
             "of Kaneohe's surface air lifted dry through it: the arithmetic behind the Thermals row on the main "
             "page. Pick the source and hour, then dial in the temperature you expect. The dew point is not a "
             "choice: Kaneohe measured for hours already gone, its forecast for hours ahead.</p>")
    thermal = re.sub(r"<p>The air above Kula.*?</p>", lambda m: intro, kd[a:b], count=1, flags=re.S)
    thermal = thermal.replace('<span id="raoblbl">Hilo balloon</span>', '<span id="raoblbl">Lihue balloon</span>')
    page = (HERE / "makapuu_details.html").read_text(encoding="utf-8").replace("{style}", style).replace("{updated}", f"{now:%a %d %b %Y · %H:%M} HST") \
        .replace("<!--THERMAL-->", thermal)
    return page.replace("ʻ", "&#8216;")



PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta http-equiv="refresh" content="900">
<title>Makapuʻu Today</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
{style}
<style>.back {{ color: var(--sky); text-decoration: none; font: 600 .95rem var(--display); }} .downret {{ margin-top: 6px; }}</style>
</head>
<body>
<div class="wrap">
  <header>
    <div><a class="homebtn" href="/">&larr; Home</a><div class="tag">Makapu&#8216;u soaring briefing</div><h1>Makapu&#8216;u Today</h1></div>
    <div class="date">Updated {updated}</div>
  </header>
  <div class="databadge"><div class="big">Under construction</div></div>
  <div>
    <div class="cam"><iframe src="{webcam}" title="Makapuʻu webcam, live, looking northwest toward Waimānalo" onload="setTimeout(function(){{var c=document.getElementById('camcover');if(c){{c.style.opacity=0;setTimeout(function(){{c.remove()}},700)}}}},3500)"></iframe><img id="camcover" src="camera.jpg?{built}" alt=""></div>
    <div class="camcap">Makapuʻu webcam, live, looking northwest. FAA WeatherCam. <a href="{webcam}" target="_blank" rel="noopener">Open full size</a></div>
  </div>
  <div class="cols">
    <section class="panel today" aria-labelledby="t-today">
      <div class="kind">{kind}</div>
      <div class="head"><h2 id="t-today">Today</h2>{chip}</div>
      <dl>
        <dt>On the ridge</dt><dd>{fly}</dd>
        <dt>Wind</dt><dd>{wind}</dd>
        {dirnote}
        <dt>Downrange</dt><dd>{base}</dd>
        <dt>Thermals</dt><dd>{therm}</dd>
        <dt>Inversion</dt><dd>{inv}</dd>
      </dl>
      <div class="kind" style="color:var(--muted)">Rest of today (forecast)</div>
      {rest}
    </section>
    <section class="panel tomorrow" aria-labelledby="t-tomorrow">
      <div class="kind">Forecast</div>
      <div class="head"><h2 id="t-tomorrow">{tom_title}</h2>{tom_chip}</div>
      {tomorrow}
    </section>
  </div>
  <section class="block" aria-labelledby="t-week">
    <h2 id="t-week">The week</h2>
    <div class="tablewrap"><table>
      <thead><tr><th>Day</th><th>Midday wind</th><th>Best</th><th>Hours</th><th class="n">Downrange base</th><th>Thermals</th></tr></thead>
      <tbody>{week}</tbody>
    </table></div>
    <p>Rated hour by hour, 08:00–19:00, from the NWS forecast grid at Makapuʻu: wind from 0–70° only (east of 070° turbulence off Rabbit Island reaches launch), speed and gusts against the site's limits, under 10 mph is a run-off launch. Downrange base and thermals from the RRFS sounding at Kaneohe, as far as it runs (about three days).</p>
  </section>
  <a class="detailslink" href="details.html">Details: thermal profile and station map &rarr;</a>
  <div class="notes">
    <p><strong>Where the numbers come from.</strong> Wind: Bellows (BELH1) via api.weather.gov, or the NWS forecast grid for the hour when Bellows is not reporting. Downrange base: the Kāneʻohe Bay camera's view of the Koʻolau, and Kaneohe (PHNG) temperature and dew point via aviationweather.gov. Thermals, inversion and the forecast downrange base: RRFS forecast soundings at Kaneohe (BUFKIT, Penn State). Limits: the Makapuʻu site table.</p>
    <p><strong>Notice.</strong> Information only, provided as is, with no guarantee of accuracy or availability. This is not an official forecast. Every pilot is responsible for their own decision to fly.</p>
  </div>
</div>
<!-- build problems: {diag} -->
<script>
(function () {{
  var built = "{built}";
  function check() {{
    fetch("summary.json?" + Date.now(), {{cache: "no-store"}}).then(function (r) {{ return r.json(); }})
      .then(function (j) {{ if (j && j.built && j.built > built) location.replace(location.pathname + "?v=" + Date.now()); }})
      .catch(function () {{}});
  }}
  setInterval(check, 120000);
  document.addEventListener("visibilitychange", function () {{ if (!document.hidden) {{ check(); }} }});
}})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.environ.get("KULAMODEL", str(HERE.parent / "KulaModel")))
    write(datetime.now(HST))
    print("wrote", OUT / "index.html")
