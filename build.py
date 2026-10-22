"""Build the Kula Today page: measured conditions now, and the outlook.

Runs on GitHub Actions every 15 minutes in daylight, hourly overnight, and writes
docs/index.html, which GitHub Pages serves. All the meteorology comes from
the KulaModel repository (checked out beside this one, or wherever
$KULAMODEL points), so there is one copy of the forecast code, not two.

Every section is built on its own: a source that fails shows as
"unavailable" and the rest of the page still publishes.
"""

from __future__ import annotations

import html
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from string import Template
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("KULAMODEL", str(HERE / "KulaModel")))

import functools                        # noqa: E402

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import requests                         # noqa: E402

import convergence as cv                # noqa: E402
import mesonet                          # noqa: E402
import sampler                          # noqa: E402
import sounding as sd                   # noqa: E402
import thermal as th                    # noqa: E402
import patterns                         # noqa: E402
import rules                            # noqa: E402
from kula import OFF_SLOPE_STATIONS     # noqa: E402

def _retrying(fn, tries: int = 3):
    """Retry a forecast fetch that times out; Open-Meteo sometimes hangs a request."""
    @functools.wraps(fn)
    def run(*a, **k):
        for i in range(tries):
            try:
                return fn(*a, **k)
            except requests.RequestException:
                if i == tries - 1:
                    raise
                time.sleep(5 * (i + 1))
    return run


# One fetch of the surface forecast per build, shared by soaring() and
# base_by_hour(). Two fetches doubled the chance of a timeout, and on
# 2026-10-12 13:31 the second one hung and blanked every forecast cloudbase.
cv.slope_parcel = functools.lru_cache(maxsize=4)(_retrying(cv.slope_parcel))
cv.sun_forecast = _retrying(cv.sun_forecast)
cv.trade_direction = _retrying(cv.trade_direction)

HST = ZoneInfo("Pacific/Honolulu")
RELEASE_FT = 3162                       # Kula Ag
# Added to the forecast cloudbase: Open-Meteo's air runs cool at Kula Ag,
# putting the base low. Over Oct 2025 - Oct 2026 the monthly median miss
# was +150 to +740 ft except March and April (-540, -320: forecast too
# high). +650, from Sept-Oct alone, would put spring ~1,000 ft high; a
# monthly table was tried and judged not worth its upkeep (Sam,
# 2026-10-13). A flat +400 is within ~150 ft most of the year and ~450 ft
# high in spring. Revisit occasionally (scratchpad method: seasonal_poc.py).
# Off (Sam, 2026-10-21: "Let's get rid of the 400 ft correction and see what
# happens"): over the 28 days to 10-21 the page ran +495 ft against the
# derived base with it, +95 without.
FORECAST_BASE_LOW_FT = 0
WEBCAM = ("https://weathercams.faa.gov/map/-128.93458,33.06367,-103.99562,49.85195"
          "/cameraSite/931/details/camera/13415/full")
STALE = pd.Timedelta("75min")
GOOD_VERDICTS = {"Moderate", "Strong", "Boom!"}

# What the page says for thermal.py's words. Strength is climb rate, which
# cannot be predicted (Sam), and the top of thermal.py's scale gave "Boom!"
# too easily; three plain words say only as much as is known.
RATING = {"Flat": "Nope", "Light": "OK", "Moderate": "Good", "Strong": "Good", "Boom!": "Good"}


def rating(verdict) -> str:
    return RATING.get(str(verdict), "--")


# Thermals are rated on the energy reaching the slope, not thermal.py's
# parcel. Kula Ag's midday net radiation split Sam's calls cleanly (good
# days >= 490 W/m2, capped/overcast/weak/socked <= 360) and tracks its share
# of clear-sky sunlight at r 0.98; the share corrects for time of day, so it
# carries the rating. Good days measured 79-81%, the rest 20-51%. GFS's sun
# forecast lands in the same class 71% of days one day ahead (61% at two),
# against 7% for the old forecast rating from model air temperature.
def sun_rating(share) -> str:
    if share is None or pd.isna(share):
        return "--"
    return "Good" if share >= 0.75 else "OK" if share >= 0.55 else "Nope"


def window_sun(now: datetime) -> float | None:
    """Kula Ag's share of clear-sky sun over the rating window, 09:00-15:00."""
    day = now.replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
    start, end = day + timedelta(hours=9), min(day + timedelta(hours=15), now.replace(tzinfo=None))
    rows = [(t, w) for t, st, w in _retrying(sampler.read_mesonet_sun)(start, end)
            if st.name == "Kula Ag" and start <= t < end]
    if not rows:
        return None
    ts = pd.DatetimeIndex([t for t, _ in rows]).tz_localize(HST)
    clear = mesonet.clear_sky(ts).sum()
    return sum(w for _, w in rows) / clear if clear > 50 else None


# Typical upslope start at Kula Ag by month (median, Oct 2025-Oct 2026,
# 346 days; July interpolated): it follows the sun clearing Haleakala.
# When direct sun clears Haleakala onto Kula Ag, by month (median, same year).
# The first sunlight reading shown is the first full hour after it.
SUN_ON = {1: "08:45", 2: "08:30", 3: "08:15", 4: "07:15", 5: "07:30", 6: "07:15",
          7: "07:20", 8: "07:30", 9: "08:00", 10: "08:00", 11: "08:15", 12: "08:45"}


# Stations that come out of shadow later than Kula Ag, in minutes after SUN_ON
# (Sam, 2026-10-20: "Could be some local shading"). 10-17 to 10-20, clear
# mornings: Keokea 21-39% at 08:00 while Kula Ag read 85-95%; Kula 1 9-14% at
# 08:00 and ~65% at 09:00 every day whatever the sky -- terrain or trees. Before
# these times a station's reading is shadow, not cloud, and is left out.
LATE_SUN_MIN = {"Keokea": 60, "Kula 1": 120}
SHADOWED: set = set()


def first_sun_hour(month: int) -> int:
    h, m = map(int, SUN_ON[month].split(":"))
    return h if m == 0 else h + 1


UPSLOPE_TYPICAL = {1: "09:15", 2: "08:50", 3: "08:30", 4: "08:30", 5: "08:00", 6: "08:00",
                   7: "08:00", 8: "08:00", 9: "08:20", 10: "08:30", 11: "08:45", 12: "09:15"}

# The day is rated from upslope (about 08:00-09:15 by season) to 15:00 (Sam), not 09:00-16:00.
_assess = th.assess


@functools.wraps(_assess)
def _assess_in_window(surface, release_ft, lift_ft=0.0, window=(9, 15), path=None):
    return _assess(surface, release_ft, lift_ft, window, path)


th.assess = _assess_in_window

esc = html.escape


# What went wrong in this build, written into the page as an HTML comment.
# Actions logs need a GitHub sign-in to read; the page does not.
PROBLEMS: list[str] = []


def safely(fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        traceback.print_exc()
        PROBLEMS.append(f"{fn.__name__}: {type(e).__name__}: {e}")
        return None


# -- today, measured ---------------------------------------------------------

# The Mesonet's own server has its stations about 15 minutes behind;
# api.weather.gov about 50. These four match api.weather.gov's values to
# rounding (Kula Ag, Keokea, Pulehu, 2026-10-12: T and Td within 0.03 F,
# direction exact), so they are the same sensors, just sooner.
HCDP_VARS = "Tair_1_Avg,RH_1_Avg,WS_1_Avg,WDrs_1_Avg,WS_1_Max"


def dewpoint_c(t_c: float, rh: float) -> float:
    a, b = 17.625, 243.04
    g = np.log(rh / 100) + a * t_c / (b + t_c)
    return b * g / (a - g)


# Mesonet-only stations for the map, not in the sampler (no NWS id, so the
# sampler's api.weather.gov poll would fail on them). Nakula, high on the south
# flank, is where cloud shows first coming round the backside below the
# summit (Sam, 2026-10-15). Off the Kula slope: above the core's 4,400 ft cut,
# and not on the sunlight ladder.
MAP_ONLY = [sampler.Station("HCDP0143", "Nakula", 5355, 20.67474, -156.23349, "0143")]
# Never in the Kula calculations (Sam): Nakula and the Kahikinui pair are
# there only to watch summit cloud being pushed over Kula. By id, not left to
# the elevation cut.
OFF_SLOPE = OFF_SLOPE_STATIONS | {s.id for s in MAP_ONLY}


def read_hcdp(start: datetime) -> dict[str, list]:
    """Today's Mesonet readings by station, oldest first, as sampler Readings."""
    by_hcdp = {s.hcdp: s for s in sampler.STATIONS + MAP_ONLY if s.hcdp}
    rows = mesonet.get("/mesonet/db/measurements", location="hawaii",
                       station_ids=",".join(by_hcdp), var_ids=HCDP_VARS,
                       start_date=start.strftime("%Y-%m-%dT%H:%M:%S-10:00"),
                       row_mode="json", limit=100000)
    vals: dict[tuple, dict] = {}
    for r in rows:
        if r.get("value") is None or r["station_id"] not in by_hcdp:
            continue
        t = (datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00"))
             .astimezone(HST).replace(tzinfo=None, second=0, microsecond=0))
        vals.setdefault((r["station_id"], t), {})[r["variable"]] = float(r["value"])
    out: dict[str, list] = {}
    for (hid, t), v in sorted(vals.items(), key=lambda x: x[0][1]):
        if "Tair_1_Avg" not in v or not 0 < v.get("RH_1_Avg", 0) <= 100:
            continue
        st = by_hcdp[hid]
        t_c = v["Tair_1_Avg"]
        rd = sampler.Reading(st, t, sampler.f32(sampler.net_round(t_c * 9 / 5 + 32, 1)),
                             sampler.f32(sampler.net_round(
                                 dewpoint_c(t_c, v["RH_1_Avg"]) * 9 / 5 + 32, 1)))
        if "WS_1_Max" in v:
            rd.gust_mph = sampler.f32(round(v["WS_1_Max"] * 2.23694))
        if "WS_1_Avg" in v:
            rd.wind_mph = sampler.f32(round(v["WS_1_Avg"] * 2.23694))
            if "WDrs_1_Avg" in v:
                rd.wind_dir = sampler.f32(v["WDrs_1_Avg"] % 360)
        out.setdefault(st.id, []).append(rd)
    return out


def observations(now: datetime) -> pd.DataFrame:
    """Today's readings: Mesonet stations from HCDP, the rest and any HCDP gap
    from api.weather.gov."""
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    start = midnight.astimezone(timezone.utc)
    try:
        hcdp = _retrying(read_hcdp)(midnight)
    except Exception as e:
        hcdp = {}
        PROBLEMS.append(f"HCDP observations, using api.weather.gov: {type(e).__name__}: {e}")
    rows = []
    for st in sampler.STATIONS + MAP_ONLY:
        got = hcdp.get(st.id)
        if st in MAP_ONLY and not got:
            continue                              # Mesonet only; remote, often down
        for attempt in range(0 if got else 3):   # api.weather.gov drops single requests
            try:
                got, _ = sampler.read_nws(st, start)
                break
            except Exception as e:
                fault = f"{type(e).__name__}: {e}"
                time.sleep(2 * (attempt + 1))
        if got is None:
            PROBLEMS.append(f"no observations for {st.id} ({fault})")
            continue
        for r in got:
            rows.append({"station_id": st.id, "name": st.name, "elev_ft": st.elev_ft,
                         "observed": r.observed, "temp_f": float(r.temp_f),
                         "dew_f": float(r.dew_f), "lcl_ft": float(r.lcl_ft),
                         "wind_dir": None if r.wind_dir is None else float(r.wind_dir),
                         "wind_mph": None if r.wind_mph is None else float(r.wind_mph),
                         "gust_mph": None if r.gust_mph is None else float(r.gust_mph),
                         "suspect": r.suspect})
    return pd.DataFrame(rows)


def cloudbase_now(obs: pd.DataFrame) -> dict | None:
    """Middle of the slope stations' latest condensation levels, with their range.

    Not the highest station alone: on 2026-10-12 that matched the webcam at
    10:45 (6,000 ft) but read 6,100 at 12:45 when Sam saw 4,900 -- the low
    slope had moistened under it and Crater Rd sat above the new moist air.
    The middle of the group was about 4,600.
    """
    fresh_after = obs.observed.max() - STALE
    core = obs[~obs.station_id.isin(OFF_SLOPE | {"TT925"}) & (obs.elev_ft <= 4400)
               & (obs.suspect == "") & (obs.observed >= fresh_after)]
    latest = core.sort_values("observed").groupby("station_id").tail(1)
    if len(latest) < 3:
        return None
    # The range drops the single lowest and highest station when 7+ report
    # (Sam, 2026-10-14). The usual odd ones out are Pulehu in a trade-air pulse
    # (4,160 ft at 12:00 that day) and Crater Rd, the top station; the middle
    # is unchanged and "Stations disagree" stops firing on one outlier.
    v = latest.lcl_ft.sort_values()
    span = v.iloc[1:-1] if len(v) >= 7 else v
    return {"mid": float(v.median()), "lo": float(span.min()),
            "hi": float(span.max()), "n": len(latest),
            "at": latest.observed.max()}


def west_intrusion(obs: pd.DataFrame) -> str | None:
    """Pulehu leaving the upslope sector: trade air coming round the low west slope.

    The ladder runs down the middle of the slope and misses this route. On
    2026-10-12 Pulehu turned N at 6-8 mph while the ladder still read upslope
    and Sam saw the shear move far south; on 9/26, 10/4 and 10/5 it read NE.
    """
    p = obs[(obs.station_id == "042HI") & obs.wind_dir.notna()].sort_values("observed").tail(2)
    if p.empty or p.observed.max() < obs.observed.max() - STALE:
        return None
    d, mph = cv.mean_direction(p.wind_dir), p.wind_mph.mean()
    if cv._side(d) == "upslope" or mph < 5:
        return None
    return f"Trade air reaching the lower west slope: Pulehu {cv.compass(d)} at {mph:.0f} mph"


# The line is read by where it crosses the road up the slope -- Makawao
# Avenue from Makawao to Pukalani, then Kula Highway south -- and, north of
# Makawao, by where it reaches the shore (Sam, 2026-10-12). By latitude,
# like convergence.LANDMARKS, which these come from.
# "Kula" alone was too vague to place on the road (Sam). Kamehameha Schools
# (OSM 20.8216) and the middle junction of Lower Kula Road with the highway
# (OSM 20.767-20.774) are Sam's picks to fill the gaps.
ROAD = sorted([(n, lat) for n, lat in cv.LANDMARKS
               if n in ("Makawao", "Pukalani", "Omaopio", "Waiakoa", "Keokea", "Ulupalakua")]
              + [("Kamehameha Schools", 20.8216),
                 ("the middle Lower Kula Road junction", 20.7707)], key=lambda m: -m[1])
# North of Makawao the line is named by the coast (Sam: "It just keeps moving
# north until it goes away"; his order Paia, Kuau, Hookipa, Haiku runs EAST
# along the shore). The forecast only gives a latitude, and even at 132 deg,
# near the edge of the trade sector, it reaches only Paia's -- so a forecast
# line north of Makawao is "near Paia". Kuau/Hookipa/Haiku come only from the
# measured coastal stations below (map drawn 2026-10-14).
SHORE = [("Paia", 20.9033)]
# The ladder's coastal stations, by where they actually sit. "Paia" (GW5395,
# 20.938 -156.348) is out by Hookipa; Hamakuapoko Rd and Peahi Rd are on the
# Haiku side.
COAST_STATION = {"Paia": "Hookipa", "Hamakuapoko Rd": "Haiku", "Peahi Rd": "Haiku"}
NORTH_OF_ROAD = 20.866          # past Makawao, about halfway to Haliimaile


# Sam: "if the shear is there it's not flyable" -- the line at Naalae Road
# (MECO Naalaea Rd, 20.767) or further south puts Kula in the trades.
# Measured: trades already at MECO Kula (20.771) or south of it.
NOT_FLYABLE_LAT = 20.775
NOT_FLYABLE = "The line is at Naalae Road or further south, where Kula is usually not flyable."


def _place(lat: float) -> tuple[str, str]:
    marks = SHORE if lat > NORTH_OF_ROAD else ROAD
    return ("shore" if lat > NORTH_OF_ROAD else "road",
            min(marks, key=lambda m: abs(m[1] - lat))[0])


def line_at(lat: float) -> str:
    """Where a forecast line crosses the road, lower case, for a sentence."""
    kind, name = _place(lat)
    if kind == "shore":
        return f"north of Makawao, reaching the coast near {name}"
    if name == "Makawao":
        return "on Makawao Avenue near Makawao"
    if name == "Pukalani":
        return "at Pukalani"
    if name.startswith("the "):
        return f"crossing Kula Highway at {name}"
    return f"crossing Kula Highway near {name}"


def line_between(north_lat: float, south_lat: float,
                 north_name: str | None = None, south_name: str | None = None) -> str:
    """The same for a measured bracket between two ladder stations."""
    (nk, nn), (sk, sn) = _place(north_lat), _place(south_lat)
    if north_name in COAST_STATION:
        nk, nn = "shore", COAST_STATION[north_name]
    if south_name in COAST_STATION:
        sk, sn = "shore", COAST_STATION[south_name]
    if (nk, nn) == (sk, sn):
        return (f"north of Makawao, reaching the coast near {nn}" if nk == "shore"
                else line_at(north_lat))
    if nk == "shore" and sk == "shore":
        return f"north of Makawao, reaching the coast between {sn} and {nn}"
    if nk == "shore":
        # Not "running from" the road: the line is anchored to the north (Sam).
        # Name the coast only when the south end is at Pukalani or north of
        # it; a bracket from Omaopio to the Haiku coast is a gap, not a line.
        if south_lat >= dict(ROAD)["Pukalani"] - 0.003:
            return f"north of {sn}, reaching the coast near {nn}"
        return f"north of {sn}"
    road = ("Makawao Avenue" if sn in ("Makawao", "Pukalani") else
            "Kula Highway" if nn != "Makawao" else "the road")
    return f"crossing {road} between {nn} and {sn}"


def line_short(name) -> str:
    """A forecast landmark as the week table shows it."""
    lat = dict(cv.LANDMARKS).get(name)
    if lat is None:
        return str(name)
    kind, place = _place(lat)
    if kind == "shore":
        return f"{place} (coast)"
    return "Lower Kula Rd (middle)" if place.startswith("the middle") else place


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _core(obs: pd.DataFrame) -> pd.DataFrame:
    return obs[~obs.station_id.isin(OFF_SLOPE | {"TT925"}) & (obs.elev_ft <= 4400)
               & (obs.suspect == "")]


def _sun_words(f: float) -> str:
    return ("full sun" if f >= 0.9 else "mostly sun" if f >= 0.65 else
            "about half sun" if f >= 0.4 else "mostly shade" if f >= 0.15 else "deep shade")


# Off the Kula side: shaded by trade cloud to the north and west, not by the
# Kula deck. With them in, the 11:30 replay of 2026-10-12 said the cloud
# "tops out between Piiholo and Keokea".
OFF_DECK = {"Piiholo", "Pulehu"}

# The sun trend averages these. Kula Ag alone flipped fading / returning /
# fading at 11:30, 12:30 and 13:30 on 2026-10-12 as gaps in broken cloud passed.
SUN_TREND_STATIONS = {"Kula Ag", "Keokea", "Park HQ"}


def sun_trend(now: datetime) -> float | None:
    """Share of clear-sky sun over the last 30 min minus the 30 min an hour earlier."""
    end = now.replace(tzinfo=None, second=0, microsecond=0)
    rows = [(t, w) for t, st, w in _retrying(sampler.read_mesonet_sun)(end - timedelta(minutes=90), end)
            if st.name in SUN_TREND_STATIONS]
    if not rows:
        return None
    s = pd.Series([w for _, w in rows], index=pd.DatetimeIndex([t for t, _ in rows]))
    clear = pd.Series(mesonet.clear_sky(s.index.tz_localize(HST)), index=s.index)

    def share(a, b):
        m = (s.index > a) & (s.index <= b)
        return s[m].sum() / clear[m].sum() if clear[m].sum() > 100 else None

    late = share(end - timedelta(minutes=30), end)
    early = share(end - timedelta(minutes=90), end - timedelta(minutes=60))
    return None if late is None or early is None else late - early


def _stations_at(core: pd.DataFrame, t) -> int:
    c = core[(core.observed <= t) & (core.observed > t - pd.Timedelta("20min"))]
    return int(c.station_id.nunique())


# The chart's derived points, kept through the day (Sam, 2026-10-17): an
# update that can't reach the Mesonet falls back to api.weather.gov, ~50 min
# late, and a point that needed the Mesonet stations vanished (10:45 that
# day). A point is only ever replaced by one built from more stations.
DERIVED_LOG = HERE / "docs" / "derived_log.json"


def _kept_points(day, fresh: list) -> list:
    """[[minute, ft]] for the chart: today's saved points merged with this
    update's (t, base, n_stations), keeping the most stations per point."""
    d = f"{day:%Y-%m-%d}"
    try:
        log = json.loads(DERIVED_LOG.read_text(encoding="utf-8"))
        pts = log.get("points", {}) if log.get("date") == d else {}
    except Exception:
        pts = {}
    for t, v, n in fresh:
        if v is None:
            continue
        k = str(t.hour * 60 + t.minute)
        if k not in pts or n >= pts[k][1]:
            pts[k] = [round(v, -1), n]
    try:
        DERIVED_LOG.write_text(json.dumps({"date": d, "points": pts}), encoding="utf-8")
    except Exception:
        pass
    return [[int(k), v] for k, (v, n) in sorted(pts.items(), key=lambda kv: int(kv[0]))]


# Dew-morning base phases (Sam, 2026-10-17): "Two transitions." 1: the
# saturated morning -- every station near saturation, the median is right.
# 2: convection starts while the lower slope is still damp from the dew --
# the median undershoots and the upper stations show the moisture above the
# surface (Crater Rd, the highest, matched his 4,900 that day). 3: the lower
# slope has dried -- the median is right again. Only after a wet morning (the
# morning-cloud line fired). Switches: highest station >= 500 ft above the median
# (2), back within 300 ft (3) -- fitted to one day, logged to be checked.
BASE_PHASE_FILE = HERE / "docs" / "base_phase.json"
PHASE2_SPLIT_FT = 500
PHASE3_CLOSE_FT = 300
PHASE = {}


def _upper_pair(latest: pd.DataFrame):
    """The highest station reporting (normally Crater Rd). A pair of the two
    highest depended on who had reported: Waipoli Rd (~4,360) pulled it to
    4,560 on 2026-10-17 while Crater Rd (4,740-4,920) matched Sam's 4,900."""
    top = latest.sort_values("elev_ft", ascending=False).head(1)
    return float(top.lcl_ft.iloc[0]), list(top.name)


def update_phase(obs: pd.DataFrame, now: datetime, wet_now: bool, up, day_over: bool,
                 morning_now: bool = False) -> dict:
    """wet: a dew morning (the morning line after a saturated night) -- only
    then may phase 2 read the base from the highest stations. Sam, 2026-10-18:
    "morning derived heights should not ignore lower stations" -- on a damp
    but dewless morning phase 2 ran 08:30-11:45 and put the base at
    4,700-5,400 ft while it sat at 4,000 or less. morning: the morning line
    fired at all today (for the hand-over "First clouds" line)."""
    d = f"{now:%Y-%m-%d}"
    try:
        ph = json.loads(BASE_PHASE_FILE.read_text(encoding="utf-8"))
        if ph.get("date") != d:
            raise ValueError
    except Exception:
        ph = {"date": d, "wet": False, "phase": 1, "intervals": []}
    if wet_now:
        ph["wet"] = True
    if morning_now or wet_now:
        ph["morning"] = True
    if ph["wet"] and up is not None and not day_over and len(obs):
        core = _core(obs)
        core = core[core.observed >= obs.observed.max() - STALE]
        latest = core.sort_values("observed").groupby("station_id").tail(1)
        if len(latest) >= 4:
            med = float(latest.lcl_ft.median())
            upper, _ = _upper_pair(latest)
            stamp = f"{latest.observed.max():%H:%M}"
            if ph["phase"] == 1 and upper - med >= PHASE2_SPLIT_FT:
                ph["phase"] = 2
                ph["intervals"].append([stamp, None])
            elif ph["phase"] == 2 and upper - med <= PHASE3_CLOSE_FT:
                ph["phase"] = 3
                ph["intervals"][-1][1] = stamp
    BASE_PHASE_FILE.write_text(json.dumps(ph), encoding="utf-8")
    return ph


def _in_phase2(t) -> bool:
    hm = f"{t:%H:%M}"
    return any(a <= hm and (b is None or hm < b) for a, b in PHASE.get("intervals", []))


def _derived_at(core: pd.DataFrame, t) -> float | None:
    """The derived base at t: the median of all stations, dew or not (Sam,
    2026-10-18: "Even with dew, stations provide correct base"). The highest
    stations are a forecast of where it lifts to, shown beside it in phase 2."""
    return _base_at(core, t)


def _base_at(core: pd.DataFrame, t) -> float | None:
    """Median of each core station's latest LCL at t, readings up to 20 min old.

    Per station, not per clock slot: the MECO stations report every 10
    minutes and the rest every 5, so slot medians jumped between station sets.
    """
    c = core[(core.observed <= t) & (core.observed > t - pd.Timedelta("20min"))]
    latest = c.sort_values("observed").groupby("station_id").tail(1)
    return float(latest.lcl_ft.median()) if len(latest) >= 3 else None


def summary(obs: pd.DataFrame, ladder, up, verdict, pos, west, now: datetime,
            sun_change: float | None = None, evening=None, cu_clear: bool = False) -> str:
    """A few plain sentences on how the day is going, from rules, not a model.

    No cloudbase figures: an estimate from ground stations is not good to
    the foot, and the Cloudbase line already gives its range (Sam,
    2026-10-12). The base is told by its trend, the deck by the stations
    it lies between. When two of base, sun and upslope move the same way
    over the last hour and none the other, it opens by saying so.
    """
    out, worse, better = [], [], []

    core = _core(obs)
    upslope_on = up is not None and not (evening and evening[0] in ("over", "drain"))
    if upslope_on and cu_clear:
        # No cloud yet: a "cloudbase rising/falling" sentence would describe
        # cloud that isn't there (Sam, 2026-10-14).
        out.append("No cumulus over the slope yet.")
    elif not core.empty and upslope_on:            # no base talk outside the upslope
        t = core.observed.max()
        # The trend describes the END of the chart's line: the last 30 minutes,
        # two chart steps (Sam, 2026-10-17: "The today description window
        # should match the chart"). Subject to change.
        cur, ago = _derived_at(core, t), _derived_at(core, t - pd.Timedelta("30min"))
        if cur is not None and ago is not None and t >= obs.observed.max() - STALE:
            d = cur - ago
            trend = ("rising" if d >= 200 else "edging up" if d >= 100 else
                     "lowering" if d <= -200 else "edging down" if d <= -100 else "steady")
            if d <= -200:
                worse.append("base lowering")
            elif d >= 200:
                better.append("base rising")
            day = t.normalize()
            morning = [x for x in (_derived_at(core, day + pd.Timedelta(minutes=m))
                                   for m in range(600, 721, 30)) if x is not None]
            if t.hour >= 12 and morning and max(morning) - cur >= 700:
                out.append(f"Cloudbase has come down from its late-morning high and is {trend} now.")
            elif t.hour < 12 and morning and cur >= max(morning) - 200 and d >= 100:
                out.append(f"Cloudbase is at its highest of the day and still {trend}.")
            else:
                out.append(f"Cloudbase is {trend}.")

    if sun_change is not None:
        if sun_change <= -0.2:
            worse.append("sun fading")
        elif sun_change >= 0.2:
            better.append("sun returning")

    if up is not None and now.hour >= 10:
        w = core.dropna(subset=["wind_mph"])
        t = obs.observed.max()
        w_now = w[w.observed > t - pd.Timedelta("30min")].wind_mph.mean()
        w_then = w[(w.observed > t - pd.Timedelta("90min"))
                   & (w.observed <= t - pd.Timedelta("60min"))].wind_mph.mean()
        if pd.notna(w_now) and pd.notna(w_then):
            if w_now - w_then <= -2:
                worse.append("upslope fading")
            elif w_now - w_then >= 2:
                better.append("upslope building")

    if len(worse) >= 2 and not better:
        out.insert(0, f"Conditions are going downhill: {' and '.join(worse)}.")
    elif len(better) >= 2 and not worse:
        out.insert(0, f"Conditions are improving: {' and '.join(better)}.")

    lit = [(n, s) for n, _, s in ladder if s is not None and n not in OFF_DECK]  # top down
    ka = next((s for n, s in lit if n == "Kula Ag"), None)
    if ka is not None:
        if all(s >= 0.75 for _, s in lit):
            out.append("The whole slope is in sun.")
        else:
            text = f"Kula Ag had {_sun_words(ka)} over the last hour"
            # Where the deck tops out (Sam, 2026-10-17): heights, and only from
            # the sunlight stations at 4,000 ft and up -- a shaded station
            # lower down sits UNDER the cloud ("between about 2,800 and 3,200
            # ft" with the base at 3,800). When they can't place it, fall back
            # to the Puunianiau camera (6,854 ft): in cloud or clear.
            high = [(n, e, s) for n, e, s in ladder
                    if s is not None and e >= 4000 and n not in OFF_DECK]      # top down
            cb = safely(cloudbase_now, obs)
            shaded = [i for i, (_, _, s) in enumerate(high) if s < 0.5]
            top = shaded[0] if shaded else None
            above = [(n, e) for n, e, s in high[:top] if s >= 0.75] if top else []
            if cu_clear:
                pass        # no cumulus: a low reading is terrain shadow or haze, not a deck
            elif top == 0:
                text += "; cloud reaches all the way to the Summit"
            else:
                placed = False
                if top is not None and above:
                    floor = max(high[top][1], cb["mid"]) if cb else high[top][1]
                    if floor < above[-1][1] - 200:
                        text += (f"; the cloud tops out between about {round(floor, -2):,.0f} and "
                                 f"{round(above[-1][1], -2):,.0f} ft, with sun above it")
                        placed = True
                # The lower Kula slope's average sun, as P4 uses: Kula Ag alone at
                # 53% hid the tops while Keokea sat at 31% (Sam, 2026-10-17).
                lowk = [min(s, 1.0) for n, _, s in ladder if n in ("Kula Ag", "Keokea") and s is not None]
                shaded_low = bool(lowk) and sum(lowk) / len(lowk) < 0.6
                cam = safely(camera_now, "Puunianiau") if not placed and (shaded_low or top is not None) else None
                if cam is not None:
                    text += ("; cloud reaches about 6,900 ft (Puunianiau camera in cloud)" if cam
                             else "; the cloud tops out below about 6,900 ft (Puunianiau camera clear)")
            out.append(text + ".")

    if evening and up is not None:
        kind, since = evening
        out.append({
            "drain": f"Drainage has set in since about {since:%H:%M}; the upslope is over for the day.",
            "over": f"The slope has been losing heat since about {since:%H:%M}, so the upslope is "
                    f"over; drainage usually follows within about two hours.",
            "ending": f"The upslope is ending: the slope has nearly stopped gaining heat since "
                      f"about {since:%H:%M}.",
        }[kind])
    elif up is not None:
        w = core[core.observed >= obs.observed.max() - pd.Timedelta("30min")].wind_mph.dropna()
        if len(w):
            mph = w.mean()
            word = "a light" if mph < 6 else "a moderate" if mph < 10 else "a brisk"
            line = f"Upslope has run since about {up:%H:%M} at {word} {mph:.0f} mph"
        else:
            line = f"Upslope has run since about {up:%H:%M}"
        out.append(line + (f", and thermals seem {verdict.lower()}." if verdict and verdict != "--" else "."))
    elif now.hour < 11:
        out.append("Upslope has not set in yet.")

    if pos and pos.get("found"):
        tail = ("with trade air reaching the lower west slope at Pulehu" if west
                else "with no trade air on the lower west slope")
        out.append(f"Trades are holding the shear line "
                   f"{line_between(pos['north_lat'], pos['south_lat'], pos['north_of'], pos['south_of'])}, {tail}.")
        if pos["north_lat"] <= NOT_FLYABLE_LAT:
            out.append(NOT_FLYABLE)
    elif west:
        out.append("Trade air is reaching the lower west slope at Pulehu.")

    return " ".join(out)


def forecast_summary(r, pair, today_pair, inv, today_word: str) -> str:
    """The forecast panel's few sentences, in the same style as summary().

    Base by comparison only -- with the other day and through the day --
    never as a figure.
    """
    out = []
    sun = r["sun"]
    if sun is not None and not pd.isna(sun):
        # Qualified (Sam): GFS cannot see the slope's own cumulus. On the 7
        # days it gave 90%+ in Sept-Oct 2026, Kula Ag measured a median 81%.
        sky = ("The model sees little cloud over Kula; the slope's own cumulus are too "
               "small for it to show, so expect some" if sun >= 0.8 else
               "The model sees some cloud over Kula, on top of the slope's own cumulus"
               if sun >= 0.6 else
               "The model sees a partly cloudy sky over Kula" if sun >= 0.4 else
               "The model sees mostly cloudy skies over Kula")
    else:
        sky = None
    am, pm = pair if pair else (None, None)
    bits = []
    if am is not None and today_pair and today_pair[0] is not None:
        d = am - today_pair[0]
        if d >= 500:
            bits.append(f"a higher base than {today_word}")
        elif d <= -500:
            bits.append(f"a lower base than {today_word}")
    # No "rising / lowering through the afternoon": the forecast's afternoon
    # shape had no skill over 32 days (Sam, 2026-10-17).
    # No sky sentence (Sam, 2026-10-17: "Unnecessary"): the model cannot see
    # Kula's own cumulus, and the Sun row already says so.
    if bits:
        out.append(_cap(", ".join(bits)) + ".")
    word = strength_word(r["thermal"])
    if word in ("Flat", "Light", "Moderate", "Strong"):
        out.append(f"Thermals probably {word.lower()}.")
    line = r["shear_location"]
    if line is None or pd.isna(line):
        out.append("No trade line is expected.")
    elif not pd.isna(r["dir_deg"]):
        where = (line_at(dict(cv.LANDMARKS)[line]) if line in dict(cv.LANDMARKS)
                 else f"near {line}")
        out.append(f"Trades from the {cv.compass(r['dir_deg'])} should put the shear line {where}.")
        if dict(cv.LANDMARKS).get(line, 99) <= NOT_FLYABLE_LAT:
            out.append(NOT_FLYABLE)
    return " ".join(out)


def thermal_today(obs: pd.DataFrame, now: datetime, path=None):
    ka = obs[obs.station_id == "001HI"].copy()
    ka["hour"] = ka.observed.dt.hour
    surf = ka.groupby("hour").agg(temp_f=("temp_f", "mean"),
                                  dewpoint_f=("dew_f", "mean")).reset_index()
    surf["day"] = now.date()
    got = th.assess(surf, RELEASE_FT, path=path)
    return None if got.empty else got


def thermal_now(obs: pd.DataFrame, path=None):
    """(row, newest) -- the rating from Kula Ag's last 60 minutes of readings.

    The clock-hour rating rested on 1-3 readings early in each hour; a full
    rolling hour is steady, and is labelled by its newest reading (Sam).
    """
    ka = obs[obs.station_id == "001HI"].dropna(subset=["temp_f", "dew_f"])
    if ka.empty:
        return None
    end = ka.observed.max()
    w = ka[ka.observed > end - pd.Timedelta("60min")]
    surf = pd.DataFrame({"hour": [end.hour], "temp_f": [w.temp_f.mean()],
                         "dewpoint_f": [w.dew_f.mean()], "day": [end.date()]})
    # thermal.py's window stops at 16:00; the live call runs to the upslope's end.
    got = th.assess(surf, RELEASE_FT, window=(9, 18), path=path)
    return None if got.empty else (got.iloc[0], end)


# Gustiness on the lower slope (Sam: "indications of relatively gusty
# conditions may be useful"). Kula's midday wind is light but its gusts run
# well above it: over a year at Kula Ag and Keokea, 10:00-15:00, the typical
# gust (90th percentile) ran ~6.5 mph above the average, and 7.3-7.4 mph or
# more on the gustiest quarter of days. Stations below 3,600 ft on the core
# slope -- Mesonet, MECO, Kaonoulu -- over the last hour.
GUSTY_EXCESS_MPH = 7.5


def lower_wind(obs: pd.DataFrame, since=None, until=None):
    """Lower-slope wind over the last hour, or over [since, until] for the day review."""
    w = _core(obs)
    since = obs.observed.max() - pd.Timedelta("60min") if since is None else since
    until = obs.observed.max() if until is None else until
    w = w[(w.elev_ft <= 3600) & (w.observed >= since) & (w.observed <= until)]
    w = w.dropna(subset=["wind_mph"])
    if w.station_id.nunique() < 3:
        return None
    gusts = w.gust_mph.dropna() if "gust_mph" in w else pd.Series(dtype=float)
    avg = float(w.wind_mph.mean())
    typ = float(gusts.quantile(0.9)) if len(gusts) >= 5 else None
    # The flag is judged at Kula Ag and Keokea only, where the threshold was
    # measured; MECO's gusts are reported differently and may always run high.
    ref = w[w.station_id.isin(["001HI", "016HI"])].dropna(subset=["gust_mph"])
    gusty = (len(ref) >= 6 and
             ref.gust_mph.quantile(0.9) - ref.wind_mph.mean() >= GUSTY_EXCESS_MPH)
    return {"avg": avg, "gust": typ, "max": float(gusts.max()) if len(gusts) else None,
            "n": int(w.station_id.nunique()), "gusty": bool(gusty)}


def upslope_since(obs: pd.DataFrame):
    core = obs[~obs.station_id.isin(OFF_SLOPE) & (obs.elev_ft <= 4400)].dropna(
        subset=["wind_dir", "wind_mph"]).copy()
    core = core[core.observed.dt.hour.between(5, 11)]
    if core.empty:
        return None, None
    core["t"] = core.observed.dt.floor("15min")
    r = np.radians(core.wind_dir)
    core["u"], core["v"] = core.wind_mph * np.sin(r), core.wind_mph * np.cos(r)
    g = core.groupby("t")[["u", "v"]].mean()
    d = (np.degrees(np.arctan2(g.u, g.v)) % 360)
    phase = d.map(lambda x: "drain" if 70 <= x <= 160 else ("up" if 190 <= x <= 320 else "mix"))
    drains = [t for t, p in phase.items() if p == "drain"]
    last_drain = drains[-1] if drains else None
    ts = [t for t in phase.index if last_drain is None or t > last_drain]
    # Two quarter hours (30 min) of upslope confirm it (Sam, 2026-10-14);
    # three made the page trail Weather Guesser's cusp by ~45-60 min.
    for i in range(len(ts) - 1):
        if all(phase[ts[i + k]] == "up" for k in range(2)):
            return ts[i], last_drain
    return None, last_drain


# Whether cumulus have formed over the slope yet today, from sunlight. The
# derived cloudbase is a condensation level -- where cumulus WOULD form -- and
# on a clear morning it read as cloud that wasn't there (Sam, 2026-10-14).
# A station sees cloud between itself and the sun: share of clear-sky under
# 75%, or Kula Ag's sky radiating warm (emissivity over 0.85; clear ~0.75-0.8).
# The radiometer version was checked against Sam's days (first cu 09:00-13:20).
CU_STATIONS = {"0119": "Kula Ag", "0116": "Keokea", "0151": "Park HQ"}
KULA_SIDE = {"Keokea", "Kula Ag", "Kula 1", "Kamehamenui 1", "Park HQ"}


def first_cumulus(now: datetime):
    """("clear", None) if no cumulus yet since the sun cleared Haleakala,
    ("cumulus", time) once seen, or None if there is too little data."""
    day = now.replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
    h, m = map(int, SUN_ON[now.month].split(":"))
    start = day + timedelta(hours=h, minutes=m) - timedelta(minutes=30)
    end = now.replace(tzinfo=None)
    if end - start < timedelta(minutes=50):
        return None
    rows = _retrying(mesonet.get)("/mesonet/db/measurements", location="hawaii",
                                  station_ids=",".join(CU_STATIONS),
                                  var_ids="SWin_1_Avg,LWin_1_Avg,Tair_1_Avg",
                                  start_date=start.strftime("%Y-%m-%dT%H:%M:%S-10:00"),
                                  end_date=end.strftime("%Y-%m-%dT%H:%M:%S-10:00"),
                                  row_mode="json", limit=100000)
    if not rows:
        return None
    x = pd.DataFrame(rows)
    x["t"] = pd.to_datetime(x.timestamp).dt.tz_convert(HST)
    x["value"] = pd.to_numeric(x.value, errors="coerce")
    p = x.pivot_table(index=["t", "station_id"], columns="variable", values="value").reset_index()
    p["clear"] = mesonet.clear_sky(pd.DatetimeIndex(p.t))
    p["share"] = p.SWin_1_Avg / p.clear.where(p.clear > 50)
    p["slot"] = p.t.dt.floor("15min")
    g = p.groupby(["slot", "station_id"]).share.mean().unstack()
    # Each station counts only after its own sunrise over Haleakala: Keokea
    # sat in shadow until 08:50 on 2026-10-14, Kula Ag lit at 08:05.
    lit = g.ge(0.85).cummax()
    shaded = (g.lt(0.75) & lit).any(axis=1)
    ka = p[p.station_id == "0119"].copy()
    if "LWin_1_Avg" in ka:
        ka["emis"] = ka.LWin_1_Avg / (5.67e-8 * (ka.Tair_1_Avg + 273.15) ** 4)
        shaded = shaded | ka.groupby("slot").emis.mean().gt(0.85).reindex(shaded.index, fill_value=False)
    seen = shaded[shaded & shaded.shift(-1, fill_value=False)]       # two quarter hours in a row
    return ("cumulus", seen.index[0].tz_localize(None)) if len(seen) else ("clear", None)


def net_radiation(now: datetime) -> pd.Series:
    """Kula Ag's net radiation (W/m2) since noon, quarter-hour means, from HCDP.

    The energy the slope surface is banking: positive heats it and drives
    the upslope, negative cools it and builds the drainage.
    """
    noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
    rows = _retrying(mesonet.get)("/mesonet/db/measurements", location="hawaii",
                                  station_ids="0119", var_ids="Rnet_1_Avg",
                                  start_date=noon.strftime("%Y-%m-%dT%H:%M:%S-10:00"),
                                  row_mode="json", limit=100000)
    s = pd.Series({pd.Timestamp(r["timestamp"]).tz_convert(HST).tz_localize(None): float(r["value"])
                   for r in rows if r.get("value") is not None}, dtype=float).sort_index()
    return s.resample("15min").mean().dropna() if len(s) else s


def _first_two(s: pd.Series):
    for a, b in zip(s.index[:-1], s.index[1:]):
        if s[a] and s[b]:
            return a
    return None


def sundown(now: datetime):
    """When the clear-sky sun at Kula fell below 10 W/m2 this evening, or None
    while it is still up. The page's own clock for dark: on 2026-10-15 the
    Mesonet timed out, net radiation never arrived, and at 18:37 the page
    still said the upslope was ending (Sam: "It's dark out")."""
    ts = pd.date_range(now.replace(hour=15, minute=0, second=0, microsecond=0), now, freq="5min")
    if len(ts) == 0:
        return None
    cs = np.asarray(mesonet.clear_sky(ts))
    if cs[-1] >= 10:
        return None
    return ts[np.argmax(cs < 10)].tz_localize(None)


def evening_state(obs: pd.DataFrame, rnet: pd.Series | None):
    """("ending" | "over" | "drain", since) once the slope stops heating.

    Net radiation is the signal (29 days, 9/14-10/12): it falls below
    50 W/m2 about 17:15, within 15 minutes of the upslope weakening, and
    turns negative about 17:45 every day, +-15 min; drainage follows about
    1 h 45 min later and varies far more. From 15:00 only, so a thick deck
    at midday is not read as the evening. Drainage itself, and the old
    wind rule when net radiation is missing, come from evening_flow().
    """
    wind = evening_flow(obs)
    if wind and wind[0] == "drain":
        return wind
    if rnet is None or rnet.empty:
        return ("ending", wind[1]) if wind else None
    # Half-hour means (Sam), stamped at the half hour's start. Back-tested on
    # Oct 2025 - Oct 2026: below 100 W/m2 the wind weakened a median 45 min
    # later; the upslope ran on for an hour after only 5% of the time.
    r = rnet.rolling(2).mean().shift(-1).dropna()
    r = r[r.index.hour >= 15]
    over = r.index[(r < 0).values]
    if len(over):
        return "over", over[0]
    # A cloud can do the same for half an hour and the sun bring the upslope
    # back (2026-10-14: 15:00-15:45 under 100, calm at Kula Ag, then 210 W/m2
    # and SSW 5 mph by 16:15), so "ending" holds only while it stays low and
    # dates from the last time it was still above.
    if len(r) and r.iloc[-1] < 100:
        above = r.index[(r >= 100).values]
        low = r[r.index > above[-1]] if len(above) else r
        return "ending", low.index[0]
    return None


def evening_flow(obs: pd.DataFrame):
    """After 14:00: ("cusp", since) or ("drain", since) once the upslope gives way.

    The evening mirror of upslope_since, at Kula Ag, where Sam watches it.
    Kula Ag's afternoon upslope is itself SSW (203-217 deg at 3-6 mph on
    10-03, 10-05, 10-11), so direction alone cannot see the end. The cusp
    is the wind dying to about 2 mph and backing below 200 deg (17:30-18:30
    on those days); drainage from the east follows within half an hour.
    Needs two quarter hours in a row out of the upslope, so one lull does
    not end the day.
    """
    ka = obs[(obs.station_id == "001HI")].dropna(subset=["wind_dir", "wind_mph"]).copy()
    ka = ka[ka.observed.dt.hour >= 12]
    if ka.empty:
        return None
    ka["t"] = ka.observed.dt.floor("15min")
    r = np.radians(ka.wind_dir)
    ka["u"], ka["v"] = ka.wind_mph * np.sin(r), ka.wind_mph * np.cos(r)
    g = ka.groupby("t").agg(u=("u", "mean"), v=("v", "mean"), mph=("wind_mph", "mean"))
    g["d"] = np.degrees(np.arctan2(g.u, g.v)) % 360
    # The same weak wind means more as the day runs down (Sam: include time
    # of day): a lull at 14:00, the end of the upslope by 17:00.
    def floor_mph(t):
        return 2.5 if t.hour < 16 else 3.5 if t.hour < 17 else 4.5

    phase = pd.Series({t: "drain" if 70 <= x.d <= 160 else
                       "up" if 185 <= x.d <= 320 and x.mph > floor_mph(t) else "mix"
                       for t, x in g.iterrows()})
    if len(phase) < 2 or phase.index[-1].hour < 14 or "up" not in phase.values:
        return None
    # From 16:00, the first two quarter hours in a row out of the upslope end
    # it for the day; a later gust does not bring it back (it flickered back
    # on 10-11 and 10-12). Before 16:00 a weak spell is only a lull.
    since = None
    for a, b in zip(phase.index[:-1], phase.index[1:]):
        if a.hour >= 16 and phase[a] != "up" and phase[b] != "up" and (phase[:a] == "up").any():
            since = a
            break
    if since is None:
        return None
    drains = phase[(phase.index >= since) & (phase == "drain")]
    if phase.iloc[-1] == "drain" and len(drains):
        return "drain", drains.index[0]
    return "cusp", since


def sun_ladder(now: datetime) -> list[tuple[str, int, float | None]]:
    """Share of clear-sky sun over the 60 minutes up to the newest reading,
    per station, top down.

    Not the last clock hour (Sam, 2026-10-15): at :52 that was 12:00-13:00,
    up to two hours old. The Mesonet runs about 15 minutes behind, so the
    window ends at its newest reading rather than now. RAWS reports by the
    hour only, so Kula 1 and Kamehamenui keep their latest whole hour.
    Mesonet and RAWS each fail on their own; a HCDP connect timeout at
    14:23 on 2026-10-12 used to blank the whole ladder, RAWS included.
    """
    now_l = now.replace(tzinfo=None, second=0, microsecond=0)
    got_m = []
    try:
        got_m = _retrying(sampler.read_mesonet_sun)(now_l - timedelta(minutes=100), now_l)
    except Exception as e:
        PROBLEMS.append(f"Mesonet sun: {type(e).__name__}: {e}")
    stamps = [t for t, st, _ in got_m if st.name == "Kula Ag"] or [t for t, _, _ in got_m]
    end = max(stamps) if stamps else now_l
    start = end - timedelta(hours=1)
    rows: dict[str, list] = {}
    for obs_t, st, w in got_m:
        if start < obs_t <= end:
            rows.setdefault(st.id, []).append((obs_t, w))
    for st in (s for s in sampler.STATIONS if s.raws):
        try:
            got = _retrying(sampler.read_raws_sun)(st.raws, now_l.date(), now_l.date())
        except Exception as e:
            PROBLEMS.append(f"RAWS sun {st.raws}: {type(e).__name__}: {e}")
            continue
        done = [(t, w) for t, w in got if t <= now_l]      # hour-ending stamps
        if done:
            t, w = max(done)
            rows.setdefault(st.id, []).append((t - timedelta(minutes=30), w))
    ladder = []
    for st in sampler.STATIONS:
        if st.id not in rows or st.id in ("019HI",):
            continue
        ts = pd.DatetimeIndex([t for t, _ in rows[st.id]]).tz_localize(HST)
        clear = mesonet.clear_sky(ts).sum()
        share = sum(w for _, w in rows[st.id]) / clear if clear > 50 else None
        ladder.append((st.name, int(st.elev_ft), share))
    return sorted(ladder, key=lambda x: -x[1]), start


# The last good ladder, reused for up to two hours when both sources fail.
# The heading names the hour it covers, so an old one shows as old.
LADDER_FILE = HERE / "docs" / "ladder.json"
LADDER_MAX_AGE = timedelta(hours=2)


def ladder_or_last(now: datetime):
    got = safely(sun_ladder, now)
    # Good only with Kula Ag in it: RAWS alone answering made a two-station
    # ladder that replaced the full one at 14:54 on 2026-10-12.
    if got and any(name == "Kula Ag" and s is not None for name, _, s in got[0]):
        LADDER_FILE.write_text(json.dumps({"start": got[1].isoformat(), "ladder": got[0]}),
                               encoding="utf-8")
        return got
    try:
        saved = json.loads(LADDER_FILE.read_text(encoding="utf-8"))
        start = datetime.fromisoformat(saved["start"])
    except (OSError, ValueError, KeyError):
        return got
    if now.replace(tzinfo=None) - start > LADDER_MAX_AGE:
        return got
    PROBLEMS.append(f"sun ladder from {start:%H:%M} reused")
    return [tuple(r) for r in saved["ladder"]], start


# The line "flaps like a flag on the micro level" (Sam): one reading per
# station moved the bracket around from build to build. Each station's
# wind is averaged over its last hour instead.
LINE_SMOOTH = pd.Timedelta("60min")

# The FAA weather camera Sam watches, Kekoa (HNMA), has its own wind, temperature
# and humidity every 10 minutes ("advisory, not certified"; the last 6 hours
# only). It sits at Waihou Springs' latitude, in the ladder's widest gap: on
# 2026-10-14 it turned from SSW 13-19 mph to NNE-NE at 12:30-12:40 and stayed
# in the trades while the ladder still bracketed the line north of it. Ladder
# and map only -- like Piiholo it is often in trade air, so it stays out of
# the cloudbase and the upslope clock.
KEKOA = ("HNMA", "Red Hill", 20.79729, 2819)   # FAA calls it Kekoa; Sam: "Red Hill"
KEKOA_LON = -156.31142
KEKOA_URL = "https://weathercams.faa.gov/api/summary"
# More rungs for the shear line (Sam), read from api.weather.gov like the
# coastal three: (id, name, lat, elev_ft, lon). Ladder and map only.
EXTRA_LADDER = [
    ("015HE", "MECO Makawao Ave", 20.83766, 1595, -156.32803),
    ("129HE", "MECO Pulehu Rd 2", 20.82525, 647, -156.39461),
    ("003HE", "MECO Kula 2", 20.82302, 2320, -156.30763),
    ("094HE", "MECO Olinda Rd", 20.80301, 4021, -156.27382),
]
KEKOA_PAGE = "https://weathercams.faa.gov/cameras/state/HI/cameraSite/931/summary"


@functools.cache
def read_kekoa() -> pd.DataFrame:
    # The API answers 401 without the camera page as the referrer.
    r = requests.get(KEKOA_URL, params={"siteId": 931},
                     headers={**cv.HEAD, "Referer": KEKOA_PAGE}, timeout=30)
    r.raise_for_status()
    rows = []
    for o in (r.json().get("payload") or {}).get("site", {}).get("advisoryWeather") or []:
        if o.get("windDirDegrees") is None or o.get("windSpeedKnots") is None:
            continue
        rows.append({"observed": pd.Timestamp(o["observationDateTime"]).tz_convert(HST).tz_localize(None),
                     "wind_dir": float(o["windDirDegrees"]),
                     "wind_mph": float(o["windSpeedKnots"]) * 1.15078})
    return pd.DataFrame(rows)


def _ladder_readings(sid: str, since: datetime) -> pd.DataFrame:
    """One ladder station's wind since a time, from api.weather.gov."""
    r = requests.get(f"https://api.weather.gov/stations/{sid}/observations",
                     params={"start": since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
                     headers=cv.HEAD, timeout=45)
    r.raise_for_status()
    rows = []
    for f in r.json().get("features", []):
        p = f["properties"]
        d, s = p["windDirection"]["value"], p["windSpeed"]["value"]
        if d is None:
            continue
        rows.append({"observed": pd.Timestamp(p["timestamp"]).tz_convert(HST).tz_localize(None),
                     "wind_dir": float(d), "wind_mph": (s or 0) * 0.621371})
    return pd.DataFrame(rows)


def smoothed_ladder(obs: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Each ladder station's vector-mean wind over the last hour.

    The page's own readings (HCDP and api.weather.gov, already fetched) for
    the stations it has; api.weather.gov for the coastal three it does not.
    """
    since = now - LINE_SMOOTH
    cut = since.replace(tzinfo=None)
    rows = []
    for sid, name, lat, elev in cv.LADDER + [KEKOA] + [e[:4] for e in EXTRA_LADDER]:
        have = (obs[(obs.station_id == sid) & obs.wind_dir.notna()]
                [["observed", "wind_dir", "wind_mph"]] if len(obs) else pd.DataFrame())
        if sid == KEKOA[0]:
            have = safely(read_kekoa)
            if have is None:
                continue
        elif have.empty:
            try:
                have = _retrying(_ladder_readings)(sid, since)
            except Exception:
                continue
        if have.empty:
            continue
        recent = have[have.observed >= cut]
        if recent.empty:
            # No reading in the hour: use up to three hours rather than drop
            # the rung. Piiholo missing once let the bracket jump from the
            # Haiku coast to Omaopio (12:00 on 2026-10-14).
            recent = have[have.observed >= cut - timedelta(hours=2)]
        have = recent
        if have.empty:
            continue
        d = cv.mean_direction(have.wind_dir)
        rows.append({"station": sid, "name": name, "lat": lat, "elev_ft": elev,
                     "observed_local": have.observed.max(), "wind_dir": d,
                     "wind_mph": float(have.wind_mph.mean()), "side": cv._side(d),
                     "n": len(have)})
    return pd.DataFrame(rows)


# Off (Sam, 2026-10-21, "Stale-stations, let's see what happens"): it sat a
# station out of 62% of updates over the week to 10-21.
STALE_RUNGS_SIT_OUT = False


def best_split(lad: pd.DataFrame) -> dict:
    """The line where the most ladder stations agree: trades north of it,
    upslope south of it. Shear readings count for neither side.

    convergence.position() stops at the first non-trade station walking
    south, so one coastal station on a local southwest wind (Hamakuapoko Rd,
    afternoon of 2026-10-14) put the line "north of Makawao" while the trades
    reached Piiholo and Pulehu (Sam: "Contradiction").
    """
    f = lad.sort_values("lat", ascending=False).reset_index(drop=True)
    # A rung whose newest reading lags the ladder's typical one by 30 min or
    # more sits out (Sam, option 2): at 17:07 on 2026-10-16 Lower Kimo Dr's
    # 16:10 trades and a lone stale Waihou reading outvoted Red Hill's fresh
    # upslope and put the line at Omaopio. Measured from the median, not the
    # freshest: api.weather.gov runs ~50 min behind the Mesonet and FAA, so
    # "45 min behind the freshest" would have dropped most MECO and coast rungs.
    dropped = 0
    if STALE_RUNGS_SIT_OUT and len(f) >= 4:
        typical = f.observed_local.median()
        keep = f.observed_local > typical - pd.Timedelta("30min")
        dropped = int((~keep).sum())
        f = f[keep].reset_index(drop=True)
    if len(f) < 2:
        return {"found": False, "why": "too few ladder stations reporting"}
    side = f.side.tolist()
    # A shear reading counts half a vote for the side its direction is
    # nearer (Sam, 2026-10-17): Pulehu Rd 2 NNE 17 and Olinda Rd N 10 sat just
    # outside the trades sector, counted for nothing, and with Makawao Ave and
    # Kula 2 missing a tie sent the line 2.5 mi north of Red Hill, where the
    # webcam had it. Gaps: 330-20 splits at 355, 140-180 at 160.
    def lean(d):
        if d is None or pd.isna(d):
            return None
        d = d % 360
        if d >= 330 or d < 20:
            return "trades" if (d >= 355 or d < 20) else "upslope"
        if 140 < d < 180:
            return "trades" if d < 160 else "upslope"
        return None
    leans = [lean(d) if sd == "shear" else None for d, sd in zip(f.wind_dir, side)]
    best, k_best = -1, None
    for k in range(1, len(f)):
        score = (side[:k].count("trades") + side[k:].count("upslope")
                 + 0.5 * leans[:k].count("trades") + 0.5 * leans[k:].count("upslope"))
        if score > best:
            best, k_best = score, k
    if "trades" not in side[:k_best] or "upslope" not in side[k_best:]:
        return {"found": False, "why": f"no transition in the ladder; all stations read {sorted(set(side))}"}
    north, south = f.loc[k_best - 1], f.loc[k_best]
    return {"found": True, "north_lat": north.lat, "south_lat": south.lat,
            "north_of": north["name"], "south_of": south["name"],
            "landmarks": [m for m, lat in cv.LANDMARKS if south.lat <= lat <= north.lat],
            "in_shear": south.side == "shear",
            "span_miles": abs(north.lat - south.lat) * cv.MILES_PER_DEG,
            "as_of": f.observed_local.max(), "dropped": dropped}


# Where the traced line meets Kula Highway (Pukalani-Omaopio-Waiakoa run
# close to this longitude).
KULA_HWY_LON = -156.333


def line_now(obs: pd.DataFrame, now: datetime):
    """The traced line (Sam, 2026-10-21: "build it"), as the page's bracket:
    where the anchor->origin line crosses Kula Highway. The ladder's best split
    stands in when the trace finds no line."""
    lad = smoothed_ladder(obs, now)
    if lad.empty:
        return None, lad
    tr = safely(trace_line, station_rows(obs, lad))
    if not tr or not tr.get("found"):
        got = best_split(lad)
        if got.get("found"):
            got["traced"] = False
        return got, lad
    (a_lat, a_lon), (o_lat, o_lon) = tr["anchor_ll"], tr["origin_ll"]
    lat = (a_lat if o_lon == a_lon else
           a_lat + (o_lat - a_lat) * (KULA_HWY_LON - a_lon) / (o_lon - a_lon))
    return {"found": True, "north_lat": lat, "south_lat": lat,
            "north_of": tr["origin"], "south_of": tr["anchor"], "traced": True, "trace": tr,
            "landmarks": [], "in_shear": False, "span_miles": 0.0,
            "as_of": lad.observed_local.max(), "dropped": 0}, lad


# -- forecast ------------------------------------------------------------------

def outlook() -> pd.DataFrame:
    return cv.soaring()


def base_by_hour() -> dict:
    """Forecast cloudbase at fixed hours, 11:00 and 14:00, by day, adjusted.

    soaring() reports the base at each day's strongest hour, which moved
    between 11:00 and 14:00 across one week -- not comparable day to day,
    and it hid afternoon drops of up to 1,500 ft. 11:00 is about when
    Kula's base peaks; 14:00 shows how far it comes down.
    """
    out = {}
    for day, part in cv.slope_parcel().groupby("day"):
        got = th.assess(part, RELEASE_FT)
        out[day] = tuple(round(got.loc[h, "lcl_ft"] + FORECAST_BASE_LOW_FT, -2)
                         if h in got.index else None for h in (11, 14))
    return out


def am_pm(pair) -> str:
    if not pair or pair[0] is None:
        return "--"
    am, pm = pair
    return ft(am) if pm is None or pm == am else f"{ft(am)} → {ft(pm)}"


# Summit cloud pushed over Kula (Sam, 2026-10-15; site lore: "early clouds on
# the summit mean a short day in Kula"). On 10-15 trade moisture lifted up the
# windward slope came over the ridge on an easterly and shaded Kula from
# 15:00. The forecast column is Kahului's free air, which cannot show the
# slope's own lifting, so the test is orographic: would the 6,000-8,500 ft
# trade-layer air condense before an 8,500 ft crest (LCL = z + 410 ft per C of
# spread), with an easterly >= 5 kt at 8,000-10,000 ft and no inversion below
# 10,000 ft to hold the tops under the ridge. RRFS 18Z, 2026-08-14..10-15:
# flagged 13 of 63 afternoons and caught 4 of the 5 measured intrusions
# (Summit moist on an easterly, Kula Ag's sun halving), so "possible" only.
SUMMIT_CREST_FT = 8500
SUMMIT_EAST_KT = 5
# The forecast day's summit-cloud call, for Sam's session-start reminder to
# check the webcam while its short loop still holds the morning
# (KulaModel/.claude/hooks/session_alert.py reads it from the live site).
# One day only, overwritten each build: "If I catch the note, great. If
# not, next time" (Sam) -- no history kept.
ALERTS_FILE = HERE / "docs" / "alerts.json"


def log_alert(day, push, now: datetime) -> None:
    ALERTS_FILE.write_text(json.dumps({f"{day}": {"summit_push_from": push,
                                                  "updated": f"{now:%Y-%m-%d %H:%M}"}}),
                           encoding="utf-8")


def summit_push(path, day) -> int | None:
    """First hour, 12:00-16:00 HST on day, when the RRFS column says summit
    cloud could be pushed over Kula; None if it never does."""
    prof = sd.profiles(path)
    for h in (12, 13, 14, 15, 16):
        key = (pd.Timestamp(day) + pd.Timedelta(hours=h + 10)).tz_localize("UTC")
        if key not in prof:
            continue
        q = prof[key]
        lay = q[(q.height_ft >= 6000) & (q.height_ft <= 8500)]
        ridge = q[(q.height_ft >= 8000) & (q.height_ft <= 10000)]
        if lay.empty or ridge.empty:
            continue
        lcl = (lay.height_ft + 410 * (lay.temp_c - lay.dewpoint_c)).min()
        east = (ridge.wind_kt * np.sin(np.radians(ridge.wind_dir))).mean()
        if lcl <= SUMMIT_CREST_FT and east >= SUMMIT_EAST_KT and not sd.inversion(q):
            return h
    return None


# The Ceiling line's sounding: NAM, then RRFS once NAM retires (check.py does
# the same). Against Hilo and Lihue balloons (Jan-Oct 2026, 14:00 HST) NAM put
# the trade inversion within ~75-280 ft and RRFS ~200-315 ft (20 afternoons),
# where GFS was 560-1,700 ft too low and often missed it; GFS stays for sun.
def cap_model(now: datetime) -> str:
    utc = now.astimezone(timezone.utc).replace(tzinfo=None)
    return "NAM" if utc < sd.RETIRED["NAM"] else "RRFS"


def inversion_on(day, path, model):
    found = []
    for h in ("11:00", "14:00"):
        try:
            inv = sd.inversion(sd.at(pd.Timestamp(f"{day} {h}"), path=path, model=model), floor_ft=4000)
            found.append(None if inv is None else {k: float(v) for k, v in inv.items()})
        except Exception:
            found.append(None)
    return found


# The forecasts change on model cycles a few times a day, not every 15
# minutes. Refetching them each build only added chances for a timeout, and
# one blanked every forecast cloudbase at 13:31 on 2026-10-12. So they are
# refreshed hourly, saved beside the page, and reused when a fetch fails.
FORECAST_FILE = HERE / "docs" / "forecast.json"
FORECAST_MAX_AGE = timedelta(minutes=60)


# The forecast thermal word, like Today's: thermal.py on Open-Meteo's
# forecast air for Kula Ag, warmed 3 F (it runs cool), through the NAM/RRFS
# sounding where its run reaches (~3.5 days) and GFS beyond. Tested 30 days
# (9/14-10/13) against the measured-air word through NAM: same word 47%,
# within one category every day, no lean; raw air leaned 0.7 category weak.
FORECAST_AIR_WARM_F = 3.0


def forecast_words(cap) -> dict:
    parcel = cv.slope_parcel()
    gfs = safely(sd.fetch, "GFS")
    out = {}
    for day, part in parcel.groupby("day"):
        part = part.assign(temp_f=part.temp_f + FORECAST_AIR_WARM_F)
        for path in (cap, gfs):
            if path is None:
                continue
            got = th.assess(part, RELEASE_FT, window=(10, 14), path=path)
            if not got.empty:
                out[day] = strength_word(th.describe(got.ti_c.mean()))
                break
    return out


def fresh_forecast(now: datetime, cap) -> dict:
    week = outlook()
    words = safely(forecast_words, cap) or {}
    week["thermal"] = [words.get(d, "--") for d in week.index]
    hours = base_by_hour()
    days = [now.date(), now.date() + timedelta(days=1)]
    return {
        "made": now.isoformat(),
        "week": [{"day": str(d), "dir_deg": None if pd.isna(r["dir_deg"]) else float(r["dir_deg"]),
                  "thermal": str(r["thermal"]),
                  "sun": None if pd.isna(r["sun"]) else float(r["sun"]),
                  "line": None if r["shear_location"] is None or pd.isna(r["shear_location"])
                  else str(r["shear_location"])}
                 for d, r in week.iterrows()],
        "hours": {str(d): list(v) for d, v in hours.items()},
        "inversions": {str(d): inversion_on(d, cap, cap_model(now)) if cap else [None, None] for d in days},
    }


def forecast(now: datetime, cap) -> dict | None:
    """Last saved forecast if under an hour old, else a fresh one, else the saved one."""
    saved = None
    try:
        saved = json.loads(FORECAST_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if saved and now - datetime.fromisoformat(saved["made"]) < FORECAST_MAX_AGE:
        return saved
    got = safely(fresh_forecast, now, cap)
    if got and got["week"] and any(v[0] is not None for v in got["hours"].values()):
        FORECAST_FILE.write_text(json.dumps(got), encoding="utf-8")
        return got
    return saved


def unpack(fc: dict | None, now: datetime):
    """Saved forecast back into the shapes the page code uses."""
    if not fc:
        return None, {}, [None, None], [None, None], None
    week = pd.DataFrame([{**w, "shear_location": w["line"]} for w in fc["week"]])
    week.index = pd.to_datetime(week.pop("day")).dt.date
    hours = {datetime.fromisoformat(d).date(): tuple(v) for d, v in fc["hours"].items()}
    inv = fc.get("inversions", {})
    return (week, hours, inv.get(str(now.date()), [None, None]),
            inv.get(str(now.date() + timedelta(days=1)), [None, None]),
            datetime.fromisoformat(fc["made"]))


# -- page ------------------------------------------------------------------------

def ft(x) -> str:
    return "--" if x is None or pd.isna(x) else f"{x:,.0f}"


def pct(x) -> str:
    return "--" if x is None or pd.isna(x) else f"{x:.0%}"


def chip(kind: str, text: str) -> str:
    return f'<span class="chip {kind}">{esc(text)}</span>'


def rating_chip(word):
    return {"Good": chip("good", "Good"), "OK": chip("fair", "OK"), "Nope": chip("warn", "Nope"),
            "Strong": chip("good", "Strong"), "Moderate": chip("good", "Moderate"),
            "Light": chip("fair", "Light"), "Flat": chip("warn", "Flat")}.get(word, "")


# Today's rating is thermal.py again (Sam, 2026-10-14: sunlight alone "too
# simplistic"): Kula Ag's measured air through the NAM / RRFS Kahului
# sounding -- the model Weather Guesser now uses -- in its words, with
# "Boom!" folded into Strong (it came too easily). Sunlight is the check
# that thermal.py lacks: under 55% of clear sky the slope isn't heating
# whatever the air says (9/30: overcast, warm air, thermal.py said Light).
def strength_word(verdict) -> str:
    return "Strong" if verdict == "Boom!" else str(verdict)


def day_chip(verdict, sun):
    if sun is not None and not pd.isna(sun) and sun < 0.55:
        return chip("warn", "Shaded")
    if verdict in ("Flat",):
        return chip("warn", "Weak")
    if verdict in GOOD_VERDICTS and (sun is None or pd.isna(sun) or sun >= 0.7):
        return chip("good", "Good")
    return chip("fair", "Fair")


# The station map (Sam, 2026-10-14: "for identification of stations referenced
# in the report(s)"). Each build writes every station's position and latest
# wind; the page draws them over Esri's topographic basemap with Leaflet.
COAST_POS = {"Paia": (20.9380, -156.3478), "Hamakuapoko Rd": (20.9200, -156.3410),
             "Peahi Rd": (20.9076, -156.2896)}
STATIONS_FILE = HERE / "docs" / "stations.json"


def write_station_map(obs: pd.DataFrame, lad: pd.DataFrame, now: datetime, pos=None, west=None) -> None:
    rows = station_rows(obs, lad)
    _write_map(rows, now, pos, west)


def station_rows(obs: pd.DataFrame, lad: pd.DataFrame) -> list:
    rows = []
    fresh = obs.observed.max() - pd.Timedelta("90min") if len(obs) else None
    for st in sampler.STATIONS + MAP_ONLY:
        o = obs[(obs.station_id == st.id)] if len(obs) else obs
        o = o[o.observed >= fresh].dropna(subset=["wind_mph"]) if len(o) else o
        last = o.loc[o.observed.idxmax()] if len(o) else None
        rows.append({"name": st.name, "lat": st.lat, "lon": st.lon, "elev_ft": int(st.elev_ft),
                     "dir": None if last is None or pd.isna(last.wind_dir) else round(float(last.wind_dir)),
                     "mph": None if last is None else round(float(last.wind_mph)),
                     "at": None if last is None else f"{last.observed:%H:%M}"})
    for name, (lat, lon) in COAST_POS.items():
        r = lad[lad.name == name] if len(lad) else lad
        last = r.iloc[0] if len(r) else None
        rows.append({"name": name, "lat": lat, "lon": lon, "elev_ft": None,
                     "dir": None if last is None else round(float(last.wind_dir)),
                     "mph": None if last is None else round(float(last.wind_mph)),
                     "at": None if last is None else f"{last.observed_local:%H:%M}"})
    for sid, name, lat, elev, lon in EXTRA_LADDER:
        r = lad[lad.name == name] if len(lad) else lad
        last = r.iloc[0] if len(r) else None
        rows.append({"name": name, "lat": lat, "lon": lon, "elev_ft": elev,
                     "dir": None if last is None else round(float(last.wind_dir)),
                     "mph": None if last is None else round(float(last.wind_mph)),
                     "at": None if last is None else f"{last.observed_local:%H:%M}"})
    r = lad[lad.name == KEKOA[1]] if len(lad) else lad
    last = r.iloc[0] if len(r) else None
    rows.append({"name": KEKOA[1], "lat": KEKOA[2], "lon": KEKOA_LON, "elev_ft": KEKOA[3],
                 "dir": None if last is None else round(float(last.wind_dir)),
                 "mph": None if last is None else round(float(last.wind_mph)),
                 "at": None if last is None else f"{last.observed_local:%H:%M}"})
    return rows


def _write_map(rows: list, now: datetime, pos, west) -> None:
    # The shear line, drawn as a stylised stroke through the middle of the
    # bracket (Sam liked it with "a little artistic license"); none when the
    # page reports no line.
    line = None
    if pos and pos.get("found"):
        (_, nn), (_, sn) = _place(pos["north_lat"]), _place(pos["south_lat"])
        nn = COAST_STATION.get(pos["north_of"], nn)
        line = {"mid": (pos["north_lat"] + pos["south_lat"]) / 2,
                "label": "Shear line · " + (nn if nn == sn else f"{nn}–{sn}"),
                # Trade air at Pulehu (the west-slope alert): the line swings
                # south round the lower west slope, passing below Pulehu (Sam).
                # Always south of where the line would otherwise end: once the
                # line itself is south of Pulehu, "below Pulehu" bent it north.
                "west_lat": None}
        if west:
            west_end = line["mid"] - 0.006          # the template's plain west end
            line["west_lat"] = min(20.7953 - 0.005, west_end - 0.005)   # Sam: less bend
    STATIONS_FILE.write_text(json.dumps({"made": f"{now:%H:%M}", "built": f"{now:%Y-%m-%dT%H:%M}",
                                         "stations": rows, "line": line}),
                             encoding="utf-8")


# The traced line (Sam, 2026-10-21, "Shearline analysis needs work. New
# plan."): (1) walk the lowest band north to south to the first backwind --
# anything not in the trades, since in the shear the direction can read
# "other" -- the anchor; (2) the highest station above it still in the
# trades, the origin; (3) the line runs from anchor to origin, and odd
# readings near it are expected and help confirm it. Logged beside the
# ladder's bracket to compare, not on the page yet.
TRACE_LOG = HERE / "docs" / "line_trace_log.json"
LOW_BAND_FT = 2400           # coast to ~2,400 ft: Paia ... MECO Kula 2
CORRIDOR_TOP_FT = 4500       # Crater Rd; above it the summit stations sit over the trade layer
TRACE_NEAR_MI = 1.0
# Trades proper, for the origin and the stations near the line (Sam, 10-21:
# "Red Hill is not in the trades. 1. Note wind speed 2. Note direction is not
# trades" -- 119-132 deg at 7-8 mph -- then: "There is no minimum speed for
# trades, use a comparison to other 'true' trade reports. Same applies to
# direction. Remember, under the shear the wind can be variable."). The true
# trades are the low band north of the anchor; a station is in them when its
# direction is within TRACE_DIR_DEG of their median and its speed at least
# TRACE_SPEED_FRAC of theirs. The anchor's test stays any non-trade direction.
TRACE_DIR_DEG = 45
TRACE_SPEED_FRAC = 0.5


def trace_line(rows: list) -> dict:
    live = [r for r in rows if r.get("dir") is not None and r.get("mph") is not None]
    trades = lambda r: cv._side(r["dir"]) == "trades"
    elev = lambda r: r["elev_ft"] or 0
    low = sorted((r for r in live if elev(r) <= LOW_BAND_FT), key=lambda r: -r["lat"])
    anchor, seen = None, False
    for r in low:
        if trades(r):
            seen = True
        elif seen:
            anchor = r
            break
    if anchor is None:
        return {"found": False, "why": "no backwind in the low band" if seen else "no trades in the low band"}
    ref = [r for r in low if r["lat"] > anchor["lat"] and trades(r)]
    ref_dir = cv.mean_direction(pd.Series([r["dir"] for r in ref], dtype=float))
    ref_mph = float(np.median([r["mph"] for r in ref]))
    solid = lambda r: (abs((r["dir"] - ref_dir + 180) % 360 - 180) <= TRACE_DIR_DEG
                       and r["mph"] >= TRACE_SPEED_FRAC * ref_mph)
    # East of the anchor (Sam, 2026-10-21: "For the origin, try highest station
    # east of anchor").
    up = [r for r in live if solid(r) and r["lon"] > anchor["lon"]
          and elev(anchor) < elev(r) <= CORRIDOR_TOP_FT]
    if not up:
        return {"found": False, "why": "no trades above the anchor", "anchor": anchor["name"]}
    # Highest; of stations within 300 ft of the highest, the southernmost --
    # Olinda Rd (4,021) and Waihou (4,011) both in the trades means the line
    # passes south of Waihou.
    top = max(elev(r) for r in up)
    origin = min((r for r in up if elev(r) >= top - 300), key=lambda r: r["lat"])
    # Stations near the line: which side of it they sit, and what they read.
    (a_lat, a_lon), (o_lat, o_lon) = (anchor["lat"], anchor["lon"]), (origin["lat"], origin["lon"])
    k = math.cos(math.radians(a_lat))
    dx, dy = (o_lon - a_lon) * k, o_lat - a_lat
    near = []
    for r in live:
        if r is anchor or r is origin:
            continue
        px, py = (r["lon"] - a_lon) * k, r["lat"] - a_lat
        t = (px * dx + py * dy) / (dx * dx + dy * dy)
        if not 0 <= t <= 1:
            continue
        off = (px * dy - py * dx) / math.hypot(dx, dy)       # > 0: south/right of anchor->origin
        if abs(off) * cv.MILES_PER_DEG > TRACE_NEAR_MI:
            continue
        side = "south" if off > 0 else "north"
        fits = solid(r) if side == "north" else not trades(r)
        near.append({"name": r["name"], "side": side, "dir": r["dir"], "mph": r["mph"],
                     "miles": round(abs(off) * cv.MILES_PER_DEG, 2), "fits": fits})
    return {"found": True, "trades": [round(ref_dir), round(ref_mph, 1)],
            "anchor": anchor["name"], "anchor_dir": anchor["dir"],
            "anchor_ll": [a_lat, a_lon], "origin": origin["name"], "origin_dir": origin["dir"],
            "origin_ll": [o_lat, o_lon], "near": near}


def log_trace(now: datetime, pos) -> None:
    today = f"{now:%Y-%m-%d}"
    try:
        log = json.loads(TRACE_LOG.read_text(encoding="utf-8"))
        rows = json.loads(STATIONS_FILE.read_text(encoding="utf-8"))["stations"]
    except Exception:
        return
    entries = log.get("entries", []) if log.get("date") == today else []
    got = trace_line(rows)
    got["t"] = f"{now:%H:%M}"
    got["ladder"] = ([str(pos["north_of"]), str(pos["south_of"])] if pos and pos.get("found") else None)
    entries.append(got)
    TRACE_LOG.write_text(json.dumps({"date": today, "entries": entries}), encoding="utf-8")


# -- the day in review -----------------------------------------------------------
# Once the upslope is over the live rows describe the drainage, not the flying
# day, so the Today panel shows the whole day instead and keeps it through the
# night until the next upslope begins; the Forecast panel has the new day by
# then (Sam, 2026-10-15).

REVIEW_FILE = HERE / "docs" / "review.json"
LINE_LOG = HERE / "docs" / "line_log.json"


def log_line(now: datetime, pos) -> list:
    """Add this update's shear-line bracket to today's log; return today's entries."""
    today = f"{now:%Y-%m-%d}"
    try:
        log = json.loads(LINE_LOG.read_text(encoding="utf-8"))
    except Exception:
        log = {}
    entries = log.get("entries", []) if log.get("date") == today else []
    if pos and pos.get("found"):
        entries.append({"t": f"{now:%H:%M}", "north_lat": float(pos["north_lat"]),
                        "south_lat": float(pos["south_lat"]), "north_of": str(pos["north_of"]),
                        "south_of": str(pos["south_of"]), "dropped": int(pos.get("dropped", 0))})
    LINE_LOG.write_text(json.dumps({"date": today, "entries": entries}), encoding="utf-8")
    return entries


def upslope_end(rnet, evening):
    """When the day's upslope ended: the start of the last net-radiation dip
    under 100 W/m2 before it turned negative; else the evening state's time."""
    if rnet is not None and len(rnet):
        r = rnet.rolling(2).mean().shift(-1).dropna()
        r = r[r.index.hour >= 15]
        neg = r.index[(r < 0).values]
        if len(neg):
            r = r[r.index < neg[0]]
            above = r.index[(r >= 100).values]
            low = r[r.index > above[-1]] if len(above) else r
            return low.index[0] if len(low) else neg[0]
    return evening[1] if evening else None


def line_moves(entries: list) -> str | None:
    """The day's shear-line landmarks in words (Sam, 2026-10-20: the wrap-up
    should mention shear movement): where it sat most of the day, and its
    furthest north and south with times, when they differ."""
    if not entries:
        return None
    text = lambda e: line_between(e["north_lat"], e["south_lat"], e["north_of"], e["south_of"])
    texts = [text(e) for e in entries]
    common = max(dict.fromkeys(texts), key=texts.count)
    south = min(entries, key=lambda e: e["south_lat"])
    north = max(entries, key=lambda e: e["north_lat"])
    out = f"Most of the day the shear line was {common}"
    for label, e in (("furthest north", north), ("furthest south", south)):
        if text(e) != common:
            out += f"; {label} about {e['t']}, {text(e)}"
    return out + "."


THERM_HOURS = None      # thermal_today() for the review's thermal landmarks


def thermal_moves(hours) -> str | None:
    """Thermal landmarks for the wrap-up (Sam, 2026-10-20): when the measured
    rating reached its strongest, or that it stayed light, 10:00-15:00."""
    if hours is None or len(hours) == 0 or "ti_c" not in hours:
        return None
    h = hours[(hours.index >= 10) & (hours.index <= 15)].ti_c.dropna()
    if h.empty:
        return None
    words = {int(k): th.describe(v) for k, v in h.items()}
    order = ["Flat", "Light", "Moderate", "Strong", "Boom!"]
    best = max(words.values(), key=order.index)
    first = min(k for k, w in words.items() if w == best)
    last = max(k for k, w in words.items() if w == best)
    if best in ("Flat", "Light"):
        return f"Thermals stayed {best.lower()} all day."
    span = f"about {first:02d}:00" + (f" to {last + 1:02d}:00" if last > first else "")
    return f"Thermals were strongest ({best.lower()}) {span}."


def line_day(entries: list):
    """(where it sat most of the day, the furthest south and when) from the log."""
    if not entries:
        return None
    texts = [line_between(e["north_lat"], e["south_lat"], e["north_of"], e["south_of"]) for e in entries]
    common = max(dict.fromkeys(texts), key=texts.count)
    far = min(entries, key=lambda e: e["south_lat"])
    far_text = line_between(far["north_lat"], far["south_lat"], far["north_of"], far["south_of"])
    return common, (None if far_text == common else (far_text, far["t"]))


# The review's cloudbase chart (Sam, 2026-10-15: "a direct comparison
# between derived and predicted would create (or destroy) interest"; sun
# dropped). The forecast is the one a pilot saw: saved each morning update
# until the upslope begins, then frozen for the day.
MORNING_BASE_FILE = HERE / "docs" / "forecast_base.json"


@functools.lru_cache(maxsize=1)
def _slope_parcel():
    return cv.slope_parcel()


def forecast_base_hours(day) -> dict:
    """Forecast cloudbase by hour, with the same +400 ft as the page, only
    from the first to the last hour thermal.py has thermals working: before
    and after, a condensation level is no cloudbase (Sam: "meaningless until
    thermal activity begins")."""
    part = _slope_parcel()
    part = part[part.day == day]
    got = th.assess(part, RELEASE_FT, window=(8, 17))
    on = got.index[got.working.astype(bool)] if "working" in got else got.index
    if not len(on):
        return {}
    got = got.loc[on.min():on.max()]
    return {int(h): round(float(v) + FORECAST_BASE_LOW_FT, -1) for h, v in got.lcl_ft.items()
            if not pd.isna(v)}


def forecast_cu_end(day):
    """The hour the forecast thermals stop working for the rest of the day
    (no cumulus from then), or None if they run to 16:00."""
    part = cv.slope_parcel()
    part = part[part.day == day]
    got = th.assess(part, RELEASE_FT, window=(8, 17))
    on = got.index[got.working.astype(bool)] if "working" in got else got.index
    # Only when at least an hour is left before 16:00, where the charts end.
    return int(on.max()) + 1 if len(on) and on.max() < 15 else None


def core_stations() -> list:
    """The slope stations behind the derived base (as _core selects them)."""
    return [st for st in sampler.STATIONS
            if st.id not in OFF_SLOPE and st.id != "TT925" and st.elev_ft <= 4400]


def station_forecast_lcl(day) -> dict:
    """Forecast LCL by hour (09-15) at each core station's own position and
    elevation, one Open-Meteo request for all (sampler's formula: station
    elevation + 1000/4.5 ft per F of spread)."""
    sts = core_stations()
    r = requests.get(cv.OPEN_METEO, timeout=60, params={
        "latitude": ",".join(f"{st.lat:.5f}" for st in sts),
        "longitude": ",".join(f"{st.lon:.5f}" for st in sts),
        "elevation": ",".join(str(round(st.elev_ft / 3.28084)) for st in sts),
        "hourly": "temperature_2m,dew_point_2m", "timezone": "Pacific/Honolulu",
        "temperature_unit": "fahrenheit", "start_date": f"{day}", "end_date": f"{day}"}).json()
    out = {}
    for st, loc in zip(sts, r if isinstance(r, list) else [r]):
        hh = loc["hourly"]
        out[st.id] = {int(t[11:13]): round(st.elev_ft + (tt - td) / 4.5 * 1000, -1)
                      for t, tt, td in zip(hh["time"], hh["temperature_2m"], hh["dew_point_2m"])
                      if 9 <= int(t[11:13]) <= 15 and tt is not None and td is not None}
    return out


def morning_inversions(now: datetime) -> dict:
    """{hour: [base_ft, top_ft]} 09-15 from the morning's cap-model sounding
    (Kahului), for the capped-day log (Sam, 2026-10-17: "get a few more capped
    days logged")."""
    prof = sd.profiles(sd.fetch(cap_model(now)))
    out = {}
    for h in range(9, 16):
        k = (pd.Timestamp(now.date()) + pd.Timedelta(hours=h + 10)).tz_localize("UTC")
        if k in prof and (inv := sd.inversion(prof[k])):
            out[h] = [round(inv["base_ft"]), round(inv["top_ft"])]
    return out


# The latest forecast for each hour, beside the morning save (Sam, 2026-10-17:
# "Are we recording latest model cloudbase to compare with derived?"). Each
# build recomputes today's forecast base and keeps it only for hours not yet
# begun -- Open-Meteo's past hours are partly filled from what happened, which
# would flatter the score. Tells whether later runs fix the morning's misses
# (stale forecast) or repeat them (model bias a correction can fix).
FORECAST_LATEST_FILE = HERE / "docs" / "forecast_latest.json"


def log_forecast_latest(now: datetime) -> None:
    if not 6 <= now.hour < 16:
        return
    d = f"{now:%Y-%m-%d}"
    try:
        fl = json.loads(FORECAST_LATEST_FILE.read_text(encoding="utf-8"))
        if fl.get("date") != d:
            raise ValueError
    except Exception:
        fl = {"date": d, "hours": {}, "made": {}}
    hours = forecast_base_hours(now.date()) or {}
    for h, v in hours.items():
        if h > now.hour:
            fl["hours"][str(h)] = v
            fl["made"][str(h)] = f"{now:%H:%M}"
    FORECAST_LATEST_FILE.write_text(json.dumps(fl), encoding="utf-8")


def save_morning_forecast(now: datetime) -> None:
    hours = forecast_base_hours(now.date())
    if not hours:
        return
    part = cv.slope_parcel()
    part = part[(part.day == now.date()) & part.hour.between(9, 16)]
    rrfs = safely(sd.fetch, "RRFS")
    MORNING_BASE_FILE.write_text(json.dumps({
        "date": f"{now:%Y-%m-%d}", "made": f"{now:%H:%M}", "hours": hours,
        "temp": {int(r.hour): float(r.temp_f) for r in part.itertuples()},
        "dew": {int(r.hour): float(r.dewpoint_f) for r in part.itertuples()},
        "sun": None if (fs := safely(cv.sun_forecast)) is None or now.date() not in fs.index
               else float(fs.loc[now.date()]),
        "summit_push": safely(summit_push, rrfs, now.date()) if rrfs else None,
        "stations": safely(station_forecast_lcl, now.date()),
        "inv": safely(morning_inversions, now)}), encoding="utf-8")


# -- the automated review (Sam, 2026-10-16: "Can your review be automated?") --
# Each evening the day review also writes the day's scorecard: hour rows
# (09-15) and one day row, kept in docs/ for good. Every build then
# recomputes a short summary that KulaModel's session-start hook reports --
# flags only, and it never changes a correction itself.
SCORE_HOURS = HERE / "docs" / "scorecard_hours.csv"
SCORE_DAYS = HERE / "docs" / "scorecard_days.csv"
PATTERN_LOG = HERE / "docs" / "patterns_log.json"

# The FAA cameras' image analytics -- visibility and sky cover every 10 min --
# as a CHECK on the stations, not shown on the page (Sam, 2026-10-17: "Use it
# as a check on other sources"). Puunianiau (6,854 ft) in cloud means the
# deck's top reaches it, against Park HQ's humidity a mile away; Red Hill
# (2,819 ft) in cloud means the base is down to it, against the derived base.
# The FAA keeps ~6 h, so every build saves what it sees.
CAMERA_LOG = HERE / "docs" / "camera_log.csv"
CAMERAS = {"Puunianiau": (850, 6854), "Red Hill": (931, 2819)}
CAM_IN_CLOUD_MI = 3          # visibility below this = camera in cloud
CAM_MIN_CONFIDENCE = 55      # the software's own confidence, per cent


def log_cameras(now: datetime) -> None:
    rows = []
    for name, (sid, elev) in CAMERAS.items():
        page = f"https://weathercams.faa.gov/cameras/state/HI/cameraSite/{sid}/summary"
        r = requests.get(KEKOA_URL, params={"siteId": sid}, timeout=30,
                         headers={**cv.HEAD, "Referer": page})
        r.raise_for_status()
        for v in (r.json().get("payload") or {}).get("site", {}).get("visibilities") or []:
            if v.get("visibilityStatuteMi") is None:
                continue
            t = pd.Timestamp(v["processedTime"]).tz_convert(HST).tz_localize(None)
            sc = v.get("skyCondition") or {}
            rows.append({"time": f"{t:%Y-%m-%d %H:%M}", "camera": name, "elev_ft": elev,
                         "vis_mi": v["visibilityStatuteMi"], "confidence": v.get("confidence"),
                         "cloud_pct": sc.get("cloudPercentage")})
    if not rows:
        return
    old = pd.read_csv(CAMERA_LOG) if CAMERA_LOG.exists() else pd.DataFrame()
    allr = pd.concat([old, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(["time", "camera"], keep="last")
    cut = f"{now - timedelta(days=60):%Y-%m-%d}"
    allr[allr.time >= cut].sort_values(["time", "camera"]).to_csv(CAMERA_LOG, index=False)


def camera_now(camera: str):
    """True/False: the camera's latest confident reading (within 40 min) puts
    it in cloud; None if there is none."""
    if not CAMERA_LOG.exists():
        return None
    c = pd.read_csv(CAMERA_LOG)
    c = c[(c.camera == camera) & (c.confidence >= CAM_MIN_CONFIDENCE)]
    if c.empty:
        return None
    last = c.iloc[-1]
    if pd.Timestamp(last.time) < datetime.now(HST).replace(tzinfo=None) - timedelta(minutes=40):
        return None
    return bool(last.vis_mi < CAM_IN_CLOUD_MI)


def camera_cloud_minutes(day: str, camera: str):
    """Minutes, 09:00-16:00, the camera's analytics put it in cloud (None if no data)."""
    if not CAMERA_LOG.exists():
        return None
    c = pd.read_csv(CAMERA_LOG)
    c = c[(c.camera == camera) & c.time.str.startswith(day)]
    c = c[(c.time.str[11:16] >= "09:00") & (c.time.str[11:16] < "16:00")]
    if len(c) < 12:
        return None
    ok = c[c.confidence >= CAM_MIN_CONFIDENCE]
    return int(10 * (ok.vis_mi < CAM_IN_CLOUD_MI).sum())


def summit_spill(now: datetime, obs: pd.DataFrame):
    """{"su_rh", "su_dir", "dew"} for P8 once the 08:59 summit reading is in
    (Mesonet runs ~15 min behind): the Summit station 07:00-08:59 and whether
    Kula Ag started saturated (95%+ at 06:00-06:59)."""
    if now.hour * 60 + now.minute < 9 * 60 + 15:
        return None
    r = mesonet.get("/mesonet/db/measurements", location="hawaii", station_ids="0153",
                    var_ids="RH_1_Avg,WDrs_1_Avg,WS_1_Avg",
                    start_date=f"{now:%Y-%m-%d}T07:00:00-10:00", end_date=f"{now:%Y-%m-%d}T08:59:59-10:00",
                    row_mode="json", limit=5000)
    x = pd.DataFrame(r)
    x["value"] = pd.to_numeric(x.value)
    v = x.pivot_table(index="timestamp", columns="variable", values="value")
    rad = np.radians(v.WDrs_1_Avg)
    u, vv = float((v.WS_1_Avg * np.sin(rad)).mean()), float((v.WS_1_Avg * np.cos(rad)).mean())
    ka = obs[(obs.station_id == "001HI") & (obs.observed.dt.hour == 6)] if len(obs) else obs
    dew = bool(len(ka) and float(_rh(ka.temp_f, ka.dew_f).mean()) >= 95)
    return {"su_rh": float(v.RH_1_Avg.mean()), "su_dir": float(np.degrees(np.arctan2(u, vv)) % 360), "dew": dew}


def log_patterns(now: datetime, fired: list) -> dict:
    """{id: first HH:MM it fired today}, kept for the day's scorecard."""
    today = f"{now:%Y-%m-%d}"
    try:
        log = json.loads(PATTERN_LOG.read_text(encoding="utf-8"))
    except Exception:
        log = {}
    seen = log.get("fired", {}) if log.get("date") == today else {}
    for pid, _, _ in fired:
        seen.setdefault(pid, f"{now:%H:%M}")
    PATTERN_LOG.write_text(json.dumps({"date": today, "fired": seen}), encoding="utf-8")
    return seen
# Per-station forecast vs measured LCL, for a per-station forecast base once
# 6-8 weeks are logged (Sam, 2026-10-16: test on 31 days had per-station
# corrections ~12% better than the flat +400, but too few days to trust).
SCORE_STATIONS = HERE / "docs" / "scorecard_stations.csv"
STATION_DAYS_NEEDED = 42
REVIEW_SUMMARY = HERE / "docs" / "review_summary.json"


def _replace_day(path, rows: list, day: str) -> None:
    old = pd.read_csv(path, dtype={"date": str}) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[old.date != day]
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


# The page's own calls, kept for the monthly check (monthly.py): what the
# Forecast panel said the evening before each day.
CALLS_LOG = HERE / "docs" / "calls_log.json"


def log_call(day, base: str, thermal: str, now: datetime) -> None:
    try:
        log = json.loads(CALLS_LOG.read_text(encoding="utf-8"))
    except Exception:
        log = {}
    # The scored call is the one standing at 18:00 the evening before, when
    # pilots plan (Sam, 2026-10-20: "use 1800"); later runs no longer replace
    # it. Before 18:00 each build updates it, so a missed evening still leaves
    # the latest daytime call.
    old = log.get(f"{day}") or {}
    if str(old.get("made", "")) >= f"{day - timedelta(days=1)} 18:00":
        return
    log[f"{day}"] = {"base": base, "rule": CALL_RULE.get(f"{day}"), "thermal": thermal,
                     "made": f"{now:%Y-%m-%d %H:%M}"}
    keep = sorted(log)[-120:]
    CALLS_LOG.write_text(json.dumps({k: log[k] for k in keep}, indent=0), encoding="utf-8")


def rules_line() -> str:
    """A quiet line saying the rules are still being checked (monthly.py)."""
    st = rules.status()
    if not st.get("checked"):
        return ""
    when = datetime.strptime(st["checked"], "%Y-%m")
    off = [f"{rules.RULES[k]['name']}" for k, v in st.get("rules", {}).items()
           if not v.get("active", True) and k in rules.RULES]
    tail = "all holding" if not off else "switched off: " + ", ".join(off)
    return f'<div class="sub">Forecast rules last checked {when:%B %Y}: {tail}.</div>'


def write_scorecard(obs, now, up, end, word, entries) -> None:
    day = now.replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
    d = f"{day:%Y-%m-%d}"
    try:
        mf = json.loads(MORNING_BASE_FILE.read_text(encoding="utf-8"))
        mf = mf if mf.get("date") == d else {}
    except Exception:
        mf = {}
    core = _core(obs)
    core = core[core.observed.dt.date == day.date()].copy()
    core["h"] = core.observed.dt.hour
    derived = (core.groupby(["h", "station_id"]).lcl_ft.mean().groupby("h").median()
               if len(core) else pd.Series(dtype=float))
    # Logged for later (Sam, 2026-10-17): the mean of the two highest core
    # stations reporting each hour -- best of five methods against his 13
    # sightings (330 ft typical miss vs the median's 395), not yet the page's.
    if len(core):
        hs = core.groupby(["h", "station_id"]).agg(lcl=("lcl_ft", "mean"), elev=("elev_ft", "first")).reset_index()
        top2 = hs.sort_values("elev", ascending=False).groupby("h").head(2).groupby("h").lcl.mean()
    else:
        top2 = pd.Series(dtype=float)
    ka = obs[(obs.station_id == "001HI") & (obs.observed.dt.date == day.date())]
    ka_t = ka.groupby(ka.observed.dt.hour).temp_f.mean()
    ka_td = ka.groupby(ka.observed.dt.hour).dew_f.mean()
    sun = {}
    try:
        got = [(t, w) for t, st, w in _retrying(sampler.read_mesonet_sun)(day + timedelta(hours=9), day + timedelta(hours=16))
               if st.name == "Kula Ag"]
        sx = pd.DataFrame(got, columns=["t", "w"])
        sx["clear"] = mesonet.clear_sky(pd.DatetimeIndex(sx.t).tz_localize(HST))
        sun = {h: g.w.sum() / g.clear.sum() for h, g in sx.groupby(sx.t.dt.hour) if g.clear.sum() > 50}
    except Exception:
        pass
    try:
        _dl = json.loads(DERIVED_LOG.read_text(encoding="utf-8"))
        shown = {int(k): v[0] for k, v in _dl.get("points", {}).items()} if _dl.get("date") == d else {}
    except Exception:
        shown = {}
    try:
        fl = json.loads(FORECAST_LATEST_FILE.read_text(encoding="utf-8"))
        fl = fl.get("hours", {}) if fl.get("date") == d else {}
    except Exception:
        fl = {}
    hrs = []
    for h in range(9, 16):
        fb = mf.get("hours", {}).get(str(h))
        hrs.append({"date": d, "hour": h,
                    "fc_base": fb, "fc_base_raw": None if fb is None else fb - FORECAST_BASE_LOW_FT,
                    "fc_base_latest": fl.get(str(h)),
                    "derived": None if h not in derived.index else round(float(derived[h]), -1),
                    "derived_top2": None if h not in top2.index else round(float(top2[h]), -1),
                    "derived_shown": shown.get(h * 60),
                    "inv_base": (mf.get("inv") or {}).get(str(h), [None, None])[0],
                    "inv_top": (mf.get("inv") or {}).get(str(h), [None, None])[1],
                    "fc_capped": (None if fb is None else
                                  min(fb, (mf.get("inv") or {}).get(str(h), [None, 1e9])[1])
                                  if (mf.get("inv") or {}).get(str(h)) and fb >= (mf.get("inv") or {})[str(h)][0]
                                  else fb),
                    "fc_temp": mf.get("temp", {}).get(str(h)), "temp": None if h not in ka_t.index else round(float(ka_t[h]), 1),
                    "fc_dew": mf.get("dew", {}).get(str(h)), "dew": None if h not in ka_td.index else round(float(ka_td[h]), 1),
                    "sun": None if h not in sun else round(float(sun[h]), 2)})
    _replace_day(SCORE_HOURS, hrs, d)

    fcs = mf.get("stations") or {}
    per = (core.groupby(["station_id", "h"]).lcl_ft.mean() if len(core) else pd.Series(dtype=float))
    srows = [{"date": d, "hour": h, "station_id": st.id,
              "fc_lcl": (fcs.get(st.id) or {}).get(str(h)),
              "lcl": None if (st.id, h) not in per.index else round(float(per[(st.id, h)]), -1)}
             for st in core_stations() for h in range(9, 16)]
    _replace_day(SCORE_STATIONS, srows, d)

    # Did summit cloud come over? Summit near saturation on an easterly while
    # Kula Ag's sun halved from late morning to its worst afternoon hour.
    intrusion = None
    try:
        r = mesonet.get("/mesonet/db/measurements", location="hawaii", station_ids="0153",
                        var_ids="RH_1_Avg,WDrs_1_Avg,WS_1_Avg",
                        start_date=f"{d}T13:00:00-10:00", end_date=f"{d}T16:00:00-10:00",
                        row_mode="json", limit=5000)
        x = pd.DataFrame(r)
        x["value"] = pd.to_numeric(x.value)
        v = x.pivot_table(index="timestamp", columns="variable", values="value")
        east = float((v.WS_1_Avg * np.sin(np.radians(v.WDrs_1_Avg))).mean()) > 0
        early = np.mean([sun[h] for h in (11, 12) if h in sun]) if any(h in sun for h in (11, 12)) else None
        worst = min([sun[h] for h in (14, 15) if h in sun], default=None)
        if early is not None and worst is not None:
            intrusion = bool(v.RH_1_Avg.max() >= 93 and east and early >= 0.5 and worst <= early / 2)
    except Exception:
        pass
    dropped = sum(1 for e in entries if e.get("dropped"))
    ph = obs[(obs.station_id == "002HI") & (obs.observed.dt.date == day.date())
             & obs.observed.dt.hour.between(9, 15)]
    ph_cloud = (int(10 * (_rh(ph.temp_f, ph.dew_f) >= 95).groupby(ph.observed.dt.floor("10min")).max().sum())
                if len(ph) else None)
    rh_cloud = int(60 * sum(1 for r in hrs if r["derived"] is not None and r["derived"] <= 3300))
    _replace_day(SCORE_DAYS, [{
        "date": d, "word": word, "fc_morning_made": mf.get("made"), "fc_sun": mf.get("sun"),
        "summit_alert": mf.get("summit_push"), "intrusion": intrusion,
        "upslope": None if up is None else f"{up:%H:%M}", "upslope_end": None if end is None else f"{end:%H:%M}",
        "line_updates": len(entries), "line_updates_with_dropped": dropped,
        "patterns": " ".join(f"{k}@{v}" for k, v in sorted(_patterns_today(d).items())),
        "cam_puunianiau_cloud_min": camera_cloud_minutes(d, "Puunianiau"), "parkhq_cloud_min": ph_cloud,
        "cam_redhill_cloud_min": camera_cloud_minutes(d, "Red Hill"), "derived_low_min": rh_cloud,
        "base_phase2": " ".join(f"{a}-{b or ''}" for a, b in PHASE.get("intervals", [])) if PHASE.get("date") == d else "",
        **_call_scores(d, hrs)}], d)


def _call_scores(d: str, hrs: list) -> dict:
    """The evening-before calls for day d against what it did."""
    try:
        call = json.loads(CALLS_LOG.read_text(encoding="utf-8")).get(d, {})
    except Exception:
        call = {}
    by_h = {r["hour"]: r["derived"] for r in hrs if r["derived"] is not None}
    mid = [by_h[h] for h in (11, 12, 13, 14) if h in by_h]
    fired = _patterns_today(d).get("P6")
    fell = None
    if fired:
        h0 = int(fired[:2])
        if h0 in by_h and 15 in by_h:
            fell = bool(by_h[15] - by_h[h0] <= -300)
    return {"base_call": call.get("base"), "base_rule": call.get("rule"), "thermal_call": call.get("thermal"),
            "base_cat": rules.category(sum(mid) / len(mid), int(d[5:7])) if mid else None,
            "p6_fell": fell}


def _patterns_today(d: str) -> dict:
    try:
        log = json.loads(PATTERN_LOG.read_text(encoding="utf-8"))
        return log.get("fired", {}) if log.get("date") == d else {}
    except Exception:
        return {}


def review_summary(now: datetime) -> None:
    flags, stats = [], {}
    if SCORE_HOURS.exists():
        h = pd.read_csv(SCORE_HOURS, dtype={"date": str})
        h = h[pd.to_datetime(h.date) >= pd.Timestamp(now.date()) - pd.Timedelta(days=30)].dropna(
            subset=["fc_base", "derived"])
        days = h.date.nunique()
        stats["base_days"] = int(days)
        if days >= 10:
            err = h.fc_base - h.derived
            m, raw = float(err.mean()), float((h.fc_base_raw - h.derived).mean())
            stats.update(base_mean=round(m), base_raw_mean=round(raw), base_sd=round(float(err.std())))
            if abs(m) > 300:
                flags.append(f"Forecast base has run {m:+.0f} ft against the derived base over {days} days "
                             + (f"(the +{FORECAST_BASE_LOW_FT} ft correction included; without it {raw:+.0f})."
                                if FORECAST_BASE_LOW_FT else "(no correction)."))
            am, pm = err[h.hour <= 11].mean(), err[h.hour >= 13].mean()
            stats.update(base_morning=round(float(am)), base_afternoon=round(float(pm)))
            if abs(am - pm) > 600:
                flags.append(f"Forecast base error differs by time of day over {days} days: morning {am:+.0f} ft, "
                             f"afternoon {pm:+.0f} ft.")
    if SCORE_DAYS.exists():
        dd = pd.read_csv(SCORE_DAYS, dtype={"date": str})
        al = dd[dd.summit_alert.notna()]
        hits = int((al.intrusion == True).sum())  # noqa: E712
        missed = int(((dd.intrusion == True) & dd.summit_alert.isna()).sum())  # noqa: E712
        stats.update(alert_days=len(al), alert_hits=hits, intrusions_missed=missed)
        if len(al) >= 5:
            flags.append(f"Summit-cloud alert so far: {hits} of {len(al)} alerted days had cloud come over; "
                         f"{missed} intrusions came without an alert.")
        rec = dd.tail(7)
        if rec.line_updates.sum() > 0:
            share = float(rec.line_updates_with_dropped.sum() / rec.line_updates.sum())
            stats["line_dropped_share_7d"] = round(share, 2)
            if share > 0.3:
                flags.append(f"Shear line: a stale station sat out of {share:.0%} of updates over the last week "
                             "(the rule under review).")
    if SCORE_DAYS.exists():
        dd = pd.read_csv(SCORE_DAYS, dtype={"date": str}).tail(14)
        for cam, stn, what in (("cam_puunianiau_cloud_min", "parkhq_cloud_min", "cloud at the tops (Puunianiau camera vs Park HQ)"),
                               ("cam_redhill_cloud_min", "derived_low_min", "cloud down to 2,800 ft (Red Hill camera vs derived base)")):
            if cam in dd and stn in dd:
                both = dd.dropna(subset=[cam, stn])
                off = both[(both[cam] - both[stn]).abs() >= 120]
                stats[f"{cam}_days"] = int(len(both))
                if len(both) >= 5 and len(off) >= 3:
                    flags.append(f"Camera check: {what} disagreed by 2 h or more on {len(off)} of {len(both)} days "
                                 f"({', '.join(off.date.str[5:])}).")
    if SCORE_STATIONS.exists() and SCORE_HOURS.exists():
        sc = pd.read_csv(SCORE_STATIONS, dtype={"date": str}).dropna(subset=["fc_lcl", "lcl"])
        nd = sc.date.nunique()
        stats["station_days"] = int(nd)
        if nd >= STATION_DAYS_NEEDED:
            days = sorted(sc.date.unique())
            first, second = set(days[:nd // 2]), set(days[nd // 2:])
            bias = (sc[sc.date.isin(first)].assign(e=lambda q: q.fc_lcl - q.lcl)
                    .groupby("station_id").e.mean())
            t = sc[sc.date.isin(second)].copy()
            t["fc_corr"] = t.fc_lcl - t.station_id.map(bias).fillna(0)
            fc_base = t.groupby(["date", "hour"])["fc_corr"].median()
            der = t.groupby(["date", "hour"]).lcl.median()
            hh = pd.read_csv(SCORE_HOURS, dtype={"date": str}).set_index(["date", "hour"])
            now_err = (hh.fc_base - hh.derived).reindex(fc_base.index).dropna()
            st_err = (fc_base - der).reindex(now_err.index)
            flags.append(f"Per-station forecast base: {nd} days logged. On the later half, per-station "
                         f"corrections miss the derived base by {st_err.abs().mean():.0f} ft typically, "
                         f"the page's Kula Ag + {FORECAST_BASE_LOW_FT} ft by {now_err.abs().mean():.0f} ft. "
                         "Ready to decide whether to switch.")
    # The monthly rule check's events, for a week after it runs (monthly.py).
    try:
        mr = json.loads((HERE / "docs" / "monthly_report.json").read_text(encoding="utf-8"))
        if mr.get("events") and now.replace(tzinfo=None) - datetime.fromisoformat(mr["made"]) <= timedelta(days=7):
            flags += [f"Monthly rule check ({mr['month']}): {e}" for e in mr["events"]]
    except Exception:
        pass
    REVIEW_SUMMARY.write_text(json.dumps({"made": f"{now:%Y-%m-%d %H:%M}", "flags": flags, "stats": stats},
                                         indent=1), encoding="utf-8")


# The morning before the upslope (Sam, 2026-10-17: "the today tile should not
# wait for upslope"). The derived base is shown only when the slope is near
# saturation -- dew-fed low cloud, as at 08:30 that day (Kula Ag 64/64 F,
# median LCL 3,300 ft); on dry mornings the stations disagree wildly (3,000-
# 7,900 ft at 07:17 on 10-14, "absurd") and it waits for the upslope.
MORNING_SATURATED_RH = 90       # per cent (95 missed 10-17: Keokea, Waipoli, Crater sat at 93)
MORNING_SATURATED_N = 3         # stations at or above it
SKY_OVERCAST = 0.93             # cloudiest Kula sky emissivity that counts as cloud overhead


def _rh(t_f, td_f):
    t, td = (t_f - 32) / 1.8, (td_f - 32) / 1.8
    return 100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))


def slope_lcl(obs: pd.DataFrame):
    """The middle of the slope stations' condensation levels now, or None."""
    core = _core(obs)
    last = core.sort_values("observed").groupby("station_id").tail(1)
    last = last[last.observed > obs.observed.max() - pd.Timedelta("45min")]
    return float(last.lcl_ft.median()) if len(last) >= 4 else None


def morning_cloud(obs: pd.DataFrame, now: datetime):
    """Low morning cloud from a near-saturated slope, or None. Returns the
    median LCL, how many stations are near saturation, Kula Ag's air and its
    dew-point rise since the morning low (the night's dew evaporating)."""
    core = _core(obs)
    last = core.sort_values("observed").groupby("station_id").tail(1)
    last = last[last.observed > obs.observed.max() - pd.Timedelta("45min")]
    if len(last) < 4:
        return None
    wet = last[_rh(last.temp_f, last.dew_f) >= MORNING_SATURATED_RH]
    if len(wet) < MORNING_SATURATED_N:
        return None
    ka = obs[(obs.station_id == "001HI") & (obs.observed.dt.hour >= 4)].sort_values("observed")
    out = {"base": float(last.lcl_ft.median()), "n_wet": len(wet), "n": len(last)}
    if len(ka):
        k = ka.iloc[-1]
        low = ka.loc[ka.dew_f.idxmin()]
        out.update(ka_t=float(k.temp_f), ka_td=float(k.dew_f), ka_rh=float(_rh(k.temp_f, k.dew_f)),
                   rise=float(k.dew_f - low.dew_f), since=low.observed)
    # Was there dew? Only if Kula Ag reached saturation overnight (any half
    # hour at 95%+ from midnight to 06:30). Sam, 2026-10-18: on a dry night
    # the dew point still rose 4 F -- trade air and crater spillover arriving,
    # not dew -- and the page blamed the dew.
    night = obs[(obs.station_id == "001HI") & (obs.observed.dt.hour * 60 + obs.observed.dt.minute < 390)]
    out["dewy"] = bool(len(night) and (_rh(night.temp_f, night.dew_f)
                                       .groupby(night.observed.dt.floor("30min")).mean() >= 95).any())
    # Is there cloud overhead, or only damp air? (Sam, 2026-10-18: "There are
    # no clouds in Kula yet".) The cloudiest Kula sky over the last 45 min;
    # 0.97-0.98 under last evening's low cloud, ~0.81 clear. 0.93 is a
    # judgment between them, not a tested value.
    end = now.replace(tzinfo=None)
    sky = safely(kula_sky, end - timedelta(minutes=45), end)
    out["overhead"] = bool(sky is not None and len(sky) and float(sky.iloc[-1]) >= SKY_OVERCAST)
    return out


# -- sky over the Kula stations ---------------------------------------------
# Sky emissivity (LWin / sigma T^4) from the Mesonet radiometers: high under
# cloud, low under clear sky, whatever the sun angle. One station alone is
# fooled by cloud streets and the gaps between them (2026-10-17: Keokea under
# the south street, Kula Ag in the gap), so the cloudiest of three is used.
SKY_STATIONS = ("0119", "0116", "0118")      # Kula Ag, Keokea, Pulehu
DECAY_DROP = 0.05                             # cleared this much in an hour


def kula_sky(start: datetime, end: datetime) -> pd.Series:
    """The cloudiest Kula station's sky emissivity per 15 minutes."""
    rows = _retrying(mesonet.get)("/mesonet/db/measurements", location="hawaii",
                                  station_ids=",".join(SKY_STATIONS), var_ids="LWin_1_Avg,Tair_1_Avg",
                                  start_date=start.strftime("%Y-%m-%dT%H:%M:%S-10:00"),
                                  end_date=end.strftime("%Y-%m-%dT%H:%M:%S-10:00"),
                                  row_mode="json", limit=100000)
    if not rows:
        return pd.Series(dtype=float)
    x = pd.DataFrame(rows)
    x["t"] = pd.to_datetime(x.timestamp).dt.tz_convert(HST).dt.tz_localize(None)
    x["value"] = pd.to_numeric(x.value, errors="coerce")
    w = x.pivot_table(index=["t", "station_id"], columns="variable", values="value").reset_index()
    w["emis"] = w.LWin_1_Avg / (5.67e-8 * (w.Tair_1_Avg + 273.15) ** 4)
    return w.groupby([w.t.dt.floor("15min"), "station_id"]).emis.mean().unstack().max(axis=1)


def sky_now(now: datetime):
    """{"drop", "cloudiest"}: how much the cloudiest Kula sky cleared over
    the last hour, or None before 14:00 or without data."""
    if now.hour < 14:
        return None
    end = now.replace(tzinfo=None)
    s = kula_sky(end - timedelta(minutes=90), end)
    if len(s) < 5:
        return None
    return {"drop": float(s.iloc[-5] - s.iloc[-1]), "cloudiest": float(s.iloc[-1])}


def decay_time(day: datetime, derived: list):
    """When the late-day decay began ("HH:MM"), for the review: the first
    quarter hour from 14:30 where the cloudiest Kula sky had cleared by
    DECAY_DROP over the hour while the base held within 300 ft."""
    s = kula_sky(day + timedelta(hours=13, minutes=15), day + timedelta(hours=18, minutes=30))
    base = dict(derived)
    for t in s.index:
        m = t.hour * 60 + t.minute
        if m < 14 * 60 + 30 or (t - timedelta(hours=1)) not in s.index:
            continue
        held = [v for k, v in base.items() if m - 60 <= k <= m]
        if s[t - timedelta(hours=1)] - s[t] >= DECAY_DROP and len(held) >= 3 and max(held) - min(held) <= 300:
            return f"{t:%H:%M}"
    return None


# -- the day's base as a category ----------------------------------------------
# Sam, 2026-10-17: "rather than worrying about feet we start with general
# rules." Each month's normal range is the middle 60% of the 1,020 history
# days (KulaModel history.py, 2024-01 to 2026-10). Low is called only when
# two independent signals agree -- moist air aloft (the evening-before rule
# that recurred in all three held-back years) and a forecast base in the
# month's lowest fifth: right 89% of the time on 8% of days. Either alone
# was right ~62%, so then it says normal.
MONTH_BASE_RANGE = rules.MONTH_BASE_RANGE


def _layer_rh(prof: pd.DataFrame, lo: float, hi: float) -> float:
    q = prof[(prof.height_ft >= lo) & (prof.height_ft <= hi)]
    rh = 100 * np.exp(17.625 * q.dewpoint_c / (243.04 + q.dewpoint_c) - 17.625 * q.temp_c / (243.04 + q.temp_c))
    return float(rh.mean())


@functools.lru_cache(maxsize=4)
def _profiles(path):
    return sd.profiles(path, model=cap_model(datetime.now(HST)))


def airmass_noon(path, day) -> dict | None:
    """Kahului humidity 5,000-7,000 and 7,000-10,000 ft at noon HST on `day`."""
    prof = _profiles(path)
    k = pd.Timestamp(f"{day} 12:00").tz_localize(HST).tz_convert("UTC")
    near = min(prof, key=lambda t: abs(t - k))
    if abs(near - k) > pd.Timedelta("90min"):
        return None
    return {"rh_5_7k": _layer_rh(prof[near], 5000, 7000), "rh_7_10k": _layer_rh(prof[near], 7000, 10000)}


def today_level(now: datetime):
    """Today's midday base so far (11:00-14:59 derived points), or None."""
    try:
        dl = json.loads(DERIVED_LOG.read_text(encoding="utf-8"))
    except Exception:
        return None
    if dl.get("date") != f"{now:%Y-%m-%d}":
        return None
    v = [p[0] for m, p in dl.get("points", {}).items() if 11 * 60 <= int(m) < 15 * 60]
    return sum(v) / len(v) if v else None


CALL_RULE = {}          # the rule behind the last cloudbase_call, for the call log
# Flicker (Sam, 2026-10-20: "Fix the flicker, but not stuck on a single
# occurrence"): on 10-19 the call for 10-20 swung Leaning high <-> Normal a
# dozen times as model runs nudged B4's inputs across its edges. A call is now
# shown once it has held for 3 of the last 4 builds (~an hour by day); until
# then the previous call stays. One build can neither switch it on nor off.
CALL_HISTORY = HERE / "docs" / "call_history.json"
CALL_VOTES, CALL_NEEDED = 4, 3
SETTLED: dict = {}


def settled_call(day, *args):
    key = f"{day}"
    if key in SETTLED:                      # Forecast panel and week table share a build
        return SETTLED[key]
    head, sub = cloudbase_call(day, *args)
    try:
        hist = json.loads(CALL_HISTORY.read_text(encoding="utf-8"))
    except Exception:
        hist = {}
    h = hist.get(key) or {"recent": [], "shown": None, "texts": {}, "rules": {}}
    h["recent"] = (h["recent"] + [head])[-CALL_VOTES:]
    h["texts"][head] = [head, sub]
    h["rules"][head] = CALL_RULE.get(key)
    shown = h["shown"] if h["shown"] in h["texts"] else head
    for lab in dict.fromkeys(h["recent"]):
        if lab != shown and h["recent"].count(lab) >= CALL_NEEDED:
            shown = lab
    h["shown"] = shown
    hist[key] = h
    keep = f"{datetime.now(HST).date() - timedelta(days=1)}"
    hist = {k: v for k, v in hist.items() if k >= keep}
    try:
        CALL_HISTORY.write_text(json.dumps(hist), encoding="utf-8")
    except Exception:
        pass
    CALL_RULE[key] = h["rules"].get(shown)
    SETTLED[key] = tuple(h["texts"][shown])
    return SETTLED[key]


def cloudbase_call(day, path, fc_hours: dict, level_today, leaning=False) -> tuple[str, str]:
    """The day's base as a category, from the rules in rules.py (B1-B4).
    `leaning` allows B4, the weaker forecast-only high call, after B2 and B3
    (Sam, 2026-10-17: tomorrow falls back to it too)."""
    CALL_RULE[f"{day}"] = None
    lo, hi = MONTH_BASE_RANGE[day.month]
    air = airmass_noon(path, day) if path else None
    fc = [v - FORECAST_BASE_LOW_FT for h, v in fc_hours.items() if 11 <= h <= 14]
    if air:
        f = {"month": day.month, "rh_5_7k": air["rh_5_7k"], "rh_7_10k": air["rh_7_10k"],
             "fc_level": sum(fc) / len(fc) if fc else None,      # B1 needs it, B2 does not
             "today_cat": rules.category(level_today, datetime.now(HST).month),
             "today_level": level_today}
        if rules.says("B1", f):
            CALL_RULE[f"{day}"] = "B1"
            return ("Probably a low day",
                    "Moist air aloft and a low forecast base both point to it: right about 8 times in 10.")
        if rules.says("B2", f):
            CALL_RULE[f"{day}"] = "B2"
            return ("May be a high day", "Dry air aloft after a high day today: right about 2 times in 3.")
        if rules.says("B3", f):
            CALL_RULE[f"{day}"] = "B3"
            return ("May be a high day", "High base today and a high forecast: right about half the time.")
        if leaning and f["fc_level"] is not None and rules.says("B4", f):
            CALL_RULE[f"{day}"] = "B4"
            near = (day - datetime.now(HST).date()).days <= 1
            return ("Leaning high", "Dry air aloft and a high forecast base: right "
                    + ("a little under half the time." if near else "about 4 times in 10."))
    return (f"Probably normal for {day:%B}, about {lo:,}–{hi:,} ft",
            "Most days the base levels off by mid-morning and holds. Nothing in the forecast points to "
            "a low or a high day.")


def base_chart(obs, now, up, end, cid="basechart", title="Cloudbase, derived vs forecast",
               with_forecast=True) -> tuple[str, list, list]:
    """(html, derived, forecast): the derived base every 15 minutes through
    the upslope against that morning's forecast by hour."""
    day = now.replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
    core = _core(obs)
    stop = end if end is not None else obs.observed.max()
    derived = _kept_points(day, [(t, _derived_at(core, t), _stations_at(core, t))
                                 for t in pd.date_range(up.ceil("15min"), stop, freq="15min")])
    src = "that morning"
    try:
        saved = json.loads(MORNING_BASE_FILE.read_text(encoding="utf-8"))
        assert saved["date"] == f"{day:%Y-%m-%d}"
        fc = {int(h): v for h, v in saved["hours"].items()}
        src = f"the {saved['made']} morning forecast"
    except Exception:
        fc = safely(forecast_base_hours, day.date()) or {}
        src = "a later run (the morning forecast was not saved)"
    forecast = [[h * 60, v] for h, v in sorted(fc.items())]
    span = [up.hour * 60 + up.minute, stop.hour * 60 + stop.minute]
    note = (f"Derived from the stations every 15 minutes during the upslope; forecast by hour, "
            f"{src}." if with_forecast else "Derived from the stations every 15 minutes during the upslope.")
    # The forecast is still returned for the review notes and the scorecard,
    # just not drawn (Sam, 2026-10-17: derived-only charts).
    return (chart_html(cid, derived, forecast if with_forecast else [], span, note, title=title),
            derived, forecast)


# A forecast base is an educated guess (Sam): hour by hour it missed the
# stations by 680-850 ft either way over 31 days (9/13-10/13), so the
# Forecast panel says "give or take" this much. The chart band that drew it
# was dropped (Sam, 2026-10-17).
FORECAST_BAND_FT = 700


def chart_html(cid, derived, forecast, span, note, nocu=None, title="") -> str:
    """Cloudbase line chart: derived (solid) and forecast (dashed) by minute of
    the day, the upslope shaded; derived and span may be empty for a forecast.

    One fixed scale, 3,500-7,000 ft, for every chart (Sam): the station
    estimate's 15-minute wobble reads as wobble, and days compare at a
    glance. It stretches only when a value falls outside."""
    # Every chart shows 09:00-16:00 (Sam): the flying day, nothing outside it.
    clip = lambda pts: [q for q in pts if 540 <= q[0] <= 960]
    derived, forecast = clip(derived), clip(forecast)
    if span:
        span = [max(span[0], 540), min(span[1], 960)]
    data = json.dumps({"derived": derived, "forecast": forecast, "up": span, "nocu": nocu})
    legend = ("" if not derived else
              '<span><span style="display:inline-block;width:16px;height:2px;background:#2a78d6;vertical-align:middle"></span> Derived (stations)</span>'
              '')
    head = (f'<div style="font-weight:600;font-size:.9rem;margin:10px 0 2px">{esc(title)}</div>'
            if title else "")
    # The Forecast key only when there is a forecast line (Sam, 2026-10-17:
    # the derived-only charts still showed it).
    fkey = ('<span><span style="display:inline-block;width:16px;border-top:2px dashed #eb6834;'
            'vertical-align:middle"></span> Forecast</span>' if forecast else "")
    aria = "Derived cloudbase against the forecast through the day" if forecast else "Derived cloudbase through the day"
    return f"""{head}<div style="display:flex;gap:14px;flex-wrap:wrap;font-size:.8rem;color:var(--muted,#6b6b66);margin:2px 0 4px">
{fkey}
{legend}</div>
<div style="position:relative;width:100%;min-width:0;height:220px"><canvas id="{cid}" role="img" aria-label="{aria}"></canvas></div>
{f'<div class="sub">{esc(note)}</div>' if note else ""}
<script>
(function () {{
  var d = {data}, pt = function (a) {{ return {{x: a[0], y: a[1]}}; }};
  var dark = matchMedia('(prefers-color-scheme: dark)').matches, muted = '#898781', grid = dark ? '#2c2c2a' : '#e1e0d9';
  var nocu = {{ id: 'nocu', beforeDatasetsDraw: function (c) {{ if (!d.nocu) return; var x = c.scales.x, a = c.chartArea, g = c.ctx,
    x0 = x.getPixelForValue(d.nocu[0]), x1 = x.getPixelForValue(d.nocu[1]);
    g.save(); g.fillStyle = 'rgba(137,135,129,0.10)'; g.fillRect(x0, a.top, x1 - x0, a.bottom - a.top);
    g.fillStyle = muted; g.font = '11px sans-serif'; g.textAlign = 'center';
    g.fillText('No cumulus', (x0 + x1) / 2, a.top + (a.bottom - a.top) / 2); g.restore(); }} }};
  var band = {{ id: 'band', beforeDatasetsDraw: function (c) {{ if (!d.up) return; var x = c.scales.x, a = c.chartArea, g = c.ctx;
    g.save(); g.fillStyle = 'rgba(137,135,129,0.14)';
    g.fillRect(x.getPixelForValue(d.up[0]), a.top, x.getPixelForValue(d.up[1]) - x.getPixelForValue(d.up[0]), a.bottom - a.top); g.restore(); }} }};
  var hm = function (m) {{ return String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0'); }};
  var ys = d.derived.map(function (a) {{ return a[1]; }}).concat(d.forecast.map(function (a) {{ return a[1]; }}));
  var xs = [540, 960];
  new Chart(document.getElementById('{cid}'), {{
    type: 'line',
    data: {{ datasets: [
      {{ label: 'Derived', data: d.derived.map(pt), borderColor: '#2a78d6', borderWidth: 2, pointRadius: 0, tension: 0.3 }},
      {{ label: 'Forecast', data: d.forecast.map(pt), borderColor: '#eb6834', borderWidth: 2, borderDash: [6, 4], pointRadius: 3, pointBackgroundColor: '#eb6834', tension: 0.3 }}
    ] }},
    options: {{ responsive: true, maintainAspectRatio: false, interaction: {{ mode: 'nearest', axis: 'x', intersect: false }},
      plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{
        title: function (i) {{ return hm(i[0].parsed.x); }},
        label: function (i) {{ return i.dataset.label + ': ' + Math.round(i.parsed.y).toLocaleString() + ' ft'; }} }} }} }},
      scales: {{
        x: {{ type: 'linear', min: xs[0], max: xs[1], ticks: {{ stepSize: 60, color: muted, callback: function (v) {{ return hm(v); }} }}, grid: {{ display: false }} }},
        y: {{ min: Math.min(3500, Math.floor(Math.min.apply(null, ys) / 500) * 500),
              max: Math.max(7000, Math.ceil(Math.max.apply(null, ys) / 500) * 500),
              ticks: {{ stepSize: 500, color: muted, callback: function (v) {{ return v.toLocaleString(); }} }}, grid: {{ color: grid }} }}
      }} }},
    plugins: [nocu]
  }});
}})();
</script>"""




def _hm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


FLAT_RANGE_FT = 300
FLAT_TREND_FT_H = 75


def day_notes(derived: list, forecast: list, inv, phase: dict, fired: dict, decay=None) -> list:
    """What the day did, for the review (Sam, 2026-10-17: "The daily recap can
    include such observations"). Fixed rules on the saved data, one per kind
    of observation; an ordinary day gives few or none."""
    notes = []
    if phase.get("wet"):
        start2 = phase["intervals"][0][0] if phase.get("intervals") else None
        notes.append("Saturated, dew-fed morning" + (f": low cloud until about {start2}." if start2 else "."))
    pts = sorted(derived)
    # the biggest 30-minute rise of the base
    best = None
    for (m0, v0), (m1, v1) in zip(pts, pts[2:]):
        if m1 - m0 == 30 and (best is None or v1 - v0 > best[2]):
            best = (m0, m1, v1 - v0)
    if best and best[2] >= 700:
        notes.append(f"Base stepped up about {round(best[2], -2):,.0f} ft between {_hm(best[0])} and "
                     f"{_hm(best[1])} as mixing took over.")
    # a plateau: the base held flat for 2 h or more -- the stations'
    # own sign of a cap, with no model lid needed (Sam, 2026-10-17: the
    # Kahului inversion is useless for our purposes)
    # Flat means not changing (Sam, 2026-10-18: "Held flat has nothing to do
    # with the height of the base. It has to do with change in height"): the
    # 400 ft band alone called a steady 150 ft/h climb (4,920 -> 5,510,
    # 11:30-15:30) "held flat". Now within 300 ft AND a fitted trend under
    # 75 ft an hour.
    def _flat(r):
        if max(x for _, x in r) - min(x for _, x in r) > FLAT_RANGE_FT:
            return False
        ms, vs = np.array([m for m, _ in r], float), np.array([x for _, x in r], float)
        return abs(np.polyfit(ms, vs, 1)[0] * 60) <= FLAT_TREND_FT_H if len(r) >= 3 else True
    run, longest = [], []
    for m, v in pts:
        run = run + [(m, v)] if run and m - run[-1][0] == 15 else [(m, v)]
        while len(run) > 1 and not _flat(run):
            run = run[1:]
        longest = run if len(run) > len(longest) else longest
    if len(longest) >= 9:
        mean = sum(v for _, v in longest) / len(longest)
        hrs = (longest[-1][0] - longest[0][0]) / 60
        held = (f"Base held flat at about {round(mean, -2):,.0f} ft from {_hm(longest[0][0])} "
                f"for about {hrs:.0f} hour{'s' if round(hrs) != 1 else ''}")
        # The late-day decay (Sam, 2026-10-17: "dissipating, from base up"):
        # the stations' sky clearing while the base held.
        notes.append(held + (f", then the cloud decayed from the base up from about {decay}." if decay else "."))
    elif decay:
        notes.append(f"The cloud decayed from the base up from about {decay}.")
    # (The "forecast peaked at X" note went with the hourly forecast, 2026-10-20.)
    if fired:
        parts = []
        # Named, not just numbered (Sam, 2026-10-17: "p3 and p4 are claimed
        # but not defined"): the ID stays for finding it in patterns.py.
        names = {q["id"]: q["name"] for q in patterns.PATTERNS if q["active"]}
        for pid in sorted({k.rstrip("c") for k in fired} & set(names)):
            t = fired.get(pid)
            c = fired.get(pid + "c")
            parts.append(f"{names[pid]} ({pid})" + (f" from {t}" if t else "") + (f", confirmed {c}" if c else ""))
        if parts:
            notes.append("Watched for: " + "; ".join(parts) + ".")
    return notes


def day_review(obs, now, up, evening, rnet, word, model, fc_today, fc_pair, entries, inv=None) -> dict:
    """The whole flying day, for the Today panel after the upslope ends."""
    day = now.replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
    end = upslope_end(rnet, evening) if up is not None else None
    fc_word = strength_word(fc_today["thermal"]) if fc_today is not None else None
    said = []

    therm = "Unavailable"
    if word:
        therm = (f"<strong>{word}</strong> for the day, 10:00–14:00<div class=\"sub\">Kula Ag's measured "
                 f"air through the {model} sounding (thermal.py)"
                 + (f"; forecast was {fc_word}" if fc_word else "") + "</div>")
        said.append(f"A {word.lower()} day" + (f" (forecast {fc_word.lower()})" if fc_word else "") + ".")
        tm = safely(thermal_moves, THERM_HOURS)
        if tm:
            said.append(tm)

    if up is not None:
        up_html = (f'<span class="num">{up:%H:%M}</span> to about <span class="num">{end:%H:%M}</span>'
                   if end is not None else f'From about <span class="num">{up:%H:%M}</span>')
        if evening and evening[0] == "drain":
            up_html += f'<div class="sub">Drainage from about {evening[1]:%H:%M}</div>'
        said.append(f"The upslope ran from about {up:%H:%M}"
                    + (f" to about {end:%H:%M}." if end is not None else "."))
    else:
        up_html = "No clear upslope today"
        said.append("No clear upslope today.")

    chart_html = "Not derived<div class=\"sub\">No upslope, so no slope air to derive it from</div>"
    if up is not None and len(obs):
        got = safely(base_chart, obs, now, up, end, "basechart", "Cloudbase, derived", False)
        if got and got[1]:
            chart, derived, forecast = got
            dv = [v for _, v in derived]
            lo_m, hi_m = up.hour * 60 + up.minute, (end or obs.observed.max()).hour * 60 + (end or obs.observed.max()).minute
            fv = [v for m, v in forecast if lo_m <= m <= hi_m]
            # Not repeated in words (Sam, 2026-10-17: "since derived cloudbase is
            # charted it does not need to be repeated in text"): no "Cloudbase
            # held X-Y" sentence and no Cloudbase row in the review.
            chart_html = ""
            # But say what kind of day it was for the month, and what the call
            # the evening before said (Sam, 2026-10-20: "daily wrap up ignores
            # the fact that base was high, very high").
            mid = [v for m, v in derived if 11 * 60 <= m <= 14 * 60]
            cat = rules.category(sum(mid) / len(mid), now.month) if mid else None
            if cat in ("high", "low"):
                lo, hi = MONTH_BASE_RANGE[now.month]
                said.append(f"A {cat} day for {now:%B}: the base reached about {ft(round(max(dv), -2))} ft "
                            f"(normal {lo:,}–{hi:,} ft)." if cat == "high" else
                            f"A low day for {now:%B}: the base stayed down near {ft(round(sum(mid) / len(mid), -2))} ft "
                            f"(normal {lo:,}–{hi:,} ft).")
            try:
                call = json.loads(CALLS_LOG.read_text(encoding="utf-8")).get(f"{now:%Y-%m-%d}") or {}
            except Exception:
                call = {}
            if cat and call.get("base"):
                want = {"low": ("Low",), "high": ("May be high", "Leaning high")}.get(cat, ("Normal",))
                said.append(f"The evening before, the page called it {call['base'].lower()}"
                            + (", a hit." if call["base"] in want else ", a miss."))

    ld = line_day(entries)
    if ld:
        common, far = ld
        line_html = (f"{esc(_cap(common))}<div class=\"sub\">Most of the day, {len(entries)} updates"
                     + (f". Furthest south about {far[1]}: {esc(_cap(far[0]))}" if far else "") + "</div>")
        said.append(line_moves(entries))
    else:
        line_html = "Not located<div class=\"sub\">No readings logged during the upslope</div>"

    lw = (safely(lower_wind, obs, up, end if end is not None else obs.observed.max())
          if up is not None and len(obs) else None)
    wind_html = ((f'<span class="num">{lw["avg"]:.0f} mph</span>'
                  + (f', gusts to {lw["gust"]:.0f}' if lw["gust"] is not None else "")
                  + '<div class="sub">Lower elevations, during the upslope</div>')
                 if lw else "Unavailable")

    _day0 = now.replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
    decay = safely(decay_time, _day0, derived) if derived else None
    notes = (safely(day_notes, derived, forecast, inv, PHASE, _patterns_today(f"{now:%Y-%m-%d}"), decay) or []
             if up is not None and len(obs) and got and got[1] else [])
    notes_html = ('<div class="watch"><div class="kind">What the day did</div><ul>'
                  + "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul></div>") if notes else ""
    return {"date": f"{now:%Y-%m-%d}", "word": word, "summary": " ".join(said),
            "therm": therm, "base": chart_html, "notes": notes,
            "chart": notes_html + (chart if up is not None and len(obs) and got and got[1] else ""),
            "line": line_html,
            "up": up_html, "wind": wind_html}


def build() -> str:
    now = datetime.now(HST)
    obs = safely(observations, now)
    obs = obs if obs is not None else pd.DataFrame()
    cap = safely(sd.fetch, cap_model(now))

    # today
    base = safely(cloudbase_now, obs) if len(obs) else None
    therm = safely(thermal_today, obs, now, cap) if len(obs) else None
    global THERM_HOURS
    THERM_HOURS = therm
    up, last_drain = safely(upslope_since, obs) or (None, None) if len(obs) else (None, None)
    rnet = safely(net_radiation, now) if now.hour >= 15 and up is not None and len(obs) else None
    evening = safely(evening_state, obs, rnet) if up is not None and len(obs) else None
    down = safely(sundown, now) if now.hour >= 15 else None
    if up is not None and down is not None and not (evening and evening[0] in ("over", "drain")):
        evening = ("over", evening[1] if evening else down)     # it's dark, whatever reported
    # Night (Sam: the page should know): from the end of the upslope -- or 18:00
    # on a day without one -- until the next upslope, the Today panel shows the
    # day in review and the Forecast panel the next flying day.
    day_over = ((up is not None and evening is not None and evening[0] in ("over", "drain"))
                or (up is None and (now.hour >= 18 or down is not None)))
    before_up = up is None and now.hour < 12
    ladder_start = ladder_or_last(now)
    ladder, sun_hour = ladder_start if ladder_start else ([], None)
    # Haleakala's shadow: direct sun reaches Kula Ag 07:15-08:45 by season (SUN_ON),
    # so before the first full sunlit hour the stations get only skylight and read as
    # if under cloud -- 26% at Kula Ag with a clear sky on 2026-10-14 (Sam).
    sun_on, first_hr = SUN_ON[now.month], first_sun_hour(now.month)
    sun_on_at = now.replace(tzinfo=None, hour=int(sun_on[:2]), minute=int(sun_on[3:]),
                            second=0, microsecond=0)
    # The window must start after the sun is on Kula; with the Mesonet's lag
    # the first one lands about 75 minutes after.
    first_at = f"{sun_on_at + timedelta(minutes=75):%H:%M}"
    in_shadow = sun_hour is None or sun_hour < sun_on_at
    if in_shadow:
        ladder = []
    global SHADOWED
    SHADOWED = ({n for n, late in LATE_SUN_MIN.items() if sun_hour < sun_on_at + timedelta(minutes=late)}
                if ladder and sun_hour is not None else set())
    ladder = [(n, e, None if n in SHADOWED else v) for n, e, v in ladder]
    night = now.hour >= 19 or down is not None   # sun down: no sunlight ladder
    if night:
        ladder = []
    pos_lad = safely(line_now, obs, now)
    pos, lad = pos_lad if pos_lad else (None, pd.DataFrame())
    week, hours, inv_today, inv_tom, fc_made = unpack(forecast(now, cap), now)
    # Today's forecast, shown in place of each measured item until its gate
    # opens (Sam: "I'm drinking my coffee and thinking about going flying").
    fc_today = week.loc[now.date()] if week is not None and now.date() in week.index else None
    fc_pair = hours.get(now.date()) if hours else None

    ka_sun = next((s for n, e, s in ladder if n == "Kula Ag"), None)
    # The Kula side of the slope, Keokea to Park HQ (Sam, 2026-10-14): Kula Ag
    # alone sat in a gap at 83% while the side averaged 56%. Each station is
    # capped at 100% so cumulus-edge spikes don't lift the average. Pulehu and
    # Piiholo (trade air) and Nene/Summit (above the cumulus) stay out.
    side = [min(s, 1.0) for n, e, s in ladder if n in KULA_SIDE and s is not None]
    side_best = max(side) if side else None              # for the Flat check
    # The Sun row averages the Kula-side stations BELOW the cloudbase -- Kula
    # Ag and Keokea (Sam, 2026-10-14): the upper ones (Kula 1, Kamehamenui,
    # Park HQ, 6,550-6,930 ft) sit up among the cumulus, and their shade is
    # not the ground pilots thermal from.
    # Pulehu joins when its wind was upslope through that hour (Sam): it is
    # Kula-side terrain below the base, but trade air pulses round it.
    sun_names, pulehu_out = ["Kula Ag", "Keokea"], False
    if sun_hour is not None and len(obs):
        pu = obs[(obs.station_id == "042HI") & obs.wind_dir.notna()
                 & obs.observed.between(sun_hour, sun_hour + timedelta(hours=1))]
        if len(pu) >= 3 and 185 <= cv.mean_direction(pu.wind_dir) <= 320:
            sun_names.append("Pulehu")
        elif len(pu) >= 3:
            pulehu_out = True
    low_sun = [(n, min(s, 1.0)) for n, e, s in ladder if n in sun_names and s is not None]
    side_sun = sum(v for _, v in low_sun) / len(low_sun) if low_sun else None
    latest = None if therm is None else therm.iloc[-1]
    verdict_now = None if latest is None else latest["verdict"]

    # cloudbase text -- an estimate from ground stations, never stated as seen
    if base:
        mid = round(base["mid"], -2)
        if base["hi"] - base["lo"] > 1000:
            base_html = (f'<span class="num">{ft(round(base["lo"], -2))}–{ft(round(base["hi"], -2))} ft</span>'
                         f'<div class="sub">Stations disagree; middle about {ft(mid)} ft '
                         f'({base["n"]} stations, {base["at"]:%H:%M})</div>')
        else:
            base_html = (f'<span class="num">About {ft(mid)} ft</span>'
                         f'<div class="sub">Station estimate, {ft(round(base["lo"], -2))}–'
                         f'{ft(round(base["hi"], -2))} ft ({base["n"]} stations, {base["at"]:%H:%M})</div>')
    else:
        base_html = "Unavailable"
    # Before the upslope the slope's air is the night's drainage layer and the
    # stations disagree wildly (3,000-7,900 ft at 07:17 on 2026-10-14); Sam:
    # "absurd". The derived base starts once the upslope sets in.
    base_label = "Derived<br>Cloudbase"
    cu = safely(first_cumulus, now) if up is not None else None
    cu_clear = bool(cu) and cu[0] == "clear"
    mc = safely(morning_cloud, obs, now) if up is None and 6 <= now.hour < 12 and len(obs) else None
    global PHASE
    # Dew-morning phases: the base is always all stations; in phase 2 the
    # highest stations forecast how far it lifts once the dew dries (Sam,
    # 2026-10-18: "It was used to provide forecast of lifting").
    PHASE = safely(update_phase, obs, now, bool(mc and mc.get("dewy")), up, day_over, bool(mc)) or {}
    if PHASE.get("phase") == 2 and base and len(obs):
        _c = _core(obs)
        _c = _c[_c.observed >= obs.observed.max() - STALE]
        _lat = _c.sort_values("observed").groupby("station_id").tail(1)
        if len(_lat) >= 3:
            _up, _names = _upper_pair(_lat)
            base_html = (f'<span class="num">About {ft(round(base["mid"], -2))} ft</span>'
                         f'<div class="sub">The lower slope is still damp from the dew overnight. The highest '
                         f'station ({", ".join(_names)}) is at about {ft(round(_up, -2))} ft: the base should '
                         f'lift toward that as the dew dries</div>')
    if mc:
        base_label = "Morning<br>cloud" if mc.get("overhead") else "First<br>clouds"
        dew = (f"; Kula Ag {mc['ka_t']:.0f}/{mc['ka_td']:.0f} °F ({mc['ka_rh']:.0f}%)"
               + (f", dew point up {mc['rise']:.0f} °F since {mc['since']:%H:%M}"
                  + (" as the night's dew evaporates" if mc.get("dewy") else "")
                  if mc.get("rise", 0) >= 1.5 else "") if "ka_t" in mc else "")
        # No hourly forecast feet here (Sam, 2026-10-18: Today showed 5,600 -> 5,400
        # while the Forecast panel's category said 4,500-5,400); the Forecast
        # panel carries today's call until the upslope.
        base_html = (f'Low cloud down to about <span class="num">{ft(round(mc["base"], -2))} ft</span>'
                     f'<div class="sub">{mc["n_wet"]} of {mc["n"]} slope stations near saturation{dew}</div>'
                     if mc.get("overhead") else
                     f'About <span class="num">{ft(round(mc["base"], -2))} ft</span>'
                     f'<div class="sub">{mc["n_wet"]} of {mc["n"]} slope stations near saturation; '
                     f'no cloud overhead yet{dew}</div>')
    elif up is None and now.hour < 12 and PHASE.get("morning") and (_fc := safely(slope_lcl, obs)):
        # The morning line has fired today but the slope has dried below it in
        # the sun, before the upslope registers (Sam, 2026-10-18: "latest
        # update lost the low cloud call" just as the first wisps formed).
        base_label = "First<br>clouds"
        base_html = (f'About <span class="num">{ft(round(_fc, -2))} ft</span>'
                     '<div class="sub">The slope is drying as it warms; the derived base takes over '
                     'once the upslope sets in</div>')
    elif up is None and now.hour < 12:
        base_html = ('Not yet<div class="sub">Derived once the upslope sets in, '
                     f"usually about {UPSLOPE_TYPICAL[now.month]}. Today's call is in the Forecast panel</div>")
    elif base and cu_clear:
        base_html = (f'No cumulus yet<div class="sub">They would form near about '
                     f'{ft(round(base["mid"], -2))} ft, the condensation level derived from '
                     f'the stations; no cloud over the slope in the sunlight readings</div>')
    elif evening and evening[0] in ("over", "drain"):
        # Same for the evening: once the slope is losing heat the stations
        # read the drainage layer, not a cloudbase (Sam: gate the evening).
        base_html = ('Done for the day<div class="sub">The upslope is over; the derived '
                     "base returns with tomorrow's upslope</div>")

    # No Ceiling row (Sam, 2026-10-17: "Kahului inversion is useless for our
    # purposes"): too low on 9/21 (the slope broke through), and on 10-17 it
    # came and went run to run while Kula sat capped at ~5,000 ft. The
    # inversion is still logged (scorecard inv_base/inv_top) to keep checking.
    ceil_html = ""

    # upslope text
    if evening and up is not None:
        kind, since = evening
        up_html = {
            # In time order (Sam): when it began, then how it is ending.
            "drain": f'Began about <span class="num">{up:%H:%M}</span>, now over'
                     f'<div class="sub">Drainage since about {since:%H:%M}</div>',
            "over": f'Began about <span class="num">{up:%H:%M}</span>, now over'
                    f'<div class="sub">Slope losing heat since about {since:%H:%M}</div>',
            "ending": f'Began about <span class="num">{up:%H:%M}</span>, now ending'
                      f'<div class="sub">Slope heating nearly stopped about {since:%H:%M}</div>',
        }[kind]
    elif up is not None:
        up_html = f'On since about <span class="num">{up:%H:%M}</span>'
        if last_drain is not None:
            up_html += f'<div class="sub">Drainage ended about {last_drain:%H:%M}</div>'
    elif now.hour < 9:
        up_html = f"Not yet<div class=\"sub\">Usually starts around {UPSLOPE_TYPICAL[now.month]}</div>"
    else:
        up_html = "No clear upslope today"

    # line text
    if pos and pos.get("found"):
        line_html = (f"{esc(_cap(line_between(pos['north_lat'], pos['south_lat'], pos['north_of'], pos['south_of'])))}"
                     + (f"<div class=\"sub\">Traced from the backwind at {esc(pos['south_of'])} "
                        f"up to the trades at {esc(pos['north_of'])}</div>" if pos.get("traced") else
                        f"<div class=\"sub\">Last hour: trades at {esc(pos['north_of'])}, "
                        f"upslope at {esc(pos['south_of'])}</div>"))
        if pos["north_lat"] <= NOT_FLYABLE_LAT:
            line_html += f'<div class="alert">{esc(NOT_FLYABLE)}</div>'
        paia = lad[lad.name == "Paia"]
        if len(paia):
            p = paia.iloc[0]
            line_html += (f"<div class=\"sub\">Trades at the coast: Paia "
                          f"{p.wind_dir:03.0f}° at {p.wind_mph:.0f} mph</div>")
    elif pos:
        line_html = f"Not found<div class=\"sub\">{esc(pos.get('why', ''))}</div>"
    else:
        line_html = "Unavailable"
    west = safely(west_intrusion, obs) if len(obs) else None
    if west:
        line_html += f'<div class="alert">{esc(west)}; the line may be further south than shown</div>'
    # Like the base, the line only means something while the upslope runs:
    # before it the slope stations read the night's drainage, which the ladder
    # took for trades (07:55 on 2026-10-14 put the line "north of Pukalani").
    line_on = up is not None and not (evening and evening[0] in ("over", "drain"))
    # Before the upslope the measured line is shown only when the south side
    # already reads upslope -- at least two rungs south of the line, Kula Ag
    # among them (Sam, 2026-10-17: trades north of Red Hill, faint WSW-WNW
    # south of it). On a drainage morning the slope reads easterly, nothing
    # reads upslope, and the forecast line stays (07:55 on 2026-10-14).
    early_line = False
    if (not line_on and up is None and now.hour >= 6 and pos and pos.get("found")
            and isinstance(lad, pd.DataFrame) and len(lad)):
        south = lad[lad.lat <= pos["south_lat"]]
        ups = south[south.side == "upslope"]
        early_line = len(ups) >= 2 and "Kula Ag" in set(ups.name)
    if early_line:
        line_html = line_html.replace("</div>", " (before the upslope)</div>", 1)
    if not line_on and not early_line:
        pos, west = None, None
        line_html = 'Done for the day<div class="sub">The upslope is over</div>'
        if up is None:
            line_html = 'Not yet<div class="sub">Located once the upslope sets in</div>'
            if fc_today is not None:
                name = fc_today["shear_location"]
                lat = dict(cv.LANDMARKS).get(name) if isinstance(name, str) else None
                where = _cap(line_at(lat)) if lat else "No trade line (trades absent)"
                trades = ("" if pd.isna(fc_today["dir_deg"]) else
                          f"Trades {cv.compass(fc_today['dir_deg'])} ({fc_today['dir_deg']:03.0f}°). ")
                line_html = (f"{esc(where)} (forecast)<div class=\"sub\">{trades}"
                             "Measured once the upslope sets in</div>")
                if lat and lat <= NOT_FLYABLE_LAT:
                    line_html += f'<div class="alert">{esc(NOT_FLYABLE)}</div>'

    # lower-slope wind and gusts
    lw = safely(lower_wind, obs) if len(obs) else None
    if lw:
        wind_html = (f'<span class="num">{lw["avg"]:.0f} mph</span>'
                     + (f', gusts to {lw["gust"]:.0f}' if lw["gust"] is not None else "")
                     + f'<div class="sub">Lower elevations, {lw["n"]} stations, last hour</div>'
                     + ('<div class="alert">Gustier than usual</div>' if lw["gusty"] else ""))
    else:
        wind_html = "Unavailable"

    # thermals text
    thermal_word, day_word = None, None
    rated = therm[therm.index >= 10] if therm is not None else None
    model = cap_model(now)
    # The day's rating (10:00-14:00) is for the day in review; the live row
    # keeps the current call until the upslope ends (Sam: at 16:07 it said
    # "10:00-14:00" beside the newest readings).
    if rated is not None and len(rated) and now.hour >= 15:
        mid = rated[(rated.index >= 10) & (rated.index <= 14)]
        day_word = strength_word(th.describe(mid.ti_c.mean())) if len(mid) else None
        day_sun = safely(window_sun, now)
        if day_sun is not None and day_sun < 0.40:      # broad overcast all day
            day_word = "Flat"
    if rated is not None and len(rated):
        hour = int(rated.index[-1])
        thermal_word = strength_word(rated.iloc[-1]["verdict"])
        newest = None
        tn = safely(thermal_now, obs, cap)
        if tn is not None:
            thermal_word, newest = strength_word(tn[0]["verdict"]), tn[1]
        # Flat only when NO part of the Kula side is in sun: under broken cloud,
        # differential heating still makes thermals where the sun hits (Sam).
        shaded = side_best is not None and not in_shadow and side_best < 0.55
        if shaded:
            thermal_word = "Flat"
        therm_html = ((f"Seem <strong>{thermal_word.lower()}</strong> as of {newest:%H:%M}" if newest is not None
                       else f"Seem <strong>{thermal_word.lower()}</strong> at {hour:02d}:00")
                      + f"<div class=\"sub\">Kula Ag's last hour of air through the {model} sounding (thermal.py)"
                      + (f"; no sun anywhere on the Kula side (best {pct(side_best)})" if shaded else "")
                      + "</div>")
    if thermal_word:
        pass
    elif now.hour < 10:
        word = strength_word(fc_today["thermal"]) if fc_today is not None else "--"
        therm_html = (f"Probably <strong>{word.lower()}</strong> (forecast)<div class=\"sub\">thermal.py on the forecast "
                      "air; measured rating just after 10:00</div>"
                      if word in ("Flat", "Light", "Moderate", "Strong") else
                      "Not yet<div class=\"sub\">First rating just after 10:00, for 09:00–10:00</div>")
    else:
        therm_html = "Unavailable<div class=\"sub\">Kula Ag's readings did not come through this update</div>"

    fc_sun = fc_today["sun"] if fc_today is not None else None
    sun_html = ((f'<span class="num">{pct(fc_sun)}</span> of clear sky forecast, 10:00–14:00 (GFS)'
                 f"<div class=\"sub\">Measured from about {first_at}; Kula is in "
                 f"Haleakala's shadow until about {sun_on}</div>")
                if in_shadow and fc_sun is not None and not pd.isna(fc_sun) else
                f"Not yet<div class=\"sub\">Kula is in Haleakala's shadow until about {sun_on}; "
                f"first reading about {first_at}</div>" if in_shadow else
                f'<span class="num">{pct(side_sun)}</span> of clear sky, lower Kula slope'
                f'<div class="sub">Last hour, as of {sun_hour + timedelta(hours=1):%H:%M}; average of '
                + ", ".join(f"{n} {pct(v)}" for n, v in low_sun)
                + ("; Pulehu left out, in trade air" if pulehu_out else "")
                + '</div>'
                if side_sun is not None else
                f'<span class="num">{pct(ka_sun)}</span> of clear sky at Kula Ag'
                f'<div class="sub">Last hour, as of {sun_hour + timedelta(hours=1):%H:%M}</div>'
                if ka_sun is not None else "Unavailable")

    today_chip = rating_chip(thermal_word or (strength_word(fc_today["thermal"])
                                              if fc_today is not None and now.hour < 10 else None))
    said = safely(summary, obs, ladder, up, thermal_word, pos, west, now,
                  None if now.hour < first_hr + 1 else safely(sun_trend, now), evening,
                  cu_clear) if len(obs) else None
    if up is None and now.hour < 12 and fc_today is not None:
        ahead = safely(forecast_summary, fc_today, fc_pair, hours.get(now.date() - timedelta(days=1)),
                       next((i for i in inv_today if i), None), "yesterday")
        if ahead:
            said = f"Today's forecast: {ahead}" + (f" {said}" if said else "")
    if not mc and up is None and now.hour < 12 and PHASE.get("morning") and (_fc := safely(slope_lcl, obs)):
        said = (f"First clouds will form at about {ft(round(_fc, -2))} ft; the slope is drying as it warms."
                + (f" {said}" if said else ""))
    if mc:
        said = ((f"Low cloud is down to about {ft(round(mc['base'], -2))} ft" if mc.get("overhead") else
                 f"No cloud over Kula yet. First clouds will form at about {ft(round(mc['base'], -2))} ft, "
                 "a little higher if the slope warms first")
                + (("; the night's dew is evaporating -- " if mc.get("dewy") else
                    "; " if mc.get("overhead") else ". ")
                   + f"Kula Ag's dew point is up {mc['rise']:.0f} °F "
                   f"since {mc['since']:%H:%M} and it is still {mc['ka_rh']:.0f}% humid."
                   if mc.get("rise", 0) >= 1.5 else ".")
                + (f" {said}" if said else ""))
    summary_html = f'<p class="summary">{esc(said)}</p>' if said else ""
    # The pattern vocabulary (patterns.py): situations the page recognises,
    # each shown with its ID so it can be found, changed or dropped there.
    ka = obs[obs.station_id == "001HI"].sort_values("observed") if len(obs) else obs
    recent = ka[ka.observed >= ka.observed.max() - pd.Timedelta("30min")] if len(ka) else ka
    ctx = {"now": now, "up": up, "morning": mc, "base": base["mid"] if base else None,
           "ka_td_trend": float(recent.dew_f.iloc[-1] - recent.dew_f.iloc[0]) if len(recent) >= 2 else None,
           "ka_sun": ka_sun, "inv": next((i for i in inv_today if i), None),
           "upslope_typical": UPSLOPE_TYPICAL[now.month]}
    _core_now = _core(obs) if len(obs) else obs
    _end = obs.observed.max() if len(obs) else None
    ctx["base_hist"] = ([v for v in (_derived_at(_core_now, _end - pd.Timedelta(minutes=m)) for m in (0, 15, 30, 45))
                         if v is not None] if _end is not None else [])
    _sun = {n: min(v, 1.0) for n, e, v in ladder if v is not None}
    _low = [_sun[n] for n in ("Kula Ag", "Keokea") if n in _sun]
    _high = [_sun[n] for n in ("Kula 1", "Kamehamenui 1", "Park HQ") if n in _sun]
    ctx["low_sun"] = sum(_low) / len(_low) if _low else None
    ctx["high_sun"] = max(_high) if _high else None
    ctx["p4_since"] = _patterns_today(f"{now:%Y-%m-%d}").get("P4c")
    ctx["rh_5_7k"] = ((safely(airmass_noon, cap, now.date()) or {}).get("rh_5_7k")
                      if cap and 11 <= now.hour < 13 else None)
    ctx["sky"] = safely(sky_now, now) if up is not None else None
    ctx["spill"] = safely(summit_spill, now, obs) if len(obs) else None
    fired = (safely(patterns.watch, ctx) or []) if not day_over else []
    safely(log_patterns, now, fired)
    if fired:
        summary_html += ('<div class="watch"><div class="kind">What to watch</div><ul>'
                         + "".join(f'<li>{esc(t)} <span class="pid">{pid} · {esc(n)}</span></li>'
                                   for pid, n, t in fired) + "</ul></div>")
    # The day so far against the morning's forecast (Sam): the review's chart,
    # drawn live from the upslope to the newest reading.
    if up is not None and len(obs) and not day_over:
        live = safely(base_chart, obs, now, up, None, "livechart", "Cloudbase so far, derived", False)
        if live and live[1]:
            summary_html += live[0]

    safely(write_station_map, obs, lad if isinstance(lad, pd.DataFrame) else pd.DataFrame(), now, pos, west)
    safely(log_trace, now, pos)

    entries = safely(log_line, now, pos) or []
    safely(log_cameras, now)
    safely(log_forecast_latest, now)
    safely(review_summary, now)
    review, today_kind, today_title = None, f"Measured today · {now:%H:%M} HST", "Today"
    if day_over:
        review = safely(day_review, obs, now, up, evening, rnet, day_word, model,
                        fc_today, fc_pair, entries, next((i for i in inv_today if i), None))
        if review:
            safely(REVIEW_FILE.write_text, json.dumps(review), "utf-8")
            end_t = upslope_end(rnet, evening) if up is not None else None
            safely(write_scorecard, obs, now, up, end_t, day_word, entries)
    elif before_up and now.hour < 6:            # live from 06:00 (Sam)
        try:
            saved = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
            if saved.get("date") == f"{now.date() - timedelta(days=1):%Y-%m-%d}":
                review = saved
        except Exception:
            pass
    if review:
        rd = datetime.strptime(review["date"], "%Y-%m-%d").date()
        today_kind = "The day in review"
        today_title = f"{'Today' if rd == now.date() else 'Yesterday'} · {rd:%a %d %b}"
        today_chip = rating_chip(review["word"])
        summary_html = f'<p class="summary">{esc(review["summary"])}</p>' + review.get("chart", "")
        therm_html, line_html = review["therm"], review["line"]
        up_html, wind_html = review["up"], review["wind"]
        base_label, base_html, ceil_html = "Cloudbase", review.get("base", ""), ""
        sun_html = None                                   # in the history

    # tomorrow + outlook
    if before_up:
        safely(save_morning_forecast, now)
    # The next flying day: today until its upslope begins, then tomorrow.
    tom = now.date() + timedelta(days=0 if before_up else 1)
    t_html = "<p>Forecast unavailable.</p>"
    rows_html = ""
    tom_chip = ""
    if week is not None and tom in week.index:
        r = week.loc[tom]
        inv_t = next((i for i in (inv_tom if tom > now.date() else inv_today) if i), None)
        line_t = r["shear_location"]
        line_t = ("No trade line (trades absent)" if line_t is None or pd.isna(line_t) else
                  _cap(line_at(dict(cv.LANDMARKS)[line_t])) if line_t in dict(cv.LANDMARKS)
                  else f"Near {line_t}")
        line_alert = (f'<div class="alert">{esc(NOT_FLYABLE)}</div>'
                      if dict(cv.LANDMARKS).get(r["shear_location"], 99) <= NOT_FLYABLE_LAT else "")
        wind_t = "--" if pd.isna(r["dir_deg"]) else f"{cv.compass(r['dir_deg'])} ({r['dir_deg']:03.0f}°)"
        tom_chip = rating_chip(strength_word(r["thermal"]))
        prev = tom - timedelta(days=1)
        said_t = safely(forecast_summary, r, hours.get(tom), hours.get(prev), inv_t,
                        "today" if prev == now.date() else "yesterday")
        rrfs = safely(sd.fetch, "RRFS")
        push = safely(summit_push, rrfs, tom) if rrfs else None
        if rrfs:
            safely(log_alert, tom, push, now)
        # Sam, 2026-10-18: say only that a push-over is possible during the day,
        # and flag the morning look -- cloud on the summit ridge in the morning is
        # the sign; later Kula's own cumulus hide the summit. The model's hour
        # (push) is still logged for scoring.
        push_html = ('<div class="alert">Possible summit cloud push-over during the day: the model has '
                     'moist trade air pushed over the summit ridge on an easterly. If it comes, Kula can '
                     'lose its sun, thermals weaken and the day ends early; the moist air mixed in also '
                     'holds the base lower. <b>Check the summit ridge in the morning</b> (Red Hill camera): '
                     'clouds there are the sign. <span class="pid">P1 · Summit cloud pushed over</span></div>'
                     if push else "")
        # The hourly forecast chart is gone (Sam, 2026-10-17): over 32 days its
        # afternoon shape was right less often than "always flat", and its
        # level lost to the season average. The base is a category instead.
        fc_hours = safely(forecast_base_hours, tom) or {}
        cb_head, cb_sub = (safely(settled_call, tom, cap, fc_hours,
                                  safely(today_level, now) if tom > now.date() else None, True)
                           or (f"{am_pm(hours.get(tom))} ft", ""))
        # No summary paragraph (Sam, 2026-10-17: "the whole summary paragraph
        # is unnecessary"): the rows below say it. said_t is still worked out
        # but not shown.
        t_html = push_html + f"""<dl>
        <dt>Thermals</dt><dd>Probably {esc(strength_word(r['thermal']).lower())}<div class="sub">thermal.py on the forecast air (+3 F, it runs cool); within one category of the measured rating on every test day</div></dd>
        <dt>Cloudbase</dt><dd>{esc(cb_head)}<div class="sub">{esc(cb_sub)}</div></dd>
        <dt>Sun</dt><dd><span class="num">{pct(r['sun'])}</span> of clear sky, 10:00–14:00 (GFS)<div class="sub">The model can't see the slope's own cumulus: when it says 90% or more, Kula has measured about 80%. Good flying days measure about 80%, overcast about 50%, socked in 20%.</div></dd>
        <dt>Shear line</dt><dd>{esc(line_t)}<div class="sub">Trades {esc(wind_t)}</div>{line_alert}</dd>
      </dl>
      <div class="sub">Forecast as of {fc_made.astimezone(HST):%H:%M} HST</div>{rules_line()}"""
        if tom > now.date():
            safely(log_call, tom, ("Low" if cb_head.startswith("Probably a low") else
                                   "May be high" if cb_head.startswith("May be a high") else
                                   "Leaning high" if cb_head == "Leaning high" else "Normal"),
                   strength_word(r["thermal"]), now)
        for day, w in week.iterrows():
            if day < tom:
                continue
            lt = w["shear_location"]
            lt = "--" if lt is None or pd.isna(lt) else esc(line_short(lt))
            # The base as the same category as the Forecast panel (Sam,
            # 2026-10-17: "categorize The Week"). Beyond the model run's reach
            # there is no air aloft to check, so it can only say Normal.
            cat = safely(settled_call, day, cap, safely(forecast_base_hours, day) or {},
                         safely(today_level, now) if day == now.date() + timedelta(days=1) else None,
                         True)
            # "Normal" again (Sam, 2026-10-18), after a day of "--"; "--" only
            # when there is no call at all.
            cat = ("Low" if cat and cat[0].startswith("Probably a low") else
                   "May be high" if cat and cat[0].startswith("May be a high") else
                   "Leaning high" if cat and cat[0] == "Leaning high" else
                   "Normal" if cat else "--")
            rows_html += (f"<tr><td>{day:%a %d}</td><td class=\"n\">{pct(w['sun'])}</td>"
                          f"<td>{cat}</td>"
                          f"<td>{esc(strength_word(w['thermal']))}</td><td>{lt}</td></tr>")

    ladder_html = ""
    for name, elev, share in ladder:
        w = 0 if share is None else max(0, min(1, share)) * 100
        cls = "shaded" if share is not None and share < 0.55 else ""
        # Clear-sky is computed for Kula Ag's height; higher stations exceed 100%.
        shown = None if share is None else min(share, 1.0)
        ladder_html += (f'<div class="row"><span>{esc(name)}</span><span class="elev">{elev:,}</span>'
                        f'<div class="bar {cls}"><span style="width:{w:.0f}%"></span></div>'
                        f'<span class="pct">{"in shadow" if name in SHADOWED else pct(shown)}</span></div>')
    if base and ladder:
        b = round(base["mid"], -2)
        rows = ladder_html.split('<div class="row">')
        out, placed = "", False
        for chunk in rows[1:]:
            elev = int(chunk.split('class="elev">')[1].split("<")[0].replace(",", ""))
            if not placed and elev < b:
                out += (f'<div class="base"><span class="lbl">Cloudbase now</span>'
                        f'<span class="elev">{ft(b)}</span><div class="line"></div><span></span></div>')
                placed = True
            out += '<div class="row">' + chunk
        ladder_html = out

    return PAGE.substitute(
        updated=f"{now:%a %d %b %Y · %H:%M} HST", today_kind=today_kind, today_title=today_title,
        built=f"{now:%Y-%m-%dT%H:%M}",
        today_chip=today_chip, summary=summary_html, base_label=base_label,
        therm=therm_html, base=base_html, up=up_html, line=line_html,
        base_row=(f"<dt>{base_label}</dt><dd>{base_html}</dd>" if base_html else ""),
        sun_row=(f"<dt>Sun</dt><dd>{sun_html}</dd>" if sun_html is not None else ""),
        wind=wind_html,
        ceil_row=(f"<dt>Ceiling</dt><dd>{ceil_html}</dd>" if ceil_html else ""),
        tom_title=f"{'Tomorrow' if tom > now.date() else 'Today'} · {tom:%a %d %b}", tom_chip=tom_chip, tomorrow=t_html,
        rows=rows_html, ladder=ladder_html or (f"<p>Night. First reading tomorrow about {first_at}.</p>"
                                               if night else
                                               f"<p>Not yet: the slope is in Haleakala's shadow until "
                                               f"about {sun_on}. First reading about {first_at}.</p>"
                                               if in_shadow else "<p>Unavailable.</p>"),
        ladder_hour=(f"as of {sun_hour + timedelta(hours=1):%H:%M}"
                     if sun_hour and not in_shadow and not night else "not yet"),
        # From 15:00 the sun is 40 deg up in the WSW and its path crosses a
        # 4,500 ft base 0.4 mi or more downslope -- usually the deck's edge.
        ladder_note=(" Late in the day the sun is low in the west, so a station under the deck "
                     "can read full sun coming in under the deck's edge."
                     if sun_hour and sun_hour.hour >= 15 else ""),
        webcam=WEBCAM,
        diag=("; ".join(PROBLEMS) or "none").replace("--", "-").replace(">", ")"))


PAGE = Template((HERE / "template.html").read_text(encoding="utf-8"))

if __name__ == "__main__":
    out = HERE / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print("wrote", out)
