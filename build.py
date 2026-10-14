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
FORECAST_BASE_LOW_FT = 400
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
HCDP_VARS = "Tair_1_Avg,RH_1_Avg,WS_1_Avg,WDrs_1_Avg"


def dewpoint_c(t_c: float, rh: float) -> float:
    a, b = 17.625, 243.04
    g = np.log(rh / 100) + a * t_c / (b + t_c)
    return b * g / (a - g)


def read_hcdp(start: datetime) -> dict[str, list]:
    """Today's Mesonet readings by station, oldest first, as sampler Readings."""
    by_hcdp = {s.hcdp: s for s in sampler.STATIONS if s.hcdp}
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
    for st in sampler.STATIONS:
        got = hcdp.get(st.id)
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
    core = obs[~obs.station_id.isin(OFF_SLOPE_STATIONS | {"TT925"}) & (obs.elev_ft <= 4400)
               & (obs.suspect == "") & (obs.observed >= fresh_after)]
    latest = core.sort_values("observed").groupby("station_id").tail(1)
    if len(latest) < 3:
        return None
    return {"mid": float(latest.lcl_ft.median()), "lo": float(latest.lcl_ft.min()),
            "hi": float(latest.lcl_ft.max()), "n": len(latest),
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
        return f"north of {sn}, reaching the coast near {nn}"
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
    return obs[~obs.station_id.isin(OFF_SLOPE_STATIONS | {"TT925"}) & (obs.elev_ft <= 4400)
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
        cur, ago = _base_at(core, t), _base_at(core, t - pd.Timedelta("60min"))
        if cur is not None and ago is not None and t >= obs.observed.max() - STALE:
            d = cur - ago
            trend = ("rising" if d >= 300 else "edging up" if d >= 150 else
                     "lowering" if d <= -300 else "edging down" if d <= -150 else "steady")
            if d <= -300:
                worse.append("base lowering")
            elif d >= 300:
                better.append("base rising")
            day = t.normalize()
            morning = [x for x in (_base_at(core, day + pd.Timedelta(minutes=m))
                                   for m in range(600, 721, 30)) if x is not None]
            if t.hour >= 12 and morning and max(morning) - cur >= 700:
                out.append(f"Cloudbase has come down from its late-morning high and is {trend} now.")
            elif t.hour < 12 and morning and cur >= max(morning) - 200 and d >= 150:
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
            shaded = [i for i, (_, s) in enumerate(lit) if s < 0.5]
            top = shaded[0] if shaded else None
            above = [n for n, s in lit[:top] if s >= 0.75] if top else []
            if cu_clear:
                pass        # no cumulus: a low reading is terrain shadow or haze, not a deck
            elif top is not None and above:
                text += (f"; the cloud tops out between {lit[top][0]} and {above[-1]}, "
                         "with sun above it")
            elif top == 0:
                text += "; cloud reaches all the way to the Summit"
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
        out.append(line + (f", and thermals rate {verdict}." if verdict and verdict != "--" else "."))
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
    if am is not None and pm is not None:
        d = pm - am
        drift = ("rising through the afternoon" if d >= 300 else
                 "lowering after midday" if d <= -300 else "holding through the day")
        bits.append(drift if bits else f"a base {drift}")
    if sky:
        out.append(f"{sky}.")
    if bits:
        out.append(_cap(", ".join(bits)) + ".")
    word = sun_rating(sun)
    if word != "--":
        out.append({"Good": "Thermals look good.", "OK": "Thermals look OK.",
                    "Nope": "Thermals: Nope."}[word])
    if inv and am is not None:
        out.append("The Kahului inversion sits at or above the expected base, so it may cap the tops."
                   if inv["base_ft"] >= am - 300 else
                   "The Kahului inversion is below the expected base, so probably no cap.")
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


def upslope_since(obs: pd.DataFrame):
    core = obs[~obs.station_id.isin(OFF_SLOPE_STATIONS) & (obs.elev_ft <= 4400)].dropna(
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
    ending = r.index[(r < 100).values]
    if len(ending):
        return "ending", ending[0]
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
    """Share of clear-sky sun over the last full hour, per station, top down.

    Mesonet and RAWS each fail on their own; a HCDP connect timeout at
    14:23 on 2026-10-12 used to blank the whole ladder, RAWS included.
    """
    end = now.replace(minute=0, second=0, microsecond=0).replace(tzinfo=None)
    start = end - timedelta(hours=1)
    rows: dict[str, list] = {}
    try:
        for obs_t, st, w in _retrying(sampler.read_mesonet_sun)(start, end):
            if start <= obs_t < end:
                rows.setdefault(st.id, []).append((obs_t, w))
    except Exception as e:
        PROBLEMS.append(f"Mesonet sun: {type(e).__name__}: {e}")
    for st in (s for s in sampler.STATIONS if s.raws):
        try:
            got = _retrying(sampler.read_raws_sun)(st.raws, end.date(), end.date())
        except Exception as e:
            PROBLEMS.append(f"RAWS sun {st.raws}: {type(e).__name__}: {e}")
            continue
        for t, w in got:
            if t == end:                                  # hour-ending stamp
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
    for sid, name, lat, elev in cv.LADDER:
        have = (obs[(obs.station_id == sid) & obs.wind_dir.notna()]
                [["observed", "wind_dir", "wind_mph"]] if len(obs) else pd.DataFrame())
        if have.empty:
            try:
                have = _retrying(_ladder_readings)(sid, since)
            except Exception:
                continue
        if have.empty:
            continue
        have = have[have.observed >= cut]
        if have.empty:
            continue
        d = cv.mean_direction(have.wind_dir)
        rows.append({"station": sid, "name": name, "lat": lat, "elev_ft": elev,
                     "observed_local": have.observed.max(), "wind_dir": d,
                     "wind_mph": float(have.wind_mph.mean()), "side": cv._side(d),
                     "n": len(have)})
    return pd.DataFrame(rows)


def line_now(obs: pd.DataFrame, now: datetime):
    lad = smoothed_ladder(obs, now)
    if lad.empty:
        return None, lad
    return cv.position(lad), lad


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


def fresh_forecast(now: datetime, cap) -> dict:
    week = outlook()
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


def build() -> str:
    now = datetime.now(HST)
    obs = safely(observations, now)
    obs = obs if obs is not None else pd.DataFrame()
    cap = safely(sd.fetch, cap_model(now))

    # today
    base = safely(cloudbase_now, obs) if len(obs) else None
    therm = safely(thermal_today, obs, now, cap) if len(obs) else None
    up, last_drain = safely(upslope_since, obs) or (None, None) if len(obs) else (None, None)
    evening = (safely(evening_state, obs, safely(net_radiation, now) if now.hour >= 15 else None)
               if up is not None and len(obs) else None)
    ladder_start = ladder_or_last(now)
    ladder, sun_hour = ladder_start if ladder_start else ([], None)
    # Haleakala's shadow: direct sun reaches Kula Ag 07:15-08:45 by season (SUN_ON),
    # so before the first full sunlit hour the stations get only skylight and read as
    # if under cloud -- 26% at Kula Ag with a clear sky on 2026-10-14 (Sam).
    sun_on, first_hr = SUN_ON[now.month], first_sun_hour(now.month)
    in_shadow = sun_hour is None or sun_hour.hour < first_hr
    if in_shadow:
        ladder = []
    pos_lad = safely(line_now, obs, now)
    pos, lad = pos_lad if pos_lad else (None, pd.DataFrame())
    week, hours, inv_today, inv_tom, fc_made = unpack(forecast(now, cap), now)
    # Today's forecast, shown in place of each measured item until its gate
    # opens (Sam: "I'm drinking my coffee and thinking about going flying").
    fc_today = week.loc[now.date()] if week is not None and now.date() in week.index else None
    fc_pair = hours.get(now.date()) if hours else None

    ka_sun = next((s for n, e, s in ladder if n == "Kula Ag"), None)
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
    if up is None and now.hour < 12:
        if fc_pair and fc_pair[0] is not None:
            base_label = "Forecast<br>Cloudbase"      # not derived until the upslope
        base_html = (f'<span class="num">{am_pm(fc_pair)} ft</span> forecast'
                     '<div class="sub">11:00 → 14:00. Derived from the stations once the '
                     f'upslope sets in, usually about {UPSLOPE_TYPICAL[now.month]}</div>'
                     if fc_pair and fc_pair[0] is not None else
                     'Not yet<div class="sub">Derived once the upslope sets in, '
                     f'usually about {UPSLOPE_TYPICAL[now.month]}</div>')
    elif base and cu_clear:
        base_html = (f'No cumulus yet<div class="sub">They would form near about '
                     f'{ft(round(base["mid"], -2))} ft, the condensation level derived from '
                     f'the stations; no cloud over the slope in the sunlight readings</div>')
    elif evening and evening[0] in ("over", "drain"):
        # Same for the evening: once the slope is losing heat the stations
        # read the drainage layer, not a cloudbase (Sam: gate the evening).
        base_html = ('Done for the day<div class="sub">The upslope is over; the derived '
                     "base returns with tomorrow's upslope</div>")

    # ceiling text -- shown only when the model has a lid in reach (Sam,
    # 2026-10-14): "none" is the usual case, and NAM/RRFS miss about half
    # the real ones anyway, so a standing "no lid" line said little.
    inv = next((i for i in inv_today if i), None)
    ceil_html = ""
    if inv is not None:
        kb = base["mid"] if base else None
        rng = f"{ft(inv['base_ft'])}–{ft(inv['top_ft'])} ft"
        if kb is not None and inv["base_ft"] < kb - 300:
            ceil_html = (f"Probably no cap<div class=\"sub\">Kahului inversion {rng}, "
                         f"below Kula's base</div>")
        else:
            ceil_html = (f"Possible cap near the base<div class=\"sub\">Kahului inversion "
                         f"{rng}</div>")

    # upslope text
    if evening and up is not None:
        kind, since = evening
        up_html = {
            "drain": f'Over<div class="sub">Drainage since about {since:%H:%M}; '
                     f'upslope began about {up:%H:%M}</div>',
            "over": f'Over<div class="sub">Slope losing heat since about {since:%H:%M}; '
                    f'upslope began about {up:%H:%M}</div>',
            "ending": f'Ending<div class="sub">Slope heating nearly stopped about {since:%H:%M}; '
                      f'upslope began about {up:%H:%M}</div>',
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
                     f"<div class=\"sub\">Last hour: trades at {esc(pos['north_of'])}, "
                     f"upslope at {esc(pos['south_of'])}</div>")
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
    if not line_on:
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

    # thermals text
    thermal_word = None
    rated = therm[therm.index >= 10] if therm is not None else None
    model = cap_model(now)
    if rated is not None and len(rated) and now.hour >= 15:
        mid = rated[(rated.index >= 10) & (rated.index <= 14)]
        thermal_word = strength_word(th.describe(mid.ti_c.mean())) if len(mid) else None
        day_sun = safely(window_sun, now)
        if day_sun is not None and day_sun < 0.55:
            thermal_word = "Flat"
        if thermal_word:
            therm_html = (f"<strong>{thermal_word}</strong> for the day, 10:00–14:00"
                          f"<div class=\"sub\">Kula Ag's measured air through the {model} sounding "
                          f"(thermal.py)" + (f"; {pct(min(day_sun, 1))} of clear-sky sun" if day_sun is not None else "")
                          + "</div>")
    elif rated is not None and len(rated):
        hour = int(rated.index[-1])
        thermal_word = strength_word(rated.iloc[-1]["verdict"])
        shaded = ka_sun is not None and not in_shadow and ka_sun < 0.55
        if shaded:
            thermal_word = "Flat"
        therm_html = (f"<strong>{thermal_word}</strong> at {hour:02d}:00"
                      f"<div class=\"sub\">Kula Ag's measured air through the {model} sounding (thermal.py)"
                      + (f"; shaded, {pct(ka_sun)} of clear-sky sun" if shaded else "") + "</div>")
    if thermal_word:
        pass
    elif now.hour < 10:
        word = sun_rating(fc_today["sun"]) if fc_today is not None else "--"
        therm_html = (f"<strong>{word}</strong> forecast<div class=\"sub\">From the GFS sunlight "
                      "forecast; measured rating just after 10:00</div>" if word != "--" else
                      "Not yet<div class=\"sub\">First rating just after 10:00, for 09:00–10:00</div>")
    else:
        therm_html = "Unavailable<div class=\"sub\">Kula Ag's readings did not come through this update</div>"

    fc_sun = fc_today["sun"] if fc_today is not None else None
    sun_html = ((f'<span class="num">{pct(fc_sun)}</span> of clear sky forecast, 10:00–14:00 (GFS)'
                 f"<div class=\"sub\">Measured from just after {first_hr + 1:02d}:00; Kula is in "
                 f"Haleakala's shadow until about {sun_on}</div>")
                if in_shadow and fc_sun is not None and not pd.isna(fc_sun) else
                f"Not yet<div class=\"sub\">Kula is in Haleakala's shadow until about {sun_on}; "
                f"first reading just after {first_hr + 1:02d}:00</div>" if in_shadow else
                f'<span class="num">{pct(ka_sun)}</span> of clear sky at Kula Ag'
                f'<div class="sub">{sun_hour:%H:%M}–{sun_hour + timedelta(hours=1):%H:%M}</div>'
                if ka_sun is not None else "Unavailable")

    today_chip = rating_chip(thermal_word or (sun_rating(fc_today["sun"])
                                              if fc_today is not None and now.hour < 10 else None))
    said = safely(summary, obs, ladder, up, thermal_word, pos, west, now,
                  None if now.hour < first_hr + 1 else safely(sun_trend, now), evening,
                  cu_clear) if len(obs) else None
    if up is None and now.hour < 12 and fc_today is not None:
        ahead = safely(forecast_summary, fc_today, fc_pair, hours.get(now.date() - timedelta(days=1)),
                       next((i for i in inv_today if i), None), "yesterday")
        if ahead:
            said = f"Today's forecast: {ahead}" + (f" {said}" if said else "")
    summary_html = f'<p class="summary">{esc(said)}</p>' if said else ""

    # tomorrow + outlook
    # Before 06:00 the next flying day is today, so the forecast panel shows it.
    tom = now.date() + timedelta(days=0 if now.hour < 6 else 1)
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
        cap_t = (f"Kahului inversion {ft(inv_t['base_ft'])}–{ft(inv_t['top_ft'])} ft"
                 if inv_t else "")
        tom_chip = rating_chip(sun_rating(r["sun"]))
        prev = tom - timedelta(days=1)
        said_t = safely(forecast_summary, r, hours.get(tom), hours.get(prev), inv_t,
                        "today" if prev == now.date() else "yesterday")
        t_html = (f'<p class="summary">{esc(said_t)}</p>' if said_t else "") + f"""<dl>
        <dt>Thermals</dt><dd>{esc(sun_rating(r['sun']))}<div class="sub">From the GFS sunlight forecast; right class about 7 days in 10</div></dd>
        <dt>Cloudbase</dt><dd><span class="num">{am_pm(hours.get(tom))} ft</span><div class="sub">11:00 → 14:00</div></dd>
        <dt>Sun</dt><dd><span class="num">{pct(r['sun'])}</span> of clear sky, 10:00–14:00 (GFS)<div class="sub">The model can't see the slope's own cumulus: when it says 90% or more, Kula has measured about 80%. Good flying days measure about 80%, overcast about 50%, socked in 20%.</div></dd>
        <dt>Shear line</dt><dd>{esc(line_t)}<div class="sub">Trades {esc(wind_t)}</div>{line_alert}</dd>
        {f"<dt>Ceiling</dt><dd>{esc(cap_t)}</dd>" if inv_t else ""}
      </dl>
      <div class="sub">Forecast as of {fc_made.astimezone(HST):%H:%M} HST</div>"""
        for day, w in week.iterrows():
            if day < tom:
                continue
            lt = w["shear_location"]
            lt = "--" if lt is None or pd.isna(lt) else esc(line_short(lt))
            rows_html += (f"<tr><td>{day:%a %d}</td><td class=\"n\">{pct(w['sun'])}</td>"
                          f"<td class=\"n\">{am_pm(hours.get(day))}</td>"
                          f"<td>{esc(sun_rating(w['sun']))}</td><td>{lt}</td></tr>")

    ladder_html = ""
    for name, elev, share in ladder:
        w = 0 if share is None else max(0, min(1, share)) * 100
        cls = "shaded" if share is not None and share < 0.55 else ""
        # Clear-sky is computed for Kula Ag's height; higher stations exceed 100%.
        shown = None if share is None else min(share, 1.0)
        ladder_html += (f'<div class="row"><span>{esc(name)}</span><span class="elev">{elev:,}</span>'
                        f'<div class="bar {cls}"><span style="width:{w:.0f}%"></span></div>'
                        f'<span class="pct">{pct(shown)}</span></div>')
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
        updated=f"{now:%a %d %b %Y · %H:%M} HST", updated_time=f"{now:%H:%M}",
        today_chip=today_chip, summary=summary_html, base_label=base_label,
        therm=therm_html, base=base_html, sun=sun_html, up=up_html, line=line_html,
        ceil_row=(f"<dt>Ceiling</dt><dd>{ceil_html}</dd>" if ceil_html else ""),
        tom_title=f"{'Tomorrow' if tom > now.date() else 'Today'} · {tom:%a %d %b}", tom_chip=tom_chip, tomorrow=t_html,
        rows=rows_html, ladder=ladder_html or (f"<p>Not yet: the slope is in Haleakala's shadow until "
                                               f"about {sun_on}. First reading just after {first_hr + 1:02d}:00.</p>"
                                               if in_shadow else "<p>Unavailable.</p>"),
        ladder_hour=(f"{sun_hour:%H:%M}–{sun_hour + timedelta(hours=1):%H:%M}"
                     if sun_hour and not in_shadow else "not yet"),
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
