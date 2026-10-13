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
# Open-Meteo's default model has put Kula Ag's midday cloudbase about this
# far too low (28 days, Sept-Oct 2026). A month's average, not a fitted
# correction -- shown beside the raw number, never instead of it.
FORECAST_BASE_LOW_FT = 650
WEBCAM = ("https://weathercams.faa.gov/map/-128.93458,33.06367,-103.99562,49.85195"
          "/cameraSite/931/details/camera/13415/full")
STALE = pd.Timedelta("75min")
GOOD_VERDICTS = {"Moderate", "Strong", "Boom!"}

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


def thermal_today(obs: pd.DataFrame, now: datetime):
    ka = obs[obs.station_id == "001HI"].copy()
    ka["hour"] = ka.observed.dt.hour
    surf = ka.groupby("hour").agg(temp_f=("temp_f", "mean"),
                                  dewpoint_f=("dew_f", "mean")).reset_index()
    surf["day"] = now.date()
    got = th.assess(surf, RELEASE_FT)
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
    for i in range(len(ts) - 2):
        if all(phase[ts[i + k]] == "up" for k in range(3)):
            return ts[i], last_drain
    return None, last_drain


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
    if got and got[0]:
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


def fresher_ladder(lad: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """The ladder's Mesonet stations from today's readings when those are newer.

    convergence.observed() reads api.weather.gov, about 50 minutes behind
    for Piiholo, Kula Ag and Keokea; the page's own readings come from HCDP.
    """
    if obs.empty:
        return lad
    rows = lad.to_dict("records")
    for sid, name, lat, elev in cv.LADDER:
        o = obs[(obs.station_id == sid) & obs.wind_dir.notna()]
        if o.empty:
            continue
        r = o.loc[o.observed.idxmax()]
        new = {"station": sid, "name": name, "lat": lat, "elev_ft": elev,
               "observed_local": pd.Timestamp(r.observed), "wind_dir": r.wind_dir,
               "wind_mph": r.wind_mph, "side": cv._side(r.wind_dir)}
        old = next((x for x in rows if x["station"] == sid), None)
        if old is None:
            rows.append(new)
        elif new["observed_local"] > old["observed_local"]:
            old.update(new)
    return pd.DataFrame(rows)


def line_now(obs: pd.DataFrame):
    lad = fresher_ladder(cv.observed(), obs)
    if lad.empty:
        return None, lad
    lad = lad[lad.observed_local >= lad.observed_local.max() - STALE]
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


def inversion_on(day, path):
    found = []
    for h in ("11:00", "14:00"):
        try:
            inv = sd.inversion(sd.at(pd.Timestamp(f"{day} {h}"), path=path))
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


def fresh_forecast(now: datetime, gfs) -> dict:
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
        "inversions": {str(d): inversion_on(d, gfs) if gfs else [None, None] for d in days},
    }


def forecast(now: datetime, gfs) -> dict | None:
    """Last saved forecast if under an hour old, else a fresh one, else the saved one."""
    saved = None
    try:
        saved = json.loads(FORECAST_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if saved and now - datetime.fromisoformat(saved["made"]) < FORECAST_MAX_AGE:
        return saved
    got = safely(fresh_forecast, now, gfs)
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
    gfs = safely(sd.fetch, "GFS")

    # today
    base = safely(cloudbase_now, obs) if len(obs) else None
    therm = safely(thermal_today, obs, now) if len(obs) else None
    up, last_drain = safely(upslope_since, obs) or (None, None) if len(obs) else (None, None)
    ladder_start = ladder_or_last(now)
    ladder, sun_hour = ladder_start if ladder_start else ([], None)
    pos_lad = safely(line_now, obs)
    pos, lad = pos_lad if pos_lad else (None, pd.DataFrame())
    week, hours, inv_today, inv_tom, fc_made = unpack(forecast(now, gfs), now)

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

    # ceiling text
    inv = next((i for i in inv_today if i), None)
    if gfs is None:
        ceil_html = "Unavailable"
    elif inv is None:
        ceil_html = "No cap in the model<div class=\"sub\">No inversion over Kahului today</div>"
    else:
        kb = base["mid"] if base else None
        rng = f"{ft(inv['base_ft'])}–{ft(inv['top_ft'])} ft"
        if kb is not None and inv["base_ft"] < kb - 300:
            ceil_html = (f"Probably no cap<div class=\"sub\">Kahului inversion {rng}, "
                         f"below Kula's base</div>")
        else:
            ceil_html = (f"Possible cap near the base<div class=\"sub\">Kahului inversion "
                         f"{rng}</div>")

    # upslope text
    if up is not None:
        up_html = f'On since about <span class="num">{up:%H:%M}</span>'
        if last_drain is not None:
            up_html += f'<div class="sub">Drainage ended about {last_drain:%H:%M}</div>'
    elif now.hour < 9:
        up_html = "Not yet<div class=\"sub\">Usually starts around 08:15</div>"
    else:
        up_html = "No clear upslope today"

    # line text
    if pos and pos.get("found"):
        marks = ", ".join(pos["landmarks"]) or "no named place"
        line_html = (f"Between <strong>{esc(pos['north_of'])}</strong> and "
                     f"<strong>{esc(pos['south_of'])}</strong>"
                     f"<div class=\"sub\">Covers {esc(marks)}</div>")
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

    # thermals text
    if latest is not None:
        therm_html = (f"<strong>{esc(verdict_now)}</strong> at {int(therm.index[-1]):02d}:00"
                      f"<div class=\"sub\">From Kula Ag's measured air (thermal.py)</div>")
    elif now.hour < 9:
        therm_html = "Not yet<div class=\"sub\">Rated from 09:00</div>"
    else:
        therm_html = "Unavailable<div class=\"sub\">Kula Ag's readings did not come through this update</div>"

    sun_html = (f'<span class="num">{pct(ka_sun)}</span> of clear sky at Kula Ag'
                f'<div class="sub">{sun_hour:%H:%M}–{sun_hour + timedelta(hours=1):%H:%M}</div>'
                if ka_sun is not None else "Unavailable")

    today_chip = day_chip(verdict_now, ka_sun)

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
        line_t = "No trade line (trades absent)" if line_t is None or pd.isna(line_t) else f"Near {line_t}"
        wind_t = "--" if pd.isna(r["dir_deg"]) else f"{cv.compass(r['dir_deg'])} ({r['dir_deg']:03.0f}°)"
        cap_t = ("No inversion in the model" if inv_t is None else
                 f"Kahului inversion {ft(inv_t['base_ft'])}–{ft(inv_t['top_ft'])} ft")
        tom_chip = day_chip(r["thermal"], r["sun"])
        t_html = f"""<dl>
        <dt>Thermals</dt><dd>{esc(str(r['thermal']))}<div class="sub">From forecast surface air, which runs a few degrees cool</div></dd>
        <dt>Cloudbase</dt><dd><span class="num">{am_pm(hours.get(tom))} ft</span><div class="sub">11:00 → 14:00</div></dd>
        <dt>Sun</dt><dd><span class="num">{pct(r['sun'])}</span> of clear sky, 10:00–14:00 (GFS)<div class="sub">Sun share on past days: good flying days about 80%, overcast about 50%, socked in 20%</div></dd>
        <dt>Shear line</dt><dd>{esc(line_t)}<div class="sub">Trades {esc(wind_t)}</div></dd>
        <dt>Ceiling</dt><dd>{esc(cap_t)}</dd>
      </dl>
      <div class="sub">Forecast as of {fc_made.astimezone(HST):%H:%M} HST</div>"""
        for day, w in week.iterrows():
            if day < tom:
                continue
            lt = w["shear_location"]
            lt = "--" if lt is None or pd.isna(lt) else esc(lt)
            rows_html += (f"<tr><td>{day:%a %d}</td><td class=\"n\">{pct(w['sun'])}</td>"
                          f"<td class=\"n\">{am_pm(hours.get(day))}</td>"
                          f"<td>{esc(str(w['thermal']))}</td><td>{lt}</td></tr>")

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
        today_chip=today_chip,
        therm=therm_html, base=base_html, sun=sun_html, up=up_html, line=line_html,
        ceil=ceil_html, tom_title=f"{'Tomorrow' if tom > now.date() else 'Today'} · {tom:%a %d %b}", tom_chip=tom_chip, tomorrow=t_html,
        rows=rows_html, ladder=ladder_html or "<p>Unavailable.</p>",
        ladder_hour=(f"{sun_hour:%H:%M}–{sun_hour + timedelta(hours=1):%H:%M}" if sun_hour else ""),
        webcam=WEBCAM,
        diag=("; ".join(PROBLEMS) or "none").replace("--", "-").replace(">", ")"))


PAGE = Template((HERE / "template.html").read_text(encoding="utf-8"))

if __name__ == "__main__":
    out = HERE / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print("wrote", out)
