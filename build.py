"""Build the Kula Today page: measured conditions now, and the outlook.

Runs on GitHub Actions every 30 minutes in daylight and writes
docs/index.html, which GitHub Pages serves. All the meteorology comes from
the KulaModel repository (checked out beside this one, or wherever
$KULAMODEL points), so there is one copy of the forecast code, not two.

Every section is built on its own: a source that fails shows as
"unavailable" and the rest of the page still publishes.
"""

from __future__ import annotations

import html
import os
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from string import Template
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ.get("KULAMODEL", str(HERE / "KulaModel")))

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

import convergence as cv                # noqa: E402
import mesonet                          # noqa: E402
import sampler                          # noqa: E402
import sounding as sd                   # noqa: E402
import thermal as th                    # noqa: E402
from kula import OFF_SLOPE_STATIONS     # noqa: E402

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


def safely(fn, *a):
    try:
        return fn(*a)
    except Exception:
        traceback.print_exc()
        return None


# -- today, measured ---------------------------------------------------------

def observations(now: datetime) -> pd.DataFrame:
    start = now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    rows = []
    for st in sampler.STATIONS:
        try:
            got, _ = sampler.read_nws(st, start)
        except Exception:
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
    """Share of clear-sky sun over the last full hour, per station, top down."""
    end = now.replace(minute=0, second=0, microsecond=0).replace(tzinfo=None)
    start = end - timedelta(hours=1)
    rows: dict[str, list] = {}
    for obs_t, st, w in sampler.read_mesonet_sun(start, end):
        if start <= obs_t < end:
            rows.setdefault(st.id, []).append((obs_t, w))
    for st in (s for s in sampler.STATIONS if s.raws):
        for t, w in sampler.read_raws_sun(st.raws, end.date(), end.date()):
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


def line_now():
    lad = cv.observed()
    if lad.empty:
        return None, lad
    lad = lad[lad.observed_local >= lad.observed_local.max() - STALE]
    return cv.position(lad), lad


# -- forecast ------------------------------------------------------------------

def outlook() -> pd.DataFrame:
    return cv.soaring()


def inversion_on(day, path):
    found = []
    for h in ("11:00", "14:00"):
        try:
            found.append(sd.inversion(sd.at(pd.Timestamp(f"{day} {h}"), path=path)))
        except Exception:
            found.append(None)
    return found


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
    ladder_start = safely(sun_ladder, now)
    ladder, sun_hour = ladder_start if ladder_start else ([], None)
    pos_lad = safely(line_now)
    pos, lad = pos_lad if pos_lad else (None, pd.DataFrame())
    inv_today = inversion_on(now.date(), gfs) if gfs else [None, None]
    week = safely(outlook)

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
                      f"<div class=\"sub\">Lift to about {ft(latest['usable_top_ft'])} ft, "
                      f"from Kula Ag's measured air (thermal.py)</div>")
    else:
        therm_html = "Not yet<div class=\"sub\">Rated from 09:00</div>"

    sun_html = (f'<span class="num">{pct(ka_sun)}</span> of clear sky at Kula Ag'
                f'<div class="sub">{sun_hour:%H:%M}–{sun_hour + timedelta(hours=1):%H:%M}</div>'
                if ka_sun is not None else "Unavailable")

    today_chip = day_chip(verdict_now, ka_sun)

    # tomorrow + outlook
    tom = now.date() + timedelta(days=1)
    t_html = "<p>Forecast unavailable.</p>"
    rows_html = ""
    tom_chip = ""
    if week is not None and tom in week.index:
        r = week.loc[tom]
        inv_t = next((i for i in (inversion_on(tom, gfs) if gfs else []) if i), None)
        raw = r["cloudbase_ft"]
        adj = None if pd.isna(raw) else raw + FORECAST_BASE_LOW_FT
        line_t = r["shear_location"]
        line_t = "No trade line (trades absent)" if line_t is None or pd.isna(line_t) else f"Near {line_t}"
        wind_t = "--" if pd.isna(r["dir_deg"]) else f"{cv.compass(r['dir_deg'])} ({r['dir_deg']:03.0f}°)"
        cap_t = ("No inversion in the model" if inv_t is None else
                 f"Kahului inversion {ft(inv_t['base_ft'])}–{ft(inv_t['top_ft'])} ft")
        tom_chip = day_chip(r["thermal"], r["sun"])
        t_html = f"""<dl>
        <dt>Thermals</dt><dd>{esc(str(r['thermal']))}<div class="sub">From forecast surface air, which runs a few degrees cool</div></dd>
        <dt>Cloudbase</dt><dd>About <span class="num">{ft(round(adj, -2)) if adj else '--'} ft</span><div class="sub">Model {ft(raw)} ft; it has run about {FORECAST_BASE_LOW_FT} ft low at Kula</div></dd>
        <dt>Sun</dt><dd><span class="num">{pct(r['sun'])}</span> of clear sky, 10:00–14:00 (GFS)<div class="sub">Good Kula days have measured about 80%; overcast 20–48%</div></dd>
        <dt>Shear line</dt><dd>{esc(line_t)}<div class="sub">Trades {esc(wind_t)}</div></dd>
        <dt>Ceiling</dt><dd>{esc(cap_t)}</dd>
      </dl>"""
        for day, w in week.iterrows():
            if day < tom:
                continue
            lt = w["shear_location"]
            lt = "--" if lt is None or pd.isna(lt) else esc(lt)
            rows_html += (f"<tr><td>{day:%a %d}</td><td class=\"n\">{pct(w['sun'])}</td>"
                          f"<td class=\"n\">{ft(round(w['cloudbase_ft'] + FORECAST_BASE_LOW_FT, -2) if not pd.isna(w['cloudbase_ft']) else None)}</td>"
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
        updated=f"{now:%a %d %b %Y · %H:%M} HST", today_chip=today_chip,
        therm=therm_html, base=base_html, sun=sun_html, up=up_html, line=line_html,
        ceil=ceil_html, tom_title=f"{tom:%a %d %b}", tom_chip=tom_chip, tomorrow=t_html,
        rows=rows_html, ladder=ladder_html or "<p>Unavailable.</p>",
        ladder_hour=(f"{sun_hour:%H:%M}–{sun_hour + timedelta(hours=1):%H:%M}" if sun_hour else ""),
        webcam=WEBCAM)


PAGE = Template((HERE / "template.html").read_text(encoding="utf-8"))

if __name__ == "__main__":
    out = HERE / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print("wrote", out)
