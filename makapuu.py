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
WEBCAM = ("https://weathercams.faa.gov/map/-158.09368,20.54805,-156.28094,22.06794"
          "/cameraSite/588/details/camera/12153/full")     # Waimanalo (NW), the card's view (Sam)
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


def bellows_now(now: datetime):
    r = requests.get(f"https://api.weather.gov/stations/{BELLOWS}/observations/latest", headers=HEAD, timeout=30)
    p = r.json()["properties"]
    t = datetime.fromisoformat(p["timestamp"]).astimezone(HST)
    if now - t > OBS_MAX_AGE or p["windSpeed"]["value"] is None:
        return None, t
    mph = p["windSpeed"]["value"] * 0.621371
    gust = None if p["windGust"]["value"] is None else p["windGust"]["value"] * 0.621371
    return {"dir": p["windDirection"]["value"], "mph": mph, "gust": gust, "at": t}, t


def kaneohe() -> dict | None:
    rows = requests.get(METAR, timeout=30).json()
    if not rows:
        return None
    m = max(rows, key=lambda r: r["obsTime"])
    return {"t_c": m.get("temp"), "td_c": m.get("dewp"), "clouds": m.get("clouds") or [], "cover": m.get("cover"),
            "at": datetime.fromtimestamp(m["obsTime"], timezone.utc).astimezone(HST)}


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
    obs, obs_t = got if got else (None, None)
    k = _try(kaneohe, problems=problems)
    hour_now = now.replace(minute=0, second=0, microsecond=0)

    # --- Today: the ridge now
    if obs:
        w, src = obs, f'Bellows, {obs["at"]:%H:%M}'
    elif grid.get(hour_now):
        w = grid[hour_now]
        src = ("NWS forecast for this hour: Bellows not reporting"
               + (f" since {obs_t:%d %b %H:%M}" if obs_t else ""))
    else:
        w, src = None, "No wind available"
    word, why = rate(w["dir"], w["mph"], w["gust"]) if w else ("Unknown", "")
    night = not (6 <= now.hour < 19)
    today_chip = chip(word) if w and not night else ""
    fly_html = (f"<strong>{esc(word)}</strong>" + (f'<div class="sub">{esc(why)}</div>' if why else "")) if w else "Unavailable"
    wind_html = (f'<span class="num">{esc(wind_text(w))}</span><div class="sub">{esc(src)}</div>') if w else "Unavailable"
    dir_html = esc(return_note(w["dir"])) if w and w.get("dir") is not None else ""

    base = crest_base(k)
    if base is not None:
        if base < CREST_LOW_FT:
            cap = f"below the crest behind Waimānalo ({CREST_LOW_FT:,}–{CREST_HIGH_FT:,} ft): cloud on the Koʻolau likely"
        elif base < CREST_HIGH_FT:
            cap = "cloud likely on the higher crest downrange"
        else:
            cap = "above the crest downrange"
        layers = ", ".join(f'{c["cover"]} {c["base"]:,} ft' for c in k["clouds"] if c.get("base"))
        ceil = layers or ("clear below 12,000 ft" if (k.get("cover") or "").startswith(("CLR", "SKC")) else "")
        base_html = (f'<span class="num">About {base:,} ft</span>, {esc(cap)}'
                     f'<div class="sub">Kaneohe\'s air at {k["at"]:%H:%M}, lifted over the Koʻolau. '
                     f'Downrange only; it does not decide flyability.'
                     + (f" Kaneohe's ceilometer: {esc(ceil)}." if ceil else "") + "</div>")
    else:
        base_html = "Unavailable"

    rest = [(t, grid[t]) for t in sorted(grid) if t.date() == now.date() and t.hour in DAY_HOURS and t > hour_now]
    rest_html = hour_table(rest) if rest else '<span class="sub">Nothing left of today\'s daylight.</span>'

    # --- Forecast: tomorrow, then the week
    tom = now.date() + timedelta(days=1)
    trows = [(t, grid[t]) for t in sorted(grid) if t.date() == tom and t.hour in DAY_HOURS]
    tword, tline = day_line([(t, rate(v["dir"], v["mph"], v["gust"])[0]) for t, v in trows]) if trows else ("", "")
    noon = grid.get(datetime.combine(tom, datetime.min.time()).replace(hour=12, tzinfo=HST))
    tom_html = (f'<p class="summary">{esc(tline)} Midday: {esc(wind_text(noon))}. '
                f'{esc(return_note(noon["dir"]) if noon and noon.get("dir") is not None else "")}</p>'
                + hour_table(trows)) if trows else "<p>No forecast available.</p>"

    week = ""
    for k_day in range(1, 7):
        d = now.date() + timedelta(days=k_day)
        rows = [(t, grid[t]) for t in sorted(grid) if t.date() == d and t.hour in DAY_HOURS]
        if not rows:
            continue
        bw, line = day_line([(t, rate(v["dir"], v["mph"], v["gust"])[0]) for t, v in rows])
        mid = grid.get(datetime.combine(d, datetime.min.time()).replace(hour=12, tzinfo=HST))
        week += (f"<tr><td>{d:%a %d}</td><td>{esc(wind_text(mid))}</td><td>{chip(bw)}</td>"
                 f"<td>{esc(line)}</td></tr>")

    summary = (f"Now: {word.lower() if w else 'no wind reading'}"
               + (f", {wind_text(w)}" if w else "") + (" (forecast, Bellows out)." if w and not obs else ".")
               + (f" Tomorrow: {tline}" if tline else ""))

    raw = (HERE / "template.html").read_text(encoding="utf-8")
    style = "".join("<style>" + part.split("</style>")[0] + "</style>" for part in raw.split("<style>")[1:])
    built = f"{now:%Y-%m-%dT%H:%M}"
    page = PAGE.format(style=style, updated=f"{now:%a %d %b %Y · %H:%M} HST", webcam=WEBCAM,
                       kind=f"Now · {now:%H:%M} HST", chip=today_chip, fly=fly_html, wind=wind_html,
                       dirnote=(f"<dt>Return</dt><dd>{dir_html}</dd>" if dir_html else ""),
                       base=base_html, rest=rest_html, tom_title=f"Tomorrow · {tom:%a %d %b}",
                       tom_chip=chip(tword) if tword else "", tomorrow=tom_html, week=week, built=built,
                       diag=esc("; ".join(problems) or "none").replace("--", "-"))
    # IBM Plex has no ʻokina (U+02BB); its fallback glyph stands apart from the
    # letters, so the visible text uses the left quote instead.
    page = page.replace("ʻ", "&#8216;")
    (OUT / "index.html").write_text(page, encoding="utf-8")
    camera = _try(save_camera, problems=problems)
    (OUT / "summary.json").write_text(json.dumps({
        "site": "Makapuu", "title": "Makapu2018u Today", "page": "kula-today/makapuu/", "built": built,
        "updated": f"{now:%a %d %b %Y · %H:%M} HST", "kind": f"Now · {now:%H:%M} HST",
        "chip": {"class": RATING_CLASS.get(word, "fair"), "word": word} if w and not night else None,
        "summary": summary, "limited": not obs, "camera": camera}), encoding="utf-8")


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
<style>.back {{ color: var(--sky); text-decoration: none; font: 600 .95rem var(--display); }}</style>
</head>
<body>
<div class="wrap">
  <header>
    <div><a class="homebtn" href="/">&larr; Home</a><div class="tag">Makapu&#8216;u soaring briefing</div><h1>Makapu&#8216;u Today</h1></div>
    <div class="date">Updated {updated}</div>
  </header>
  <div class="databadge"><div class="big">Under construction</div></div>
  <div>
    <div class="cam"><iframe src="{webcam}" title="Makapuʻu webcam, live, looking northwest toward Waimānalo" loading="lazy"></iframe></div>
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
        <dt>Crest base</dt><dd>{base}</dd>
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
      <thead><tr><th>Day</th><th>Midday wind</th><th>Best</th><th>Hours</th></tr></thead>
      <tbody>{week}</tbody>
    </table></div>
    <p>Rated hour by hour, 08:00–19:00, from the NWS forecast grid at Makapuʻu: wind from 0–70° only (east of 070° turbulence off Rabbit Island reaches launch), speed and gusts against the site's limits, under 10 mph is a run-off launch.</p>
  </section>
  <div class="notes">
    <p><strong>Where the numbers come from.</strong> Wind: Bellows (BELH1) via api.weather.gov, or the NWS forecast grid for the hour when Bellows is not reporting. Crest base: Kaneohe (PHNG) temperature and dew point via aviationweather.gov. Limits: the Makapuʻu site table.</p>
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
    write(datetime.now(HST))
    print("wrote", OUT / "index.html")
