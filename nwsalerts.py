"""Active NWS watches, warnings and advisories for a site, as a line for the
Today panel. Sam, 2026-10-27: a Flood Watch covered Kula and Makapuʻu all day
and neither page said so. api.weather.gov's point query returns the alerts
whose zones contain the point -- the site's own area, not the whole state.
"""
from __future__ import annotations

import html
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

HST = ZoneInfo("Pacific/Honolulu")
HEAD = {"User-Agent": "KulaToday (site pages)", "Accept": "application/geo+json"}
# Warnings first, then watches, then advisories and the rest.
RANK = ("Warning", "Watch", "Advisory")


def active(lat: float, lon: float) -> list[dict]:
    r = requests.get("https://api.weather.gov/alerts/active", params={"point": f"{lat:.4f},{lon:.4f}"},
                     headers=HEAD, timeout=30)
    r.raise_for_status()
    out, seen = [], set()
    for f in r.json()["features"]:
        p = f["properties"]
        if p.get("status") != "Actual" or p.get("messageType") == "Cancel" or p["event"] in seen:
            continue
        seen.add(p["event"])
        end = p.get("ends") or p.get("expires")
        out.append({"event": p["event"],
                    "until": datetime.fromisoformat(end).astimezone(HST) if end else None})
    rank = lambda a: next((i for i, w in enumerate(RANK) if a["event"].endswith(w)), len(RANK))
    return sorted(out, key=rank)


def line_html(lat: float, lon: float) -> str:
    """'<div class="alert">Flood Watch until Thu 06:00 · NWS</div>', or ""."""
    got = active(lat, lon)
    if not got:
        return ""
    link = f"https://forecast.weather.gov/MapClick.php?lat={lat:.4f}&lon={lon:.4f}"
    parts = [html.escape(a["event"]) + (f' until {a["until"]:%a %H:%M}' if a["until"] else "") for a in got]
    return (f'<div class="alert" style="font-size:.95rem;margin:4px 0 6px">{" · ".join(parts)} '
            f'<a href="{link}" target="_blank" rel="noopener" style="font-weight:500">NWS details</a></div>')


# ---------------------------------------------------------------- forecast icons
# The NWS point forecast's daytime icon, shown only when its words say the day
# is out of the ordinary (Sam, 10-27: "Use the words to filter the icon
# display"). Makapuʻu: thunder, showers likely or heavy, rain chance 40% or
# more, cloudy all day, breezy or windy, haze. Kula, whose ordinary afternoon
# is cloudy and whose zone wording is written for the coast: only thunder,
# "likely", "occasional", heavy or flooding, or 60% or more.
FORECAST = {"kula": "https://api.weather.gov/gridpoints/HFO/219,120/forecast",
            "makapuu": "https://api.weather.gov/gridpoints/HFO/162,145/forecast"}
BIG_WORDS = ("thunder", "likely", "occasional", "numerous", "heavy", "flood")


def notable(short: str, pop, site: str) -> bool:
    s, pop = short.lower(), pop or 0
    if any(w in s for w in BIG_WORDS):
        return True
    if site == "kula":
        return pop >= 60
    if pop >= 40:
        return True
    if s.startswith(("cloudy", "mostly cloudy")) and " then " not in s:
        return True
    return any(w in s for w in ("breezy", "windy", "haze", "smoke"))


def day_icons(site: str) -> dict:
    """{date: (icon url, words)} for daytime periods worth showing."""
    r = requests.get(FORECAST[site], headers=HEAD, timeout=30)
    r.raise_for_status()
    out = {}
    for p in r.json()["properties"]["periods"]:
        if not p["isDaytime"]:
            continue
        d = datetime.fromisoformat(p["startTime"]).astimezone(HST).date()
        pop = (p.get("probabilityOfPrecipitation") or {}).get("value")
        if d not in out and notable(p["shortForecast"], pop, site):
            words = p["shortForecast"] + (f" ({pop}%)" if pop else "")
            out[d] = (p["icon"].replace("size=medium", "size=small"), words)
    return out


def icon_html(got, size: int = 30) -> str:
    if not got:
        return ""
    url, words = got
    w = html.escape(words)
    return (f'<img src="{html.escape(url)}" width="{size}" height="{size}" alt="{w}" title="NWS: {w}" '
            f'loading="lazy" style="vertical-align:middle;border-radius:4px;margin-left:6px">')


def today_html(got) -> str:
    """The line under the alerts: the icon and the words, or ""."""
    if not got:
        return ""
    return (f'<div class="sub" style="display:flex;align-items:center;gap:6px;margin:2px 0 6px">'
            f'{icon_html(got, 34)}<span>NWS today: {html.escape(got[1])}</span></div>')
