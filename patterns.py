"""Kula Today's pattern vocabulary -- named situations the page recognises.

Sam, 2026-10-17: "over time, various edge cases will become part of the app's
vocabulary" ... "Each pattern should be easily identifiable in case it needs
to be changed or discarded."

Every pattern is one entry in PATTERNS below: an ID that never changes (P1,
P2, ...), a short name, where it came from, and a function that looks at the
day's context and returns the sentence to show, or None. The page shows the
ID and name beside each sentence, and the scorecard logs which IDs fired each
day, so a pattern can be checked against what happened and changed or
dropped here without touching anything else.

To retire a pattern: set "active": False (keep the entry, so old scorecard
rows still mean something). To add one: append with the next free ID.

Context keys (filled by build.py; any may be None):
  now               datetime, HST
  up                upslope start (Timestamp) or None if not yet
  morning           morning_cloud() result: base, n_wet, n, ka_t, ka_td,
                    ka_rh, rise, since -- or None when the slope is not wet
  ka_td_trend       Kula Ag dew-point change over the last 30 min (F)
  ka_sun            Kula Ag's share of clear-sky sun, last 60 min (0-1)
  base              current derived base (ft) or None
  base_hist         derived base every 15 min over the last 45 min (ft)
  low_sun, high_sun lower Kula slope sun (Kula Ag, Keokea) and upper slope sun
                    (best of Kula 1, Kamehamenui 1, Park HQ), last 60 min
  p4_since          when P4 was first confirmed today ("HH:MM")
  inv               Kahului inversion today: {"base_ft", "top_ft"} or None
  summit_push       first hour the summit-cloud forecast holds, or None
  upslope_typical   "HH:MM" for this month
"""

KULA_AG_FT = 3162      # stand-in for launch height until Sam gives one


def _late_start(c):
    """Low cloud shading the slope after sunrise: heating, and so the
    upslope and thermals, run late."""
    if c["up"] is not None or not c["morning"] or c["ka_sun"] is None:
        return None
    if c["now"].hour < 8 or c["now"].hour >= 11 or c["ka_sun"] >= 0.6:
        return None
    return (f"Low, wet start and the slope is shaded (Kula Ag {c['ka_sun']:.0%} of clear sky): "
            f"the upslope is likely late (usually about {c['upslope_typical']}).")


def _base_will_lift(c):
    """Dew-fed morning cloud lifts once the dew is spent: watch Kula Ag's
    dew point stop rising."""
    m, tr = c["morning"], c["ka_td_trend"]
    if not m or tr is None or m.get("rise", 0) < 1.5:
        return None
    if tr <= 0.3:
        return "Kula Ag's dew point has stopped rising: the night's dew is about spent, so the base should start to lift."
    return "The night's dew is still evaporating (Kula Ag's dew point still rising): the base stays low until it stops."


def _deck_under_lid(c):
    """Moist low layer under a low Kahului inversion: cumulus can't grow and
    spread into a flat deck at the lid."""
    inv = c["inv"]
    base = c["morning"]["base"] if c["morning"] else c["base"]
    if not inv or base is None:
        return None
    gap = inv["base_ft"] - base
    if not -300 <= gap <= 1500:          # 1,000 missed 10-17 (1,180 ft gap, base rising into it)
        return None
    # Confirmed (Sam, 2026-10-17: "A confirmation of p4 should be in the
    # tile"): the base has held for the last 45 min, within 1,000 ft under the
    # lid; the lower Kula slope is shaded; the upper slope is in sun.
    hist = c.get("base_hist") or []
    low, high = c.get("low_sun"), c.get("high_sun")
    if (len(hist) >= 4 and max(hist) - min(hist) <= 300 and gap <= 1000
            and low is not None and low < 0.6 and high is not None and high >= 0.75):
        # Dated from the start of the 45-minute stall that confirmed it.
        first = c.get("p4_since") or c["now"].strftime("%H:%M")
        h, m = map(int, first.split(":"))
        since = f"{(h * 60 + m - 45) // 60:02d}:{(h * 60 + m - 45) % 60:02d}"
        return (f"Deck under the lid since about {since}: base held at about {round(base, -2):,.0f} ft "
                f"for the last 45 minutes under the Kahului inversion ({inv['base_ft']:,.0f} ft); lower Kula "
                "slope shaded, sun above it.", "confirmed")
    return (f"Moisture under a low lid (Kahului inversion {inv['base_ft']:,.0f}–{inv['top_ft']:,.0f} ft): "
            "risk of cumulus spreading into a flat deck at the lid by midday.")


def _no_headroom(c):
    """Base too close to the slope to climb."""
    base = c["morning"]["base"] if c["morning"] else c["base"]
    if base is None or base - KULA_AG_FT >= 1000:
        return None
    return f"Little room to climb: cloud about {round(base - KULA_AG_FT, -2):,.0f} ft above Kula Ag ({KULA_AG_FT:,} ft)."


PATTERNS = [
    {"id": "P1", "name": "Summit cloud pushed over", "active": True, "where": "forecast",
     "origin": "2026-10-15 (Sam): backside cloud over the summit ridge shaded Kula from 15:00",
     "fn": None},  # drawn by build.summit_push on the Forecast panel; listed here so it has an ID
    {"id": "P2", "name": "Late start", "active": True, "where": "today",
     "origin": "2026-10-17 (Sam): dense low cloud, dew-fed morning", "fn": _late_start},
    {"id": "P3", "name": "Base will lift", "active": True, "where": "today",
     "origin": "2026-10-17: Kula Ag dew point +6 F after sunrise, 98% humid", "fn": _base_will_lift},
    {"id": "P4", "name": "Deck under the lid", "active": True, "where": "today",
     "origin": "2026-10-17: moist slope under a 4,500-5,000 ft Kahului inversion; capped days 10-03/04",
     "note": "Sam 2026-10-17: keep as a \"take your chances\" warning; no nowcast of break-through",
     "fn": _deck_under_lid},
    {"id": "P5", "name": "No headroom", "active": False, "where": "today",
     "origin": "2026-10-17: morning base 3,400 ft (Kula Ag height stands in for launch)",
     "retired": "2026-10-17, Sam: \"p5 is pointless\" -- a base that low means launch is in cloud, "
                "not launchable, which the morning-cloud line already says", "fn": _no_headroom},
]


def watch(context: dict) -> list:
    """[(id, name, sentence)] for every active Today pattern that fires. A
    pattern may return (sentence, "confirmed") once observations bear it out;
    it then shows as "<name> · confirmed" and is logged as "<id>c"."""
    out = []
    for p in PATTERNS:
        if not p["active"] or p["fn"] is None:
            continue
        try:
            text = p["fn"](context)
        except Exception:
            text = None
        if isinstance(text, tuple):
            out.append((p["id"] + "c", f"{p['name']} · confirmed", text[0]))
        elif text:
            out.append((p["id"], p["name"], text))
    return out
