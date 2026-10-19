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
  rh_5_7k           model humidity 5,000-7,000 ft at Kahului, noon today (%)
  sky               {"drop", "cloudiest"}: the cloudiest Kula station's sky
                    emissivity and how much it fell over the last hour
  inv               Kahului inversion today: {"base_ft", "top_ft"} or None
                    (no pattern uses it since 2026-10-17)
  summit_push       first hour the summit-cloud forecast holds, or None
  spill             {"su_rh", "su_dir", "dew"}: the Summit station 07:00-08:59
                    and whether Kula started saturated -- or None before ~09:15
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
    # Only after a saturated night (Sam, 2026-10-18): a rising dew point on a
    # dry night is moisture arriving, not dew, and it need not stop.
    if not m or tr is None or m.get("rise", 0) < 1.5 or not m.get("dewy"):
        return None
    # For planning (Sam, 2026-10-17): after a heavy dew the wet mid-slope can
    # keep feeding moisture once the base lifts -- on 10-17 the base held at
    # about 5,000 ft for three hours while the ground kept warming.
    cap = (" Possible cap: once it lifts, the wet slope can keep feeding moisture and hold the base "
           "down for hours.")
    import rules
    if tr <= 0.3:
        if c.get("spill") and rules.says("P8", c["spill"]):
            return None     # P8 outranks it: moisture from above keeps the base down
        return "Kula Ag's dew point has stopped rising: the night's dew is about spent, so the base should start to lift." + cap
    return "The night's dew is still evaporating (Kula Ag's dew point still rising): the base stays low until it stops." + cap


def _deck_under_lid(c):
    """Cumulus capped and spread into a flat deck. From the stations only
    (Sam, 2026-10-17: "Kahului inversion is useless for our purposes"): the
    base has held for the last 45 min, the lower Kula slope is shaded and the
    upper slope is in sun. No forecast warning -- the model's lid can't be
    trusted to give one."""
    base = c["base"]
    hist = c.get("base_hist") or []
    low, high = c.get("low_sun"), c.get("high_sun")
    if base is None:
        return None
    # Flat = not changing (Sam, 2026-10-18): base_hist is newest first, 15 min
    # apart; a steady climb of 150 ft/h stayed inside 300 ft over 45 min and
    # P4 fired on a day with no cap. Now also under 75 ft an hour net.
    trend = (hist[0] - hist[-1]) / (15 * (len(hist) - 1)) * 60 if len(hist) >= 2 else 0
    if (len(hist) >= 4 and max(hist) - min(hist) <= 300 and abs(trend) <= 75
            and low is not None and low < 0.6 and high is not None and high >= 0.75):
        # Dated from the start of the 45-minute stall that confirmed it.
        first = c.get("p4_since") or c["now"].strftime("%H:%M")
        h, m = map(int, first.split(":"))
        since = f"{(h * 60 + m - 45) // 60:02d}:{(h * 60 + m - 45) % 60:02d}"
        return (f"Deck at a lid since about {since}: base held at about {round(base, -2):,.0f} ft "
                "for the last 45 minutes; lower Kula slope shaded, sun above it.", "confirmed")
    return None


def _no_headroom(c):
    """Base too close to the slope to climb."""
    base = c["morning"]["base"] if c["morning"] else c["base"]
    if base is None or base - KULA_AG_FT >= 1000:
        return None
    return f"Little room to climb: cloud about {round(base - KULA_AG_FT, -2):,.0f} ft above Kula Ag ({KULA_AG_FT:,} ft)."


def _high_base_drop(c):
    """A base already high by late morning under moist air aloft comes back
    down. From the history rule tree (KulaModel ruletree.py --rotate): in all
    three held-back years, base over ~5,100-5,600 ft at 10-11 h with
    5,000-7,000 ft humidity over ~45% fell 300+ ft by 15:00 on 81-97% of
    training days; across 1,020 days it fired on 10% and fell on 84% (37% of
    all days)."""
    import rules
    hist, rh = c.get("base_hist") or [], c.get("rh_5_7k")
    if not 11 <= c["now"].hour < 13 or len(hist) < 3 or rh is None:
        return None
    base = sum(hist) / len(hist)
    if not rules.says("P6", {"base_late": base, "rh_5_7k": rh}):
        return None
    return (f"Base already high (about {round(base, -2):,.0f} ft) with moist air above it: on days like "
            "this it usually comes down 300 ft or more by mid-afternoon.")


def _late_decay(c):
    """Late-day decay: the base holds while the Kula stations' sky clears.
    Sam, 2026-10-17 ~17:00: "Clouds are dissipating, from base up it seems.
    A false rise." The cloudiest of Kula Ag / Keokea / Pulehu (sky emissivity
    from the radiometers) cleared by 0.05 or more in an hour while the base
    stayed within 300 ft -- one station alone is fooled by streets and gaps
    (Keokea under one, Kula Ag in the gap, that afternoon)."""
    hist, sky = c.get("base_hist") or [], c.get("sky")
    if c["now"].hour * 60 + c["now"].minute < 14 * 60 + 30 or len(hist) < 4 or not sky:
        return None
    if max(hist) - min(hist) > 300 or sky["drop"] < 0.05:
        return None
    return (f"Clouds decaying: the base has held near {round(sum(hist) / len(hist), -2):,.0f} ft and the sky "
            "is clearing at the stations. A base that looks higher now is cloud evaporating from below, "
            "not new lift.")


def _summit_spillover(c):
    """Crater spillover (Sam, 2026-10-18): the summit moist on an easterly
    07:00-08:59. See rules.P8 for the history behind the numbers."""
    import rules
    s = c.get("spill")
    if not s or not rules.says("P8", s):
        return None
    if s.get("dew"):
        return ("Summit spillover on a dew morning: probably not flyable; the base is likely to stay "
                "under 4,000 ft.")
    return ("Summit spillover: moist trade air has been over the summit since 07:00. Crater cloud spilling "
            "west gets mixed into Kula's thermals, so expect a lower base that stalls rather than climbs. "
            "Low day about 4 times in 10 (usually under 2 in 10).")


PATTERNS = [
    {"id": "P1", "name": "Summit cloud pushed over", "active": True, "where": "forecast",
     "origin": "2026-10-15 (Sam): backside cloud over the summit ridge shaded Kula from 15:00",
     "fn": None},  # drawn by build.summit_push on the Forecast panel; listed here so it has an ID
    {"id": "P2", "name": "Late start", "active": True, "where": "today",
     "origin": "2026-10-17 (Sam): dense low cloud, dew-fed morning", "fn": _late_start},
    {"id": "P3", "name": "Base will lift", "active": True, "where": "today",
     "origin": "2026-10-17: Kula Ag dew point +6 F after sunrise, 98% humid", "fn": _base_will_lift},
    {"id": "P4", "name": "Deck under the lid", "active": False, "where": "today",
     "retired": "2026-10-18, Sam: \"switch it off until we can better define what we're talking about "
                "with a lid.\" A deck is normal in Kula; a lid means the deck's height is limited. P4 "
                "tested base flatness and shading (an ordinary deck), not the tops. Redesign around the "
                "tops: the upper-slope ladder (Kula 1, Park HQ, Nene, Summit) sunny and dry just above the base.",
     "origin": "2026-10-17: moist slope under a 4,500-5,000 ft Kahului inversion; capped days 10-03/04",
     "note": "Sam 2026-10-17: keep as a \"take your chances\" warning; no nowcast of break-through. "
             "Stations only since 2026-10-17 (Kahului inversion dropped)",
     "fn": _deck_under_lid},
    {"id": "P5", "name": "No headroom", "active": False, "where": "today",
     "origin": "2026-10-17: morning base 3,400 ft (Kula Ag height stands in for launch)",
     "retired": "2026-10-17, Sam: \"p5 is pointless\" -- a base that low means launch is in cloud, "
                "not launchable, which the morning-cloud line already says", "fn": _no_headroom},
    {"id": "P6", "name": "High base, likely to drop", "active": True, "where": "today",
     "origin": "2026-10-17: history rule tree, 1,020 days 2024-26, recurring in all three held-back years",
     "fn": _high_base_drop},
    {"id": "P7", "name": "Late-day decay", "active": True, "where": "today",
     "origin": "2026-10-17 (Sam): cloud dissipating from the base up, base lingering at 4,900 ft; "
               "Keokea's sky cleared 0.95 -> 0.86 at 16:30. Thresholds from that one day",
     "fn": _late_decay},
    {"id": "P8", "name": "Summit spillover", "active": True, "where": "today",
     "origin": "2026-10-18 (Sam): crater full of cloud spilling west over the rim; tested on 1,019 days",
     "fn": _summit_spillover},
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
