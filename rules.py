"""The page's history rules, in one place for the page and the monthly check.

Each rule is learned from the KulaModel history (history.py, 1,020 days from
2024-01) and was kept because it held in every held-back year
(ruletree.py --rotate, 2026-10-17). The page (build.py, patterns.py) asks
these functions; monthly.py scores them on the latest 1,000 days and on the
page's own logged calls, and may switch one OFF in docs/rules_status.json --
never on, and never changes a threshold. Changes to thresholds, new rules and
switching one back on are for a person: see docs/monthly_report.md.

Sam, 2026-10-17: "I may lose the high level of interest I have in the app
right now." So the check runs unattended, and its only power is to take a
claim off the page.

Features (the page computes them live; monthly.py from the history tables):
    month       1-12, the day forecast for
    rh_5_7k     Kahului model humidity 5,000-7,000 ft at noon HST (%)
    rh_7_10k    the same, 7,000-10,000 ft
    fc_level    the page's forecast base, 11-14 HST, without the +400 (ft)
    today_cat   "low" / "normal" / "high": the day the forecast is made on
    today_level the day the forecast is made on: its midday derived base (ft)
    base_late   the derived base around 10:00-12:00 (ft)
"""
import json
from pathlib import Path

STATUS_FILE = Path(__file__).parent / "docs" / "rules_status.json"

# Each month's normal base: the middle 60% of the history days (2024-01 to
# 2026-10). Below the first is a low day for the month, above the second high.
MONTH_BASE_RANGE = {1: (4100, 5500), 2: (4200, 5100), 3: (4200, 5100), 4: (3900, 5200),
                    5: (4300, 5100), 6: (4300, 5300), 7: (4600, 5500), 8: (4300, 5500),
                    9: (4700, 5300), 10: (4500, 5400), 11: (4500, 5500), 12: (4500, 5600)}


def category(level, month) -> str | None:
    if level is None or level != level:
        return None
    lo, hi = MONTH_BASE_RANGE[month]
    return "low" if level <= lo else "high" if level >= hi else "normal"


def _low_day(f) -> bool:
    """Moist air aloft and a low forecast base agree (the evening before)."""
    moist = f["rh_7_10k"] > 55 and (f["rh_5_7k"] > 80 or f.get("today_cat") == "low")
    return moist and f["fc_level"] <= MONTH_BASE_RANGE[f["month"]][0]


def _may_be_high(f) -> bool:
    """Dry air aloft after a high day (tomorrow only: needs today's result)."""
    return f.get("today_cat") == "high" and f["rh_5_7k"] < 40 and f["rh_7_10k"] < 20


def _high_forecast(f) -> bool:
    """A high base today and a forecast base well above the month's middle
    (tomorrow only). Sam, 2026-10-17: "high day rule should be a priority"."""
    lo, hi = MONTH_BASE_RANGE[f["month"]]
    return f["today_level"] > 5150 and f["fc_level"] - (lo + hi) / 2 > 950


def _leaning_high(f) -> bool:
    """2-3 days out, the forecast alone: dry aloft and a forecast base 600+ ft
    above the month's middle. About twice the usual chance of a high day but
    right only ~4 in 10, so the week table says "Leaning high", not "May be"
    (Sam, 2026-10-17)."""
    lo, hi = MONTH_BASE_RANGE[f["month"]]
    return f["rh_5_7k"] < 40 and f["rh_7_10k"] < 20 and f["fc_level"] - (lo + hi) / 2 > 600


def _high_base_drop(f) -> bool:
    """A base already high by late morning under moist air aloft comes down."""
    return f["base_late"] > 5400 and f["rh_5_7k"] > 45


RULES = {
    "B1": {"name": "Low day", "says": "low", "fn": _low_day,
           "history": "right 80% on 6% of days (low days 20% of all), 2024-01 to 2026-10",
           "origin": "2026-10-17: moist-aloft rule recurred in all three held-back years; with the forecast base, both must agree"},
    "B2": {"name": "May be high", "says": "high", "fn": _may_be_high,
           "history": "right 66% on 6% of days (high days 21% of all); low on 2%",
           "origin": "2026-10-17 (Sam: \"may be a high day\")"},
    "B3": {"name": "May be high (high base, high forecast)", "says": "high", "fn": _high_forecast,
           "history": "with B2: right 52% on 18% of days, catching 96 of 213 high days (45%); "
                      "B3 alone 45%; the three slices 49/52/67%; low on 5%",
           "origin": "2026-10-17: high-day search over 1,020 days (Sam: a priority)"},
    "B4": {"name": "Leaning high (2-3 days out)", "says": "high", "fn": _leaning_high,
           "history": "2 days out right 41% on 12% of days (high 21% of all), low 5%; 3 days out 38%",
           "origin": "2026-10-17 (Sam: \"use 'leaning high'\")", "lead": 2},
    "P6": {"name": "High base, likely to drop", "says": "falls", "fn": _high_base_drop,
           "history": "fell 300+ ft by 15:00 on 84% of the 10% of days it fired (37% of all days)",
           "origin": "2026-10-17: recurred in all three held-back years at 11:00 and 12:00"},
}


def status() -> dict:
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def active(rule_id: str) -> bool:
    """False only if the monthly check has switched the rule off."""
    return status().get("rules", {}).get(rule_id, {}).get("active", True)


def says(rule_id: str, features: dict) -> bool:
    try:
        return active(rule_id) and bool(RULES[rule_id]["fn"](features))
    except (KeyError, TypeError):
        return False
