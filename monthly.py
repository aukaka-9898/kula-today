"""The monthly check of the page's history rules (rules.py).

Sam, 2026-10-17: "a rolling 1,000 day check run monthly", and "I may lose the
high level of interest I have in the app right now" -- so it runs unattended.

    python monthly.py [YYYY-MM]     (default: last month)

1. Adds the month to the history tables (KulaModel history.py update).
2. Scores each rule on the latest 1,000 days, in thirds, against how often
   the thing it calls happens anyway.
3. Scores the page's own logged calls for the month (scorecard_days.csv).
4. Switches a rule OFF in docs/rules_status.json if it no longer beats the
   plain answer in the history, or failed live two months running. It never
   switches one on and never moves a threshold: those are proposals in
   docs/monthly_report.md, for a person.
5. Lists events (switched off, proposals, live failures) in
   docs/monthly_report.json for the notifier.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, os.environ.get("KULAMODEL", str(HERE.parent / "KulaModel")))
os.environ.setdefault("KULA_HISTORY", str(HERE / "history"))

import history  # noqa: E402
import rules    # noqa: E402

WINDOW_DAYS = 1000
MARGIN = 0.10           # a rule must beat the plain answer by this much
MIN_FIRES = 10          # per third, to judge it there
REPORT_MD = HERE / "docs" / "monthly_report.md"
REPORT_JSON = HERE / "docs" / "monthly_report.json"
SCORE_DAYS = HERE / "docs" / "scorecard_days.csv"


def features() -> pd.DataFrame:
    """One row per history day with the rules' inputs and outcomes."""
    H = Path(os.environ["KULA_HISTORY"])
    d = pd.read_csv(H / "days.csv", parse_dates=["date"])
    d["month"] = d.date.dt.month
    d["cat"] = [rules.category(v, m) for v, m in zip(d.level, d.month)]
    prev = d.set_index("date")["cat"].asfreq("D").shift(1)
    d["today_cat"] = d.date.map(prev)                      # the day the forecast was made on
    d["today_level"] = d.date.map(d.set_index("date")["level"].asfreq("D").shift(1))
    a = pd.read_csv(H / "nam12_ahead.csv", parse_dates=["date"])
    a = a[a.lead == 1][["date", "rh_5_7k_12", "rh_7_10k_12"]].rename(
        columns={"rh_5_7k_12": "rh_5_7k", "rh_7_10k_12": "rh_7_10k"})
    n = pd.read_csv(H / "nam12.csv", parse_dates=["date"])[["date", "rh_5_7k_12"]].rename(
        columns={"rh_5_7k_12": "rh_5_7k_same"})
    a2 = pd.read_csv(H / "nam12_ahead.csv", parse_dates=["date"])
    a2 = a2[a2.lead == 2][["date", "rh_5_7k_12", "rh_7_10k_12"]].rename(
        columns={"rh_5_7k_12": "rh_5_7k_2", "rh_7_10k_12": "rh_7_10k_2"})
    fc = pd.read_csv(H / "fc_level.csv", parse_dates=["date"])
    x = (d.merge(a, on="date", how="left").merge(a2, on="date", how="left")
         .merge(n, on="date", how="left").merge(fc, on="date", how="left"))
    if "fc_level_2" not in x:
        x["fc_level_2"] = np.nan
    if (H / "summit.csv").exists():
        x = x.merge(pd.read_csv(H / "summit.csv", parse_dates=["date"]), on="date", how="left")
    for c in ("su_rh", "su_dir"):
        if c not in x:
            x[c] = np.nan
    x["base_late"] = x[["b10", "b11"]].mean(axis=1)
    x["rest"] = x.b15 - x.b11
    return x


def fires(rid: str, x: pd.DataFrame) -> pd.Series:
    fn = rules.RULES[rid]["fn"]
    out = []
    for r in x.itertuples():
        f = {"month": r.month, "rh_5_7k": r.rh_5_7k_same if rid == "P6" else r.rh_5_7k,
             "rh_7_10k": r.rh_7_10k, "fc_level": r.fc_level, "today_cat": r.today_cat,
             "today_level": r.today_level, "base_late": r.base_late}
        if rid == "P8":           # its own inputs only; dew starts are scored with the rest
            f = {"su_rh": r.su_rh, "su_dir": r.su_dir}
        if rid == "B4":           # judged 2 days out, as the week table uses it
            f.update(rh_5_7k=r.rh_5_7k_2, rh_7_10k=r.rh_7_10k_2, fc_level=r.fc_level_2)
        try:
            out.append(bool(fn(f)) if not any(pd.isna(v) for k, v in f.items() if k != "today_cat") else False)
        except TypeError:
            out.append(False)
    return pd.Series(out, index=x.index)


def outcome(rid: str, x: pd.DataFrame) -> pd.Series:
    says = rules.RULES[rid]["says"]
    return x.rest <= -300 if says == "falls" else x.cat == says


def score_history(x: pd.DataFrame) -> dict:
    out = {}
    thirds = np.array_split(x.sort_values("date").index, 3)
    for rid in rules.RULES:
        f, ok = fires(rid, x), outcome(rid, x)
        parts = []
        for idx in thirds:
            n = int(f[idx].sum())
            parts.append({"from": f"{x.date[idx].min():%Y-%m}", "fired": n,
                          "right": float(ok[idx][f[idx]].mean()) if n else None,
                          "base_rate": float(ok[idx].mean())})
        n = int(f.sum())
        out[rid] = {"fired": n, "share": round(n / len(x), 3),
                    "right": round(float(ok[f].mean()), 2) if n else None,
                    "base_rate": round(float(ok.mean()), 2), "thirds": parts}
    return out


def proposals(x: pd.DataFrame) -> list:
    """Changes for a person to consider: month ranges that moved."""
    out = []
    q = x.groupby("month")["level"].quantile([0.2, 0.8]).unstack().round(-2)
    for m, (lo, hi) in rules.MONTH_BASE_RANGE.items():
        if m not in q.index:
            continue
        nlo, nhi = int(q.loc[m, 0.2]), int(q.loc[m, 0.8])
        if abs(nlo - lo) >= 200 or abs(nhi - hi) >= 200:
            out.append(f"{datetime(2000, m, 1):%B} normal range: now {lo:,}-{hi:,} ft, "
                       f"the latest {WINDOW_DAYS} days say {nlo:,}-{nhi:,} ft.")
    return out


def score_live(month: str) -> dict:
    """The page's own calls for the month (logged by build.py)."""
    if not SCORE_DAYS.exists():
        return {}
    s = pd.read_csv(SCORE_DAYS, dtype={"date": str})
    s = s[s.date.str.startswith(month)]
    out = {}
    if "base_rule" in s and "base_cat" in s:
        for rid, cat in (("B1", "low"), ("B2", "high"), ("B3", "high"), ("B4", "high")):
            c = s[s.base_rule == rid]
            out[rid] = {"calls": len(c), "right": int((c.base_cat == cat).sum())}
    if "p6_fell" in s:
        c = s[s.p6_fell.notna()]
        out["P6"] = {"calls": len(c), "right": int(c.p6_fell.astype(bool).sum())}
    if "thermal_call" in s and "word" in s:
        order = ["Flat", "Light", "Moderate", "Strong"]
        c = s.dropna(subset=["thermal_call", "word"])
        c = c[c.thermal_call.isin(order) & c.word.isin(order)]
        gap = (c.thermal_call.map(order.index) - c.word.map(order.index)).abs()
        out["thermal"] = {"days": len(c), "same": int((gap == 0).sum()), "off_by_2plus": int((gap >= 2).sum())}
    return out


def main(month: str) -> None:
    history.update(month)
    x = features()
    x = x[x.date > x.date.max() - pd.Timedelta(days=WINDOW_DAYS)].reset_index(drop=True)
    hist = score_history(x)
    live = score_live(month)
    st = rules.status()
    st.setdefault("rules", {})
    events = []
    for rid, h in hist.items():
        r = st["rules"].setdefault(rid, {"active": True})
        if not r.get("active", True):
            continue
        bad = [p for p in h["thirds"] if p["right"] is not None and p["fired"] >= MIN_FIRES
               and p["right"] < p["base_rate"] + MARGIN]
        overall_bad = h["right"] is None or h["right"] < h["base_rate"] + MARGIN
        lv = live.get(rid, {})
        live_bad = lv.get("calls", 0) >= 3 and lv["right"] / lv["calls"] < 1 / 3
        r["live_fail_months"] = r.get("live_fail_months", 0) + 1 if live_bad else 0
        why = None
        if overall_bad or len(bad) >= 2:
            why = (f"no longer beats the plain answer in the latest {WINDOW_DAYS} days "
                   f"(right {h['right']}, happens anyway {h['base_rate']})")
        elif r["live_fail_months"] >= 2:
            why = "failed on the page two months running"
        if why:
            r.update(active=False, off=month, why=why)
            events.append(f"Switched off {rid} ({rules.RULES[rid]['name']}): {why}.")
        elif live_bad:
            events.append(f"{rid} ({rules.RULES[rid]['name']}) did badly on the page in {month}: "
                          f"right {lv['right']} of {lv['calls']}.")
    th = live.get("thermal", {})
    if th.get("days", 0) >= 10 and th["off_by_2plus"] / th["days"] > 0.10:
        events.append(f"Thermal words were off by two categories or more on {th['off_by_2plus']} of "
                      f"{th['days']} days in {month}.")
    props = proposals(x)
    if props:
        events.append(f"{len(props)} change(s) proposed -- see the monthly report.")
    st["checked"] = month
    rules.STATUS_FILE.write_text(json.dumps(st, indent=1), encoding="utf-8")

    held = [rid for rid, r in st["rules"].items() if r.get("active", True)]
    lines = [f"# Monthly rule check, {month}", "",
             f"History window: {x.date.min():%Y-%m-%d} to {x.date.max():%Y-%m-%d} ({len(x)} days).", ""]
    for rid, h in hist.items():
        on = "on" if st["rules"][rid].get("active", True) else f"OFF since {st['rules'][rid].get('off')}"
        lines.append(f"## {rid} {rules.RULES[rid]['name']} ({on})")
        lines.append(f"- History when built: {rules.RULES[rid]['history']}")
        lines.append(f"- Latest window: fired on {h['fired']} days ({h['share']:.0%}), right "
                     f"{'--' if h['right'] is None else f'{h['right']:.0%}'}, happens anyway {h['base_rate']:.0%}")
        for p in h["thirds"]:
            lines.append(f"  - from {p['from']}: fired {p['fired']}, right "
                         f"{'--' if p['right'] is None else f'{p['right']:.0%}'}, anyway {p['base_rate']:.0%}")
        lv = live.get(rid)
        if lv:
            lines.append(f"- On the page in {month}: {lv['right']} right of {lv['calls']} calls")
        lines.append("")
    if th:
        lines += [f"## Thermal words on the page, {month}",
                  f"- {th['days']} days: same word {th['same']}, off by two or more {th['off_by_2plus']}", ""]
    lines += ["## Proposed (for a person; nothing here changes the page)"] + [f"- {p}" for p in props or ["none"]]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    REPORT_JSON.write_text(json.dumps({"month": month, "made": f"{datetime.now():%Y-%m-%d %H:%M}",
                                       "events": events, "active": held, "history": hist, "live": live},
                                      indent=1, default=str), encoding="utf-8")
    print("\n".join(lines))
    print("events:", events or "none")


if __name__ == "__main__":
    last = (pd.Timestamp.now() - pd.offsets.MonthBegin(1)).to_period("M") - 1
    main(sys.argv[1] if len(sys.argv) > 1 else str(last))
