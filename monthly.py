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


# The camera base against moisture aloft and the trade flow (Sam, 2026-11-01:
# "Let's see how moisture aloft combined with surface flow come together,
# over time"). 10-23..10-31 the camera base swung 2-5x more hour to hour
# than the derived one, and its gap from derived followed Park HQ/Summit RH
# (r -0.5) -- in a week Sam called atypical, so this only accumulates and
# reports; nothing on the page reads it.
CAMERA_BASE = HERE / "docs" / "camera_base.csv"
SCORE_HOURS = HERE / "docs" / "scorecard_hours.csv"
GRID_FROM = "2026-10-23"                 # first full day of the Kekoa camera base
GRID_HOURS = (9, 15)
UPPER_RH = {"0151": "Park HQ", "0153": "Summit"}   # mean of the two
RH_BANDS = ((60, "Dry aloft (under 60%)"), (90, "Moist (60-90%)"), (101, "Saturated (over 90%)"))
# Offshore of Maui's north shore, the gradient wind before the island
# bends it; Open-Meteo's archived model wind, hour by hour.
FLOW_AT = (21.05, -156.12)              # convergence.MARINE_GRID's point, ~10 mi out
FLOW_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
TRADE_DIRS = (20, 120)                  # from NNE through ESE
LIGHT_MPH = 8
FLOWS = ("Trades", "Light / variable", "Non-trade")
MIN_CELL_HOURS = 20


# The combination (Sam, 11-01, "log the combo", after the base rose ~1,400 ft
# above derived in the 10:00 sun with Park HQ drying, then fell back as the
# sun faded): stirring x dryness of what is stirred in = Kula Ag's share of
# clear-sky sun x (100 - upper RH)/100. Logged hourly beside the gap.
SUN_STATION = "0119"                    # Kula Ag
MIXING_LOG = HERE / "docs" / "base_mixing.csv"


def _upper_rh(first: str, last: str) -> pd.DataFrame:
    """Hourly mean upper-station RH and Kula Ag's sun fraction."""
    import mesonet
    parts = []
    for m in pd.period_range(first, last, freq="M"):
        s = max(pd.Timestamp(first), m.start_time)
        e = min(pd.Timestamp(last) + pd.Timedelta(days=1), m.end_time.normalize() + pd.Timedelta(days=1))
        rows = mesonet.get("/mesonet/db/measurements", location="hawaii",
                           station_ids=",".join([*UPPER_RH, SUN_STATION]), var_ids="RH_1_Avg,SWin_1_Avg",
                           start_date=f"{s:%Y-%m-%d}T00:00:00-10:00",
                           end_date=f"{e:%Y-%m-%d}T00:00:00-10:00", row_mode="json", limit=200000)
        parts.append(pd.DataFrame(rows))
    w = pd.concat(parts, ignore_index=True)
    w["value"] = pd.to_numeric(w.value, errors="coerce")
    t = pd.to_datetime(w.timestamp).dt.tz_convert("Pacific/Honolulu")
    w["date"], w["hour"] = t.dt.strftime("%Y-%m-%d"), t.dt.hour
    up =w[w.station_id.isin(UPPER_RH) & (w.variable == "RH_1_Avg")]
    rh = up.groupby(["date", "hour", "station_id"]).value.mean().groupby(["date", "hour"]).mean()
    sw = w[(w.station_id == SUN_STATION) & (w.variable == "SWin_1_Avg")].dropna(subset=["value"]).copy()
    sw["clear"] = mesonet.clear_sky(pd.DatetimeIndex(t.loc[sw.index]))
    g = sw.groupby(["date", "hour"])[["value", "clear"]].sum()
    sun = (g.value / g.clear).where(g.clear > 50)
    return pd.DataFrame({"rh": rh, "sun": sun})


def _flow(first: str, last: str) -> pd.Series:
    import requests
    h = requests.get(FLOW_URL, timeout=60, params={
        "latitude": FLOW_AT[0], "longitude": FLOW_AT[1], "timezone": "Pacific/Honolulu",
        "hourly": "wind_speed_10m,wind_direction_10m", "wind_speed_unit": "mph",
        "start_date": first, "end_date": last}).json()["hourly"]
    t = pd.to_datetime(h["time"])
    f = pd.DataFrame({"date": t.strftime("%Y-%m-%d"), "hour": t.hour,
                      "spd": h["wind_speed_10m"], "dir": h["wind_direction_10m"]}).dropna()
    f["flow"] = np.where(f.spd < LIGHT_MPH, FLOWS[1],
                         np.where(f.dir.between(*TRADE_DIRS), FLOWS[0], FLOWS[2]))
    return f.set_index(["date", "hour"]).flow


def base_vs_aloft(last: str) -> dict:
    """Every camera hour since GRID_FROM, binned by upper-station RH and the
    offshore flow: hours, mean camera-minus-derived, and how much more the
    camera base moves hour to hour than the derived one."""
    c = pd.read_csv(CAMERA_BASE, dtype={"date": str}).dropna(subset=["camera_ft"])
    c["hour"] = c.time.str[:2].astype(int)
    cam = c.groupby(["date", "hour"]).camera_ft.median()
    der = pd.read_csv(SCORE_HOURS, dtype={"date": str}).set_index(["date", "hour"]).derived
    j = pd.concat([cam.rename("cam"), der], axis=1).dropna().reset_index()
    j = j[j.hour.between(*GRID_HOURS) & (j.date >= GRID_FROM) & (j.date <= last)]
    if j.empty:
        return {}
    j = j.sort_values(["date", "hour"])
    step = j.groupby("date").hour.diff() == 1
    j["d_cam"] = j.groupby("date").cam.diff().where(step)
    j["d_der"] = j.groupby("date").derived.diff().where(step)
    j["gap"] = j.cam - j.derived
    j = j.join(_upper_rh(j.date.min(), j.date.max()), on=["date", "hour"])
    j = j.join(_flow(j.date.min(), j.date.max()), on=["date", "hour"]).dropna(subset=["rh", "flow"])
    j["aloft"] = pd.cut(j.rh, [-1] + [b for b, _ in RH_BANDS], labels=[n for _, n in RH_BANDS])
    j["combo"] = j.sun * (100 - j.rh) / 100
    j[["date", "hour", "cam", "derived", "gap", "rh", "sun", "combo", "flow"]].round(
        {"cam": 0, "derived": 0, "gap": 0, "rh": 1, "sun": 2, "combo": 3}).to_csv(MIXING_LOG, index=False)
    m = j.dropna(subset=["sun"])
    r = {k: (round(float(m.gap.corr(m[k])), 2) if len(m) >= 10 else None) for k in ("combo", "rh", "sun")}
    cells = []
    for (aloft, flow), g in j.groupby(["aloft", "flow"], observed=True):
        moves = g.dropna(subset=["d_cam", "d_der"])
        cells.append({"aloft": str(aloft), "flow": flow, "hours": len(g), "days": g.date.nunique(),
                      "gap": round(float(g.gap.mean()), -1),
                      "swing": (round(float(moves.d_cam.std() / moves.d_der.std()), 1)
                                if len(moves) >= 5 and moves.d_der.std() > 0 else None)})
    return {"from": j.date.min(), "to": j.date.max(), "hours": len(j), "days": int(j.date.nunique()),
            "cells": cells, "r_gap": r, "r_hours": len(m)}


def grid_lines(g: dict) -> list:
    if not g:
        return []
    out = [f"## Camera base vs moisture aloft and the flow ({g['from']} to {g['to']}, "
           f"{g['hours']} camera hours on {g['days']} days, {GRID_HOURS[0]:02d}-{GRID_HOURS[1]:02d})",
           f"Aloft = mean RH at {' and '.join(UPPER_RH.values())}. Flow = model wind off Maui's north shore: "
           f"trades from {TRADE_DIRS[0]}-{TRADE_DIRS[1]} deg at {LIGHT_MPH} mph or more. Each cell: camera minus "
           f"derived, then how many times more the camera base moves hour to hour. Under {MIN_CELL_HOURS} "
           f"hours: too few.", "",
           "| | " + " | ".join(FLOWS) + " |", "|---|" + "---|" * len(FLOWS)]
    by = {(c["aloft"], c["flow"]): c for c in g["cells"]}
    for _, band in RH_BANDS:
        row = []
        for flow in FLOWS:
            c = by.get((band, flow))
            if not c:
                row.append("--")
            elif c["hours"] < MIN_CELL_HOURS:
                row.append(f"too few ({c['hours']} h)")
            else:
                row.append(f"{c['gap']:+,.0f} ft, x{c['swing'] if c['swing'] is not None else '--'} "
                           f"({c['hours']} h, {c['days']} d)")
        out.append(f"| {band} | " + " | ".join(row) + " |")
    r = g.get("r_gap") or {}
    if r.get("combo") is not None:
        out += ["", f"Camera minus derived against the mixing combo (Kula Ag sun share x dryness aloft): "
                    f"r {r['combo']:+.2f}; against upper RH alone {r['rh']:+.2f}, sun alone {r['sun']:+.2f} "
                    f"({g['r_hours']} hours; hourly rows in docs/{MIXING_LOG.name})."]
    return out + [""]


# Shape of the day by flow (Sam, 11-10: on a plain strong-trade day the base
# fell from 12:30 while the usual November day only eases; "monitor to include
# in monthly review"). The history (NAM, 2024-10/2026) says the opposite:
# strong-trade days HOLD the base through the afternoon (cool season +250 ft at
# 14:00 from 10:00), light-trade and other-flow days fall off (-110 / -80; -480
# / -430 by 16:00), and a flow-class shape beat the month's shape (402 vs 413 ft,
# 331 vs 369 on cool strong-trade days). But 11-09 and 11-10, strong trades by
# RRFS (25 / 23 kt), both fell. This checks each month whether RRFS-era days
# follow the history, and whether RRFS winds sort days as NAM's did.
FLOW_HOURS = (11, 12, 13, 14, 15, 16)
FLOW_TRADE_DIR = (30, 130)


def _flow_frame() -> pd.DataFrame:
    H = Path(os.environ["KULA_HISTORY"])
    d = pd.read_csv(H / "days.csv", parse_dates=["date"])
    n = pd.read_csv(H / "nam12.csv", parse_dates=["date"])
    n["model"] = n["model"].fillna("NAM") if "model" in n else "NAM"
    x = d.merge(n[["date", "wdir_low_12", "wspd_low_12", "model"]], on="date")
    x = x.dropna(subset=["b10", "wspd_low_12"] + [f"b{h}" for h in FLOW_HOURS if f"b{h}" in x])
    x["mo"] = x.date.dt.month
    x["cool"] = x.mo.isin([10, 11, 12, 1, 2, 3])
    for h in FLOW_HOURS:
        if f"b{h}" in x:
            x[f"d{h}"] = x[f"b{h}"] - x.b10
    return x


def shape_by_flow(month: str) -> dict:
    x = _flow_frame()
    hs = [h for h in FLOW_HOURS if f"d{h}" in x]
    trade = x.wdir_low_12.between(*FLOW_TRADE_DIR)
    ref = x[trade & (x.model.fillna("NAM") == "NAM")].wspd_low_12
    lo, hi = (float(v) for v in ref.quantile([1 / 3, 2 / 3]))
    x["cls"] = np.where(~trade, "other flow", np.where(x.wspd_low_12 >= hi, "strong trades",
                                                     np.where(x.wspd_low_12 >= lo, "moderate trades", "light trades")))
    m = x[x.date.dt.to_period("M") == pd.Period(month)]
    past = x[x.date < pd.Period(month).start_time]
    season = m.cool.iloc[0] if len(m) else pd.Period(month).month in (10, 11, 12, 1, 2, 3)
    hist = past[past.cool == season].groupby("cls")[[f"d{h}" for h in hs]].mean()
    hn = past[past.cool == season].cls.value_counts()
    # the month's days against the history's class shapes and the plain month shape
    ms = {h: past.groupby("mo")[f"d{h}"].mean() for h in hs}
    smean = past[past.cool == season][[f"d{h}" for h in hs]].mean()
    rows, e_month, e_cls = [], [], []
    for _, r in m.iterrows():
        pm = [r.b10 + ms[h].get(r.mo, 0) for h in hs]
        pc = [pm[i] + (hist.loc[r.cls, f"d{h}"] - smean[f"d{h}"] if r.cls in hist.index else 0) for i, h in enumerate(hs)]
        ob = [r[f"b{h}"] for h in hs]
        e_month += [abs(a - b) for a, b in zip(pm, ob)]
        e_cls += [abs(a - b) for a, b in zip(pc, ob)]
        rows.append({"date": f"{r.date:%Y-%m-%d}", "cls": r.cls, "wind_kt": round(float(r.wspd_low_12), 1),
                     "dir": round(float(r.wdir_low_12)), "model": r.model,
                     "d14": round(float(r.d14), -1) if "d14" in r else None,
                     "d16": round(float(r.d16), -1) if "d16" in r and r.d16 == r.d16 else None})
    by = (m.groupby("cls")[[f"d{h}" for h in hs]].mean().round(-1)).to_dict("index")
    return {"cuts_kt": [round(lo, 1), round(hi, 1)], "season": "cool" if season else "warm",
            "history": {c: {**{f"{h}": round(float(v), -1) for h, v in zip(hs, hist.loc[c])}, "n": int(hn.get(c, 0))}
                        for c in hist.index},
            "month": {c: {**{k[1:]: v for k, v in v_.items()}, "n": int((m.cls == c).sum())} for c, v_ in by.items()},
            "days": rows, "mae_month_shape": round(float(np.mean(e_month)), -1) if e_month else None,
            "mae_flow_shape": round(float(np.mean(e_cls)), -1) if e_cls else None,
            "rrfs_speed_median": round(float(m[m.model == "RRFS"].wspd_low_12.median()), 1)
            if (m.model == "RRFS").any() else None,
            "nam_speed_median": round(float(ref.median()), 1)}


def flow_lines(f: dict, month: str) -> list:
    if "error" in f:
        return [f"## Shape of the day by flow: skipped ({f['error']})", ""]
    order = ["strong trades", "moderate trades", "light trades", "other flow"]
    out = [f"## Shape of the day by flow, {month}",
           f"Trades are 1,000-5,000 ft wind from {FLOW_TRADE_DIR[0]:03d}-{FLOW_TRADE_DIR[1]:03d} at noon; strong "
           f"over {f['cuts_kt'][1]} kt, light under {f['cuts_kt'][0]} kt (NAM history terciles). "
           f"Change of the 3-station base from 10:00, ft; {f['season']} season history beside this month.", "",
           "| Flow | history 12 / 14 / 16 | this month 12 / 14 / 16 | days |", "|---|---|---|---|"]
    for c in order:
        hh, mm = f["history"].get(c), f["month"].get(c)
        if not hh and not mm:
            continue
        fmt = lambda v: "--" if not v else " / ".join(f"{v.get(k, float('nan')):+,.0f}" for k in ("12", "14", "16"))
        out.append(f"| {c} | {fmt(hh)} (n {hh['n'] if hh else 0}) | {fmt(mm)} | {mm['n'] if mm else 0} |")
    num = lambda v: "--" if v is None else f"{v:,.0f}"
    out += ["", f"- Typical miss 11:00-16:00 from the 10:00 base: month shape {num(f['mae_month_shape'])} ft, "
                f"flow-class shape {num(f['mae_flow_shape'])} ft.",
            f"- Trade speed, median: NAM history {f['nam_speed_median']} kt, RRFS this month "
            + ("-- (no RRFS days yet)" if f["rrfs_speed_median"] is None else f"{f['rrfs_speed_median']} kt")
            + " (if RRFS runs stronger, the cuts need their own RRFS terciles).", ""]
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
    try:
        grid = base_vs_aloft(str(pd.Period(month).end_time.date()))
    except Exception as ex:                 # a Mesonet or Open-Meteo outage skips it, not the month
        grid = {"error": f"{type(ex).__name__}: {ex}"}
    big = [c for c in grid.get("cells", []) if c["hours"] >= MIN_CELL_HOURS and abs(c["gap"]) >= 400]
    if big:
        events.append("Camera base vs moisture aloft: " + "; ".join(
            f"{c['aloft'].split(' (')[0].lower()} with {c['flow'].lower()}, camera {abs(c['gap']):,.0f} ft "
            f"{'above' if c['gap'] > 0 else 'below'} derived ({c['hours']} h)" for c in big)
            + " -- see the monthly report.")
    try:
        flow = shape_by_flow(month)
    except Exception as ex:
        flow = {"error": f"{type(ex).__name__}: {ex}"}
    st_m, st_h = (flow.get("month") or {}).get("strong trades"), (flow.get("history") or {}).get("strong trades")
    if st_m and st_h and st_m["n"] >= 5 and abs(st_m.get("14", 0) - st_h["14"]) >= 300:
        events.append(f"Shape of the day: strong-trade days changed {st_m['14']:+,.0f} ft by 14:00 this month "
                      f"against {st_h['14']:+,.0f} in the history ({st_m['n']} days) -- see the monthly report.")
    if flow.get("mae_month_shape") and flow.get("mae_flow_shape") is not None:
        gain = flow["mae_month_shape"] - flow["mae_flow_shape"]
        if abs(gain) >= 30:
            events.append(f"Shape of the day: the flow-class shape {'beat' if gain > 0 else 'lost to'} the month's "
                          f"shape by {abs(gain):.0f} ft this month -- see the monthly report.")
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
    lines += grid_lines(grid) if "error" not in grid else [f"## Camera base vs moisture aloft: skipped ({grid['error']})", ""]
    lines += flow_lines(flow, month)
    lines += ["## Proposed (for a person; nothing here changes the page)"] + [f"- {p}" for p in props or ["none"]]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    REPORT_JSON.write_text(json.dumps({"month": month, "made": f"{datetime.now():%Y-%m-%d %H:%M}",
                                       "events": events, "active": held, "history": hist, "live": live,
                                       "base_vs_aloft": grid, "shape_by_flow": flow},
                                      indent=1, default=str), encoding="utf-8")
    print("\n".join(lines))
    print("events:", events or "none")


if __name__ == "__main__":
    last = (pd.Timestamp.now() - pd.offsets.MonthBegin(1)).to_period("M") - 1
    main(sys.argv[1] if len(sys.argv) > 1 else str(last))
