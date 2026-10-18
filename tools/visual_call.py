"""Log Sam's timed visual cloudbase calls beside the station readings.

    python tools/visual_call.py 2026-10-18 12:51 5500 [note]

Appends one row to docs/visual_base.csv: the call and, at that moment, the
all-station median condensation level, the upper half's median, the highest
station and Kula Ag. For fitting how the derived base should be weighted
through the day (Sam, 2026-10-18: "It needs to be tested").
"""
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build as b  # noqa: E402

LOG = Path(__file__).resolve().parents[1] / "docs" / "visual_base.csv"
FIELDS = ["date", "time", "visual_ft", "median", "upper_half", "highest", "highest_stn", "kula_ag", "n", "note"]


def readings(day: str, hm: str) -> dict:
    when = pd.Timestamp(f"{day} {hm}")
    now = when.to_pydatetime().replace(tzinfo=b.HST) + pd.Timedelta(minutes=30)
    obs = b.observations(min(now, b.datetime.now(b.HST)))
    core = b._core(obs)
    # Readings up to 5 min after the call; if too few have arrived (the
    # Mesonet runs ~15 min behind), step back 5 min at a time, up to 15.
    for back in (0, 5, 10, 15, 20):
        t = when + pd.Timedelta(minutes=5 - back)
        c = core[(core.observed <= t) & (core.observed > t - pd.Timedelta("20min"))]
        L = c.sort_values("observed").groupby("station_id").tail(1).sort_values("elev_ft")
        if len(L) >= 4:
            break
    else:
        return {"n": len(L)}
    ka = L[L.name == "Kula Ag"].lcl_ft
    return {"median": round(float(L.lcl_ft.median()), -1),
            "upper_half": round(float(L[L.elev_ft >= L.elev_ft.median()].lcl_ft.median()), -1),
            "highest": round(float(L.lcl_ft.iloc[-1]), -1), "highest_stn": L.name.iloc[-1],
            "kula_ag": round(float(ka.iloc[0]), -1) if len(ka) else None, "n": len(L)}


def main(day, hm, ft, note=""):
    row = {"date": day, "time": hm, "visual_ft": int(ft), "note": note, **readings(day, hm)}
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in FIELDS})
    print(row)


if __name__ == "__main__":
    main(*sys.argv[1:5])
