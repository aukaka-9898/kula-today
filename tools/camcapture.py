"""Save the FAA Kekoa south camera's frames before the feed drops them.

The feed holds only the last ~36 frames (about 5 hours), so a Windows task on
Sam's PC ("Kula camera capture") runs this at 10:00, 14:00 and 18:30 (dark by 19:00),
waking the PC if needed -- every 4 hours keeps an hour of overlap (Sam,
2026-10-18; on his PC, not in the page's repo).
Frames land in E:/dev/WX/KulaModel/review/camera_archive/<date>/<HHMM>.jpg
(HST), for the blind camera tests (camtest.py). Standard library only.
"""
import json
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ARCHIVE = Path("E:/dev/WX/KulaModel/review/camera_archive")
URL = "https://weathercams.faa.gov/api/summary?siteId=931"
PAGE = "https://weathercams.faa.gov/cameras/state/HI/cameraSite/931/summary"
CAM_ID = 13417                     # Kekoa South
HST = timezone(timedelta(hours=-10))


def get(url: str, referer: str | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (KulaToday camera archive)",
                                               **({"Referer": referer} if referer else {})})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def capture() -> int:
    j = json.loads(get(URL, PAGE))
    cam = next(c for c in j["payload"]["site"]["cameras"] if c["cameraId"] == CAM_ID)
    new = 0
    for im in cam["currentImages"]:
        t = datetime.fromisoformat(im["imageDatetime"].replace("Z", "+00:00")).astimezone(HST)
        f = ARCHIVE / f"{t:%Y-%m-%d}" / f"{t:%H%M}.jpg"
        if f.exists():
            continue
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(get(im["imageUri"]))
        new += 1
    return new


if __name__ == "__main__":
    n = capture()
    print(f"{datetime.now(HST):%Y-%m-%d %H:%M} saved {n} new frames")
