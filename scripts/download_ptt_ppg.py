"""PTT-PPG 데이터셋 다운로드 (PhysioNet, 무인증 공개).

22명 × {sit, walk, run} = 66 레코드, 각 .hea + .dat
총 약 2.9GB — IMU(가속도+자이로) + SpO2 + ECG + PPG 동시 수록

출력: data/raw/ptt_ppg/
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://physionet.org/files/pulse-transit-time-ppg/1.1.0/"
OUT_DIR = Path(os.environ.get("P2_DATA_DIR", "data")) / "raw/ptt_ppg"

HEADERS = {"User-Agent": "Mozilla/5.0"}

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def fetch(url: str, out: Path, retries: int = 3) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 0:
        return True
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as r, open(out, "wb") as f:
                f.write(r.read())
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                print(f"  FAIL {url}: {e}")
                return False
    return False


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # RECORDS
    req = urllib.request.Request(BASE_URL + "RECORDS", headers=HEADERS)
    records = (
        urllib.request.urlopen(req, timeout=10).read().decode().strip().split("\n")
    )
    print(
        f"PTT-PPG: {len(records)} records ({len(records) // 3} subjects x 3 activities)"
    )

    ok = fail = 0
    for i, rec in enumerate(records, 1):
        for ext in [".hea", ".dat"]:
            url = BASE_URL + rec + ext
            dest = OUT_DIR / (rec + ext)
            if fetch(url, dest):
                ok += 1
            else:
                fail += 1
        if i % 10 == 0:
            print(f"  {i}/{len(records)} records done (ok={ok}, fail={fail})")

    # README
    fetch(BASE_URL + "README.txt", OUT_DIR / "README.txt")

    print(f"\nDone. ok={ok}, fail={fail}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
