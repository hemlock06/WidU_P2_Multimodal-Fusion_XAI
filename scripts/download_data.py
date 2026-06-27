"""실데이터 다운로드 (2단계 현실화용).

사용:
    python scripts/download_data.py --dataset sisfall
    python scripts/download_data.py --dataset harespod
    python scripts/download_data.py --dataset ptt_ppg      # ~2.9GB 주의

소스/라이선스는 configs/data.yaml 참조. 대용량이므로 기본은 dry-run(URL만 출력),
실제 다운로드는 --run 플래그 필요. 출력: $P2_DATA_DIR/raw/<dataset>/
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(os.environ.get("P2_DATA_DIR", "data"))

SOURCES = {
    "sisfall": {
        "url": "http://sistemic.udea.edu.co/wp-content/uploads/2018/04/SisFall_dataset.zip",
        "is_zip": True,
        "note": "낙상 15종 + ADL 19종, 38명, 200Hz (가속도+자이로). 클래스 3 앵커.",
    },
    "ptt_ppg": {
        "url": "https://physionet.org/files/pulse-transit-time-ppg/1.1.0/",
        "is_zip": False,
        "note": "ECG+IMU+SpO2 동시, 22명, 500Hz, ~2.9GB. PhysioNet wget 권장: "
        "wget -r -N -c -np <url>. 클래스 0·1 앵커.",
    },
    "harespod": {
        "url": "https://www.nature.com/articles/s41597-024-03988-5",
        "is_zip": False,
        "note": "유발 저산소 SpO2/HR, 15명, 100Hz. Figshare 다운로드 링크는 논문 "
        "Data Availability 절에서 확인. 클래스 4 앵커.",
    },
}


def download_zip(url: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / Path(url).name
    print(f"  다운로드 → {zip_path}")
    urllib.request.urlretrieve(url, zip_path)
    print(f"  압축 해제 → {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print("  완료.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SOURCES))
    ap.add_argument(
        "--run",
        action="store_true",
        help="실제 다운로드 수행 (미지정 시 URL/안내만 출력)",
    )
    args = ap.parse_args()

    src = SOURCES[args.dataset]
    out_dir = DATA_DIR / "raw" / args.dataset
    print(f"[{args.dataset}] {src['note']}")
    print(f"  URL: {src['url']}")
    print(f"  출력: {out_dir}")

    if not args.run:
        print("  (dry-run) 실제 다운로드는 --run 플래그 추가.")
        return
    if src["is_zip"]:
        download_zip(src["url"], out_dir)
    else:
        print("  ※ 이 데이터셋은 자동 zip 다운로드 미지원. 위 URL에서 수동/wget 권장.")
        sys.exit(1)


if __name__ == "__main__":
    main()
