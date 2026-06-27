# -*- coding: utf-8 -*-
"""pytest 부트스트랩 — src 레이아웃 패키지(p2fusion)를 import 경로에 추가."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
