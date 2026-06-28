# HANDOFF SUMMARY — P2 Multimodal Fusion

> 개발자 인수인계 요약. 빠른 온보딩용 한 장. 상세는 각 문서 링크 참조. 작성일: 2026-06-28.
> **본 작업(브랜치 `handoff-prep`)은 기존 소스 로직을 수정하지 않고 `docs/`만 추가했다.**

---

## 1. 한 줄 요약

ECG(P1) + IMU + SpO₂ 세 모달리티를 **Cross-Modal Attention**으로 융합해 5종 응급(정상안정·정상활동·
심혈관·낙상·저산소)을 판별하고, **4-layer XAI**로 판정 근거를 부양자 눈높이로 설명하는 모듈.
P1(ECG 인코더)의 출력을 입력 계약으로 받아 P1↔P2를 단일 파이프라인으로 연결한다.

---

## 2. 문서 인덱스

| 문서 | 내용 |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 전체 파이프라인, 채택 모델·비교군, 4-layer XAI, 모듈 맵, 학습 경로 |
| [INTERFACE_CONTRACTS.md](INTERFACE_CONTRACTS.md) | P1→P2 입력 계약, 모델 batch dict 입출력, 차원·키·순서, XAI 출력 |
| [HANDOFF_ISSUES.md](HANDOFF_ISSUES.md) | P0/P1/P2 이슈 (재생성 의존성·버전 불일치·문서 드리프트·죽은 코드) |
| 기존 문서 | `README.md`(개요·결과), `PROJECT_JOURNEY.md`(설계 여정), `decisions.md`(기록 인덱스), `XAI_SHAP_vs_IG_validation.md` |

---

## 3. 핵심 구조 (3줄)

1. **데이터:** P1 캐시(실 ECG-FM 출력) + 클래스 조건부 조립(IMU/SpO₂ 사전분포) → 합성 paired `.npz`.
2. **모델:** `cross_attn`(채택) / `gated`(GMU 비교군) / `concat`(베이스라인). 입력 batch dict
   `{ecg_emb[768], ecg_aux[8], imu[12], spo2[8], mask[3], label}`, 출력 `logits[5]` + side outputs.
3. **XAI:** attention[3×3] · gate[3] · conf[3] · unimodal_logits[3,5] + IG/SHAP/hierarchical/NL.

---

## 4. 검증 결과 (출처: `README.md` 핵심 결과 — 합성 평가셋·외부 test, 고정 시드)

| 항목 | 값 |
|---|---|
| 융합 Macro-F1 | **0.939** (단일 모달 최고 0.80 대비 +0.14) |
| 응급 검출 recall | **0.965** |
| 외부 test 일반화 | Cross-Modal Attention 0.94 → **0.94 유지** · Concat 0.95 → **0.89 과적합** |
| 4-layer XAI 충실도 | IG·SHAP 순위일치 **ρ ≈ 0.99** (perturbation 포함 ρ ≥ 0.87) |
| 클래스별 1차 모달 의존도 | cardiac→ECG 0.39 · impact→IMU 0.69 · hypoxia→SpO₂ 0.42 (soft) |
| ECG 단독 baseline | 응급 5종 macro-F1 **0.394** (융합의 출발점) |

> 어텐션 채택 근거: in-dist에선 concat이 근소 우위지만 외부 test에서 과적합(0.95→0.89), 어텐션은
> 모달 간 관계를 학습으로 포착해 일반화 유지(0.94→0.94). 수치는 README 기준이며, 재측정 스크립트는
> `measure_emergency.py`·`measure_reliance.py`·`verify_xai_attn.py`.

---

## 5. 빠른 시작 (README 기준 + 주의사항 반영)

```bash
# 0) (선택) 패키지 설치 — src 레이아웃, pyproject.toml
pip install -e .   # 또는 requirements 설치 후 sys.path 부트스트랩(스크립트가 자동 처리)

# 1) 데이터 재생성 — P1 캐시는 외부 P1 레포 필요 (HANDOFF_ISSUES P0-1)
python scripts/build_p1_cache.py                       # P1_REPO_DIR·ECG-FM 가중치·fairseq_signals 필요
python scripts/build_synthetic_dataset.py --version vf --n-per-class 4000 --seed 42

# 2) 학습 — 버전을 빌더와 일치시킬 것 (HANDOFF_ISSUES P1-1)
python scripts/train_fusion.py --model cross_attn --dataset-version vf --epochs 80

# 3) 측정·XAI
python scripts/measure_emergency.py        # 5-class F1·recall·이진 응급
python scripts/verify_xai_attn.py          # SHAP·IG·perm 순위일치 ρ
python scripts/demo_xai.py                 # XAI 시연 (주의: gated 모델 로드)

# 데이터 없이도 가능한 구조 검증
pytest -q                                  # e2e 스모크 4 passed
```

> ⚠️ **기본값 함정:** 빌더 기본 version은 `vf`, 학습 기본은 `v1`. 명시적으로 맞춰야 한다.
> 학습 기본 `--model`은 `gated`(비채택)이므로 채택 모델은 `--model cross_attn` 명시 필수.

---

## 6. 인수인계 시 가장 먼저 확인할 것 (우선순위)

1. **[P0]** P1 캐시·합성셋·체크포인트의 소재 — `data/`는 git 미추적. 외부 스토리지/드라이브 위치 확보.
2. **[P0]** P1 레포 접근 — P1 캐시 재생성은 `WidU_ecg-fm_emergency-detection` + ECG-FM 가중치 + `fairseq_signals` 의존.
3. **[P1]** 데이터셋 version 일치(`vf`) 확인 후 학습.
4. **[P1]** XAI를 코드로 추적 시 README 함수명(`generate_attn_explanation`)은 부재 — 본 문서 §XAI 참조.
5. **[P1]** `xai.py` off-by-one(rel=aux[6]) 정정 여부 — 설명 텍스트 정확성 영향.

---

## 7. 범위 / 한계 (README 담백 서술 기준)

- 동시 측정 멀티모달 데이터 부재로 **클래스 조건부 조립(합성 paired)** 사용. 외부 test로 일반화는
  검증하나, 실 정렬 데이터 기반 검증은 후속 과제.
- 모달 의존도는 soft(단일 모달 독점 아님). 합성 in-dist에선 concat 근소 우위 — 어텐션의 실효는
  외부 test 일반화·해석성에서 나온다.
- 관련 레포: **P1** ECG-FM+LoRA 심장 검출기(ECG 인코더), **P3** 엣지 비전(동일 온디바이스 AI 역량 타도메인).
