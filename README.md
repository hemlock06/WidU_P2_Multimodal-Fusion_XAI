# P2-A Late Fusion — 멀티모달 응급 원인 융합 (Project 2-A)

## 개요

P1(ECG)의 출력에 IMU(가속도·자이로)와 SpO2를 결합해 응급 원인을 5분류하는 멀티모달 융합의 첫
접근(late fusion)이다. 동시 측정 멀티모달 데이터가 없는 제약 위에서, 클래스 조건부 조립으로 합성
paired 데이터를 구성하고 gated late fusion을 설계했다. 데이터 누수 점검을 포함한 검증을 거쳐 정직한
분리 성능(macro-F1 0.94)을 확보했으나, **조건부 독립 조립의 구조적 한계**(교차모달 시간 대응 부재)를
확인했다. 이 한계가 후속 cross-modal attention(P2-B)과 실데이터 검증 시스템의 동기가 되었다.

설계 근거·결과·한계의 전체 서술은 [`PROJECT_JOURNEY.md`](PROJECT_JOURNEY.md), 기록 인덱스는
[`decisions.md`](decisions.md). 교차 어텐션 접근은 별도 레포 `WidU_P2_B_Cross_Modal_Attention`로 분기했다.

---

## 핵심 기여 / 결과

| 항목 | 내용 |
|---|---|
| **데이터 전략** | 동시측정 데이터 부재 → 클래스 조건부 조립(방법 A) + 실측 IMU 사전분포 보정 |
| **융합 아키텍처** | concat(베이스라인) / 학습형 게이팅(붕괴) / confidence 라우팅(채택) — test macro-F1 0.94 |
| **정직성 검증** | 누수 3종 교정(임베딩 90.5% · bootstrap 근접 · 고차원 과적합) → 보정 후 macro-F1 0.94 |
| **핵심 한계** | 조건부 독립 조립은 교차모달 시간 대응을 담지 못함 → 융합 정교화의 효용 제한 |

---

## 출력 taxonomy (5분류)

| 클래스 | 설명 | 1차 모달리티 |
|---|---|---|
| 0 | 정상 (안정) | 전체 |
| 1 | 정상 (운동/활동) | IMU 활동량 + ECG(심박) |
| 2 | 심혈관계 응급 | ECG (P1) |
| 3 | 외부충격 (낙상·충돌) | IMU (가속도 impact + 자이로 보조) |
| 4 | 저산소 | SpO2 |

---

## 아키텍처 (Gated Late Fusion)

```
[ECG]  P1 임베딩(768) ─► 인코더 768→32→128 ┐
[IMU]  feat(12) ──────► 인코더 12→32→128   ├─► confidence(conf) 라우팅 ─► 가중합 ─► MLP ─► 5분류
[SpO2] feat(8) ───────► 인코더 8→32→128    ┘   (단일 구조: 세 인코더 동일 병목 in→32→128)
```

concat MLP(베이스라인) / Gated Late Fusion(메인). 학습형 게이팅은 합성 데이터에서 붕괴(고정 가중치로
수렴)했고, 학습 파라미터가 없는 confidence 라우팅(gate_w = softmax(conf/τ))이 클래스별 모달리티 분리를 안정적으로 달성했다.

---

## 구조

```
src/p2fusion/
├── schema.py            통합 멀티모달 샘플 스키마
├── synth/               클래스 사전분포 + 조건부 조립기 (방법 A)
├── features/            IMU·SpO2 raw→피처 추출
├── models/              concat_mlp · gated_fusion · cross_modal_attention(비교군)
└── data/                데이터셋 로더
scripts/                 합성셋 빌드 · 실데이터 다운로드 · 피처 추출 · 융합 학습 · 어블레이션
records/                 설계 결정 · 훈련 로그 · 평가 · 이슈 (decisions.md = 인덱스)
```

## 실행

```bash
python scripts/build_synthetic_dataset.py --n-per-class 4000 --seed 42   # 합성셋 생성
python scripts/train_fusion.py                                           # 융합 학습
```

### 합성셋 분리도 (sanity, 임베딩 제외 선형 분류기)

| 입력 | macro-F1 |
|---|---|
| IMU only | 0.845 |
| SpO2 only | 0.789 |
| ECG-aux only | 0.719 |
| 전체 융합 | **0.940** |

단일 모달리티는 자기 담당만 분류하고, 융합이 +0.10 향상 — 단 이 수치는 합성 데이터 기준이며 한계는 위 §개요·PROJECT_JOURNEY 참조.

---

## 주요 결과

- 데이터 전략 — 동시측정 부재 → 클래스 조건부 조립(방법 A) + 실측 사전분포 보정
- 전처리 정합 (IMU 200Hz·3초, 5클래스 실측 재보정)
- 융합 아키텍처 3종 (concat 베이스라인 / 학습형 게이팅=붕괴 / confidence 라우팅=채택)
- 데이터 누수 3종 교정 (임베딩 90.5% · bootstrap 근접 · 고차원 과적합) → macro-F1 0.94
- 구조적 한계 규명 (조건부 독립 조립 = 교차모달 시간대응 부재)
- 활동분류 IMU 임계룰 baseline (test macro-F1 0.964, 학습 0 대조군)
- 실데이터 검증 (실 ECG+IMU 운동쌍 — 합성 융합 과발화, 룰 ≥ 학습 확인)
- XAI 4계층 (게이트 라우팅 · IG 피처귀속 · P1 임상 판독 · 보호자 자연어)
- 임베딩 과적합 종결 (임베딩 병목으로 해소)

---

## 관련 레포

- **P1** (`WidU_P1_LoRA-PEFT_Foundation-Model_Adaptation`): ECG-FM + LoRA 심장 검출기 — 본 모듈의 ECG 인코더.
- **P2-B** (`WidU_P2_B_Cross_Modal_Attention`): cross-modal attention 융합 — late fusion 한계를 잇는 후속.
- **시스템 통합·배포**: 검증된 검출기를 규칙 결합기로 통합한 별도 레포.
