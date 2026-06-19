# P2-A Late Fusion — 기록 인덱스

> GitHub: `WidU_P2_A_Late-Fusion_XAI`
> 데이터·출력은 환경변수 `P2_DATA_DIR`(기본 `data/`)로 지정 — 대용량이라 미추적.
> 관련: `WidU_P1_LoRA-PEFT_Foundation-Model_Adaptation`
>
> **상태**: late fusion 접근의 설계·검증·한계 규명을 마치고 보존한다. 교차모달 상호작용 학습은
> 후속 트랙(cross-modal attention, `WidU_P2_B_Cross_Modal_Attention`)과 실데이터 검증·통합으로
> 이관했다. 개요는 [PROJECT_JOURNEY.md](PROJECT_JOURNEY.md), 수치 상세는 `records/`.

---

## records/ 폴더 구조

| 파일 | 내용 |
|---|---|
| [records/00_research_plan.md](records/00_research_plan.md) | 연구계획 — P2 설계, 데이터 전략, 아키텍처, 단계 계획 |
| [records/01_design_decisions.md](records/01_design_decisions.md) | 설계 결정 — 융합 아키텍처, 데이터 전략, 모달리티 설계 |
| [records/02_training_logs.md](records/02_training_logs.md) | 훈련 로그 — 전 에폭 기록 |
| [records/03_eval_results.md](records/03_eval_results.md) | 평가 결과 — 테스트 지표, 누수 점검, 외부검증 |
| [records/04_run_history.md](records/04_run_history.md) | 실행 기록 — 다운로드, 전처리, 환경 설정 |
| [records/05_open_issues.md](records/05_open_issues.md) | 열린 이슈 |

---

## 3-프로젝트 로드맵 (P2는 A/B 두 접근으로 분기)

| 프로젝트 | 역할 | 입력 | 출력 | 상태 |
|---|---|---|---|---|
| P1 (ECG) | 심전도 분석 | 12-lead ECG | 심장분류 + 응급점수 + 임베딩 + 신뢰도 | ✅ 완료 |
| **P2-A (late fusion, 본 레포)** | 말단 결합 융합 | P1 출력 + IMU + SpO2 | 응급 원인 5분류 | ✅ 결론·보존 |
| P2-B (cross-modal attention) | 교차 어텐션 융합 | P1 임베딩 + IMU + SpO2 | 응급 원인 분류 | 별도 레포 |
| P3 (XAI) | 설명 생성 | P2 결과 | 근거 설명 | 대기 |

---

## P1 인터페이스 계약 (late fusion 입력 규약)

```python
P1_output = {
    "cardiac_probs":   List[float],  # 길이 5 [NSR, AF, Ischemia, Conduction, Ectopic]
    "emergency_score": float,        # 0~1, 이진 응급 확률 (AUROC=0.914)
    "embedding":       List[float],  # 길이 768, ECG-FM mean-pool
    "reliability":     float,        # 0~1, ECG 신호 신뢰도
    "gate_tier":       str,          # "use" | "mask" | "alert"
    "physio": {"hr_bpm": float, "rhythm_regularity": float},
    "model_version":   str,
    "inference_ms":    float,
}
```

---

## P2 출력 taxonomy (5분류)

| 클래스 | 설명 | 1차 모달리티 |
|---|---|---|
| 0 | 정상 (안정) | 전체 |
| 1 | 정상 (운동/활동) | IMU 활동량 + ECG(심박) |
| 2 | 심혈관계 응급 | ECG (P1) |
| 3 | 외부충격 (낙상·충돌) | IMU (가속도 impact 1차 + 자이로 보조) |
| 4 | 저산소 | SpO2 |

---

## 요약 — late fusion 접근의 결과

| 항목 | 결과 |
|---|---|
| 데이터 전략 | 동시측정 데이터 부재 → 클래스 조건부 조립(방법 A) — `records/01 §1` |
| 전처리 정합 | IMU 200Hz·3초 통일 + 5클래스 실측 재보정 — `records/01·04` |
| 융합 아키텍처 | concat(베이스라인) / 학습형 게이팅(붕괴) / **신뢰도 라우팅(채택)** — 3시드 val 0.947±.001 / test 0.935±.002 (τ ablation: 성능은 τ에 둔감, τ는 라우팅 sharpness 노브) |
| 정직성 검증 | 누수 3종 교정(임베딩 90.5% · bootstrap 근접 · 고차원 과적합) → 보정 후 **macro-F1 0.94** — `records/03` |
| **핵심 한계** | 조건부 독립 조립은 교차모달 시간 대응을 담지 못함 → 융합 정교화의 효용이 제한됨 |
| 함의 | 교차모달 학습은 실 정렬 데이터 필요(P2-B) / 신뢰 가능 시스템은 실데이터 검증 검출기 + 투명 결합 |

상세 서술: [PROJECT_JOURNEY.md](PROJECT_JOURNEY.md).

---

> 이 파일은 인덱스 전용. 상세 기록은 `records/` 폴더 참조.
