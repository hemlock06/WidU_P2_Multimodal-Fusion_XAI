# P2 Multimodal Fusion — 다중모달 응급 판별 + 설명가능 AI (Project 2)

## 개요

ECG 단일 채널(P1)은 부정맥·허혈 등 **심혈관 응급**을 상시 탐지하지만, 실제 응급의 다수인
**낙상·저산소 등 비심장 응급**은 ECG 단독으로 판별이 어렵다(응급 5종·ECG 단독 macro-F1 **0.394**).
또한 원격부양 특성상 판단·대응의 주체가 **의료 비전문가인 부양자**다.

이를 풀기 위해 ECG에 **IMU(외부충격·낙상)** 와 **SpO₂(저산소)** 를 더하고, 세 모달 토큰이 서로를
참조하는 **Cross-Modal Attention**으로 동적 융합해 비심장 응급까지 커버리지를 확대한다. 나아가 판정
근거가 가려지지 않도록 **4-layer XAI**로 어느 모달·신호가 판정을 이끌었는지 부양자 눈높이 자연어로
설명한다.

> **데이터 정직성.** 동시 측정 멀티모달 데이터가 없어, 클래스 조건부 조립(방법 A)으로 합성 paired
> 데이터를 구성했다. 모든 수치는 합성 평가셋·외부 test 기준이며 고정 시드로 재현 가능하다. 설계 여정
> (concat → gated → **cross-modal attention 채택**)의 전체 서술은 [`PROJECT_JOURNEY.md`](PROJECT_JOURNEY.md),
> 기록 인덱스는 [`decisions.md`](decisions.md).

---

## 핵심 결과

| 항목 | 값 |
|---|---|
| 융합 Macro-F1 | **0.939** — 단일 모달 최고(0.80) 대비 **+0.14** |
| 응급 검출 민감도(recall) | **0.965** |
| 일반화 (val→test 격차) | 임베딩 병목(768→16)으로 **0.10 → 0.007** 해소 — 전 아키텍처 공통, 용량 맞추면 concat ≈ cross_attn |
| 4-layer XAI 충실도 | IG·SHAP 순위 일치 **ρ ≈ 0.99** (perturbation 포함 ρ ≥ 0.87) |
| 클래스별 1차 모달 의존도 | cardiac → ECG **0.39** · impact → IMU **0.69** · hypoxia → SpO₂ **0.42** (임상 정합·soft) |

> concat의 외견상 과적합은 *full-768 임베딩 용량* 탓이며, 임베딩 병목(768→16)이 전 아키텍처의 val-test
> 격차를 해소한다(0.10→0.007). 용량을 맞추면 concat ≈ cross_attn(정확도 동급). **어텐션 채택 근거는
> 정확도 우위가 아니라** ① 어텐션 가중치를 곧 판정 근거로 읽는 충실한 XAI, ② 모달이 서로 참조해야
> 풀리는 실 cross-modal joint의 잠재력(시간 정렬 실데이터 = 후속 과제)이다.

---

## 출력 taxonomy (5분류)

| 클래스 | 설명 | 1차 모달리티 |
|---|---|---|
| 0 | 정상 (안정) | 전체 |
| 1 | 정상 (운동/활동) | IMU 활동량 + ECG(심박) |
| 2 | 심혈관계 응급 | ECG (P1) |
| 3 | 외부충격 (낙상·충돌) | IMU (가속도 impact + 자이로) |
| 4 | 저산소 | SpO₂ |

---

## 아키텍처 (Cross-Modal Attention)

```
[ECG ] P1 임베딩(768)→병목(16) + 임상점수 ecg_aux(8) ┐
[IMU ] feat(12) ───────────────────────────────────┤  각 모달 → d=128 토큰 (동일 표현공간 정렬)
[SpO2] feat(8) ────────────────────────────────────┘
   → Cross-Modal Attention (Pre-LN Transformer ×2) : 토큰 상호 참조 → 맥락 표현
   → mean-pool → Fusion Head → class_probs[5]  (5종 응급 최종 판정)

   Side outputs(분석·XAI용): attention_weights[3×3] · conf[3] · unimodal_logits[3,5]
```

- **ECG 토큰**이 P1의 임베딩·임상점수(`ecg_aux` = cardiac_probs[5] + emergency·hr·rhythm = **8**)를 받아
  **Project 1 ↔ Project 2를 단일 파이프라인으로 연결**한다.
- **비교군:** `ConcatMLP`(고정 결합 베이스라인) · `GatedFusion`(GMU, conf-routed 게이팅). 합성 in-dist에선
  concat이 근소 우위였으나 용량 맞추면 동급 → 정확도가 아니라 XAI·실 joint 잠재력에서 어텐션을 채택.

---

## 4-layer XAI

| 계층 | 방법 | 답하는 질문 |
|---|---|---|
| ① Intrinsic Attention | `attention_weights` 직접 판독 | 어느 모달에 주목해 판정했나 |
| ② Post-hoc Attribution | Integrated Gradients (`xai.integrated_gradients`) — SHAP 교차검증은 `verify_shap_vs_ig.py` | 어떤 신호가 기여했나 |
| ③ Hierarchical | ECG = P1 임상확률(AF·허혈)로 해석 결합 | ECG 임베딩(불투명)의 임상 근거 |
| ④ Natural-language | `xai.generate_caregiver_message`(보호자용)·`generate_combined_explanation`(연구용) | 부양자용 일상 언어 설명 |

**별도의 설명용 모델 구축·학습 없이** 모델이 forward마다 내보내는 출력과 모델 함수 자체만으로 구성
(효율적). 임상의(성동재활의원 김보경 원장) 자문으로 임상 판단과의 정합성 검증.

---

## 구조

```
src/p2fusion/
├── schema.py            통합 멀티모달 샘플 스키마 (단일 진실원천)
├── synth/               클래스 사전분포 + 조건부 조립기 (방법 A)
├── features/            IMU·SpO₂ raw → 핸드크래프트 피처 추출
├── models/              cross_modal_attention(채택) · gated_fusion(GMU) · concat_mlp(베이스라인)
├── data/                .npz 데이터셋 로더
└── xai.py               4-layer XAI (attention·SHAP·hierarchical·NL)
scripts/                 합성셋 빌드 · 피처 추출 · 융합 학습 · 어블레이션 · 측정/검증
tests/                   e2e 스모크 (모델 forward·loss·schema, 데이터 비의존)
records/                 설계 결정 · 평가 · 이슈 (decisions.md = 인덱스)
```

## 실행

```bash
# 합성 데이터(클래스 조건부 조립) 생성
python scripts/build_synthetic_dataset.py --n-per-class 4000 --seed 42

# 융합 학습 — cross_attn(채택) / gated(GMU 비교군) / concat(베이스라인)
python scripts/train_fusion.py --model cross_attn --epochs 80

# 측정·검증 (데이터 위치는 env WIDU_P2_DATA 또는 repo/data 기본)
python scripts/measure_emergency.py     # 응급 검출 F1·recall
python scripts/measure_reliance.py      # 클래스별 모달 의존도
python scripts/verify_xai_attn.py       # XAI 순위일치 ρ·deletion

pytest -q                               # e2e 스모크 (4 passed)
```

---

## 한계 (담백)

- **합성 paired 데이터.** 동시 측정 멀티모달 데이터 부재로 클래스 조건부 조립을 사용 — 외부 test로
  일반화는 검증하나, 실 정렬 데이터 기반 검증은 후속 과제.
- **모달 의존도는 soft.** 클래스별 1차 모달에 최대 가중되나 단일 모달 독점이 아닌 분산(임상 최댓값 유지).
- 합성 in-dist 정확도는 concat과 동급 — 어텐션의 실효는 **XAI(어텐션 가중치 해석) + 실 cross-modal joint(필드 이월)**에서 나온다.

## 관련 레포

- **P1** (`WidU_P1_LoRA-PEFT_Foundation-Model_Adaptation`): ECG-FM + LoRA 심장 검출기 — 본 모듈의 ECG 인코더(임베딩·임상점수 공급).
- **P3** (`CoLot-edge-vision`): 엣지 비전 무인 주차 관제 — 동일 "온디바이스 AI 시스템 기획·통합·검증" 역량의 다른 도메인 적용.
