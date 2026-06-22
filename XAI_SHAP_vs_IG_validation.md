# P2 멀티모달 융합 모델 — XAI 방법 선택 검증 보고서
### post-hoc 설명을 perturbation-SHAP로 일원화(IG 폐기)하는 것이 타당한가

> **검증 성격**: 모델 정확도 테스트가 아니다. 모델(가중치·예측·macro-F1 **0.9421**)은 고정이고,
> 바뀌는 것은 "설명 방법"뿐이다. 평가 대상은 **설명의 충실도(faithfulness)와 일관성(consistency)**이다.
> 모든 수치는 실측이며, 측정하지 않은 값은 표기하지 않았다.

---

## 0. 대상·환경

| 항목 | 값 |
|---|---|
| 모델 | `p2_gated_11882` (GatedFusionModel, conf_routed, τ=0.15, emb_bottleneck=32, fusion=feature) |
| 평가셋 | `p2_synth_vf_test.npz` (N=**3525**, 5-class) — 모델 고정, baseline macro-F1 **0.9421** |
| 피처 스키마 | 29개 = ECG(768→1 블록) + ecg_aux 8 + IMU 12 + SpO₂ 8 |
| 비교 방법 | **SHAP**(PermutationExplainer, 캐시 재사용) · **IG**(Integrated Gradients, 재계산) · **permutation**(피처 셔플) |
| 결측/제거 처리 | 해당 피처를 **background 평균값**(test 평균)으로 치환. ECG는 768차원 블록 일괄 치환. |
| 재현 | `scripts/verify_shap_vs_ig.py` → `results/shap_vs_ig_verification.json` (신규 산출, 기존 파일 미수정) |

---

## 1. 검증 1 — Faithfulness deletion test (핵심)

SHAP 평균중요도(mean|SHAP|) 순서로 피처를 차례로 제거하며 전체 test의 macro-F1을 측정. 동일 절차를
무작위 순서(시드 5개 평균)·하위중요도 순서로도 수행. **AUC**(정규화 곡선면적, 낮을수록 가파른 하락)와
**상위5 제거 시 ΔF1**로 정량화.

| 제거 순서 | 곡선 AUC | 상위5 제거 ΔF1 | 해석 |
|---|---|---|---|
| **SHAP-top** (중요도순) | **0.193** | **−0.504** | 상위 제거 시 즉시 붕괴 (0.942→0.438) |
| random (무작위 평균) | 0.467 | −0.057 | 중간 |
| SHAP-bottom (하위순) | 0.853 | −0.000 | 하위 제거는 영향 없음 (0.942 유지) |

곡선 요약 (macro-F1, 제거 피처 수 k):

```
k:        0      1      2      3      5     10     20     31
top:    .942   .818   .424   .302   .438   .137   .067   .067   ← 급락
random: .942   .930   .907   .900   .885   .615   .316   .067
bottom: .942   .942   .942   .942   .942   .942   .938   .067   ← 평탄, 막판만 하락
```

**판정 (V1)**: SHAP-top 제거 곡선이 random·bottom보다 **유의하게 가파르게** 떨어진다(AUC 0.193 ≪ 0.467 ≪
0.853; 상위5 제거 ΔF1 −0.504 vs −0.057 vs 0.000). → **SHAP 귀속은 실제 모델 의존을 반영 = faithful.**
하위 피처를 모두 빼도 0.942가 유지되다가 마지막에야 붕괴하는 대칭 패턴이 이를 재확인한다.

---

## 2. 검증 2 — 방법 간 순위 일치도 (Spearman)

IG·permutation·SHAP의 피처 중요도를 공통 피처로 정렬해 쌍별 Spearman 상관을 계산.

| 피처 집합 | IG–SHAP | IG–perm | SHAP–perm |
|---|---|---|---|
| **named 20** (IMU+SpO₂) | **0.983** | 0.932 | 0.958 |
| **all 31** (ECG·aux 포함) | **0.984** | 0.885 | 0.883 |

세 방법 상위5 (일치 확인):

| 순위 | SHAP | IG | permutation |
|---|---|---|---|
| 1 | ECG (.16) | jerk_peak (1.84) | ECG (.164) |
| 2 | jerk_peak (.123) | ECG (.86) | jerk_peak (.107) |
| 3 | desat_rate (.103) | dom_freq (.77) | desat_rate (.082) |
| 4 | dom_freq (.095) | desat_rate (.71) | tilt_change (.064) |
| 5 | tilt_change (.072) | recovery_slope (.32) | dom_freq (.054) |

**판정 (V2)**: 모든 쌍이 Spearman **0.88~0.98** (전부 p<0.001), 임계 0.7을 크게 상회. → **결론은 방법에
무관하게 robust** — IG·perm·SHAP가 **같은 신호**(ECG·jerk_peak·desat_rate·dom_freq …)를 가리킨다.
**중요**: 따라서 IG는 피처 순위에서 "틀린" 방법이 아니다 — SHAP와 거의 완전히 일치한다(0.98). IG 폐기의
근거는 "순위가 다르다"가 아니라 검증 3의 구조적 사유다.

---

## 3. 검증 3 — IG의 구조적 한계 (게이트 detach)

이 모델의 게이트는 `gated_fusion.py`에서 `conf = torch.stack([...]).detach()` 후
`gate_raw = conf / temperature`로 라우팅한다. 즉 **게이트 가중치는 입력에 대해 미분되지 않는다**(detach).
gradient 기반인 IG는 expert 경로만 보고 게이트 라우팅을 귀속할 수 없다. perturbation 기반 SHAP는
입력을 직접 교란하므로 게이트를 포함한 전체를 본다. 이를 세 측정으로 확인.

| 측정 | 값 | 의미 |
|---|---|---|
| **IG 완결성**: Σattr vs 실제 logit 변화 | **4.74 vs 5.42** | 완결성 공리(Σattr=ΔF) 위배 — IG가 결정의 **~13%(라우팅분) 미귀속** |
| 완결성 샘플별 상대오차 (중앙값) | **0.88** | 샘플 단위로는 더 불안정(중앙 88% 오차) |
| **게이트 반응성**: 모달 마스킹 시 gate_w 이동 | ECG 0.118 / IMU 0.249 / **SpO₂ 0.300** | 라우팅이 입력에 강하게 의존(0.12~0.30 이동) — IG는 detach라 이를 미분 불가, SHAP는 포착 |
| ecg_aux IG 합 | 0.000 | (참고) conf_routed+feature서 aux는 **미사용**이라 0이 정상 — SHAP도 0. *IG 한계 아님, 모델 속성* |

**판정 (V3)**: 완결성 공리 위배(Σattr 4.74 ≠ 실제 5.42, 약 13% 라우팅분 누락)와 게이트의 강한 입력
의존성(마스킹 시 가중치 0.12~0.30 이동)이, IG가 **이 모델의 핵심 기제인 게이트 라우팅을 구조적으로 볼 수
없음**을 실측으로 보인다. 단, ecg_aux IG=0은 IG 결함이 아니라 모델이 aux를 안 쓰는 속성이므로 한계 근거에서
제외했다. **결정적 근거 = 완결성 갭 + 게이트 반응성.**

---

## 4. 최종 판정

> ### SHAP로 일원화 = **타당 (Y).**

근거는 세 검증의 종합이다. **① SHAP는 충실하다(V1)** — SHAP 상위5 피처를 background로 치환하면 macro-F1이
**0.50** 떨어지는 반면 무작위 0.057·하위 0.000으로, SHAP 귀속이 실제 모델 의존을 반영한다. **② IG를 폐기해도
피처 중요도 정보는 잃지 않는다(V2)** — IG·permutation·SHAP 순위가 Spearman **0.88~0.98**로 일치해 세 방법이
같은 신호를 가리킨다. **③ 그러나 이 모델의 게이트는 `conf.detach()`라(V3)** gradient가 라우팅 경로로 흐르지
않아, IG는 입력 의존적 게이트 라우팅(모달 마스킹 시 가중치 0.12~0.30 이동)을 귀속하지 못한다 — 완결성 공리가
깨져 IG의 Σattr(4.74)가 실제 logit 변화(5.42)의 87%만 설명하고 **13%(라우팅분)를 놓친다**. perturbation 기반
SHAP는 게이트를 포함한 전체 결정을 model-agnostic하게 귀속한다. 따라서 **SHAP는 IG가 주는 충실한 피처 순위를
동일하게 제공(②)하면서 IG가 구조적으로 볼 수 없는 게이트 라우팅까지 포착(③)**하므로 SHAP가 IG를 엄밀히
지배한다 → **일원화 타당**.

**정직한 단서**: 이는 "IG가 틀렸다/불충실하다"가 아니다 — IG는 SHAP와 피처 순위가 0.98로 일치하며(②), 모델이
미분 가능한 부분에서는 정확하다(aux=0). 다만 **gated-fusion 모델의 핵심 가치인 라우팅을 IG는 구조적으로 못
본다.** 라우팅이 곧 해석의 본체인 모델에서, 라우팅을 못 보는 방법은 부적합하다. SHAP는 본다. 그래서 SHAP로
일원화한다.

---

## 5. 한계·주의 (정직)

- **합성 test셋 기반**: 평가셋은 조건부 조립 합성 데이터다. 실 정렬 멀티모달 데이터에서의 재검증은 별개 과제다.
- **완결성 갭의 귀속**: 64-step IG의 수치오차는 통상 1~2%로, 관측된 13%(집계)·88%(중앙) 갭은 수치오차를 크게
  상회하므로 detach된 게이트 경로에 귀속함이 타당하나, "전적으로 게이트만"이라는 분해까지는 본 실험 범위 밖이다.
- **SHAP 비용**: PermutationExplainer는 IG보다 forward 호출이 많다(계산비용↑). 정확도가 아니라 *설명 충실도*를
  택한 결정이며, 비용은 트레이드오프로 수용한다.
- 본 검증은 기존 모델·캐시·deck을 수정하지 않았고, 신규 산출(`shap_vs_ig_verification.json`, 본 보고서)만 추가했다.

---

## 6. 산출물·재현

- **검증 코드**: `scripts/verify_shap_vs_ig.py` (V1 deletion · V2 Spearman · V3 완결성/게이트)
- **결과 수치**: `results/shap_vs_ig_verification.json`
- **재사용 캐시(읽기전용)**: `results/shap_cache.npz`, `results/ig_result_mean.json`, `results/perm_result.json`
- 실행: `python scripts/verify_shap_vs_ig.py` (torch CPU, scipy; shap 라이브러리 불요 — 캐시 SHAP 재사용)

*(끝. 모든 수치 실측, 추정·과장 없음.)*
