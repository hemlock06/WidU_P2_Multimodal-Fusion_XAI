# INTERFACE CONTRACTS — P2 Multimodal Fusion

> 모듈 간 입출력 계약. **소스에서 직접 검증한 차원·키·순서만 기재.** 작성일: 2026-06-28.
> 단일 진실원천(SSOT): `src/p2fusion/schema.py`.

---

## 1. P1 → P2 입력 계약 (ECG 채널)

P2는 ECG를 재인코딩하지 않고 **P1 출력을 그대로 입력**으로 받는다.

### 1.1 P1 캐시 파일 (`scripts/build_p1_cache.py` 생성)
`data/p1_cache/cpsc_mc_{train,val,test}.npz`

| 키 | shape | 의미 | 산출 |
|---|---|---|---|
| `embedding` | `[N, 768]` | ECG-FM mean-pool 임베딩 (raw) | backbone `out["x"].mean(dim=1)` |
| `cardiac_probs` | `[N, 5]` | 심장 5분류 확률 | `softmax(head_mc(emb))` |
| `emergency_score` | `[N]` | 응급 점수 (0~1) | `sigmoid(head_bin(emb))` |
| `hr_bpm` | `[N]` | 추정 심박수 (bpm, **raw 스케일 ~40–185**) | R-peak 기반 `estimate_physio` |
| `rhythm_regularity` | `[N]` | 추정 리듬 규칙성 (0~1) | `clip(1 - CV×3, 0, 1)` |
| `label_mc` | `[N]` | 심장 5분류 정답 (NSR=0,AF=1,Ischemia=2,Conduction=3,Ectopic=4) | CPSC 라벨 |
| `label_bin` | `[N]` | 이진(응급=1) | CPSC 라벨 |

- **외부 의존:** `build_p1_cache.py`는 P1 레포(`P1_REPO_DIR`, 기본 `../WidU_ecg-fm_emergency-detection`)의
  ECG-FM 체크포인트와 LoRA 가중치, `fairseq_signals` 패키지를 요구한다. 본 레포 단독으로는 실행 불가.

### 1.2 임베딩(768) + cardiac scores → 융합 모델 입력으로의 변환
- ECG 임베딩 `[768]`은 모델 내부 병목으로 들어간다(cross_attn: `768→16`, gated: `768→256` 또는 `768→bn`).
- cardiac scores·생리지표는 `MultimodalSample.flat_ecg_aux()`로 `ecg_aux[8]` 벡터가 된다(§2.2).

---

## 2. 융합 모델 입력 계약 (batch dict)

`CrossModalAttentionFusion` / `GatedFusionModel` / `ConcatMLP` 공통 입력 (`forward(batch)`):

| 키 | shape | dtype | 의미 |
|---|---|---|---|
| `ecg_emb` | `[B, 768]` | float32 | P1 임베딩 (`EMB_DIM=768`) |
| `ecg_aux` | `[B, 8]` | float32 | P1 임상점수 (§2.2 순서) |
| `imu` | `[B, 12]` | float32 | IMU 핸드크래프트 피처 (`IMU_DIM=12`) |
| `spo2` | `[B, 8]` | float32 | SpO₂ 핸드크래프트 피처 (`SPO2_DIM=8`) |
| `mask` | `[B, 3]` | float32 | 모달 가용성 `(ecg, imu, spo2)`, 1=존재 0=결측 |
| `label` | `[B]` | int64 | 정답 클래스 0~4 (`loss()` 호출 시 필요) |

> 결측 처리: 세 모델 모두 `mask`로 해당 모달 피처/토큰을 zero-out. cross_attn은 추가로
> `src_key_padding_mask`로 attention에서 제외(전부 결측이면 마스크 해제로 NaN 방지).

### 2.1 데이터셋 파일 키 (`P2Dataset`가 읽는 .npz)
`data/synthetic/p2_synth_{version}_{split}.npz` — **파일 키와 batch 키가 다름에 주의:**

| .npz 키 | → batch 키 | shape |
|---|---|---|
| `ecg_embedding` | `ecg_emb` | `[N, 768]` |
| `ecg_aux` | `ecg_aux` | `[N, 8]` |
| `imu_feat` | `imu` | `[N, 12]` |
| `spo2_feat` | `spo2` | `[N, 8]` |
| `modality_mask` | `mask` | `[N, 3]` |
| `label` | `label` | `[N]` |

### 2.2 `ecg_aux[8]` 순서 (`schema.flat_ecg_aux`, SSOT)

| idx | 이름 | 범위/단위 |
|---|---|---|
| 0 | `cardiac_probs[0]` = NSR | 0~1 |
| 1 | `cardiac_probs[1]` = AF | 0~1 |
| 2 | `cardiac_probs[2]` = Ischemia | 0~1 |
| 3 | `cardiac_probs[3]` = Conduction | 0~1 |
| 4 | `cardiac_probs[4]` = Ectopic | 0~1 |
| 5 | `emergency_score` | 0~1 |
| 6 | `hr_bpm` | bpm (raw ~40–185) |
| 7 | `rhythm_regularity` | 0~1 |

> ⚠️ `xai.py`는 일부 위치에서 `rhythm_regularity`를 idx 6(=hr_bpm)으로 잘못 참조한다(off-by-one).
> [HANDOFF_ISSUES.md](HANDOFF_ISSUES.md) **P1** 참조 — 모델 예측엔 영향 없고 XAI 설명 텍스트에만 영향.

### 2.3 IMU 입력 피처 `[12]` (`schema.IMU_FEATURES`)
순서: `smv_mean, smv_std, smv_peak, smv_min, jerk_peak, gyro_peak, gyro_energy, tilt_change, act_energy, dom_freq, spec_entropy, impact_count`.
- raw 추출: `features/imu_features.py:window_to_imu_feat(data[T,6], fs=200, accel_unit)`,
  입력 채널 순서 `[ax,ay,az,gx,gy,gz]`, 가속도 단위 g(또는 `accel_unit="ms2"`로 변환), 자이로 rad/s.

### 2.4 SpO₂ 입력 피처 `[8]` (`schema.SPO2_FEATURES`)
순서: `spo2_mean, spo2_nadir, spo2_current, desat_rate, time_below_90, time_below_88, recovery_slope, spo2_std`.
- raw 추출: `features/spo2_features.py:extract_spo2_features(spo2[T], fs=1.0)`. 입력은 % 단위 시계열.

---

## 3. 융합 모델 출력 계약

### 3.1 `CrossModalAttentionFusion.forward` → dict

| 키 | shape | 의미 |
|---|---|---|
| `logits` | `[B, 5]` | 메인 fusion logits → **class_probs**(softmax) |
| `unimodal_logits` | `[B, 3, 5]` | 각 토큰(ECG,IMU,SpO₂) 단독 예측 |
| `attention_weights` | `[B, 3, 3]` | 마지막 레이어 self-attn (head 평균). row=query, col=key |
| `gate_weights` | `[B, 3]` | attention 열 합산 후 정규화 (시각화·GatedFusion 호환) |
| `conf_per_modality` | `[B, 3]` | 각 unimodal softmax max (detached) |

### 3.2 `GatedFusionModel.forward` → dict
키: `logits[B,5]`, `gate_weights[B,3]`, `unimodal_logits[B,3,5]`, `conf_per_modality[B,3]`.
(`attention_weights` 없음 — cross_attn 전용.)

### 3.3 `ConcatMLP.forward` → **Tensor `[B, 5]`** (dict 아님)
- 다른 두 모델과 인터페이스가 다르다. 소비 코드는 `isinstance(...)` 분기 또는
  `out["logits"] if isinstance(out, dict) else out` 패턴으로 처리(`tests/test_p2.py` 참조).

### 3.4 클래스 taxonomy (`logits[5]`, `schema.CLASS_NAMES`)
| idx | 이름 | 1차 모달 |
|---|---|---|
| 0 | `normal_rest` 정상(안정) | 전체 |
| 1 | `normal_active` 정상(활동) | IMU+ECG |
| 2 | `cardiac` 심혈관 응급 | ECG(P1) |
| 3 | `impact` 외부충격(낙상) | IMU |
| 4 | `hypoxia` 저산소 | SpO₂ |

`loss(batch, out)` = `main_CE + aux_loss_weight × mean unimodal_CE` (cross_attn·gated 공통, gated는 결측 모달 제외 평균).

---

## 4. XAI 출력 계약 (`src/p2fusion/xai.py`)

| 함수 | 입력 | 출력 |
|---|---|---|
| `collect_gate(model, arrays, device)` | `{ecg_emb,ecg_aux,imu,spo2,mask}` | `(gate_w[N,3], conf[N,3], uni_logits[N,3,5], pred[N])` — 모델이 gate_weights·conf·unimodal_logits 출력해야 함 |
| `integrated_gradients(model, sample, target, device, steps=64)` | 단일 샘플 dict | `{ecg_emb[768], ecg_aux[8], imu[12], spo2[8]}` 피처별 귀속 |
| `aggregate_attribution(attr)` | IG attr dict | `(per_mod{ECG,IMU,SpO2}, feats{IMU,SpO2,ECG_aux,ECG_emb})` |
| `ig_completeness(...)` | model·sample·attr | `(Σattr, F_t(x)−F_t(base))` 완결성 갭 |
| `generate_gate_explanation(...)` | pred·gate_w·conf·uni·ecg_aux | 한국어 NL 문자열 (모달 기여·확신·단독예측) |
| `generate_ig_explanation(...)` | pred·attr | 한국어 NL 문자열 (모달 기여 + 상위 피처) |
| `generate_combined_explanation(model, sample, device)` | 단일 샘플 | 한국어 NL (게이트 라우팅 + IG + P1 임상판독 + 기여분해) |
| `generate_caregiver_message(model, sample, device)` | 단일 샘플 | 한국어 보호자 알림 (평이어 + 행동지침) |

- **side outputs로서의 XAI 산출:** `attention_weights[3,3]`(어느 모달 주목), `gate_weights[3]`(라우팅),
  `conf_per_modality[3]`(모달별 확신), `unimodal_logits[3,5]`(모달별 단독판정)은 모델 forward에서 직접 나온다.
- ② SHAP 기반 귀속은 `xai.py`가 아니라 `scripts/verify_xai_attn.py`·`verify_shap_vs_ig.py`에서 grouped
  Shapley sampling으로 별도 계산된다(모듈은 IG 사용).

---

## 5. 불변식 (테스트로 보증, `tests/test_p2.py`)

- `MultimodalSample.flat_ecg_aux().shape == (8,)` (5 cardiac + 3 physio).
- cross_attn forward 출력 키 `{logits, unimodal_logits, attention_weights, gate_weights, conf_per_modality}` 존재,
  `logits[B,5]`·`unimodal_logits[B,3,5]`·`attention_weights[B,3,3]`·`conf_per_modality[B,3]`.
- cross_attn `loss`는 유한 스칼라.
- ConcatMLP forward는 `[B,5]`(dict 또는 Tensor 모두 수용하는 패턴으로 검증).
- 데이터 비의존 e2e 스모크 4개(`pytest -q` → 4 passed). 학습 데이터 없이도 구조 검증 가능.
