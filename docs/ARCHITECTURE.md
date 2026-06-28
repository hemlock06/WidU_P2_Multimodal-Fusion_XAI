# ARCHITECTURE — P2 Multimodal Fusion

> 개발자 인수인계용 아키텍처 문서. **이 문서는 소스 코드에서 직접 검증한 사실만 담는다.**
> 추측·미검증 항목은 [HANDOFF_ISSUES.md](HANDOFF_ISSUES.md)에 별도 기록.
> 기준 브랜치: `handoff-prep` (소스 로직 무수정, 문서만 추가). 작성일: 2026-06-28.

ECG(P1) + IMU + SpO₂ 세 모달리티를 **Cross-Modal Attention**으로 융합해 5종 응급을 판별하고,
**4-layer XAI**로 판정 근거를 설명하는 모듈. P1(ECG 인코더)의 출력을 입력 계약으로 받아
Project 1 ↔ Project 2를 단일 파이프라인으로 연결한다.

---

## 1. 전체 파이프라인

```
[P1: WidU ECG-FM + LoRA]                 (외부 레포, 본 레포 밖)
   └─ scripts/build_p1_cache.py
        CPSC2018 신호 → ECG-FM(backbone) + LoRA + 2 heads
        → data/p1_cache/cpsc_mc_{train,val,test}.npz
           embedding[768] · cardiac_probs[5] · emergency_score · hr_bpm · rhythm_regularity · label_mc · label_bin
                         │
                         ▼  (P1 → P2 경계: 임베딩·임상점수를 그대로 사용, 재인코딩 없음)
[P2: 합성 paired 데이터 조립]
   └─ scripts/build_synthetic_dataset.py
        P1Cache + ConditionalAssembler(클래스 조건부 조립, 방법 A)
        IMU/SpO₂ = 클래스 사전분포(synth/class_priors.py)에서 샘플링
        → data/synthetic/p2_synth_{version}_{split}.npz   (기본 version=vf, 누출 0%)
                         │
                         ▼
[P2: 융합 학습]  scripts/train_fusion.py  (--model cross_attn | gated | concat)
   data/dataset.py(P2Dataset) → 모델 forward → main CE + aux CE
        → data/checkpoints/p2_{run_id}/best_model.pt (val macro-F1 기준)
                         │
                         ▼
[P2: 측정·XAI]  scripts/measure_*.py · verify_xai_attn.py · demo_xai.py
```

- **P1 → P2 경계(핵심):** P2는 ECG를 재인코딩하지 않는다. P1의 `embedding[768]`과 임상점수
  (`cardiac_probs[5]`, `emergency_score`, `hr_bpm`, `rhythm_regularity`)를 P1 캐시에서 그대로 받아
  ECG 채널로 쓴다. 입력 계약 상세는 [INTERFACE_CONTRACTS.md](INTERFACE_CONTRACTS.md) §1.
- **데이터 정직성:** 동시 측정 멀티모달 데이터가 없어 클래스 조건부 조립(합성 paired)을 사용한다.
  ECG 채널은 P1 실출력 캐시에서 샘플링하되, train/val은 P1 train+val 풀, test는 P1 test 풀만
  사용해 임베딩 누출을 차단한다(`assembler.P1Cache`, `version=vf`).

---

## 2. 채택 모델 — Cross-Modal Attention

소스: `src/p2fusion/models/cross_modal_attention.py` (`CrossModalAttentionFusion`)

```
[ECG] P1 embedding(768) ─► ecg_bn: Linear(768→16)+Dropout(0.5) ─┐
                            ecg_aux(8) ────────────────────────┤ concat → 24
                                                                └► ecg_proj: Linear(24→128)+LayerNorm ─┐
[IMU] feat(12) ─────────────► imu_proj:  Linear(12→128)+LayerNorm ──────────────────────────────────┤
[SpO2] feat(8) ─────────────► spo2_proj: Linear(8→128)+LayerNorm ───────────────────────────────────┤
                                                                                                     ▼
                        결측 모달 토큰 zero-out (mask) → stack [B,3,128]
                                                                                                     ▼
        Cross-Modal Transformer Encoder ×2  (nhead=4, d_model=128, dim_ff=256, GELU, Pre-LN/norm_first, dropout=0.3)
        결측 토큰은 src_key_padding_mask로 attention에서 제외 (전부 결측이면 마스크 해제로 보호)
                                                                                                     ▼
        ┌─ 메인: 결측 제외 mean-pool → cls_head: MLP(128→64→5) ──────────────► logits[B,5]
        ├─ 보조: 각 토큰 → {ecg,imu,spo2}_uni_head: Linear(128→5) ───────────► unimodal_logits[B,3,5]
        ├─ attention_weights[B,3,3]  (마지막 레이어 self-attn, head 평균, no_grad 재계산)
        ├─ gate_weights[B,3]         (attention 열 합산 후 정규화 — GatedFusion 호환·시각화용)
        └─ conf_per_modality[B,3]    (각 unimodal softmax max, detached)
```

**하이퍼파라미터(코드 기본값):** `d_model=128`, `n_heads=4`, `n_layers=2`, `dropout=0.3`,
`aux_loss_weight=0.3`, `emb_bottleneck=16`(ECG 임베딩 병목, 과적합 방지), `num_classes=5`.
train_fusion.py에서 `--d-model/--n-heads/--n-layers/--dropout/--aux-weight`로 조정 가능.

**손실:** `loss = main_CE(logits, label) + aux_loss_weight × mean_m CE(unimodal_logits[:,m,:], label)`.

**설계 동기(docstring 근거):** conf-routed 게이트는 "가장 확신하는 단일 expert에 가중치 몰아주기"라
오경보 케이스(운동 중 순간 낙상, 만성 비정상 ECG, 수면무호흡)에서 단일 지표 오답에 끌려간다.
Cross-Modal Attention은 "ECG는 응급인데 IMU/SpO₂는 정상" 같은 맥락 패턴을 joint modeling해
모달 간 거부권(veto)·협의를 가능케 한다.

> ⚠️ 코드 주석 불일치: L175 `ecg_tok` 주석이 `[B, 26]`으로 적혀 있으나 실제는 `16+8=24`.
> L200–213의 hook 등록 루프는 `pass`만 있는 죽은 코드(실제 어텐션은 L222–234에서 수동 재계산).
> 동작에는 영향 없음. [HANDOFF_ISSUES.md](HANDOFF_ISSUES.md) P2 참조.

---

## 3. 비교군 모델

| 모델 | 소스 | 구조 요지 | 위치 |
|---|---|---|---|
| **CrossModalAttentionFusion** (채택) | `models/cross_modal_attention.py` | 3토큰 Transformer 융합 | §2 |
| **GatedFusionModel** (GMU 비교군) | `models/gated_fusion.py` | 모달별 expert + confidence-aware 게이팅 | 아래 |
| **ConcatMLP** (베이스라인) | `models/concat_mlp.py` | 전 모달 concat(796) → LayerNorm → MLP | 아래 |

### GatedFusionModel (`models/gated_fusion.py`)
- 모달별 expert: ECG `768→256→128`(또는 `emb_bottleneck>0`시 `768→bn→128`), IMU `12→64→128`, SpO₂ `8→32→128`.
  `unified_experts=True`면 세 모달이 동일 병목 인코더(`in→bn→128`) 공유.
- 게이팅: `gate_mode="learned"`(gate_net MLP, 입력 `[ecg_aux(8)+mask(3)+conf(3)]=14`) 또는
  `"conf_routed"`(`softmax(conf/τ)`, 학습 파라미터 없음, τ 기본 0.15). 결측 모달은 `-inf` hard masking.
- 융합 레벨: `fusion_level="feature"`(gate 가중합 feature → fusion MLP) 또는 `"logit"`(MoE 확률 혼합 → log-prob, NLL).
- 출력 dict 키: `logits[B,5]`, `gate_weights[B,3]`, `unimodal_logits[B,3,5]`, `conf_per_modality[B,3]`.

### ConcatMLP (`models/concat_mlp.py`)
- `INPUT_DIM = 768+8+12+8 = 796` → LayerNorm → MLP(512→256→128→5). 결측 모달은 0벡터.
- **인터페이스 차이:** forward가 **dict가 아닌 `Tensor[B,5]`를 직접 반환**한다(다른 두 모델은 dict).
  학습/평가 코드는 `isinstance(model, (GatedFusionModel, CrossModalAttentionFusion))`로 분기 처리.

> `models/__init__.py`는 `ConcatMLP`·`GatedFusionModel`만 export하고
> `CrossModalAttentionFusion`은 export하지 않는다(직접 import 필요).

---

## 4. 4-layer XAI

README가 정의하는 4계층(개념)과 **실제 구현 위치**를 함께 정리한다(둘 사이 드리프트는
[HANDOFF_ISSUES.md](HANDOFF_ISSUES.md) P1 참조).

| 계층 | 답하는 질문 | 실제 구현 |
|---|---|---|
| ① Intrinsic Attention | 어느 모달에 주목해 판정했나 | 모델 출력 `attention_weights[3,3]`. 측정: `scripts/measure_matrix.py`(6-seed 평균 [3×3]) |
| ② Post-hoc Attribution | 어떤 신호가 기여했나 | `src/p2fusion/xai.py`의 **Integrated Gradients**(`integrated_gradients`, 완결성 공리 검증). cross_attn용 SHAP/IG/perm 순위일치는 `scripts/verify_xai_attn.py`(grouped Shapley sampling) |
| ③ Hierarchical | ECG 임베딩(불투명)의 임상 근거 | `xai.py:generate_combined_explanation` — 게이트 라우팅 + IG 피처 + **P1 cardiac_probs**(NSR/AF/허혈/전도/이소성)를 ECG 해석층으로 결합 |
| ④ Natural-language | 부양자용 일상 언어 설명 | `xai.py:generate_caregiver_message` — 기술 피처(`gyro_peak`…)를 평이어로 번역 + 행동지침 |

- **별도 설명 모델 없음:** 모델이 forward마다 내보내는 side outputs와 `xai.py` 함수만으로 구성.
- **`xai.py`의 게이트 기반 함수**(`collect_gate`, `generate_gate_explanation`, `gate_report`)는
  `gate_weights`·`conf_per_modality`·`unimodal_logits`를 출력하는 모델에 동작한다. GatedFusion과
  CrossModalAttentionFusion 둘 다 이 세 키를 출력하므로 두 모델 모두에 적용 가능하다.
  단 `demo_xai.py`(시연 진입점)는 **gated 모델을 로드**하도록 작성돼 있다.
- ② SHAP는 `xai.py` 모듈이 아니라 검증 스크립트(`verify_xai_attn.py`, `verify_shap_vs_ig.py`)에서
  grouped Shapley sampling으로 계산한다. `xai.py` 모듈 자체는 IG를 사용.

---

## 5. 모듈 맵

```
src/p2fusion/
├── schema.py                  단일 진실원천(SSOT): 클래스·차원·MultimodalSample
│                              CLASS_NAMES[5], CARDIAC_PROB_NAMES[5], EMB_DIM=768,
│                              IMU_FEATURES[12], SPO2_FEATURES[8], flat_ecg_aux()→[8]
├── synth/
│   ├── class_priors.py        클래스별 ECG/IMU/SpO₂ 사전분포(trunc_normal·MVN·bootstrap)
│   └── assembler.py           P1Cache(누출방지 풀 분리) · ConditionalAssembler(방법 A) · samples_to_arrays
├── features/
│   ├── imu_features.py        raw [T,6]→[12] (SMV·jerk·gyro·tilt·dom_freq·spec_entropy·impact_count)
│   └── spo2_features.py       raw [T]→[8] (mean·nadir·current·desat_rate·time_below_90/88·recovery·std)
├── models/
│   ├── cross_modal_attention.py   채택 모델(§2)
│   ├── gated_fusion.py            GMU 비교군(§3)
│   ├── concat_mlp.py              베이스라인(§3)
│   └── __init__.py                ConcatMLP·GatedFusionModel만 export
├── data/
│   └── dataset.py             P2Dataset(.npz 로더, modality_dropout) · make_loaders(train/val/test)
└── xai.py                     4-layer XAI(§4): 게이트·IG·hierarchical·caregiver NL

scripts/
├── build_p1_cache.py          P1 모델 추론 → p1_cache .npz (P1 레포 의존: P1_REPO_DIR)
├── build_synthetic_dataset.py 클래스 조건부 조립 → synthetic .npz (기본 version=vf)
├── train_fusion.py            융합 학습(concat/gated/cross_attn)
├── measure_emergency.py       5-class macro-F1·클래스 recall·이진 응급 recall (GMU vs cross_attn)
├── measure_reliance.py        클래스별 modality reliance = 평균 gate_weights[primary_mod]
├── measure_matrix.py          클래스별 cross-modal attention 행렬 [3×3] (6-seed 평균)
├── verify_xai_attn.py         cross_attn XAI 재계산(SHAP·IG·perm 순위일치 ρ·deletion·완결성)
├── verify_shap_vs_ig.py       SHAP vs IG 방법 검증(faithfulness·rank agreement·IG 구조한계)
├── demo_xai.py                XAI 시연(gated 모델 로드 — 연구용 + 보호자용 설명)
├── ablation_*.py·run_*.py·sweep_*.py  어블레이션·비교·온도 스윕
└── (P1 캐시·피처추출·다운로드 보조 스크립트)

configs/data.yaml              2단계 실데이터 소스 레지스트리(PTT-PPG·SisFall·Harespod)
tests/test_p2.py               e2e 스모크 4개(데이터 비의존, 모델 forward·loss·schema 차원)
```

---

## 6. 학습·평가 경로 (train_fusion.py)

- **데이터:** `P2_DATA_DIR/synthetic/p2_synth_{version}_{split}.npz` (기본 `version=v1`,
  누출 없는 셋은 `vf`). `modality_dropout_p`(기본 0.15)로 학습 중 결측 시뮬레이션(최소 1모달 보존).
- **옵티마이저:** AdamW(lr 3e-4, wd 1e-4) + CosineAnnealingLR. grad clip 1.0. 기본 80 epoch.
- **체크포인트:** `data/checkpoints/p2_{model}_{run_id}/best_model.pt`(val macro-F1 최대), `last_model.pt`, `train_log.csv`.
- **평가지표:** `macro_f1`(클래스별 F1 평균), 클래스별 recall. `--no-embedding`으로 ECG 임베딩
  zero-out(과적합 진단), `--emb-bottleneck`으로 ECG 병목 차원 제어.

검증된 결과 수치는 [HANDOFF_SUMMARY.md](HANDOFF_SUMMARY.md) §검증 결과 참조(출처: README.md `핵심 결과`).
