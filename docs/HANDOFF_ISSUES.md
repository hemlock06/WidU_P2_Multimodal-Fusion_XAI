# HANDOFF ISSUES — P2 Multimodal Fusion

> 인수인계 시 주의할 이슈 목록. **모두 소스/실행 환경에서 직접 검증한 관찰**이며, 추측이 섞인
> 항목은 제외했다. 권장 조치는 관찰에 근거한 제안이며, 본 작업은 소스 로직을 수정하지 않았다.
> 작성일: 2026-06-28. 우선순위: **P0**(인수인계 차단) > **P1**(정확성·재현성·문서 드리프트) > **P2**(경미·표기).

---

## P0 — 인수인계 차단 (실행 전 해결 필요)

### P0-1. 데이터·가중치·체크포인트가 git에 없음 → 재생성 경로가 외부 레포에 의존
- **관찰:** `.gitignore`가 `data/`, `*.pt`, `weights/`, `outputs/`, `checkpoints/`를 전부 제외.
  `git ls-files data/` 결과 0개. 즉 P1 캐시(`data/p1_cache/`), 합성셋(`data/synthetic/`),
  체크포인트가 레포에 포함되지 않는다.
- **관찰(재생성 의존성):** `scripts/build_p1_cache.py`는 P1 레포(`P1_REPO_DIR`,
  기본 `../WidU_ecg-fm_emergency-detection`)의 ECG-FM 체크포인트
  (`checkpoints/ecg-fm/...pt`)·LoRA 가중치(`outputs/lora_multitask_snr_a07/...pt`)와
  `fairseq_signals` 패키지를 import한다. 이들 없이는 P1 캐시 재생성 불가.
- **관찰(fallback):** P1 캐시가 없으면 `assembler.P1Cache`가 `FileNotFoundError`를 던지고,
  `build_synthetic_dataset.py`는 **합성 가우시안 임베딩 fallback**으로 전환한다(품질 저하 — ECG 채널이
  실 P1 출력이 아니게 됨).
- **영향:** 새 개발자가 빈 체크아웃에서 학습을 재현하려면 (a) P1 레포·가중치 확보 → P1 캐시 재생성,
  (b) 합성셋 재생성, (c) 학습 순으로 진행해야 한다. 데이터 없이는 `pytest`의 e2e 스모크 4개만 통과.
- **권장:** 인수인계 시 P1 캐시(`cpsc_mc_*.npz`)·합성셋(`p2_synth_*.npz`)·best 체크포인트의
  소재(외부 스토리지/드라이브)를 명시하거나, P1 레포 접근 경로를 함께 전달.

---

## P1 — 정확성·재현성·문서 드리프트

### P1-1. 합성셋 빌더 기본 version(`vf`) ≠ 학습 기본 version(`v1`) → 기본값 재현 시 파일 불일치
- **관찰:** `build_synthetic_dataset.py` `--version` 기본값 `"vf"`(누출 0% 셋),
  `train_fusion.py` `--dataset-version` 기본값 `"v1"`.
- **관찰:** `make_loaders`는 `p2_synth_{version}_{split}.npz`를 찾는다. 따라서 빈 상태에서 README
  quickstart 기본값대로 `build_synthetic_dataset.py`(→ `p2_synth_vf_*.npz` 생성) 후
  `train_fusion.py --model cross_attn`(→ `p2_synth_v1_*.npz` 탐색)을 돌리면 **FileNotFoundError**.
- **참고:** 현재 디스크에는 `p2_synth_v1_*.npz`가 존재해 우연히 동작하나, fresh rebuild 시 깨진다.
- **권장:** 재현 시 `train_fusion.py --dataset-version vf`(또는 빌더 `--version v1`)로 명시 일치.

### P1-2. `xai.py` off-by-one — `rhythm_regularity`를 `hr_bpm` 인덱스로 참조
- **위치:** `src/p2fusion/xai.py` L303 `es, rel = float(aux[5]), float(aux[6])`,
  L390 `ci, rel = ..., float(aux[6])`.
- **관찰:** `ecg_aux` 순서(SSOT)는 idx 6 = `hr_bpm`(raw bpm ~40–185), idx 7 = `rhythm_regularity`(0~1).
  `rel`은 "신뢰도(리듬 규칙성)"로 쓰이지만 실제로는 `hr_bpm`을 읽는다.
- **영향:** L309 `... 신뢰도 {rel:.2f}`가 hr_bpm을 출력. L424 `if rel > 0.6:`은 hr_bpm이 항상 0.6 초과라
  **모든 심혈관 케이스에서 "신호 품질 낮음 → 재측정" 경고가 항상 발화**한다.
  모델 예측·logits에는 영향 없고 XAI 설명/보호자 알림 텍스트에만 영향.
- **권장:** `aux[6]` → `aux[7]`로 정정(별도 PR; 본 브랜치는 무수정).

### P1-3. README 4-layer XAI 서술 ↔ 코드 드리프트
- **관찰(④):** `README.md` L74가 NL 계층을 `generate_attn_explanation`으로 명시하나, 이 함수는
  코드베이스 어디에도 없다(`grep` 0건). 실제 NL 설명은 `xai.py:generate_caregiver_message`(보호자용)·
  `generate_combined_explanation`(연구용).
- **관찰(②):** README L72는 ②를 "perturbation-SHAP"로 적으나, `xai.py` 모듈은 **Integrated Gradients**를
  사용한다. SHAP는 모듈이 아니라 검증 스크립트(`verify_xai_attn.py`, `verify_shap_vs_ig.py`)에서
  grouped Shapley sampling으로 계산된다.
- **관찰(시연 진입점):** `scripts/demo_xai.py`는 **gated 모델**을 로드한다(docstring: "late fusion XAI").
  채택 모델은 cross_attn이며, cross_attn의 attention 기반 XAI는 `measure_matrix.py`·`verify_xai_attn.py`로
  오프라인 측정된다.
- **영향:** XAI를 코드로 따라가려는 개발자가 README 함수명/방법으로 검색하면 찾지 못한다.
- **권장:** README의 XAI 표를 실제 구현(함수명·스크립트)과 일치시키거나, 본 문서
  [ARCHITECTURE.md §4](ARCHITECTURE.md)·[INTERFACE_CONTRACTS.md §4](INTERFACE_CONTRACTS.md)를 참조.

---

## P2 — 경미·죽은 코드·표기

### P2-1. `cross_modal_attention.py` 주석 차원 오기
- **위치:** L175 `ecg_tok = torch.cat([ecg_bn, ecg_aux], dim=-1)  # [B, 26]`.
- **관찰:** `ecg_bn`=16, `ecg_aux`=8 → 실제 `[B, 24]`(주석만 오기, 동작 무관).

### P2-2. `cross_modal_attention.py` 죽은 hook 루프
- **위치:** L200–213 `for layer in self.transformer.layers:` 내부 `make_hook`/`hook`이 `pass`만 수행하고
  훅을 등록하지 않는다. 실제 어텐션 가중치는 L222–234에서 `no_grad`로 수동 재계산.
- **관찰:** 결과에 영향 없는 dead code. 정리 대상.

### P2-3. `models/__init__.py`가 채택 모델을 export하지 않음
- **관찰:** `__all__ = ["ConcatMLP", "GatedFusionModel"]` — 채택 모델 `CrossModalAttentionFusion` 누락.
  사용 시 `from p2fusion.models.cross_modal_attention import CrossModalAttentionFusion` 직접 import 필요
  (`train_fusion.py`·`tests/test_p2.py`는 이미 직접 import).

### P2-4. `train_fusion.py` 기본 모델이 비채택 모델
- **관찰:** `--model` 기본값 `"gated"`(GMU 비교군). 채택 모델은 `cross_attn`. README quickstart는
  `--model cross_attn`을 명시하나 CLI 기본은 gated라, 인자 생략 시 비채택 모델이 학습된다.

---

## 검증 메모 (skip한 항목)
- `gate_weights`(cross_attn)는 attention 열 합산 기반 파생값으로, GatedFusion의 학습형 gate와 의미가
  다르다(시각화·호환용). 두 모델의 reliance 수치 비교 시 이 차이를 전제로 해석해야 하나, 구체적
  영향은 측정 스크립트 실행 결과 없이 단정 불가 → **skip(추측 배제)**.
- 합성 사전분포(`class_priors.py`)의 문헌 캘리브레이션 정확성은 원 논문 대조가 필요해 **skip**.
