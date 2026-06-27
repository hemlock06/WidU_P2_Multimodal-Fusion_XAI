# IMPROVE PROPOSALS — P2 Multimodal Fusion

> 설계결정·검증 불가 항목은 **구현하지 않고 제안만** 한다(무인 작업 원칙).
> 각 항목은 관찰(사실) + 제안(선택지) + 미구현 사유로 구성. 추측은 배제.
> 작성일: 2026-06-28. 관련: `HANDOFF_ISSUES.md`, `IMPROVE_SUMMARY.md`.

---

## 제안 1 — (P0-1) 데이터·가중치·체크포인트 재현 경로
- **관찰:** `.gitignore`가 `data/`·`*.pt`·`weights/`·`outputs/`·`checkpoints/`를 제외 → `git ls-files data/` 0개.
  P1 캐시·합성셋·체크포인트가 레포에 없음. 재생성은 외부 P1 레포(`P1_REPO_DIR`)·LoRA 가중치·
  `fairseq_signals`에 의존.
- **제안(택1 또는 조합):**
  1. 외부 스토리지(드라이브/S3 등)에 산출물을 두고 README에 다운로드 경로·체크섬 명시.
  2. `data/`에 작은 샘플 셋 + 재생성 스크립트 경로를 문서화.
  3. P1 레포 접근법(커밋 해시·가중치 위치)을 인수인계 문서에 고정.
- **미구현 사유:** 스토리지 선택·접근 권한은 인프라/운영 결정. 본 환경에서 검증 불가.

## 제안 2 — (P1-1) 정준 데이터셋 버전 정렬
- **관찰:** `build_synthetic_dataset.py --version` 기본 `"vf"`(누출 0% 셋) ≠ `train_fusion.py --dataset-version`
  기본 `"v1"`. `make_loaders`는 `p2_synth_{version}_{split}.npz`를 찾으므로 fresh rebuild 시 기본값끼리 불일치.
- **본 작업에서 한 것:** `P2Dataset`에 파일 부재 시 **버전 불일치 힌트가 담긴 명확한 에러**를 추가(강건화). 정준 버전 자체는 변경 안 함.
- **제안(택1):**
  1. 빌더·학습 기본값을 한쪽으로 통일(예: 둘 다 `v1` 또는 둘 다 `vf`).
  2. 통일하되 README quickstart도 동시 갱신.
- **미구현 사유:** `vf`(누출 0%)와 `v1`은 **의미가 다른 데이터셋**이라 어느 쪽을 정준으로 둘지는
  실험 재현성 정책 결정. 잘못 통일하면 보고 수치의 기준 셋이 조용히 바뀜 → 추측 배제.

## 제안 3 — (P2-4) `train_fusion.py --model` 기본값
- **관찰:** `--model` 기본 `"gated"`(GMU 비교군)인데 채택 모델은 `cross_attn`. README quickstart는
  `--model cross_attn` 명시. 인자 생략 시 비채택 모델이 학습됨.
- **제안:** 기본값을 `cross_attn`으로 변경(채택 모델과 일치).
- **미구현 사유:** 기본 학습 모델을 바꾸는 것은 재현·비교 워크플로에 영향을 주는 동작 변경.
  기존 스크립트/자동화가 기본 gated에 의존하는지 본 환경에서 검증 불가 → 결정 위임.

## 제안 4 — 의존성 핀(requirements)
- **관찰:** 레포에 의존성 선언 파일이 **하나도 추적되지 않음**(`requirements*.txt`·`pyproject.toml`·`setup.*` 0개).
  재현성 공백.
- **관찰(실제 import — src/+scripts/ 정적 스캔):** 3rd-party는 `numpy`, `scipy`, `torch`, `wfdb` 4종.
  `sklearn`·`shap`·`matplotlib`·`pandas`·`tqdm`·`captum`은 import 0건(README의 SHAP는 패키지가 아니라
  수동 grouped-Shapley 구현).
- **관찰(검증에 쓴 환경 = 테스트 통과 env):**
  ```
  torch==2.1.2+cu118   # CUDA 11.8 빌드 — CPU 환경은 torch==2.x (cpu) 로 대체
  numpy==1.26.4
  scipy==1.13.1
  wfdb==4.3.1           # 캐시 빌드(build_p1_cache.py)에서만 사용
  python 3.9
  ```
- **제안:** 위 4종 + python 버전을 `requirements.txt`(또는 `pyproject`)로 고정. torch는 CUDA/CPU 변형을
  README에 분기 안내.
- **미구현 사유:** torch 빌드(cu118 vs cpu)·정확한 하한 버전은 배포 타깃에 따른 결정이고, 본 환경 한 곳의
  관측치를 단일 정답으로 커밋하면 다른 타깃을 오도할 수 있음 → 관측치만 제공하고 채택은 위임.

## 제안 5 — (독립검증 발견) xai 보호자 경고 임계 방향
- **위치:** `src/p2fusion/xai.py` `generate_caregiver_message`, `if rel > 0.6:` 분기(pred==2).
- **관찰:** P1-2 인덱스 수정 후 `rel = rhythm_regularity`(0~1, 높을수록 규칙적·양호). 현재 가드는
  `rel > 0.6`일 때 "측정 신호 품질이 낮아 정확하지 않을 수 있습니다 — 재측정 권합니다"를 발화 →
  **신뢰도가 높을 때 '품질 낮음' 경고가 뜨는 내부 모순.**
- **제안(결정 필요):**
  1. 방향을 `rel < 0.6`으로 정정(신뢰도 낮을 때 경고). 단,
  2. pred==2(부정맥: AF·이소성)는 본질적으로 *낮은 규칙성*이 정상이므로, 이 분기에서 rhythm_regularity를
     신호 품질 프록시로 쓰는 것 자체가 적절한지 재검토 필요(별도 품질 지표 도입 등).
- **미구현 사유:** 올바른 임계값·올바른 품질 지표는 **저자 의도/임상 정의 검증 필요** → 추측 배제.
  (단, 본 경로는 모듈 주석상 "연구 시연"이며 배포 보호자 알림은 룰 결합기 쪽 제공.)
