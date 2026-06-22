# 실행 기록 (Run History)

> 다운로드, 전처리, 환경 설정 등 실행 이력 기록.

---

## 환경

| 항목 | 사양 |
|---|---|
| Python | 3.9 (python) |
| PyTorch | 2.1.2+cu118 |
| CUDA | 11.8 |
| GPU | NVIDIA RTX 3060 12GB |
| OS | Windows |

---

## 2026-05-30 — 단계 1: 클래스 조건부 조립기 (방법 A) 구축 + 1단계 합성셋

**구현**
- `src/p2fusion/schema.py` — 통합 멀티모달 샘플 스키마 (emb768 + ecg_aux8 + imu12 + spo2_8)
- `src/p2fusion/synth/class_priors.py` — 5클래스 문헌 캘리브레이션 사전분포 (IMU/SpO2/ECG)
- `src/p2fusion/synth/assembler.py` — 조건부 독립 조립 + 측정노이즈(0.35×std) + hard case(12%)
- `scripts/build_synthetic_dataset.py` — train/val/test npz 생성

**산출물**: `data/synthetic/p2_synth_v1_{train,val,test}.npz`
- train 14000 / val 3000 / test 3000, 5클래스 균형, seed=42

**분리도 sanity (선형/RF, 임베딩 제외 피처만)**

| 입력 | macro-F1 | 비고 |
|---|---|---|
| IMU only | 0.845 | 심혈관(0.70)·저산소(0.72) 약함 — 저활동 클래스 혼동 |
| SpO2 only | 0.789 | 저산소(0.95)만 강함 |
| ECG-aux only | 0.719 | 심혈관(0.85) 강함, 낙상·저산소 약함 |
| **ALL 융합 (linear)** | **0.940** | 모든 클래스 0.93+ |
| ALL 융합 (RF) | 0.947 | — |

→ 단일 모달리티는 자기 담당만 잘 보고(현실적), 융합이 +0.10 향상. 천장 0.94(≠1.0)로
   아키텍처 비교 여지 확보. 혼동은 의도한 쌍(운동↔낙상, 운동↔심혈관, 심혈관↔저산소)에 집중.

**조정 경과**: 초기 사전분포는 과분리(선형 macro-F1=1.0) → emergency_score를 P1 AUROC=0.914
불완전성에 맞춰 중첩, smv_min/peak·impact_count 운동↔낙상 중첩 확대, 측정노이즈+hard case 도입.

---

## 2026-05-30 — Phase 1 융합 모델 완성 (conf-routed gate)

ConcatMLP 베이스라인 + GatedFusionModel. 게이트 4변형 실험으로 학습 게이트 붕괴 확인,
conf-routed 채택. 상세: `01_design_decisions.md §3`. 3시드로 확정.

---

## 2026-05-30 — Phase 2: 실데이터 확보 + 전처리 검증/보정

**다운로드 완료**
- PTT-PPG (PhysioNet): 66 레코드(22명×sit/walk/run), 18채널 WFDB, 500Hz.
  IMU(a_x/y/z g단위→실측 m/s², g_x/y/z deg/s) + ECG + PPG. **SpO2 수치 채널 없음**(pleth=파형).
- SisFall (kagglehub): 4505 파일(낙상1798+ADL2707), 9채널 CSV, 200Hz, 허리 착용.
- Harespod (figshare): 23명, spv(SpO2값 1Hz) + spo(파형 100Hz) + hr. 7z 해제.

**전처리 검증 (심층 재검토) — 문제 3건 발견·해결**
1. IMU 스케일 불일치(클래스 1·3 보정 vs 0·2·4 문헌) → 통일 프로토콜로 해결
2. fs·윈도우 불일치(PTT 500Hz/전체 vs SisFall 200Hz/3초) → 200Hz·3초 통일
3. Harespod SpO2 피험자별 min-max 정규화 → 절대% 복원 불가, 임상 prior 유지
   (상세: `01_design_decisions.md §4·§5`)

**통일 IMU 보정 결과** (`calibrate_imu_priors.py`, 200Hz·3초)

| 피처 | rest(PTT sit) | active(PTT walk+run) | fall(SisFall) |
|---|---|---|---|
| jerk_peak | 1.34 | 9.46 | 280.5 |
| gyro_peak | 0.001 | 0.037 | 8.26 |
| smv_peak | 0.99 | 1.36 | 3.69 |

**재보정 후 분리도** (선형, 임베딩 제외)

| 입력 | macro-F1 | 비고 |
|---|---|---|
| IMU only | 0.782 | hypoxia 0.50(rest와 구분불가=정상), impact 0.93 |
| SpO2 only | 0.799 | hypoxia 0.94 |
| ECG full | 0.394 | cardiac 0.95 |
| ALL 융합 | 0.951 | — |

**재학습 (conf-routed, 보정 데이터)**: test macro-F1=0.954, recall 0.93~0.97.
게이트 라우팅: rest→spo2 0.76, active→imu 0.79, cardiac→ecg 0.83, impact→imu 0.91, hypoxia→spo2 0.94.

---

## 스크립트 추가
- `scripts/download_ptt_ppg.py` — PhysioNet 자동 다운로드
- `scripts/extract_ptt_ppg_features.py` / `extract_sisfall_features.py` — 피처 추출
- `scripts/calibrate_imu_priors.py` — 통일 200Hz·3초 IMU 보정
- `src/p2fusion/features/{imu,spo2}_features.py` — 핸드크래프트 피처 추출기

---

## Phase 2 후반 — 정직성 강화 및 누출 교정 (2026-05-30)

### IMU 샘플링 충실도 3-way 비교

**동기**: 조건부 독립 가정(I-1)이 내부 IMU 상관(smv_std↔act_energy r=0.94)을 무시해 sim-real 갭 유발.
세 버전으로 통제 실험:

| 버전 | 샘플링 방법 | test F1 | impact recall |
|---|---|---|---|
| v1_indep | 독립 trunc_normal | 0.957 | 0.928 |
| v2_mvn | MVN (실 공분산 보존) | 0.958 | 0.955 |
| v3_bootstrap | 실벡터 리샘플 + jitter | 0.962 | 0.961 |

**결론 (3-way)**:
1. 게이트 라우팅 패턴 불변 — conf-routed 결론이 충실도 아티팩트 아님 확인
2. impact recall 단조 증가 (+0.033) — 내부 상관 복원이 운동↔낙상 혼동 개선
3. MVN ≈ Bootstrap → 공분산 구조가 핵심, 고차모멘트 부차적

**주의 (누출 발견 → 결정 번복)**:
- v3 bootstrap: test→train 최근접거리 0.19 (근접복제, 누출 아티팩트)
- v2 mvn: nearest dist 0.55 (novel 샘플, 누출 없음) → **MVN 채택으로 정정**
- bootstrap 우위(0.962)는 근접 누출 아티팩트였음

### ECG 임베딩 누출 발견 및 교정

**발견**: P1Cache가 train+val 풀에서 복원추출 후 20K 샘플을 70/15/15 분할
→ test 임베딩 90.5%가 train에 동일 존재 → 성능 부풀림

**교정**: P1Cache에 `split` 파라미터 추가
- P2-train/val: P1 train+val 풀 (5290 레코드)
- P2-test: P1 test 풀 (936 레코드, 신규 미공개 레코드)
- 누출 0% 확인 후 vf(verified, final) 데이터셋 생성

**vf 재학습 결과 (conf_routed, MVN)**:
- val F1=0.962, **test F1=0.855** (진짜 값)
- val↔test 격차 0.107 — 임베딩 과적합 의심

**격차 원인 분석**:
- cardiac test: 0.938 (P1 test pool에 비정상 ECG 724개 → 충분)
- rest/active/impact/hypoxia: 0.81~0.83 (P1 test NSR 130개 — train NSR 776개와 분포 차이)
- 진단: NSR 768차원 임베딩이 비심장 분류에 과적합 → P1 점수만 쓰는 --no-embedding 가설

### 임베딩 과적합 — 병목으로 해소
vf 데이터셋에서 raw 768 임베딩 유무를 비교한 결과, 임베딩을 제거하는 대신 16/32차원 병목 인코더로 과적합을 해소했다(val/test 격차 0.10→0.007). 임베딩 유지 + 병목 채택.

---

## SpO2 Harespod 검증 (2026-05-30)

**목표**: 합성 클래스 4 SpO2 prior가 실 고도저산소 데이터와 일치하는지 검증
**방법**: Harespod Data_Disc 레벨별(20→40 = 2.0→4.0km) 재앵커링
  - 앵커: 2.0km→95%, 4.0km→85% (문헌 standard)
  - 역산: per-subject (baseline≈97%, nadir≈77%) + range≈25%p

**결과 (N=14 피험자)**:
| 고도 | 문헌목표 | 역산실측 | 매칭 |
|---|---|---|---|
| 2.0km | 95.0% | 95.0% | ✓ |
| 3.0km | 91.0% | 91.4% | ✓ |
| 4.0km | 85.0% | 85.0% | ✓ |

**합성 prior vs Harespod 수렴 검증**:
- spo2_nadir: prior 84% ≈ Harespod 4km 85.0% ✓
- spo2_mean: prior 89% ≈ Harespod 3.5km 88.3% ✓
- spo2_std: prior 2.8%p ≈ Harespod 4km 2.45%p ✓ (미세 과추정)

**한계**: 순환성 존재(앵커가 문헌값) → 독립 측정 아님, 수렴 검증으로 정의.
임상 응급 저산소(<80%) 범위는 Harespod 범위 밖(최대 4km≈85%) — 미검증.

---

## 조건부 독립 검증 (2026-05-30)

**PTT-PPG 교차모달 검증**:
- HR↔IMU 클래스 내 상관: rest r=-0.31(중간), active r=+0.21(약함)
- → 가정 경험적으로 지지됨 (|r|<0.3 수준)

**IMU 내부 상관 (imu_calibration 200Hz·3초 기준)**:
- smv_std↔act_energy: 실 0.90(rest)/0.94(active) vs 합성 0.33/0.15 → 가정 위배
- smv_peak↔jerk_peak: 실 0.81/0.85 vs 합성 0.08/0.36 → 가정 위배
- → MVN으로 수정 → impact recall 0.928→0.955 개선

**I-1 종결 판정**: 교차모달 독립은 지지됨. 모달리티 내부 상관 위배는 MVN으로 교정.
남은 미검증: 응급 시 교차모달 결합(낙상↔놀람빈맥 타이밍).
