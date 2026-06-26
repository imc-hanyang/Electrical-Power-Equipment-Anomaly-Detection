# KEPCO OPGW 이상 탐지

CLIP 기반 CLAdapter를 활용한 OPGW 전선 이상 탐지.
ViT-B / ConvNeXt-B 백본에 CLAdapter 2-stage fine-tuning 적용.

## 성능 (Dataset_0622, 10-Fold 평균 ± 표준편차)

| 모델 | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|
| PatchCore | 83.13 ± 8.06 | 93.97 ± 4.83 | 87.94 ± 5.29 | 93.69 ± 3.76 |
| DifferNet | 82.14 ± 5.57 | 91.85 ± 4.80 | 86.46 ± 2.45 | 93.19 ± 2.90 |
| ConvNeXt-B (linear) | 97.27 ± 2.02 | 97.23 ± 2.52 | 97.24 ± 2.19 | 99.74 ± 0.35 |
| ViT-B (linear) | 99.25 ± 0.94 | 99.36 ± 0.81 | 99.30 ± 0.85 | 100.00 ± 0.00 |
| **ConvNeXt-B + CLAdapter** | **99.86 ± 0.41** | **99.78 ± 0.65** | **99.82 ± 0.53** | **100.00 ± 0.00** |
| **ViT-B + CLAdapter** | **100.00 ± 0.00** | **100.00 ± 0.00** | **100.00 ± 0.00** | **100.00 ± 0.00** |

---

## 설치

```bash
conda create -n kepco_env python=3.10 -y
conda activate kepco_env
pip install -r requirements.txt
```

---

## 입력 데이터 형식

데이터셋은 반드시 아래 구조로 구성되어야 합니다.

```
dataset/
├── Anomaly/   # 이상 이미지
└── Normal/    # 정상 이미지
```

모델이 정상적으로 동작하려면 **이미지가 관심 영역 기준으로 정밀하게 crop**되어야 합니다.  
전체 사진을 그대로 입력하면 성능이 크게 저하됩니다.

### 전선 이상탐지 (Wire Anomaly Detection)

| 정상 (Normal) | 이상 (Anomaly) |
|:---:|:---:|
| ![wire_normal](docs/assets/wire_normal.jpg) | ![wire_anomaly](docs/assets/wire_anomaly.jpg) |

### 종단접속재 황변 이상탐지 (Yellow Stain Detection)

| 샘플 1 | 샘플 2 |
|:---:|:---:|
| ![hw_sample1](docs/assets/hw_sample1.jpg) | ![hw_sample2](docs/assets/hw_sample2.jpg) |

---

## Inference

6개 모델 한 번에 실행, 10-fold mean±std 테이블 자동 출력.

```bash
bash wire_inference.sh --test-dir /path/to/dataset
```

`/path/to/dataset` 하위에 `anomaly/`, `normal/` 폴더가 있으면 Precision / Recall / F1 / AUROC 자동 계산.

### 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--test-dir` | 테스트 이미지 디렉토리 | **(필수)** |
| `--fold` | fold 번호 (0~9) 또는 `auto` | `9` |
| `--metric` | `auto` 기준 지표 (`f1` / `auroc`) | `f1` |
| `--checkpoints-dir` | 체크포인트 루트 | `checkpoints/wire_final_train/` |
| `--models` | 실행할 모델 (`all` 또는 콤마 구분) | `all` |
| `--output-dir` | 결과 저장 경로 | `predictions/` |

### 모델 목록

| 키 | 모델 |
|---|---|
| `patchcore` | PatchCore |
| `differnet` | DifferNet |
| `convnextb_linear` | ConvNeXt-B (linear probe) |
| `vitb_linear` | ViT-B (linear probe) |
| `convnextb_cla_sft2` | ConvNeXt-B + CLAdapter |
| `vitb_cla_sft2` | ViT-B + CLAdapter |

---

## 프로젝트 구조

```
KEPCO-OPGW-Anomaly/
├── src/                    # CLAdapter 모델 코드
├── inference/
│   └── inference.py        # CLAdapter 단일 모델 추론
├── baselines/
│   ├── normal_only/        # PatchCore
│   └── attent_differnet/   # DifferNet
├── checkpoints/
│   └── wire_final_train/   # 6모델 통합 체크포인트 (fold_0 ~ fold_9)
├── dataset/                # 데이터셋 (anomaly / normal)
├── wire_inference.sh       # 6모델 통합 inference 스크립트
├── train_kfold.sh          # CLAdapter 10-fold 학습
└── requirements.txt
```
