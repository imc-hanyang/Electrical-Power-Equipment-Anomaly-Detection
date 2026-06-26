# KEPCO OPGW 이상 탐지

## 프로젝트 구조

```
KEPCO-OPGW-Anomaly/
├── src/                            # CLAdapter 모델 코드 (학습 진입점: train.py)
├── inference/
│   └── inference.py                # 추론 스크립트
├── baselines/
│   ├── normal_only/                # PatchCore
│   ├── differnet/                  # DifferNet
│   ├── cladapter/                  # CLAdapter 베이스라인
│   ├── supervised_classifier/      # EfficientNet 베이스라인
│   └── CLAdapter_official/         # CLAdapter 공식 구현체
├── dataset/
│   ├── Dataset_0622/               # 전선 데이터셋
│   └── Dataset_0612_all/           # 종단접속새 황변 데이터셋
├── scripts/                        # 유틸 스크립트 (split 생성, 결과 요약 등)
├── docs/
├── train_kfold.sh                  # CLAdapter 10-fold 학습 진입점
├── collect_results.py              # 성능 집계
├── requirements.txt
└── .gitignore
```

## 설치

```bash
pip install -r requirements.txt
# PatchCore / DifferNet 추가 의존성은 baselines/normal_only/, baselines/differnet/ 의 requirements.txt 참조
```

---

## 1. Inference (새 데이터 테스트)

`--test-dir` 경로만 바꿔서 실행. 코드 수정 불필요.

체크포인트 폴더 구조:
```
# ViT-B + CLAdapter (vitb_cla_sft2 모델)
checkpoints/kfold10_vitb_cladapter_dataset_0622_{timestamp}/   ← --checkpoints-dir
└── fold_{N}/
    ├── vitb_cla_stage1/                    # Stage 1 (pretraining)
    └── vitb_cla_sft2/                      # Stage 2 SFT — 최종 모델
        ├── metrics.json
        └── vit_base_patch16_clip_224.laion2b_best.pth

# ConvNeXt-B + CLAdapter / linear (remaining 모델)
checkpoints/kfold10_remaining_dataset_0622_{timestamp}/        ← --checkpoints-dir
└── fold_{N}/
    ├── convnextb_cla_sft2/
    ├── linear_vitb/                        # ViT-B linear (vitb_linear 모델)
    └── linear_convnextb/                   # ConvNeXt-B linear (convnextb_linear 모델)
```

```bash
# ViT-B + CLAdapter — F1 최고 fold 자동 선택 (권장)
python inference/inference.py \
  --test-dir /path/to/new_images \
  --model vitb_cla_sft2 \
  --auto-best \
  --checkpoints-dir checkpoints/kfold10_vitb_cladapter_dataset_0622_20260624_092152

# fold 직접 지정
python inference/inference.py \
  --test-dir /path/to/new_images \
  --model vitb_cla_sft2 \
  --fold 3 \
  --checkpoints-dir checkpoints/kfold10_vitb_cladapter_dataset_0622_20260624_092152

# ConvNeXt-B + CLAdapter
python inference/inference.py \
  --test-dir /path/to/new_images \
  --model convnextb_cla_sft2 \
  --auto-best \
  --checkpoints-dir checkpoints/kfold10_remaining_dataset_0622_20260624_092147

# ViT-B linear
python inference/inference.py \
  --test-dir /path/to/new_images \
  --model vitb_linear \
  --auto-best \
  --checkpoints-dir checkpoints/kfold10_remaining_dataset_0622_20260624_092147
```

결과는 `predictions/predictions.csv`에 저장됨.

| 인자 | 설명 | 기본값 |
|---|---|---|
| `--test-dir` | 테스트 이미지 디렉토리 | **(필수)** |
| `--checkpoints-dir` | train_kfold.sh 체크포인트 루트 | **(필수)** |
| `--model` | `vitb_cla_sft2` / `convnextb_cla_sft2` / `vitb_linear` / `convnextb_linear` | `vitb_cla_sft2` |
| `--fold` | fold 번호 (0~9) 직접 지정 | — |
| `--auto-best` | metrics.json 기준 최고 fold 자동 선택 | — |
| `--metric` | auto-best 기준 (`f1` / `auroc` / `prec` / `rec`) | `f1` |
| `--output` | 결과 CSV 경로 | `predictions/predictions.csv` |
| `--batch-size` | 배치 크기 | `16` |
| `--image-size` | 입력 이미지 크기 | `224` |
| `--device` | `cuda` / `cpu` | 자동 감지 |

---



## 2. 성능 집계

```bash
python collect_results.py
```

---
## wire data(전선 데이터) 성능 (Dataset_0622, 10-Fold 평균 ± 표준편차)

| 모델 | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|
| PatchCore | 83.13 ± 8.06 | 93.97 ± 4.83 | 87.94 ± 5.29 | 93.69 ± 3.76 |
| DifferNet | 80.93 ± 6.43 | 89.71 ± 5.00 | 84.79 ± 2.89 | 93.19 ± 2.90 |
| ConvNeXt-B (linear) | 97.27 ± 2.02 | 97.23 ± 2.52 | 97.24 ± 2.19 | 99.74 ± 0.35 |
| ViT-B (linear) | 99.25 ± 0.94 | 99.36 ± 0.81 | 99.30 ± 0.85 | 100.00 ± 0.00 |
| **ConvNeXt-B + CLAdapter** | **99.86 ± 0.41** | **99.78 ± 0.65** | **99.82 ± 0.53** | **100.00 ± 0.00** |
| **ViT-B + CLAdapter** | **100.00 ± 0.00** | **100.00 ± 0.00** | **100.00 ± 0.00** | **100.00 ± 0.00** |
