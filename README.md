# KEPCO OPGW 이상 탐지

CLIP 기반 CLAdapter를 활용한 OPGW 전선 이상 탐지.  
ViT-B / ConvNeXt-B 백본에 CLAdapter 2-stage fine-tuning 적용.

## 성능 (Dataset_0622, 10-Fold 평균 ± 표준편차)

| 모델 | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|
| PatchCore | 83.13 ± 8.06 | 93.97 ± 4.83 | 87.94 ± 5.29 | 93.69 ± 3.76 |
| DifferNet | 80.93 ± 6.43 | 89.71 ± 5.00 | 84.79 ± 2.89 | 93.19 ± 2.90 |
| ConvNeXt-B (linear) | 97.27 ± 2.02 | 97.23 ± 2.52 | 97.24 ± 2.19 | 99.74 ± 0.35 |
| ViT-B (linear) | 99.25 ± 0.94 | 99.36 ± 0.81 | 99.30 ± 0.85 | 100.00 ± 0.00 |
| **ConvNeXt-B + CLAdapter** | **99.86 ± 0.41** | **99.78 ± 0.65** | **99.82 ± 0.53** | **100.00 ± 0.00** |
| **ViT-B + CLAdapter** | **100.00 ± 0.00** | **100.00 ± 0.00** | **100.00 ± 0.00** | **100.00 ± 0.00** |

## 프로젝트 구조

```
KEPCO-OPGW-Anomaly/
├── src/                            # CLAdapter 모델 코드 (학습 진입점: train.py)
├── inference/
│   └── inference.py                # 추론 스크립트
├── baselines/
│   ├── normal_only/                # PatchCore
│   ├── differnet/                  # DifferNet
│   ├── cladapter/                  # CLAdapter 선형 분류기 베이스라인
│   ├── supervised_classifier/      # EfficientNet 지도학습 베이스라인
│   └── CLAdapter_official/         # CLAdapter 공식 구현체
├── dataset/
│   ├── Dataset_0622/               # OPGW 데이터셋 (anomaly / normal)
│   └── Dataset_0612_all/           # 초기 수집 데이터셋
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

## 2. CLAdapter 학습 (train_kfold.sh)

```bash
# 전체 10-fold, ViT-B, GPU 0
bash train_kfold.sh

# ConvNeXt-B, GPU 4개 병렬
bash train_kfold.sh --model convnextb --gpu 0,1,2,3

# 특정 fold만
bash train_kfold.sh --fold 3 --model vitb

# 환경변수 오버라이드
EPOCHS=50 bash train_kfold.sh --model vitb --gpu 0,1
```

| 인자 | 설명 | 기본값 |
|---|---|---|
| `--model` | `vitb` / `convnextb` | `vitb` |
| `--fold` | 특정 fold 번호만 실행 | 전체 (0~9) |
| `--gpu` | GPU ID (쉼표 구분) | `0` |
| `--epochs` | 에폭 수 | `100` |
| `--split-dir` | fold CSV 경로 | `dataset/splits/kfold10_train_val_test_dataset_0622` |
| `--data-root` | 데이터셋 루트 | `dataset/` |

체크포인트 출력 위치:

```
checkpoints/{model}_kfold10_{timestamp}/
└── fold_{N}/
    ├── vitb_cla_stage1/
    │   └── vit_base_patch16_clip_224.laion2b_best.pth
    └── vitb_cla_sft2/
        └── vit_base_patch16_clip_224.laion2b_best.pth
```

### src/train.py 직접 실행 인자

| 인자 | 설명 | 필수 |
|---|---|---|
| `--model-mode` | `vit` / `conv` | ✅ |
| `--finetune-mode` | `cla` | ✅ |
| `--image-size` | 입력 이미지 크기 | ✅ |
| `--csv-dir` | fold CSV 경로 | ✅ |
| `--config-name` | `config_clip_vit` / `config_clip_convnext` | ✅ |
| `--data-root` | 데이터셋 루트 | ✅ |
| `--gpu_id` | GPU ID | ✅ |
| `--backbone-name` | timm 백본 이름 | — |
| `--backbone-out-dim` | 백본 출력 차원 | — |
| `--backbone-num-patch` | 패치 수 | — |
| `--finetune-ckpt` | Stage 1 체크포인트 경로 (Stage 2에서 사용) | — |
| `--norm` | 정규화 방식 (`clip` / `imagenet`) | — |
| `--epochs` | 에폭 수 | — |
| `--batch-size` | 배치 크기 | — |
| `--init-lr` | 학습률 | — |
| `--optimizer` | `AdamW` / `Adam` / `SGD` | — |
| `--selection-metric` | 모델 선택 기준 (`acc` / `f1` / `roc` / `loss`) | — |
| `--output-dir` | 결과 저장 경로 | — |

---

## 3. 베이스라인 학습

### CLAdapter 선형 분류기 (baselines/cladapter/)

```bash
python baselines/cladapter/train_normal_only.py \
  --data-root dataset/Dataset_0622 \
  --output-dir checkpoints/cladapter_normal_only
```

| 인자 | 설명 | 기본값 |
|---|---|---|
| `--data-root` | 데이터셋 경로 | **(필수)** |
| `--output-dir` | 결과 저장 경로 | — |
| `--backbone` | 백본 이름 | `convnext_base.fb_in22k_ft_in1k` |
| `--img-size` | 이미지 크기 | `224` |
| `--resize-mode` | `stretch` / `letterbox` | `letterbox` |
| `--epochs` | 에폭 수 | `20` |
| `--batch-size` | 배치 크기 | `8` |
| `--lr` | 학습률 | `1e-4` |
| `--centers` | 클러스터 중심 수 | `20` |
| `--adapter-depth` | 어댑터 레이어 수 | `1` |
| `--adapter-style` | `residual` / `official` | `residual` |

### EfficientNet 지도학습 (baselines/supervised_classifier/)

```bash
python baselines/supervised_classifier/train.py \
  --data-root dataset/Dataset_0622 \
  --output-dir checkpoints/supervised
```

| 인자 | 설명 | 기본값 |
|---|---|---|
| `--data-root` | 데이터셋 경로 | **(필수)** |
| `--model` | `resnet50` / `efficientnet_b0` / `convnext_tiny` | `resnet50` |
| `--img-size` | 이미지 크기 | `224` |
| `--epochs` | 에폭 수 | `60` |
| `--batch-size` | 배치 크기 | `16` |
| `--lr` | 헤드 학습률 | `3e-4` |
| `--backbone-lr` | 백본 학습률 | `3e-5` |
| `--n-splits` | K-fold 수 | `5` |
| `--test-fold` | 테스트 fold 번호 | `0` |

### PatchCore (baselines/normal_only/)

```bash
python -m baselines.normal_only.patchcore \
  --data-root dataset/Dataset_0622 \
  --split-csv dataset/splits/kfold10_train_val_test_dataset_0622/fold_0.csv \
  --output-dir checkpoints/patchcore_fold0
```

| 인자 | 설명 | 기본값 |
|---|---|---|
| `--data-root` | 데이터셋 경로 | **(필수)** |
| `--split-csv` | fold CSV 경로 | — |
| `--output-dir` | 결과 저장 경로 | — |
| `--backbone` | `resnet50` / `wide_resnet50_2` | `wide_resnet50_2` |
| `--img-size` | 이미지 크기 | `224` |
| `--resize-mode` | `stretch` / `letterbox` | `letterbox` |
| `--max-memory-patches` | 메모리 뱅크 최대 패치 수 | `10000` |
| `--score-top-k-frac` | 이상 점수 top-k 비율 | `0.01` |
| `--normal-percentile` | 정상 임계값 퍼센타일 | `95.0` |
| `--batch-size` | 배치 크기 | `8` |
| `--device` | `cuda` / `cpu` / `auto` | `auto` |

### DifferNet (baselines/attent_differnet/)

```bash
# AttentDifferNet (SE attention) — 권장
python -m baselines.attent_differnet.train \
  --data-root dataset/Dataset_0622 \
  --split-csv dataset/splits/kfold10_train_val_test_dataset_0622/fold_0.csv \
  --attention se \
  --output-dir checkpoints/differnet_fold0

# 기본 DifferNet (attention 없음)
python -m baselines.attent_differnet.train \
  --attention none \
  --output-dir checkpoints/differnet_baseline_fold0

# Predict
python -m baselines.attent_differnet.predict \
  --checkpoint checkpoints/differnet_fold0/se/best.pt \
  --input dataset/Dataset_0622/anomaly \
  --output-csv checkpoints/differnet_fold0/se/scores.csv
```

| 인자 | 설명 | 기본값 |
|---|---|---|
| `--data-root` | 데이터셋 경로 | **(필수)** |
| `--split-csv` | fold CSV 경로 | — |
| `--output-dir` | 결과 저장 경로 | — |
| `--attention` | `none` (DifferNet) / `se` / `cbam` | `se` |
| `--img-size` | 이미지 크기 | `448` |
| `--resize-mode` | `stretch` / `letterbox` | `stretch` |
| `--epochs` | 에폭 수 | `100` |
| `--batch-size` | 학습 배치 크기 | `8` |
| `--n-transforms` | 학습 시 augmentation 수 | `4` |
| `--lr` | 학습률 | `2e-4` |
| `--train-backbone` | 백본 가중치 학습 여부 | `False` |
| `--device` | `cuda` / `cpu` / `auto` | `auto` |

---

## 4. 성능 집계

```bash
python collect_results.py
```

---

## 5. 데이터 증강

```bash
python dataset/augment_anomaly.py \
  --src dataset/Dataset_0622 \
  --dst dataset/Dataset_0622_aug \
  --aug-only-dst dataset/Dataset_0622_aug_only
```
