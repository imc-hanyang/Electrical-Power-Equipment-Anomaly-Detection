# ViT-B + CLAdapter 개념 정리

이 문서는 KEPCO normal/anomaly crop 데이터셋에서 사용한 `ViT-B + CLAdapter SFT` 모델의 개념, 학습 방식, 평가 방식, 재현 방법을 정리한 문서다.

## 한 줄 요약

`ViT-B + CLAdapter`는 ImageNet/CLIP 계열 사전학습 ViT-B가 이미 가진 시각 표현력을 유지하면서, KEPCO 전력설비 crop 이미지의 정상/이상 분류에 맞게 patch token feature를 adapter로 보정하는 supervised classification 모델이다.

```text
input image
-> ViT-B patch embedding + transformer backbone
-> patch token features
-> CLAdapter feature refinement
-> average pooling
-> classifier head
-> normal/anomaly probability
```

## 문제 설정

우리 데이터의 목표는 crop 이미지 한 장을 보고 `normal` 또는 `anomaly`를 판별하는 것이다.

초기에는 DifferNet, PatchCore처럼 정상 이미지만 학습하는 normal-only anomaly detection도 실험했다. 하지만 현재 KEPCO crop 데이터에서는 정상 이미지 내부의 촬영 조건, 배경, 부품 상태, scale 변화가 커서 정상 분포만으로 이상을 안정적으로 분리하기 어려웠다.

그래서 최종 성능 확보 방향은 다음처럼 정리했다.

```text
normal-only anomaly detection
-> 성능 한계 확인

normal + anomaly supervised classification
-> ViT-B / ConvNeXt / ResNet / CLAdapter 비교
-> ViT-B + CLAdapter를 핵심 후보로 정리
```

## ViT-B Backbone

ViT-B는 이미지를 patch 단위로 나눠서 token sequence로 처리한다. 우리 구현에서는 `224 x 224` 입력을 사용하므로, patch token 수는 `196`개다.

```text
image: [B, 3, 224, 224]
patch tokens: [B, 196, 768]
```

ViT의 장점은 이미지 전체를 patch token들의 관계로 볼 수 있다는 점이다. 전력설비 이미지처럼 결함 부위가 작거나 주변 구조물과 섞여 있는 경우, CNN feature map만 보는 방식보다 patch 단위 feature를 adapter로 다루는 방식이 유리할 수 있다.

## CLAdapter가 붙는 위치

우리 코드에서 CLAdapter는 ViT-B backbone 뒤에 붙는다.

구현 위치:

```text
engine/cladapter_code/build_model.py
engine/cladapter_code/models/modules.py
```

실제 forward 흐름은 다음과 같다.

```python
x = self.backbone.forward_features(x)
x = self.post(x[:, 1:, :]).mean(1)
x = self.head(x)
```

즉, CLS token을 직접 쓰지 않고, CLS를 제외한 patch token `x[:, 1:, :]`에 CLAdapter를 적용한 뒤 평균 pooling을 한다.

```text
ViT output tokens
  ├── CLS token
  └── patch tokens [B, 196, 768]
              |
              v
        CLAdapter
              |
              v
        mean pooling
              |
              v
        linear classifier
```

## CLAdapter 내부 개념

CLAdapter는 patch token feature를 그대로 classifier에 넣지 않고, 데이터셋에 맞게 feature를 보정하는 작은 적응 모듈이다.

우리 구현의 CLAdapter는 여러 개의 `FR_Resblock`으로 구성된다.

```text
CLAdapter
└── FR_Resblock x N
    ├── LayerNorm
    ├── Cluster_Attention
    ├── LayerNorm
    └── MLP residual refinement
```

핵심은 `Cluster_Attention`이다.

1. patch token들을 평균내서 이미지 전체 요약 벡터를 만든다.
2. 이 요약 벡터와 learnable cluster center들의 유사도를 계산한다.
3. cluster attention weight를 만든다.
4. 이 weight로 sample-specific channel transform을 구성한다.
5. 각 이미지의 patch token feature를 해당 이미지에 맞게 변환한다.

개념적으로는 다음과 같다.

```text
patch tokens [B, N, C]
-> image summary q = mean(tokens)
-> q와 cluster centers 비교
-> cluster attention weight
-> sample-specific feature transform
-> refined patch tokens
```

따라서 CLAdapter는 단순히 classifier head만 추가하는 것이 아니라, 사전학습 backbone이 만든 feature를 KEPCO 도메인에 맞게 한 번 더 정렬하는 역할을 한다.

## 왜 Adapter가 필요한가

데이터가 많지 않은 도메인 특화 task에서 backbone 전체를 처음부터 강하게 fine-tuning하면 과적합 위험이 있다. 반대로 backbone을 완전히 얼리고 linear classifier만 학습하면 도메인 적응력이 부족할 수 있다.

CLAdapter는 이 둘의 중간 지점이다.

```text
linear probing
  - backbone 표현을 거의 그대로 사용
  - 안정적이지만 도메인 적응력이 약할 수 있음

full fine-tuning
  - 전체 backbone을 바꿈
  - 표현력은 크지만 작은 데이터에서 과적합 위험

CLAdapter
  - pretrained feature를 보존하면서 adapter로 feature를 보정
  - 적은 데이터에서 domain-specific adaptation을 노림
```

우리 실험에서도 plain ViT-B보다 `ViT-B + CLAdapter`가 훨씬 높은 성능을 보였다.

## Stage 1 / Stage 2 SFT 학습

우리 학습은 두 단계로 구성했다.

### Stage 1

Stage 1에서는 pretrained ViT-B backbone을 freeze하고, CLAdapter와 classifier head를 학습한다.

```text
ViT-B backbone: freeze
CLAdapter: train
classifier head: train
```

목적은 사전학습 feature를 망가뜨리지 않고, KEPCO normal/anomaly task에 필요한 adapter/head를 먼저 안정적으로 맞추는 것이다.

코드상으로는 `finetune_ckpt`가 없을 때 backbone이 freeze된다.

```python
if self.f_mode != 'full' and config.MODEL.finetune is None:
    for param in self.backbone.parameters():
        param.requires_grad = False
```

### Stage 2 SFT

Stage 2에서는 Stage 1 checkpoint를 로드한 뒤 backbone까지 unfreeze해서 전체 모델을 fine-tuning한다.

```text
ViT-B backbone: train
CLAdapter: train
classifier head: train
```

Stage 2의 목적은 Stage 1에서 얻은 안정적인 adapter/head를 출발점으로 삼아, backbone feature까지 KEPCO 데이터에 조금 더 맞추는 것이다.

코드상으로는 `--finetune-ckpt`를 넘기면 `config.MODEL.finetune`이 설정되고, backbone freeze 조건을 타지 않는다. 또한 learning rate를 낮춰서 SFT를 수행한다.

```python
if args.finetune_ckpt is not None:
    config.MODEL.finetune = args.finetune_ckpt

if config.MODEL.finetune is not None:
    args.init_lr /= 10
```

## 학습 데이터와 평가 방식

중요한 점은 현재 최고 후보인 `ViT-B + CLAdapter`는 normal-only 모델이 아니라 supervised 모델이라는 것이다.

```text
train: normal + anomaly 사용
valid: threshold search에 사용
test: 최종 평가에 사용
```

데이터 split은 crop 단위 누수를 막기 위해 원본 이미지 group 단위로 나눴다. 같은 원본 이미지에서 나온 crop들이 train/test에 동시에 들어가지 않도록 구성했다.

10-fold 기준 split 위치:

```text
dataset/splits/kfold10_train_val_test_second_setting/
├── fold_0.csv
├── fold_1.csv
...
└── fold_9.csv
```

## Threshold 기준

모델 자체는 `normal/anomaly` 확률을 출력한다.

```text
prob_normal
prob_anomaly
```

일반적인 fixed 기준은 다음과 같다.

```text
prob_anomaly >= 0.5 -> anomaly
prob_anomaly < 0.5  -> normal
```

하지만 보고용 주요 성능은 validation set에서 threshold를 탐색한 뒤, 그 threshold를 test set에 고정 적용한 결과다.

```text
1. validation set에서 accuracy가 가장 좋은 threshold 선택
2. 선택한 threshold를 test set에 그대로 적용
3. test accuracy / f1 계산
```

이 방식은 test label로 threshold를 고르는 test leakage를 피하면서, 모델별 score calibration 차이를 완화한다.

## 10-Fold 핵심 결과

`Final_Dataset`에서 `etc`를 제외하고 10-fold train/validation/test 평가를 수행했다.

| Model | Test Acc. @ Val Threshold | Test F1 @ Val Threshold | Fixed 0.5 Acc. | AUROC | AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| ViT-B + CLAdapter | 89.91% ± 5.58 | 90.39% ± 5.23 | 89.91% ± 5.26 | 0.9778 ± 0.0159 | 0.9759 ± 0.0201 |

해석은 다음과 같다.

- Accuracy/F1은 validation threshold가 test 분포에 얼마나 잘 맞는지의 영향을 받는다.
- AUROC/AP는 threshold-free metric이므로 모델의 ranking 능력을 더 잘 보여준다.
- 10-fold에서 `ViT-B + CLAdapter`는 AUROC/AP가 높아 normal/anomaly score ranking 자체는 강하게 형성된 것으로 볼 수 있다.

## 기존 모델과의 개념적 차이

| 모델 | 학습 방식 | 특징 |
| --- | --- | --- |
| DifferNet | normal-only | 정상 분포 likelihood 기반. score가 확률이 아니므로 fixed 0.5 없음 |
| PatchCore | normal-only | 정상 patch memory bank와 거리 기반 anomaly score |
| ResNet50 | supervised | CNN backbone full fine-tuning |
| ViT-B | supervised | ViT backbone feature + classifier |
| ViT-B + CLAdapter | supervised | ViT patch token에 CLAdapter feature refinement 추가 |

우리 목표가 최종 성능 확보라면 `ViT-B + CLAdapter`가 가장 중요한 후보이고, normal-only 모델들은 비교군으로 두는 것이 맞다.

## 재현 명령

아래 명령은 기존 10-fold split을 그대로 사용해서 `ViT-B + CLAdapter`를 다시 학습하고 평가한다.

```bash
cd "/path/to/ViT-B + CLAdapter SFT_new"

N_SPLITS=10 \
BUILD_SPLITS=0 \
SPLIT_DIR="$PWD/dataset/splits/kfold10_train_val_test_second_setting" \
RUN_ROOT="$PWD/engine/runs/kfold10_vitb_cladapter_reproduce_$(date +%Y%m%d)" \
GPU_IDS="0,1,2,3" \
EPOCHS=100 \
bash "engine/scripts/run_vitb_cladapter_kfold_train_val_test.sh"
```

이 스크립트는 각 fold마다 다음을 수행한다.

```text
fold_i/
├── vitb_cla_stage1/
│   ├── vit_base_patch16_clip_224.laion2b_best.pth
│   └── metrics.json
└── vitb_cla_sft2/
    ├── vit_base_patch16_clip_224.laion2b_best.pth
    └── metrics.json
```

학습이 끝나면 validation threshold 평가 결과가 생성된다.

```text
VALIDATION_THRESHOLD.md
validation_threshold_metrics.csv
validation_threshold_summary.csv
fold_*/validation_threshold_predictions/vit_b_plus_cladapter_test.csv
```

## 기존 checkpoint로 다시 평가

기존 학습 결과를 다시 평가하려면 다음 명령을 사용한다.

```bash
cd "/path/to/ViT-B + CLAdapter SFT_new"

python "engine/scripts/apply_validation_thresholds.py" \
  --run-root "engine/runs/kfold10_vitb_cladapter_second_setting" \
  --split-dir "dataset/splits/kfold10_train_val_test_second_setting" \
  --data-root "dataset" \
  --models "ViT-B + CLAdapter" \
  --title "KEPCO 10-Fold ViT-B + CLAdapter Re-evaluation"
```

## 특정 fold 추론

예를 들어 `fold_3` test set만 추론하려면 다음과 같이 실행한다.

```bash
cd "/path/to/ViT-B + CLAdapter SFT_new"

python "engine/scripts/infer_vitb_cladapter_sft.py" \
  --checkpoint "engine/runs/kfold10_vitb_cladapter_second_setting/fold_3/vitb_cla_sft2/vit_base_patch16_clip_224.laion2b_best.pth" \
  --csv "dataset/splits/kfold10_train_val_test_second_setting/fold_3.csv" \
  --split test \
  --data-root "dataset" \
  --output "engine/predictions/fold3_test_predictions.csv"
```

단, 이 단일 추론 스크립트는 기본적으로 `argmax / fixed 0.5` 기준 예측을 출력한다. 보고서의 `Test Acc. @ Val Threshold`를 재현하려면 `apply_validation_thresholds.py`를 사용해야 한다.

## 발표용 설명 문장

발표에서는 다음처럼 설명할 수 있다.

> 본 실험에서는 적은 수의 도메인 특화 전력설비 이미지에서 normal/anomaly 분류 성능을 높이기 위해, 사전학습 ViT-B backbone에 CLAdapter를 결합하였다. ViT-B가 이미지의 patch token 표현을 추출하면, CLAdapter가 cluster-based feature refinement를 통해 KEPCO 데이터셋에 맞는 token representation으로 보정하고, 이를 평균 pooling한 뒤 binary classifier로 정상/이상을 판별한다. 학습은 Stage 1에서 backbone을 freeze하고 adapter/head를 먼저 학습한 뒤, Stage 2에서 Stage 1 checkpoint를 기반으로 전체 모델을 fine-tuning하는 방식으로 진행하였다. 최종 평가는 group 기반 10-fold train/validation/test split에서 validation threshold를 test에 고정 적용하는 방식으로 수행하였다.
