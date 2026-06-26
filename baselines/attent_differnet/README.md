# AttentDifferNet for KEPCO_May

This folder implements a DifferNet baseline and an AttentDifferNet-style variant for
`/home/opgw/KEPCO_May/Final_Dataset`.

## Environment

```bash
cd /home/opgw/KEPCO_May
source kepco/bin/activate
cd engine
```

The `kepco` virtual environment was created with `--system-site-packages` so it can reuse
the installed CUDA-enabled PyTorch build.

## Dataset Policy

The training code uses:

- `Final_Dataset/normal` as label `0`
- `Final_Dataset/anomaly` as label `1`
- ignores `etc` and `error`

DifferNet is trained only on normal images. To reduce leakage, training normal images are
selected only from original-image groups that contain normal samples only. Test images are
all remaining normal/anomaly images whose original-image group is not in training.

## Model

`--attention none` reproduces the DifferNet-style baseline:

- AlexNet feature extractor
- multi-scale feature averaging
- fully connected normalizing flow

`--attention se` and `--attention cbam` insert attention blocks inside `self.features`
after the three AlexNet pooling blocks:

- AB1 after the first pooling block, 64 channels
- AB2 after the second pooling block, 192 channels
- AB3 after the final pooling block, 256 channels

Convolution weights are ImageNet-initialized by default and frozen. The normalizing flow
and attention blocks are trained. Use `--train-backbone` to unfreeze the full backbone.

## Train

Recommended first run:

```bash
python -m attent_differnet.train \
  --attention se \
  --epochs 100 \
  --sub-epochs 1 \
  --batch-size 8 \
  --batch-size-test 4 \
  --n-transforms 4 \
  --n-transforms-test 16 \
  --output-dir /home/opgw/KEPCO_May/engine/runs/attent_differnet
```

Use aspect-ratio-preserving letterbox resize instead of square stretching:

```bash
python -m attent_differnet.train \
  --attention none \
  --resize-mode letterbox \
  --epochs 100 \
  --output-dir /home/opgw/KEPCO_May/engine/runs/a_letterbox_20260528
```

Baseline DifferNet:

```bash
python -m attent_differnet.train --attention none
```

CBAM variant:

```bash
python -m attent_differnet.train --attention cbam
```

Outputs are written under:

```text
engine/runs/attent_differnet/<attention>/
├── best.pt
├── last.pt
├── history.json
├── metadata.json
├── scores_best.json
└── split.csv
```

## Predict

```bash
python -m attent_differnet.predict \
  --checkpoint /home/opgw/KEPCO_May/engine/runs/attent_differnet/se/best.pt \
  --input /home/opgw/KEPCO_May/Final_Dataset/anomaly \
  --output-csv /home/opgw/KEPCO_May/engine/runs/attent_differnet/se/anomaly_scores.csv
```

If the checkpoint has an evaluation threshold, predictions are emitted as `normal` or
`anomaly`; otherwise only anomaly scores are printed.

## Smoke Test

The implementation was smoke-tested with one epoch and no ImageNet download:

```bash
python -m attent_differnet.train \
  --attention se \
  --no-pretrained \
  --epochs 1 \
  --sub-epochs 1 \
  --batch-size 4 \
  --batch-size-test 4 \
  --n-transforms 1 \
  --n-transforms-test 1 \
  --num-workers 0 \
  --hide-progress \
  --output-dir /home/opgw/KEPCO_May/engine/runs/smoke
```

## Sanity Check

Use this to verify that `--attention none` matches `torchvision` AlexNet features,
attention blocks are inserted at the expected positions, and optimizer parameters include
the normalizing flow plus attention blocks.

```bash
python -m attent_differnet.sanity_check
```
