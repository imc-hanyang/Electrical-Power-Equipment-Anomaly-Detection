"""
DATA_ROOT/DATASET_NAME/{normal,anomaly}(대소문자 무관)/*.jpg 디렉토리를 스캔하여
kfold split용 SPLIT_SOURCE_CSV(image_path,label,group,label_name) 생성.

GROUP_MODE=stem   : group = 파일명 stem 그대로 (crop 1개 = group 1개, grouping 없음)
GROUP_MODE=origin : group = 원본 사진 단위 (crop 접미사 제거, leakage 방지)

사용법:
  DATA_ROOT=/home/opgw/KEPCO_May/.../dataset \
  DATASET_NAME=Final_Dataset \
  GROUP_MODE=origin \
  OUTPUT_CSV=/path/to/dataset/splits/final_dataset_origin_split.csv \
  python 6_make_split_source_csv.py
"""

import os
import csv
import re

DATA_ROOT = os.environ.get("DATA_ROOT", "/home/opgw/KEPCO_May")
DATASET_NAME = os.environ.get("DATASET_NAME", "Dataset_0612_all")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "dataset_0612_all_split.csv")
GROUP_MODE = os.environ.get("GROUP_MODE", "stem")

# label, label_name (대소문자 무관 매칭)
LABEL_INFO = {
    "normal": (0, "normal"),
    "anomaly": (1, "anomaly"),
}

IMG_EXTS = (".jpg", ".jpeg", ".png")

# origin 모드: stem 끝의 crop 식별자 패턴 제거 -> 원본 사진 단위 그룹
# - Final_Dataset 계열: DSC_0003_crop1 -> DSC_0003
# - Dataset_0612 계열 : DSC_0003_1_AITC / 하동지사_북천선397R6R2_2_1_AITC -> 끝의 "_<번호>_<영문suffix>" 제거
ORIGIN_PATTERNS = [
    re.compile(r"_crop\d+$", re.IGNORECASE),
    re.compile(r"_\d+_[A-Za-z]+$"),
]

def make_group(stem: str) -> str:
    if GROUP_MODE == "stem":
        return stem
    g = stem
    for pat in ORIGIN_PATTERNS:
        new_g = pat.sub("", g)
        if new_g != g:
            g = new_g
            break
    return g


def main():
    rows = []
    base_dir = os.path.join(DATA_ROOT, DATASET_NAME)
    if not os.path.isdir(base_dir):
        raise SystemExit(f"[ERROR] not found: {base_dir}")

    # 실제 디스크상 폴더명을 대소문자 무관하게 찾기
    entries = {e.lower(): e for e in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, e))}

    for key, (label, label_name) in LABEL_INFO.items():
        cls_dir = entries.get(key)
        if cls_dir is None:
            print(f"[WARN] not found (case-insensitive '{key}'), skip under: {base_dir}")
            continue
        full_dir = os.path.join(base_dir, cls_dir)
        for fname in sorted(os.listdir(full_dir)):
            if not fname.lower().endswith(IMG_EXTS):
                continue
            stem = os.path.splitext(fname)[0]
            group = make_group(stem)
            image_path = f"{DATASET_NAME}/{cls_dir}/{fname}"
            rows.append({
                "image_path": image_path,
                "label": label,
                "group": group,
                "label_name": label_name,
            })

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "group", "label_name"])
        writer.writeheader()
        writer.writerows(rows)

    n_normal = sum(1 for r in rows if r["label"] == 0)
    n_anomaly = sum(1 for r in rows if r["label"] == 1)
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], 0)
        groups[r["group"]] += 1
    n_groups = len(groups)
    multi = sorted([g for g, c in groups.items() if c > 1])
    print("=" * 50)
    print(f"GROUP_MODE: {GROUP_MODE}")
    print(f"total: {len(rows)} (normal={n_normal}, anomaly={n_anomaly})")
    print(f"groups: {n_groups} (multi-member groups: {len(multi)})")
    if multi:
        print(f"  sample multi-member groups: {multi[:5]} ...")
    print(f"-> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
