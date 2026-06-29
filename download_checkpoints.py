#!/usr/bin/env python3
"""
download_checkpoints.py — Google Drive에서 데이터셋·체크포인트 자동 다운로드
"""
import argparse
import os
import shutil
import sys

try:
    import gdown
except ImportError:
    os.system(f"{sys.executable} -m pip install gdown -q")
    import gdown

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Google Drive 폴더 ID ────────────────────────────────────────────────
RESOURCES = {
    "dataset_wire": {
        "id":     "1N3rTC0d0MGG8bcE9wL_f4-VxR4MM9EHA",
        "dest":   "dataset/Dataset_0622",
        "check":  "dataset/Dataset_0622/anomaly",
        "label":  "전선 데이터셋 (Dataset_0622)",
    },
    "dataset_hwang": {
        "id":     "1-BshJFPN1rGL5m6IZWOcVIv_uHSxtyGb",
        "dest":   "dataset/Dataset_0612",
        "check":  "dataset/Dataset_0612/anomaly",
        "label":  "황변 데이터셋 (Dataset_0612)",
    },
    "wire": {
        "id":     "1O1Ar2pU-PNOmDPRFU4fQLXSRyGwIM-tf",
        "dest":   "checkpoints/wire_final_train/fold_9",
        "check":  "checkpoints/wire_final_train/fold_9",
        "label":  "전선 체크포인트 (fold_9)",
    },
    "hwang": {
        "id":     "1uKMpge1NRKV3J0hb5gGbKNqFDIUU2LSA",
        "dest":   "checkpoints/hwang_group_train/fold_9",
        "check":  "checkpoints/hwang_group_train/fold_9",
        "label":  "황변 체크포인트 (fold_9)",
    },
}

TARGET_GROUPS = {
    "wire":  ["dataset_wire",  "wire"],
    "hwang": ["dataset_hwang", "hwang"],
    "all":   ["dataset_wire", "dataset_hwang", "wire", "hwang"],
}


def download_resource(key: str, force: bool = False) -> None:
    r = RESOURCES[key]
    dest     = os.path.join(SCRIPT_DIR, r["dest"])
    check    = os.path.join(SCRIPT_DIR, r["check"])

    if not force and os.path.isdir(check) and os.listdir(check):
        print(f"[SKIP] {r['label']} 이미 존재합니다.")
        return

    print(f"[INFO] {r['label']} 다운로드 중...")

    # 깨진 symlink나 파일이 폴더 위치에 있으면 제거
    parent_dir = os.path.dirname(dest)
    if os.path.isfile(parent_dir) or os.path.islink(parent_dir):
        os.remove(parent_dir)
    os.makedirs(parent_dir, exist_ok=True)

    tmp_dir = dest + "_tmp"
    if os.path.isfile(tmp_dir) or os.path.islink(tmp_dir):
        os.remove(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    gdown.download_folder(
        url=f"https://drive.google.com/drive/folders/{r['id']}",
        output=tmp_dir,
        quiet=False,
        use_cookies=False,
    )

    # Google Drive 폴더 이름으로 하위 폴더가 생기는 경우 flatten
    sub_dirs = [d for d in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, d))]
    if len(sub_dirs) == 1:
        inner = os.path.join(tmp_dir, sub_dirs[0])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.move(inner, dest)
        shutil.rmtree(tmp_dir)
    else:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.move(tmp_dir, dest)

    print(f"[DONE] {r['label']} → {r['dest']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="데이터셋·체크포인트 다운로드")
    parser.add_argument(
        "--target",
        choices=["wire", "hwang", "all"],
        default="all",
        help="다운로드 대상 (기본: all)",
    )
    parser.add_argument("--force", action="store_true", help="이미 있어도 재다운로드")
    args = parser.parse_args()

    for key in TARGET_GROUPS[args.target]:
        download_resource(key, force=args.force)

    print("\n[완료] 모든 리소스 준비 완료.")


if __name__ == "__main__":
    main()
