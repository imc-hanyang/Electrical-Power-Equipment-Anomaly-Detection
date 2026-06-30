#!/usr/bin/env python3
"""
download_checkpoints.py — Google Drive에서 데이터셋·체크포인트 자동 다운로드
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile

try:
    import gdown
except ImportError:
    os.system(f"{sys.executable} -m pip install gdown -q")
    import gdown

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Google Drive ID ─────────────────────────────────────────────────────
RESOURCES = {
    "dataset_wire": {
        "id":     "1OvtsoNC7bwnW8t2-3i-UigEWuLhsim-a",
        "type":   "zip",
        "dest":   "dataset/Dataset_0622",
        "check":  "dataset/Dataset_0622/anomaly",
        "label":  "전선 데이터셋 (Dataset_0622)",
    },
    "dataset_hwang": {
        "id":     "1PRbTHcUCd4UcxdnSlBYtv9PBgqqEUNa6",
        "type":   "zip",
        "dest":   "dataset/Dataset_0612",
        "check":  "dataset/Dataset_0612/anomaly",
        "label":  "황변 데이터셋 (Dataset_0612)",
    },
    "wire": {
        "id":     "1w16hdFJiVKGGhANeBhI1xrsqKewPgIGo",
        "type":   "zip",
        "dest":   "checkpoints/wire_final_train/fold_9",
        "check":  "checkpoints/wire_final_train/fold_9",
        "label":  "전선 체크포인트 (fold_9)",
    },
    "hwang": {
        "id":     "1YQSUpS0BiWxRTK689YgNP_NlyKn1n-7a",
        "type":   "zip",
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


def download_zip(r: dict, dest: str) -> None:
    """curl로 zip 다운로드 후 압축 해제"""
    tmp_zip = dest + "_tmp.zip"
    url = f"https://drive.usercontent.google.com/download?id={r['id']}&export=download&confirm=t"

    print(f"[INFO] curl 다운로드 중...")
    ret = subprocess.run(["curl", "-L", url, "-o", tmp_zip], check=True)

    print(f"[INFO] 압축 해제 중...")
    extract_dir = dest + "_extract"
    os.makedirs(extract_dir, exist_ok=True)
    subprocess.run(["unzip", "-q", tmp_zip, "-d", extract_dir], check=True)
    os.remove(tmp_zip)

    # 압축 해제 결과가 단일 폴더면 flatten
    entries = os.listdir(extract_dir)
    if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
        inner = os.path.join(extract_dir, entries[0])
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.move(inner, dest)
        shutil.rmtree(extract_dir)
    else:
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.move(extract_dir, dest)


def download_folder(r: dict, dest: str) -> None:
    """Google Drive 폴더 다운로드"""
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

    sub_dirs = [d for d in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, d))]
    if len(sub_dirs) == 1:
        inner = os.path.join(tmp_dir, sub_dirs[0])
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.move(inner, dest)
        shutil.rmtree(tmp_dir)
    else:
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.move(tmp_dir, dest)


def download_resource(key: str, force: bool = False) -> None:
    r = RESOURCES[key]
    dest  = os.path.join(SCRIPT_DIR, r["dest"])
    check = os.path.join(SCRIPT_DIR, r["check"])

    if not force and os.path.isdir(check) and os.listdir(check):
        print(f"[SKIP] {r['label']} 이미 존재합니다.")
        return

    print(f"[INFO] {r['label']} 다운로드 중...")

    parent_dir = os.path.dirname(dest)
    if os.path.isfile(parent_dir) or os.path.islink(parent_dir):
        os.remove(parent_dir)
    os.makedirs(parent_dir, exist_ok=True)

    try:
        if r["type"] == "zip":
            download_zip(r, dest)
        else:
            download_folder(r, dest)
    finally:
        # 에러 발생 여부와 무관하게 _tmp 잔재 정리
        tmp_path = dest + "_tmp"
        if os.path.exists(tmp_path) and not os.path.exists(dest):
            shutil.move(tmp_path, dest)
            print(f"[INFO] {tmp_path} → {dest} rename 완료")

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
