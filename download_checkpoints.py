#!/usr/bin/env python3
"""
download_checkpoints.py — Google Drive에서 체크포인트·검증셋·학습데이터 자동 다운로드

사용:
    python download_checkpoints.py                      # 권장 체크포인트(loss2)
    python download_checkpoints.py --target val_infer   # 검증 추론셋(230)
    python download_checkpoints.py --target all         # 추론용 전부(loss2 + val_infer)
    python download_checkpoints.py --target train_data  # 학습데이터(train_data.tar · 1.3GB) → data_train/
"""
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile

try:
    import gdown
except ImportError:
    os.system(f"{sys.executable} -m pip install gdown -q")
    import gdown

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Google Drive ID ─────────────────────────────────────────────────────
RESOURCES = {
    # ID는 코드에 하드코딩하지 않고 환경변수로 주입 (run_all.sh 에서 export)
    "loss2": {
        "id":     os.environ.get("LOSS2_ID", ""),
        "type":   "zip",
        "kind":   "ckpt",
        "dest":   "checkpoints/oof_loss2_k5",
        "label":  "전선 가중치 모델 (oof_loss2_k5, 권장)",
    },
    "val_infer": {
        "id":     os.environ.get("VAL_INFER_ID", ""),
        "type":   "zip",
        "kind":   "data",
        "dest":   "val_infer",
        "label":  "검증 추론셋 val_infer (Anomaly/Normal · 230)",
    },
    "train_data": {
        "id":     os.environ.get("TRAIN_DATA_ID", ""),
        "type":   "tar",
        "kind":   "data",
        "dest":   "data_train",
        "label":  "학습데이터 train_data.tar (train/val/test · 정상/이상)",
    },
}

TARGET_GROUPS = {
    "loss2":     ["loss2"],
    "val_infer": ["val_infer"],
    "all":       ["loss2", "val_infer"],          # 추론용만 (학습데이터 1.3GB 는 별도 target)
    "train_data": ["train_data"],
}

# train_data.tar 내부 폴더 → manifest 경로(wire/normal, wire/anomaly) 매핑
TRAIN_LABEL_DIRS = {"정상": "normal", "이상": "anomaly"}


def download_zip(r: dict, dest: str) -> None:
    """gdown으로 zip 파일 다운로드 후 압축 해제 (대용량 안정 · 확인토큰 처리)"""
    tmp_zip = dest + "_tmp.zip"

    print(f"[INFO] 다운로드 중 (gdown)...")
    gdown.download(id=r["id"], output=tmp_zip, quiet=False)

    # 다운로드 무결성 검증 (zip 아니면 = Drive 공유설정/네트워크 문제)
    if not zipfile.is_zipfile(tmp_zip):
        if os.path.exists(tmp_zip):
            os.remove(tmp_zip)
        raise RuntimeError(
            f"[ERROR] 다운로드 파일이 정상 zip이 아닙니다 (id={r['id']}). "
            f"Drive 공유설정('링크 있는 모든 사용자') 또는 네트워크를 확인하세요."
        )

    print(f"[INFO] 압축 해제 중...")
    extract_dir = dest + "_extract"
    os.makedirs(extract_dir, exist_ok=True)
    subprocess.run(["unzip", "-q", tmp_zip, "-d", extract_dir], check=True)
    os.remove(tmp_zip)

    # dest로 병합 복사 (git에 있던 operating_rule.json 등 보존)
    entries = os.listdir(extract_dir)
    src_root = extract_dir
    if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
        src_root = os.path.join(extract_dir, entries[0])
    os.makedirs(dest, exist_ok=True)
    for root, dirs, files in os.walk(src_root):
        rel = os.path.relpath(root, src_root)
        out_dir = dest if rel == "." else os.path.join(dest, rel)
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(root, f), os.path.join(out_dir, f))
    shutil.rmtree(extract_dir, ignore_errors=True)


def extract_train_tar(tar_path: str, dest: str) -> None:
    """train_data.tar 를 manifest 구조로 배치.

    tar:  <top>/train/{정상,이상}/*.png, <top>/val/..., <top>/test/...
    dest: dest/wire/normal, dest/wire/anomaly   ← train (oof/reference_manifests 경로와 일치)
          dest/val/{정상,이상}, dest/test/{정상,이상} ← 원본 그대로 보존 (v2 학습에는 미사용)
    """
    extract_dir = dest + "_extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)
    print(f"[INFO] 압축 해제 중... (수 분 소요)")
    with tarfile.open(tar_path) as tf:
        tf.extractall(extract_dir)

    # 최상위 폴더(학습데이터_구성 등) 1개면 그 안을 원본으로
    src_root = extract_dir
    entries = os.listdir(extract_dir)
    if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
        src_root = os.path.join(extract_dir, entries[0])
    if not os.path.isdir(os.path.join(src_root, "train")):
        raise RuntimeError(f"[ERROR] tar 안에 train/ 폴더가 없습니다: {sorted(os.listdir(src_root))}")

    os.makedirs(os.path.join(dest, "wire"), exist_ok=True)
    for kr, en in TRAIN_LABEL_DIRS.items():
        src = os.path.join(src_root, "train", kr)
        dst = os.path.join(dest, "wire", en)
        if not os.path.isdir(src):
            raise RuntimeError(f"[ERROR] tar 안에 train/{kr} 폴더가 없습니다.")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.move(src, dst)
        print(f"[INFO] train/{kr} → {os.path.relpath(dst, SCRIPT_DIR)} ({len(os.listdir(dst))}장)")
    for split in ("val", "test"):
        src = os.path.join(src_root, split)
        if os.path.isdir(src):
            dst = os.path.join(dest, split)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.move(src, dst)
    shutil.rmtree(extract_dir, ignore_errors=True)


def download_tar(r: dict, dest: str) -> None:
    """gdown으로 tar 다운로드 후 manifest 구조로 배치"""
    tmp_tar = dest + "_tmp.tar"
    print(f"[INFO] 다운로드 중 (gdown)...")
    gdown.download(id=r["id"], output=tmp_tar, quiet=False)
    if not tarfile.is_tarfile(tmp_tar):
        if os.path.exists(tmp_tar):
            os.remove(tmp_tar)
        raise RuntimeError(
            f"[ERROR] 다운로드 파일이 정상 tar가 아닙니다 (id={r['id']}). "
            f"Drive 공유설정('링크 있는 모든 사용자') 또는 네트워크를 확인하세요."
        )
    extract_train_tar(tmp_tar, dest)
    os.remove(tmp_tar)


def download_folder(r: dict, dest: str) -> None:
    """Google Drive 폴더 다운로드 (dest에 병합 — 기존 파일 보존)"""
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

    # 다운로드 결과가 단일 하위폴더면 그 안을 원본으로
    entries = os.listdir(tmp_dir)
    src_root = tmp_dir
    if len(entries) == 1 and os.path.isdir(os.path.join(tmp_dir, entries[0])):
        src_root = os.path.join(tmp_dir, entries[0])

    # dest로 병합 복사 (git에 있던 operating_rule.json 등 보존)
    os.makedirs(dest, exist_ok=True)
    for root, dirs, files in os.walk(src_root):
        rel = os.path.relpath(root, src_root)
        out_dir = dest if rel == "." else os.path.join(dest, rel)
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(root, f), os.path.join(out_dir, f))
    shutil.rmtree(tmp_dir, ignore_errors=True)


def download_resource(key: str, force: bool = False) -> None:
    r = RESOURCES[key]
    dest = os.path.join(SCRIPT_DIR, r["dest"])

    # 이미 있으면 SKIP — 체크포인트는 .pth 존재로, 데이터는 폴더 비어있지 않음으로 판단
    if not force and os.path.isdir(dest):
        if r.get("kind") == "ckpt":
            present = any(f.endswith(".pth") for f in os.listdir(dest))
        else:
            present = len(os.listdir(dest)) > 0
        if present:
            print(f"[SKIP] {r['label']} 이미 존재합니다.")
            return

    if not r["id"]:
        print(f"[SKIP] {r['label']} — 다운로드 링크(ID) 미설정. run_all.sh 로 실행하거나 "
              f"환경변수(LOSS2_ID / VAL_INFER_ID / TRAIN_DATA_ID)를 설정하세요.")
        return

    print(f"[INFO] {r['label']} 다운로드 중...")
    parent_dir = os.path.dirname(dest)
    if parent_dir and (os.path.isfile(parent_dir) or os.path.islink(parent_dir)):
        os.remove(parent_dir)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if r["type"] == "zip":
        download_zip(r, dest)
    elif r["type"] == "tar":
        download_tar(r, dest)
    else:
        download_folder(r, dest)

    print(f"[DONE] {r['label']} → {r['dest']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="체크포인트·검증셋·학습데이터 다운로드")
    parser.add_argument(
        "--target",
        choices=["loss2", "val_infer", "all", "train_data"],
        default="loss2",
        help="다운로드 대상 (기본: loss2 = 권장 가중치 모델)",
    )
    parser.add_argument("--force", action="store_true", help="이미 있어도 재다운로드")
    args = parser.parse_args()

    for key in TARGET_GROUPS[args.target]:
        download_resource(key, force=args.force)

    print("\n[완료] 다운로드 완료.")


if __name__ == "__main__":
    main()
