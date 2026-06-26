#!/usr/bin/env python3
"""
새 데이터 추가 시 crop + 시각화 + 통계 출력 스크립트

Usage:
  # 기본: src 폴더 내 JSON 탐색 → crop/visualized/summary 생성
  python scripts/process_new_data.py --src /path/to/new_raw_data

  # 출력 경로 지정
  python scripts/process_new_data.py --src /path/to/raw --dst /path/to/output

Output (--dst 기준):
  dst/
  ├── crop/
  │   ├── df_damage/
  │   ├── df_swelling/
  │   ├── df_tracking/
  │   └── df_whitening/
  ├── visualized/        # bbox 시각화 이미지
  └── dataset_summary.txt
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


LABEL_COLORS = {
    "df_damage":    (255,  60,  60),
    "df_swelling":  ( 60, 160, 255),
    "df_tracking":  ( 60, 200,  60),
    "df_whitening": (255, 200,   0),
}
DEFAULT_COLOR = (200, 200, 200)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def find_image(json_path: Path, img_name: str) -> Path | None:
    """JSON과 같은 디렉토리 또는 부모에서 이미지 파일 탐색."""
    candidates = [
        json_path.parent / img_name,
        json_path.parent / Path(img_name).name,
    ]
    for c in candidates:
        if c.exists():
            return c
    # 확장자 다를 수 있어서 stem 기준 재탐색
    stem = Path(img_name).stem
    for ext in IMG_EXTS:
        p = json_path.parent / (stem + ext)
        if p.exists():
            return p
    return None


def points_to_bbox(points: list) -> tuple[int, int, int, int]:
    """points 리스트 → (x1, y1, x2, y2)."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def process(src: Path, dst: Path) -> dict[str, list[str]]:
    """JSON 순회 → crop + visualize. 카테고리별 저장 경로 반환."""
    crop_dir = dst / "crop"
    vis_dir  = dst / "visualized"

    results: dict[str, list[str]] = defaultdict(list)  # label → [crop_path, ...]
    skipped = []

    json_files = sorted(src.rglob("*.json"))
    print(f"JSON {len(json_files)}개 발견\n")

    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [SKIP] {jf.name}: JSON 파싱 오류 ({e})")
            continue

        img_name = data.get("imagePath", "")
        shapes   = data.get("shapes", [])
        if not img_name or not shapes:
            continue

        img_path = find_image(jf, img_name)
        if img_path is None:
            print(f"  [SKIP] 이미지 없음: {img_name}")
            skipped.append(img_name)
            continue

        img = Image.open(img_path).convert("RGB")
        vis_img = img.copy()
        draw = ImageDraw.Draw(vis_img)

        # 시각화 저장 경로 (src 하위 구조 유지)
        rel = jf.parent.relative_to(src) if jf.parent != src else Path(".")
        vis_out_dir = vis_dir / rel
        vis_out_dir.mkdir(parents=True, exist_ok=True)

        stem = img_path.stem
        for idx, shape in enumerate(shapes):
            if shape.get("shape_type") != "rectangle":
                continue
            label  = shape.get("label", "unknown")
            points = shape.get("points", [])
            if len(points) < 2:
                continue

            x1, y1, x2, y2 = points_to_bbox(points)
            color = LABEL_COLORS.get(label, DEFAULT_COLOR)

            # crop 저장
            out_dir = crop_dir / label
            out_dir.mkdir(parents=True, exist_ok=True)
            crop = img.crop((x1, y1, x2, y2))
            crop_name = f"{stem}_{idx}.jpg"
            crop.save(out_dir / crop_name, quality=95)
            results[label].append(str(out_dir / crop_name))

            # bbox 시각화
            draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
            draw.text((x1 + 4, y1 + 4), label, fill=color)

        vis_img.save(vis_out_dir / img_path.name, quality=90)

    return results, skipped


def print_summary(results: dict, normal_dir: Path | None = None) -> str:
    """[데이터 구성] 출력 및 문자열 반환."""
    lines = ["[데이터 구성]"]

    # 정상 이미지 수 (normal/ 폴더가 있으면 참조)
    if normal_dir and normal_dir.exists():
        normal_count = sum(1 for f in normal_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        lines.append(f"*정상(normal) 데이터 총 {normal_count}개")

    # 이상 이미지 수
    total_anomaly = sum(len(v) for v in results.values())
    lines.append(f"*이상(anomaly) 데이터")
    label_order = ["df_tracking", "df_damage", "df_swelling", "df_whitening"]
    seen = set()
    for label in label_order:
        if label in results:
            short = label.replace("df_", "")
            lines.append(f"  * {short} : {len(results[label])}개")
            seen.add(label)
    for label in sorted(results):
        if label not in seen:
            lines.append(f"  * {label} : {len(results[label])}개")
    lines.append(f"  총 {total_anomaly}개")

    summary = "\n".join(lines)
    print("\n" + summary)
    return summary


def main():
    p = argparse.ArgumentParser(description="새 데이터 crop + 시각화 + 통계")
    p.add_argument("--src",    type=Path, required=True,  help="원본 이미지 + JSON 폴더")
    p.add_argument("--dst",    type=Path, default=None,   help="출력 폴더 (기본: src 상위)")
    p.add_argument("--normal", type=Path, default=None,   help="정상 이미지 폴더 (통계용)")
    args = p.parse_args()

    src = args.src.resolve()
    dst = (args.dst or src.parent).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    print(f"src : {src}")
    print(f"dst : {dst}")

    results, skipped = process(src, dst)

    if skipped:
        print(f"\n[경고] 이미지 미발견 {len(skipped)}개: {skipped[:5]}")

    summary = print_summary(results, args.normal)

    # summary 저장
    summary_path = dst / "dataset_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"\n통계 저장: {summary_path}")
    print(f"crop    : {dst / 'crop'}")
    print(f"시각화  : {dst / 'visualized'}")


if __name__ == "__main__":
    main()
