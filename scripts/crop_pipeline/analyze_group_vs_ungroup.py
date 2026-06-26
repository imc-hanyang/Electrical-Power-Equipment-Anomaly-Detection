"""
experiment_registry.csv(run_full_experiment.sh가 누적 기록)를 읽어서,
각 (dataset, group_mode, model)별 prediction csv를 찾아 Precision/Recall/F1/AUROC를
집계하고 group vs ungroup 비교표를 출력/저장.

prediction csv 탐색 규칙 (실제 출력 포맷에 맞게 COLUMN_CANDIDATES만 조정하면 됨):
  - RUN_ROOT_MAIN, RUN_ROOT_BASE 하위를 재귀 탐색
  - 파일명에 "predict" 또는 "test" 포함 + 확장자 .csv
  - 컬럼: label(정답), pred 계열(예측, threshold 적용), score 계열(anomaly score, AUROC용)

사용법:
  python analyze_group_vs_ungroup.py
  (ROOT_DIR/engine/runs/experiment_registry.csv 자동 탐색)

출력:
  ROOT_DIR/engine/runs/group_vs_ungroup_summary.csv
  + stdout에 markdown 표
"""

import os
import csv
import glob

try:
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REGISTRY = os.path.join(ROOT_DIR, "engine", "runs", "experiment_registry.csv")
OUTPUT_CSV = os.path.join(ROOT_DIR, "engine", "runs", "group_vs_ungroup_summary.csv")

LABEL_CANDIDATES = ["label", "y_true", "gt", "ground_truth"]
PRED_CANDIDATES = ["pred_at_val_threshold", "pred", "y_pred", "prediction"]
SCORE_CANDIDATES = ["score", "anomaly_score", "prob", "y_score"]


def find_pred_csvs(run_root):
    if not run_root or not os.path.isdir(run_root):
        return []
    pats = ["**/*predict*.csv", "**/*test*.csv", "**/*pred*.csv"]
    files = set()
    for p in pats:
        for f in glob.glob(os.path.join(run_root, p), recursive=True):
            files.add(f)
    return sorted(files)


def guess_col(header, candidates):
    lower = {h.lower(): h for h in header}
    for c in candidates:
        if c in lower:
            return lower[c]
    return None


def infer_model_name(run_root, csv_path):
    rel = os.path.relpath(csv_path, run_root)
    parts = rel.split(os.sep)
    # 보통 RUN_ROOT/<model_name>/foldN/... 구조 가정. model_name 추정 실패 시 상위 폴더명 사용.
    for p in parts:
        for key in ["resnet50", "convnext", "vit", "cladapter", "patchcore", "differnet", "efficientnet"]:
            if key in p.lower():
                return p
    return parts[0] if parts else "unknown"


def compute_metrics(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        label_col = guess_col(header, LABEL_CANDIDATES)
        pred_col = guess_col(header, PRED_CANDIDATES)
        score_col = guess_col(header, SCORE_CANDIDATES)
        if label_col is None or pred_col is None:
            return None
        y_true, y_pred, y_score = [], [], []
        for row in reader:
            try:
                y_true.append(int(float(row[label_col])))
                y_pred.append(int(float(row[pred_col])))
                if score_col:
                    y_score.append(float(row[score_col]))
            except (ValueError, KeyError):
                continue
    if not y_true:
        return None

    if HAS_SKLEARN:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auroc = None
        if y_score and len(set(y_true)) > 1:
            try:
                auroc = roc_auc_score(y_true, y_score)
            except ValueError:
                auroc = None
    else:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        auroc = None

    return {"n": len(y_true), "precision": precision, "recall": recall, "f1": f1, "auroc": auroc}


def main():
    if not os.path.isfile(REGISTRY):
        raise SystemExit(f"[ERROR] registry not found: {REGISTRY}")

    results = []
    with open(REGISTRY, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dataset = row["dataset"]
            group_mode = row["group_mode"]
            for run_root_key in ["run_root_main", "run_root_base"]:
                run_root = row[run_root_key]
                for csv_path in find_pred_csvs(run_root):
                    metrics = compute_metrics(csv_path)
                    if metrics is None:
                        continue
                    model = infer_model_name(run_root, csv_path)
                    results.append({
                        "dataset": dataset,
                        "group_mode": group_mode,
                        "model": model,
                        "file": csv_path,
                        **metrics,
                    })

    if not results:
        print("[WARN] no prediction csv found / column matching failed.")
        print("  -> COLUMN_CANDIDATES 또는 find_pred_csvs() 패턴을 실제 출력 포맷에 맞춰 조정 필요.")
        return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    fieldnames = ["dataset", "group_mode", "model", "n", "precision", "recall", "f1", "auroc", "file"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # group vs ungroup 비교 표 (markdown)
    print(f"\n-> {OUTPUT_CSV}\n")
    print("| dataset | group_mode | model | n | precision | recall | f1 | auroc |")
    print("|---|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: (x["dataset"], x["model"], x["group_mode"])):
        auroc_str = f"{r['auroc']:.4f}" if r["auroc"] is not None else "-"
        print(f"| {r['dataset']} | {r['group_mode']} | {r['model']} | {r['n']} | "
              f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {auroc_str} |")


if __name__ == "__main__":
    main()
