import json, glob, os
import numpy as np

BASE = os.path.expanduser("~") + "/Electrical-Power-Equipment-Anomaly-Detection/ViT-B + CLAdapter SFT_new/engine/runs"
REMAINING = f"{BASE}/kfold10_remaining_dataset_0622_20260624_092147"
VITB_CLA  = f"{BASE}/kfold10_vitb_cladapter_dataset_0622_20260624_092152"
BASELINES = f"{BASE}/no_etc_kfold10_baselines_dataset_0622_20260623_211414"

def conf_to_metrics(conf, auroc):
    tp, fp, fn = conf["tp"], conf["fp"], conf["fn"]
    prec = tp/(tp+fp)*100 if tp+fp else 0
    rec  = tp/(tp+fn)*100 if tp+fn else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    return dict(prec=prec, rec=rec, f1=f1, auroc=auroc*100)

def load_patchcore(run_root):
    rows = {}
    for fold in range(10):
        paths = glob.glob(f"{run_root}/fold_{fold}/patchcore/*/metrics.json")
        if not paths: continue
        m = json.load(open(paths[0]))
        conf = m["test"]["best_threshold"]["confusion"]
        rows[fold] = conf_to_metrics(conf, m["test"]["auroc"])
    return rows

def load_differnet(run_root):
    rows = {}
    for fold in range(10):
        paths = glob.glob(f"{run_root}/fold_{fold}/differnet/*/metrics.json")
        if not paths: continue
        m = json.load(open(paths[0]))
        conf = m["test_at_val_best_threshold"]["confusion"]
        rows[fold] = conf_to_metrics(conf, m["test"]["auroc"])
    return rows

def load_cla(run_root, model_dir):
    rows = {}
    for fold in range(10):
        path = f"{run_root}/fold_{fold}/{model_dir}/metrics.json"
        if not os.path.exists(path): continue
        m = json.load(open(path))
        rows[fold] = dict(prec=m["test"]["prec"], rec=m["test"]["reca"], f1=m["test"]["f1"], auroc=m["test"]["roc"]*100)
    return rows

def show(name, rows):
    if not rows: print(f"{name}: 없음\n"); return
    folds = sorted(rows)
    P = [rows[f]["prec"] for f in folds]
    R = [rows[f]["rec"]  for f in folds]
    F = [rows[f]["f1"]   for f in folds]
    A = [rows[f]["auroc"]for f in folds]
    print(f"[{name}] n={len(folds)}")
    print(f"  Prec  {np.mean(P):.2f} +/- {np.std(P):.2f}")
    print(f"  Rec   {np.mean(R):.2f} +/- {np.std(R):.2f}")
    print(f"  F1    {np.mean(F):.2f} +/- {np.std(F):.2f}")
    print(f"  AUROC {np.mean(A):.2f} +/- {np.std(A):.2f}\n")

for name, rows in {
    "PatchCore":             load_patchcore(BASELINES),
    "DifferNet":             load_differnet(BASELINES),
    "ConvNeXt-B (linear)":  load_cla(REMAINING, "linear_convnextb"),
    "ViT-B (linear)":       load_cla(REMAINING, "linear_vitb"),
    "ConvNeXt-B+CLAdapter": load_cla(REMAINING, "convnextb_cla_sft2"),
    "ViT-B+CLAdapter":      load_cla(VITB_CLA,  "vitb_cla_sft2"),
}.items():
    show(name, rows)
