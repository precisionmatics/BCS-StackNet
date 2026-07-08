"""
Step 3: BCS classification and Applicability Domain analysis.

Requires Step 2 to have completed and saved:
  saved_models/LogS_stacking_all.pkl
  saved_models/LogPapp_stacking_all.pkl

Outputs:
  results/table3_bcs_classification.csv
  results/table_ad_analysis.csv
  saved_models/AD_model.pkl
  saved_models/BCS_direct.pkl
  saved_models/BCS_multilabel.pkl
"""

import os, sys, warnings, joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                              hamming_loss, jaccard_score, classification_report)

from src.data_loading import load_endpoint, load_bcs_drugs, ENDPOINTS, BCS_PATH
from src.featurizer import compute_features, assemble
from src.models import make_lgbm_classifier, make_stacking_classifier, make_multilabel_classifier
from src.evaluation import ApplicabilityDomain, multilabel_metrics

RESULTS_DIR = "/home/stalin/Desktop/BCS/results"
MODELS_DIR  = "/home/stalin/Desktop/BCS/saved_models"
os.makedirs(RESULTS_DIR, exist_ok=True)

LOG_S_THRESHOLD = -2.0       # BCS high solubility: log S >= -2 mol/L
LOG_PAPP_CUTOFF = -5.097     # BCS high permeability: log Papp >= -5.097 log cm/s
RANDOM_STATE    = 42

ALL_DESC = ["morgan", "rdkit", "mordred"]


def banner(msg):
    print(f"\n{'='*65}\n  {msg}\n{'='*65}")


def load_bundle(ep_name):
    path = os.path.join(MODELS_DIR, f"{ep_name}_stacking_all.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run Step 2 first: {path}")
    return joblib.load(path)


def apply_sel(sel, X_raw):
    return sel.transform(X_raw)


# ── BCS threshold classification ──────────────────────────────────────────────

def classify_threshold(log_s: np.ndarray, log_papp: np.ndarray) -> np.ndarray:
    cls = np.full(len(log_s), 4, dtype=int)
    for i, (ls, lpa) in enumerate(zip(log_s, log_papp)):
        high_sol = ls  >= LOG_S_THRESHOLD
        high_per = lpa >= LOG_PAPP_CUTOFF
        cls[i] = {(True,True):1, (False,True):2, (True,False):3, (False,False):4}[(high_sol,high_per)]
    return cls


def main():
    # ── Load BCS drugs ────────────────────────────────────────────────────────
    banner("Loading BCS validation drugs")
    bcs_df = load_bcs_drugs(BCS_PATH)
    smiles_bcs = bcs_df["smiles_clean"].tolist()
    print(f"  Total BCS drugs: {len(bcs_df)}")
    print(f"  Ambiguous: {bcs_df.is_ambiguous.sum()} ({bcs_df.is_ambiguous.mean()*100:.1f}%)")
    print(f"  Class distribution (primary class):")
    single = bcs_df[~bcs_df.is_ambiguous]
    print(f"    {dict(single['bcs_primary'].value_counts().sort_index())}")

    feat_bcs, _, valid_idx_bcs = compute_features(
        smiles_bcs, descriptor_types=ALL_DESC, use_cache=True
    )
    bcs_df = bcs_df.iloc[valid_idx_bcs].reset_index(drop=True)
    X_bcs_all = assemble(feat_bcs, ALL_DESC)

    # ── Load saved models ─────────────────────────────────────────────────────
    bundle_s  = load_bundle("LogS")
    bundle_pa = load_bundle("LogPapp")

    X_bcs_s   = apply_sel(bundle_s["selector"],  X_bcs_all)
    X_bcs_pa  = apply_sel(bundle_pa["selector"], X_bcs_all)

    pred_logS    = bundle_s["model"].predict(X_bcs_s)
    pred_logPapp = bundle_pa["model"].predict(X_bcs_pa)

    # ── Strategy A: Two-step threshold ───────────────────────────────────────
    banner("Strategy A — Two-Step Threshold")
    print("  NOTE: Using predicted log S (no dose info). FormulationBCS used")
    print("        actual drug doses from product labels for D0, making their")
    print("        77.7% accuracy a measure of dose data quality, not ML alone.")

    pred_2step = classify_threshold(pred_logS, pred_logPapp)
    single_mask = ~bcs_df["is_ambiguous"].values
    y_true_single = bcs_df["bcs_primary"].values[single_mask].astype(int)
    y_pred_single = pred_2step[single_mask]

    acc_a = accuracy_score(y_true_single, y_pred_single)
    f1_a  = f1_score(y_true_single, y_pred_single, average="macro", zero_division=0)
    print(f"\n  N single-class: {single_mask.sum()}")
    print(f"  Accuracy : {acc_a:.4f}   (FormulationBCS with dose data: 0.777)")
    print(f"  F1-macro : {f1_a:.4f}")
    cm_a = confusion_matrix(y_true_single, y_pred_single, labels=[1,2,3,4])
    print(f"  Confusion matrix:\n"
          f"{pd.DataFrame(cm_a, index=[f'T{i}' for i in [1,2,3,4]], columns=[f'P{i}' for i in [1,2,3,4]]).to_string()}")

    # ── Strategy B: Direct stacking classifier (5-fold CV) ───────────────────
    banner("Strategy B — Direct 4-Class Stacking Classifier (5-fold CV)")
    X_single  = X_bcs_all[single_mask]
    y_single  = bcs_df["bcs_primary"].values[single_mask].astype(int)

    # Apply variance filter consistent with training
    sel_b = VarianceThreshold(threshold=1e-6)
    X_single_f = sel_b.fit_transform(X_single)

    skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    clf_b    = make_stacking_classifier()
    y_cv_b   = cross_val_predict(clf_b, X_single_f, y_single, cv=skf)

    acc_b = accuracy_score(y_single, y_cv_b)
    f1_b  = f1_score(y_single, y_cv_b, average="macro", zero_division=0)
    print(f"  N: {len(y_single)}, 5-fold CV")
    print(f"  Accuracy : {acc_b:.4f}")
    print(f"  F1-macro : {f1_b:.4f}")
    print(f"\n{classification_report(y_single, y_cv_b, labels=[1,2,3,4], zero_division=0)}")

    # Retrain final direct classifier
    clf_b.fit(X_single_f, y_single)
    joblib.dump({"model": clf_b, "selector": sel_b},
                os.path.join(MODELS_DIR, "BCS_direct.pkl"), compress=3)

    # ── Strategy C: Multi-label (all 294 drugs) ───────────────────────────────
    banner("Strategy C — Multi-Label BCS (all drugs including ambiguous)")
    Y_ml = np.vstack(bcs_df["bcs_multilabel"].values)
    sel_c = VarianceThreshold(threshold=1e-6)
    X_ml = sel_c.fit_transform(X_bcs_all)

    X_tr_ml, X_te_ml, Y_tr_ml, Y_te_ml = train_test_split(
        X_ml, Y_ml, test_size=0.2, random_state=RANDOM_STATE
    )
    clf_c = make_multilabel_classifier()
    clf_c.fit(X_tr_ml, Y_tr_ml)
    Y_pred_ml = clf_c.predict(X_te_ml)
    m_ml = multilabel_metrics(Y_te_ml, Y_pred_ml, name="Multi-label")

    print(f"  N total: {len(Y_ml)}, test N: {len(Y_te_ml)}")
    print(f"  Exact match : {m_ml['Exact_Match']:.4f}")
    print(f"  Hamming loss: {m_ml['Hamming_Loss']:.4f}")
    print(f"  Jaccard     : {m_ml['Jaccard']:.4f}")

    joblib.dump({"model": clf_c, "selector": sel_c},
                os.path.join(MODELS_DIR, "BCS_multilabel.pkl"), compress=3)

    # ── Applicability Domain ──────────────────────────────────────────────────
    banner("Applicability Domain Analysis")

    # Train AD on Log S Morgan fingerprints from full training set
    df_s = load_endpoint(*ENDPOINTS["LogS"])
    feat_s, _, _ = compute_features(
        df_s["smiles_clean"].tolist(), descriptor_types=["morgan"], use_cache=True
    )
    train_fps = feat_s["morgan"].astype(np.float32)

    ad = ApplicabilityDomain(percentile=5.0)
    ad.fit(train_fps)
    joblib.dump(ad, os.path.join(MODELS_DIR, "AD_model.pkl"), compress=3)

    bcs_fps = feat_bcs["morgan"].astype(np.float32)
    in_domain = ad.predict(bcs_fps)
    sims      = ad.similarity_scores(bcs_fps)

    print(f"  AD threshold  : {ad.threshold_:.4f}")
    print(f"  In domain     : {in_domain.sum()}/{len(in_domain)} ({in_domain.mean()*100:.1f}%)")
    print(f"  Out of domain : {(~in_domain).sum()} drugs ({(~in_domain).mean()*100:.1f}%)")
    print(f"  Similarity stats: mean={sims.mean():.3f} min={sims.min():.3f} max={sims.max():.3f}")

    # 5 most out-of-domain drugs
    ood_idx = np.argsort(sims)[:5]
    print(f"\n  Most out-of-domain drugs:")
    for idx in ood_idx:
        print(f"    {bcs_df.iloc[idx]['name']:30s}  sim={sims[idx]:.3f}")

    # ── BCS classification summary table ─────────────────────────────────────
    bcs_rows = [
        {"Strategy": "Two-Step (predicted log S, no dose data)", "N": single_mask.sum(),
         "Metric": "Accuracy", "Score": f"{acc_a:.4f}", "Paper": "0.777 (with dose data)"},
        {"Strategy": "Two-Step F1-macro",   "N": single_mask.sum(),
         "Metric": "F1-macro", "Score": f"{f1_a:.4f}", "Paper": "—"},
        {"Strategy": "Direct Stacking (5-fold CV)", "N": len(y_single),
         "Metric": "Accuracy", "Score": f"{acc_b:.4f}", "Paper": "—"},
        {"Strategy": "Direct Stacking F1-macro", "N": len(y_single),
         "Metric": "F1-macro", "Score": f"{f1_b:.4f}", "Paper": "—"},
        {"Strategy": "Multi-Label (all drugs)", "N": len(Y_ml),
         "Metric": "Exact Match", "Score": f"{m_ml['Exact_Match']:.4f}", "Paper": "N/A"},
        {"Strategy": "Multi-Label Jaccard", "N": len(Y_ml),
         "Metric": "Jaccard", "Score": f"{m_ml['Jaccard']:.4f}", "Paper": "N/A"},
    ]
    df_bcs = pd.DataFrame(bcs_rows)
    df_bcs.to_csv(f"{RESULTS_DIR}/table3_bcs_classification.csv", index=False)

    # AD table
    ad_df = pd.DataFrame({
        "Drug": bcs_df["name"],
        "BCS_class": bcs_df["bcs_primary"],
        "Tanimoto_sim": sims,
        "In_domain": in_domain,
    })
    ad_df.to_csv(f"{RESULTS_DIR}/table_ad_analysis.csv", index=False)

    banner("BCS Classification Summary")
    print(df_bcs.to_string(index=False))
    print(f"\nAll files saved to {RESULTS_DIR}/ and {MODELS_DIR}/")


if __name__ == "__main__":
    main()
