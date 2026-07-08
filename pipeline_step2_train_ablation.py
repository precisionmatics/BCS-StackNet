"""
Step 2: Train all models and run ablation studies.

Ablation 1 — Descriptor families (using Stacking ensemble):
  Morgan | RDKit-2D | Mordred | Morgan+RDKit | Morgan+Mordred | RDKit+Mordred | All

Ablation 2 — Model architectures (using All descriptors):
  LightGBM | XGBoost | RandomForest | Stacking

Outputs:
  results/table_descriptor_ablation.csv
  results/table_model_ablation.csv
  results/table2_train_test_comparison.csv
  results/conformal_coverage.csv
  saved_models/{endpoint}_{model}.pkl
"""

import gc, os, sys, time, warnings, joblib
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from src.data_loading import load_endpoint, ENDPOINTS, PAPER_BENCHMARKS
from src.featurizer import compute_features, assemble, DESCRIPTOR_SETS
from src.models import REGRESSOR_CONFIGS, make_stacking_regressor
from src.evaluation import regression_metrics, coverage_table

RESULTS_DIR = "/home/stalin/Desktop/BCS/results"
MODELS_DIR  = "/home/stalin/Desktop/BCS/saved_models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

RANDOM_STATE = 42

# ── Train/cal/test split helper ───────────────────────────────────────────────

def split_data(X, y, test_frac=0.10, cal_frac=0.10):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=test_frac + cal_frac, random_state=RANDOM_STATE
    )
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE
    )
    return X_tr, X_cal, X_te, y_tr, y_cal, y_te


# ── Variance filter ───────────────────────────────────────────────────────────

def sanitize(X: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=np.float32)
    X = np.where(np.isfinite(X), X, np.float32(0.0))
    X = np.clip(X, -1e9, 1e9)
    return X


def apply_var_filter(X_tr, X_cal, X_te, thresh=1e-6):
    X_tr, X_cal, X_te = sanitize(X_tr), sanitize(X_cal), sanitize(X_te)
    sel = VarianceThreshold(threshold=thresh)
    X_tr  = sel.fit_transform(X_tr)
    X_cal = sel.transform(X_cal)
    X_te  = sel.transform(X_te)
    return X_tr, X_cal, X_te, sel


# ── Single experiment ─────────────────────────────────────────────────────────

def run_one(X_all, y_all, model_factory, label=""):
    X_tr, X_cal, X_te, y_tr, y_cal, y_te = split_data(X_all, y_all)
    X_tr, X_cal, X_te, sel = apply_var_filter(X_tr, X_cal, X_te)

    t0 = time.time()
    model = model_factory()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    m_tr  = regression_metrics(y_tr,  model.predict(X_tr),  name="Train")
    m_cal = regression_metrics(y_cal, model.predict(X_cal), name="Cal")
    m_te  = regression_metrics(y_te,  model.predict(X_te),  name="Test")

    cov_df = coverage_table(y_cal, model.predict(X_cal), y_te, model.predict(X_te))

    print(f"    {label:30s}  "
          f"R²_tr={m_tr['R2']:.4f}  R²_te={m_te['R2']:.4f}  "
          f"RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}  "
          f"[{elapsed:.0f}s]")

    return model, sel, m_tr, m_te, cov_df, (y_cal, model.predict(X_cal), y_te, model.predict(X_te))


def banner(msg):
    print(f"\n{'='*65}\n  {msg}\n{'='*65}")


# ── Main ─────────────────────────────────────────────────────────────────────

def append_csv(path, rows):
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


def main():
    # Determine which endpoints are already complete (model saved + CSV rows written)
    desc_csv = f"{RESULTS_DIR}/table_descriptor_ablation.csv"
    done_endpoints = set()
    if os.path.exists(desc_csv):
        existing = pd.read_csv(desc_csv)
        for ep in existing["Endpoint"].unique():
            model_path = os.path.join(MODELS_DIR, f"{ep}_stacking_all.pkl")
            if os.path.exists(model_path):
                done_endpoints.add(ep)

    if done_endpoints:
        print(f"  Resuming — skipping already-complete endpoints: {sorted(done_endpoints)}")
    else:
        # Fresh run — clear any partial CSVs
        for fname in ["table_descriptor_ablation.csv", "table_model_ablation.csv",
                      "table2_train_test_comparison.csv", "conformal_coverage.csv"]:
            fpath = f"{RESULTS_DIR}/{fname}"
            if os.path.exists(fpath):
                os.remove(fpath)

    all_desc_abl_rows  = []
    all_model_abl_rows = []
    all_table2_rows    = []
    all_coverage_rows  = []

    for ep_name, (path, smiles_col, target_col) in ENDPOINTS.items():
        if ep_name in done_endpoints:
            banner(f"Endpoint: {ep_name} — SKIPPED (already complete)")
            continue
        banner(f"Endpoint: {ep_name}")
        paper = PAPER_BENCHMARKS[ep_name]

        # Load data
        df = load_endpoint(path, smiles_col, target_col)
        smiles = df["smiles_clean"].tolist()
        y_all  = df["target"].values

        # Load cached features
        feat, feat_names, valid_idx = compute_features(
            smiles, descriptor_types=["morgan", "rdkit", "mordred"], use_cache=True
        )
        y_all = y_all[valid_idx]

        # ── Ablation 1: Descriptor families (Stacking) ──────────────────────
        print(f"\n  [Descriptor Ablation — {ep_name}]")
        for desc_label, desc_types in DESCRIPTOR_SETS.items():
            X_all = assemble(feat, desc_types)
            _, _, m_tr, m_te, cov_df, _ = run_one(
                X_all, y_all, make_stacking_regressor, label=desc_label
            )
            all_desc_abl_rows.append({
                "Endpoint": ep_name,
                "Descriptor Set": desc_label,
                "N_features": X_all.shape[1],
                "R2_train": m_tr["R2"],
                "R2_test":  m_te["R2"],
                "RMSE_test": m_te["RMSE"],
                "MAE_test":  m_te["MAE"],
            })
            del X_all; gc.collect()

        # ── Ablation 2: Model comparison (All descriptors) ──────────────────
        print(f"\n  [Model Ablation — {ep_name}]")
        X_all = assemble(feat, ["morgan", "rdkit", "mordred"])
        best_model, best_sel, best_m_tr, best_m_te, best_cov, best_cal_te = None, None, None, None, None, None

        for model_label, factory in REGRESSOR_CONFIGS.items():
            model, sel, m_tr, m_te, cov_df, cal_te = run_one(
                X_all, y_all, factory, label=model_label
            )
            all_model_abl_rows.append({
                "Endpoint": ep_name,
                "Model": model_label,
                "R2_train": m_tr["R2"],
                "R2_test":  m_te["R2"],
                "RMSE_test": m_te["RMSE"],
                "MAE_test":  m_te["MAE"],
            })
            if model_label == "Stacking":
                best_model, best_sel = model, sel
                best_m_tr, best_m_te = m_tr, m_te
                best_cov, best_cal_te = cov_df, cal_te
            else:
                del model; gc.collect()

        # ── Table 2: Our Stacking vs Paper (both train and test R²) ─────────
        all_table2_rows.append({
            "Endpoint": ep_name,
            "Paper_Model":    paper["model"],
            "Paper_R2_test":  paper["R2_test"],
            "Paper_RMSE":     paper["RMSE"],
            "Paper_MAE":      paper["MAE"],
            "Ours_R2_train":  best_m_tr["R2"],
            "Ours_R2_test":   best_m_te["R2"],
            "Ours_RMSE_test": best_m_te["RMSE"],
            "Ours_MAE_test":  best_m_te["MAE"],
            "Delta_R2_test":  round(best_m_te["R2"] - paper["R2_test"], 4),
            "Delta_MAE_test": round(best_m_te["MAE"] - paper["MAE"], 4),
        })

        # ── Conformal coverage for stacking model ───────────────────────────
        for _, row in best_cov.iterrows():
            all_coverage_rows.append({
                "Endpoint": ep_name, **row.to_dict()
            })

        # ── Save best (Stacking + All) model for web app ────────────────────
        y_cal_raw, cal_pred, y_te_raw, te_pred = best_cal_te
        cal_residuals = np.abs(y_cal_raw - cal_pred)
        joblib.dump(
            {"model": best_model, "selector": best_sel,
             "cal_residuals": cal_residuals},
            os.path.join(MODELS_DIR, f"{ep_name}_stacking_all.pkl"),
            compress=3
        )
        print(f"\n  Saved: saved_models/{ep_name}_stacking_all.pkl")
        del best_model, X_all; gc.collect()

        # ── Flush this endpoint's results to disk immediately ────────────────
        ep_desc_rows = [r for r in all_desc_abl_rows if r["Endpoint"] == ep_name]
        ep_model_rows = [r for r in all_model_abl_rows if r["Endpoint"] == ep_name]
        ep_table2_rows = [r for r in all_table2_rows if r["Endpoint"] == ep_name]
        ep_cov_rows = [r for r in all_coverage_rows if r["Endpoint"] == ep_name]
        append_csv(f"{RESULTS_DIR}/table_descriptor_ablation.csv", ep_desc_rows)
        append_csv(f"{RESULTS_DIR}/table_model_ablation.csv", ep_model_rows)
        append_csv(f"{RESULTS_DIR}/table2_train_test_comparison.csv", ep_table2_rows)
        append_csv(f"{RESULTS_DIR}/conformal_coverage.csv", ep_cov_rows)
        print(f"  Results flushed to CSV for {ep_name}")

    # ── Print summary tables ──────────────────────────────────────────────────
    banner("TABLE 2 — Our Stacking+All vs. FormulationBCS (test set only)")
    df_t2 = pd.DataFrame(all_table2_rows)
    print(df_t2[["Endpoint","Paper_Model","Paper_R2_test","Ours_R2_test",
                 "Delta_R2_test","Paper_MAE","Ours_MAE_test","Delta_MAE_test"]].to_string(index=False))

    banner("DESCRIPTOR ABLATION (Stacking model, test R²)")
    df_da = pd.DataFrame(all_desc_abl_rows)
    pivot = df_da.pivot(index="Descriptor Set", columns="Endpoint", values="R2_test")
    print(pivot.to_string())

    banner("MODEL ABLATION (All descriptors, test R²)")
    df_ma = pd.DataFrame(all_model_abl_rows)
    pivot_m = df_ma.pivot(index="Model", columns="Endpoint", values="R2_test")
    print(pivot_m.to_string())

    banner("CONFORMAL COVERAGE (Stacking + All, 90% nominal)")
    cov90 = pd.DataFrame(all_coverage_rows)
    cov90 = cov90[cov90["Nominal (%)"] == 90.0]
    print(cov90[["Endpoint","Nominal (%)","Empirical (%)","Δ (pp)","Mean Width"]].to_string(index=False))

    print(f"\nAll results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
