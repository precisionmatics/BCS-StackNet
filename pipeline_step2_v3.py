"""
Step 2 v3 — Enhanced training with:
  - CatBoost as 4th base learner
  - ElasticNetCV meta-learner
  - Optuna hyperparameter tuning (per endpoint)
  - Per-endpoint optimal feature sets (from ablation + CaliciBoost research)
  - Expanded datasets (LogPapp + ChEMBL, LogD + GitHub) if available

Feature sets per endpoint:
  LogS    : Morgan + RDKit-2D + Mordred (all, current best)
  LogP    : Morgan + RDKit-2D + Mordred (all, current best)
  LogD    : Morgan + Mordred (ablation best: 0.7523 vs 0.7425 for All)
  LogPapp : Morgan + RDKit-2D + Mordred-3D (research: 3D features +15% MAE for Caco-2)
"""

import gc, os, sys, time, warnings, joblib
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import multiprocessing
_NCPU = multiprocessing.cpu_count()
_NJOBS_PER_MODEL = max(1, _NCPU // 3)  # 3 folds in parallel, each model gets 1/3 of cores

from src.data_loading import load_endpoint, ENDPOINTS, PAPER_BENCHMARKS
from src.featurizer import compute_features, assemble
from src.models import make_rf_regressor, make_stacking_regressor_v3
from src.evaluation import regression_metrics, coverage_table

RESULTS_DIR = "/home/stalin/Desktop/BCS/results/v3"
MODELS_DIR  = "/home/stalin/Desktop/BCS/saved_models/v3"
AUG_DIR     = "/home/stalin/Desktop/BCS/data/augmented"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

RANDOM_STATE = 42

# ── Per-endpoint feature sets (research-informed) ─────────────────────────────

ENDPOINT_FEATURES = {
    "LogS":    ["morgan", "rdkit", "mordred"],
    "LogP":    ["morgan", "rdkit", "mordred"],
    "LogD":    ["morgan", "mordred"],               # ablation: Morgan+Mordred=0.7523 > All=0.7425
    "LogPapp": ["morgan", "rdkit", "mordred"],      # 12k compounds — enough data for Mordred not to overfit
}

# ── Expanded dataset paths (if Step 0 was run) ────────────────────────────────

def get_data_path(ep_name):
    aug = {
        "LogPapp": os.path.join(AUG_DIR, "Log_Papp_expanded.csv"),
        "LogD":    os.path.join(AUG_DIR, "Log_D_expanded.csv"),
    }
    if ep_name in aug and os.path.exists(aug[ep_name]):
        return aug[ep_name]
    # Fall back to original
    path, _, _ = ENDPOINTS[ep_name]
    return path


# ── Data splitting ────────────────────────────────────────────────────────────

def split_data(X, y, test_frac=0.10, cal_frac=0.10):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=test_frac + cal_frac, random_state=RANDOM_STATE)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE)
    return X_tr, X_cal, X_te, y_tr, y_cal, y_te


def sanitize(X):
    X = np.array(X, dtype=np.float32)
    X = np.where(np.isfinite(X), X, np.float32(0.0))
    return np.clip(X, -1e9, 1e9)


def apply_var_filter(X_tr, X_cal, X_te, thresh=1e-6):
    X_tr, X_cal, X_te = sanitize(X_tr), sanitize(X_cal), sanitize(X_te)
    sel = VarianceThreshold(threshold=thresh)
    X_tr  = sel.fit_transform(X_tr)
    X_cal = sel.transform(X_cal)
    X_te  = sel.transform(X_te)
    return X_tr, X_cal, X_te, sel


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def tune_lgbm(X_tr, y_tr, n_trials=50, cv=3):
    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 300, 1500),
            learning_rate     = trial.suggest_float("lr", 0.005, 0.2, log=True),
            max_depth         = trial.suggest_int("max_depth", 3, 10),
            num_leaves        = trial.suggest_int("num_leaves", 15, 255),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            min_child_samples = trial.suggest_int("min_child_samples", 5, 60),
        )
        mdl = lgb.LGBMRegressor(**params, random_state=42, n_jobs=_NJOBS_PER_MODEL, verbose=-1)
        return cross_val_score(mdl, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=cv).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    p = study.best_params
    return dict(
        n_estimators=p["n_estimators"], learning_rate=p["lr"],
        max_depth=p["max_depth"], num_leaves=p["num_leaves"],
        subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
        reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"],
        min_child_samples=p["min_child_samples"],
        random_state=42, n_jobs=-1, verbose=-1,
    )


def tune_xgb(X_tr, y_tr, n_trials=50, cv=3):
    def objective(trial):
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 300, 1500),
            learning_rate    = trial.suggest_float("lr", 0.005, 0.2, log=True),
            max_depth        = trial.suggest_int("max_depth", 3, 10),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 20),
            gamma            = trial.suggest_float("gamma", 0.0, 5.0),
        )
        mdl = xgb.XGBRegressor(**params, random_state=42, n_jobs=_NJOBS_PER_MODEL, verbosity=0)
        return cross_val_score(mdl, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=cv).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    p = study.best_params
    return dict(
        n_estimators=p["n_estimators"], learning_rate=p["lr"],
        max_depth=p["max_depth"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"],
        min_child_weight=p["min_child_weight"], gamma=p["gamma"],
        random_state=42, n_jobs=-1, verbosity=0,
    )


def tune_catboost(X_tr, y_tr, n_trials=50, cv=3):
    def objective(trial):
        params = dict(
            iterations       = trial.suggest_int("iterations", 300, 1500),
            learning_rate    = trial.suggest_float("lr", 0.005, 0.2, log=True),
            depth            = trial.suggest_int("depth", 3, 10),
            l2_leaf_reg      = trial.suggest_float("l2_leaf_reg", 1e-8, 20.0, log=True),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bylevel= trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        )
        mdl = CatBoostRegressor(**params, random_seed=42, verbose=0,
                                thread_count=_NJOBS_PER_MODEL, allow_writing_files=False)
        return cross_val_score(mdl, X_tr, y_tr, cv=cv, scoring="r2", n_jobs=cv).mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    p = study.best_params
    return dict(
        iterations=p["iterations"], learning_rate=p["lr"],
        depth=p["depth"], l2_leaf_reg=p["l2_leaf_reg"],
        subsample=p["subsample"], colsample_bylevel=p["colsample_bylevel"],
        random_seed=42, verbose=0, thread_count=-1, allow_writing_files=False,
    )


# ── Tune all base learners ────────────────────────────────────────────────────

def tune_all(X_tr, y_tr, ep_name, n_trials=50):
    # After subsampling to 5000, all datasets are fast — use uniform settings
    n_trials = 50
    cv = 3

    for label, fn in [
        ("LGBM",     lambda: tune_lgbm(X_tr, y_tr, n_trials=n_trials, cv=cv)),
        ("XGBoost",  lambda: tune_xgb(X_tr, y_tr, n_trials=n_trials, cv=cv)),
        ("CatBoost", lambda: tune_catboost(X_tr, y_tr, n_trials=n_trials, cv=cv)),
    ]:
        print(f"    Tuning {label:10s} ({n_trials} trials, {cv}-fold)...", end=" ", flush=True)
        t0 = time.time()
        result = fn()
        print(f"[{time.time()-t0:.0f}s]")
        if label == "LGBM":
            lgbm_p = result
        elif label == "XGBoost":
            xgb_p = result
        else:
            cat_p = result

    return lgbm_p, xgb_p, cat_p


# ── Single model run ──────────────────────────────────────────────────────────

def run_stacking(X_all, y_all, lgbm_p, xgb_p, cat_p, label="Stacking-v3"):
    X_tr, X_cal, X_te, y_tr, y_cal, y_te = split_data(X_all, y_all)
    X_tr, X_cal, X_te, sel = apply_var_filter(X_tr, X_cal, X_te)

    t0 = time.time()
    model = make_stacking_regressor_v3(
        lgbm_params=lgbm_p, xgb_params=xgb_p, cat_params=cat_p
    )
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    m_tr = regression_metrics(y_tr,  model.predict(X_tr),  name="Train")
    m_te = regression_metrics(y_te,  model.predict(X_te),  name="Test")
    cov  = coverage_table(y_cal, model.predict(X_cal), y_te, model.predict(X_te))

    print(f"    {label:30s}  "
          f"R²_tr={m_tr['R2']:.4f}  R²_te={m_te['R2']:.4f}  "
          f"RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}  [{elapsed:.0f}s]")

    return model, sel, m_tr, m_te, cov, (y_cal, model.predict(X_cal), y_te, model.predict(X_te))


# ── Also run individual models for model ablation ─────────────────────────────

def run_individual(X_all, y_all, lgbm_p, xgb_p, cat_p):
    rows = []
    X_tr, X_cal, X_te, y_tr, y_cal, y_te = split_data(X_all, y_all)
    X_tr, X_cal, X_te, sel = apply_var_filter(X_tr, X_cal, X_te)

    for label, factory in [
        ("LightGBM",  lambda: lgb.LGBMRegressor(**lgbm_p)),
        ("XGBoost",   lambda: xgb.XGBRegressor(**xgb_p)),
        ("CatBoost",  lambda: CatBoostRegressor(**cat_p)),
        ("RandomForest", make_rf_regressor),
    ]:
        t0 = time.time()
        m = factory(); m.fit(X_tr, y_tr)
        m_tr = regression_metrics(y_tr, m.predict(X_tr))
        m_te = regression_metrics(y_te, m.predict(X_te))
        print(f"    {label:30s}  "
              f"R²_tr={m_tr['R2']:.4f}  R²_te={m_te['R2']:.4f}  "
              f"RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}  [{time.time()-t0:.0f}s]")
        rows.append({"Model": label, "R2_train": m_tr["R2"], "R2_test": m_te["R2"],
                     "RMSE_test": m_te["RMSE"], "MAE_test": m_te["MAE"]})
        del m; gc.collect()
    return rows


def banner(msg):
    print(f"\n{'='*65}\n  {msg}\n{'='*65}")


def append_csv(path, rows):
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_table2 = []
    all_coverage = []
    all_model_abl = []

    for ep_name, (_, smiles_col, target_col) in ENDPOINTS.items():
        banner(f"Endpoint: {ep_name}")
        paper = PAPER_BENCHMARKS[ep_name]

        # Load data (expanded if available)
        data_path = get_data_path(ep_name)
        df = load_endpoint(data_path, smiles_col, target_col)
        smiles = df["smiles_clean"].tolist()
        y_all  = df["target"].values
        print(f"  Dataset: {len(df):,} compounds  (from: {os.path.basename(data_path)})")

        # Feature types for this endpoint
        desc_types = ENDPOINT_FEATURES[ep_name]
        print(f"  Features: {desc_types}")

        # Compute features (cached for morgan/rdkit/mordred; fresh for mordred_3d/maccs)
        use_cache = all(dt in ["morgan", "rdkit", "mordred"] for dt in desc_types)
        feat, feat_names, valid_idx = compute_features(
            smiles, descriptor_types=desc_types, use_cache=True
        )
        y_all = y_all[valid_idx]
        X_all = assemble(feat, desc_types)
        print(f"  Feature matrix: {X_all.shape[0]} × {X_all.shape[1]}")

        # ── Optuna tuning on training split ─────────────────────────────────
        # Subsample to max 5000 for tuning — hyperparameter landscape is stable,
        # 3× faster per trial than using full dataset
        X_tune, _, y_tune, _ = train_test_split(
            sanitize(X_all), y_all, test_size=0.20, random_state=RANDOM_STATE)
        sel_tune = VarianceThreshold(threshold=1e-6)
        X_tune = sel_tune.fit_transform(X_tune)
        if len(X_tune) > 5000:
            idx = np.random.RandomState(42).choice(len(X_tune), 5000, replace=False)
            X_tune, y_tune = X_tune[idx], y_tune[idx]
        print(f"\n  [Optuna Tuning — {ep_name}]  ({X_tune.shape[0]} subsample, {X_tune.shape[1]} features)")
        lgbm_p, xgb_p, cat_p = tune_all(X_tune, y_tune, ep_name)
        del X_tune, sel_tune; gc.collect()

        # ── Final stacking model ─────────────────────────────────────────────
        print(f"\n  [Final Model — {ep_name}]")
        model, sel, m_tr, m_te, cov_df, cal_te = run_stacking(
            X_all, y_all, lgbm_p, xgb_p, cat_p
        )

        # ── Individual model ablation ────────────────────────────────────────
        print(f"\n  [Model Ablation — {ep_name}]")
        indiv_rows = run_individual(X_all, y_all, lgbm_p, xgb_p, cat_p)
        for r in indiv_rows:
            r["Endpoint"] = ep_name
        all_model_abl.extend(indiv_rows)

        # ── Table 2 comparison ───────────────────────────────────────────────
        all_table2.append({
            "Endpoint":       ep_name,
            "Paper_Model":    paper["model"],
            "Paper_R2_test":  paper["R2_test"],
            "Paper_RMSE":     paper["RMSE"],
            "Paper_MAE":      paper["MAE"],
            "Ours_R2_train":  m_tr["R2"],
            "Ours_R2_test":   m_te["R2"],
            "Ours_RMSE_test": m_te["RMSE"],
            "Ours_MAE_test":  m_te["MAE"],
            "Delta_R2_test":  round(m_te["R2"] - paper["R2_test"], 4),
            "Delta_MAE_test": round(m_te["MAE"] - paper["MAE"], 4),
            "N_train":        len(df),
            "Features":       "+".join(desc_types),
        })

        for _, row in cov_df.iterrows():
            all_coverage.append({"Endpoint": ep_name, **row.to_dict()})

        # ── Save model ───────────────────────────────────────────────────────
        y_cal_raw, cal_pred, _, _ = cal_te
        cal_residuals = np.abs(y_cal_raw - cal_pred)
        joblib.dump(
            {"model": model, "selector": sel, "cal_residuals": cal_residuals,
             "features": desc_types, "lgbm_params": lgbm_p,
             "xgb_params": xgb_p, "cat_params": cat_p},
            os.path.join(MODELS_DIR, f"{ep_name}_stacking_v3.pkl"),
            compress=3,
        )
        print(f"\n  Saved: saved_models/v3/{ep_name}_stacking_v3.pkl")

        # ── Flush ────────────────────────────────────────────────────────────
        ep_rows = [r for r in all_table2 if r["Endpoint"] == ep_name]
        ep_cov  = [r for r in all_coverage if r["Endpoint"] == ep_name]
        ep_abl  = [r for r in all_model_abl if r["Endpoint"] == ep_name]
        append_csv(f"{RESULTS_DIR}/table2_v3.csv", ep_rows)
        append_csv(f"{RESULTS_DIR}/conformal_coverage_v3.csv", ep_cov)
        append_csv(f"{RESULTS_DIR}/model_ablation_v3.csv", ep_abl)
        print(f"  Results flushed to CSV for {ep_name}")

        del model, X_all, feat; gc.collect()

    # ── Summary ──────────────────────────────────────────────────────────────
    banner("TABLE 2 v3 — BCS-StackNet Enhanced vs FormulationBCS")
    df_t2 = pd.DataFrame(all_table2)
    print(df_t2[["Endpoint","Paper_Model","Paper_R2_test","Ours_R2_test",
                 "Delta_R2_test","Paper_MAE","Ours_MAE_test","Delta_MAE_test",
                 "N_train","Features"]].to_string(index=False))

    banner("CONFORMAL COVERAGE (90% nominal)")
    cov90 = pd.DataFrame(all_coverage)
    cov90 = cov90[cov90["Nominal (%)"] == 90.0]
    print(cov90[["Endpoint","Nominal (%)","Empirical (%)","Δ (pp)","Mean Width"]].to_string(index=False))

    print(f"\n  All results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
