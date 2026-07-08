"""
LogPapp fix: retrain using Wang 2016 only (2,337 compounds) with v3 improvements.
- No ChEMBL data (too noisy — multi-lab Caco-2 values are inconsistent)
- Features: Morgan + RDKit (ablation best: 0.7436; Mordred hurts for Caco-2)
- Optuna tuning (50 trials) + CatBoost + ElasticNetCV meta-learner
"""

import gc, os, sys, time, warnings, joblib
warnings.filterwarnings("ignore")
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
_NJOBS_PER_MODEL = max(1, _NCPU // 3)

from src.data_loading import load_endpoint, PAPER_BENCHMARKS
from src.featurizer import compute_features, assemble
from src.models import make_rf_regressor, make_stacking_regressor_v3
from src.evaluation import regression_metrics, coverage_table

RESULTS_DIR = "/home/stalin/Desktop/BCS/results/v3"
MODELS_DIR  = "/home/stalin/Desktop/BCS/saved_models/v3"
RANDOM_STATE = 42

# Wang 2016 only — original curated dataset
PAPP_PATH   = "/home/stalin/Desktop/BCS/OLD/data/raw/Log_Papp.csv"
FEATURES    = ["morgan", "rdkit"]   # ablation winner for Caco-2


def split_data(X, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
    X_cal, X_te, y_cal, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE)
    return X_tr, X_cal, X_te, y_tr, y_cal, y_te


def sanitize(X):
    X = np.array(X, dtype=np.float32)
    X = np.where(np.isfinite(X), X, 0.0)
    return np.clip(X, -1e9, 1e9)


def apply_var_filter(X_tr, X_cal, X_te, thresh=1e-6):
    X_tr, X_cal, X_te = sanitize(X_tr), sanitize(X_cal), sanitize(X_te)
    sel = VarianceThreshold(threshold=thresh)
    X_tr  = sel.fit_transform(X_tr)
    X_cal = sel.transform(X_cal)
    X_te  = sel.transform(X_te)
    return X_tr, X_cal, X_te, sel


def tune_lgbm(X, y, n_trials=50, cv=5):
    def obj(trial):
        p = dict(
            n_estimators      = trial.suggest_int("n_estimators", 200, 2000),
            learning_rate     = trial.suggest_float("lr", 0.005, 0.2, log=True),
            max_depth         = trial.suggest_int("max_depth", 3, 10),
            num_leaves        = trial.suggest_int("num_leaves", 15, 255),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            min_child_samples = trial.suggest_int("min_child_samples", 3, 50),
        )
        m = lgb.LGBMRegressor(**p, random_state=42, n_jobs=_NJOBS_PER_MODEL, verbose=-1)
        return cross_val_score(m, X, y, cv=cv, scoring="r2", n_jobs=cv).mean()
    s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    p = s.best_params
    return dict(n_estimators=p["n_estimators"], learning_rate=p["lr"],
                max_depth=p["max_depth"], num_leaves=p["num_leaves"],
                subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"],
                min_child_samples=p["min_child_samples"],
                random_state=42, n_jobs=-1, verbose=-1)


def tune_xgb(X, y, n_trials=50, cv=5):
    def obj(trial):
        p = dict(
            n_estimators     = trial.suggest_int("n_estimators", 200, 2000),
            learning_rate    = trial.suggest_float("lr", 0.005, 0.2, log=True),
            max_depth        = trial.suggest_int("max_depth", 3, 10),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 20),
            gamma            = trial.suggest_float("gamma", 0.0, 5.0),
        )
        m = xgb.XGBRegressor(**p, random_state=42, n_jobs=_NJOBS_PER_MODEL, verbosity=0)
        return cross_val_score(m, X, y, cv=cv, scoring="r2", n_jobs=cv).mean()
    s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    p = s.best_params
    return dict(n_estimators=p["n_estimators"], learning_rate=p["lr"],
                max_depth=p["max_depth"], subsample=p["subsample"],
                colsample_bytree=p["colsample_bytree"],
                reg_alpha=p["reg_alpha"], reg_lambda=p["reg_lambda"],
                min_child_weight=p["min_child_weight"], gamma=p["gamma"],
                random_state=42, n_jobs=-1, verbosity=0)


def tune_catboost(X, y, n_trials=50, cv=5):
    def obj(trial):
        p = dict(
            iterations        = trial.suggest_int("iterations", 200, 2000),
            learning_rate     = trial.suggest_float("lr", 0.005, 0.2, log=True),
            depth             = trial.suggest_int("depth", 3, 10),
            l2_leaf_reg       = trial.suggest_float("l2_leaf_reg", 1e-8, 20.0, log=True),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        )
        m = CatBoostRegressor(**p, random_seed=42, verbose=0,
                              thread_count=_NJOBS_PER_MODEL, allow_writing_files=False)
        return cross_val_score(m, X, y, cv=cv, scoring="r2", n_jobs=cv).mean()
    s = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    p = s.best_params
    return dict(iterations=p["iterations"], learning_rate=p["lr"],
                depth=p["depth"], l2_leaf_reg=p["l2_leaf_reg"],
                subsample=p["subsample"], colsample_bylevel=p["colsample_bylevel"],
                random_seed=42, verbose=0, thread_count=-1, allow_writing_files=False)


def banner(msg):
    print(f"\n{'='*65}\n  {msg}\n{'='*65}")


def main():
    banner("LogPapp Fix — Wang 2016 only + Optuna + CatBoost")
    paper = PAPER_BENCHMARKS["LogPapp"]

    df = load_endpoint(PAPP_PATH, "SMILES", "Log  Papp")
    smiles = df["smiles_clean"].tolist()
    y_all  = df["target"].values
    print(f"  Dataset: {len(df):,} compounds (Wang 2016 — curated, single source)")
    print(f"  Features: {FEATURES}")

    feat, _, valid_idx = compute_features(smiles, descriptor_types=FEATURES, use_cache=True)
    y_all = y_all[valid_idx]
    X_all = assemble(feat, FEATURES)
    print(f"  Feature matrix: {X_all.shape[0]} × {X_all.shape[1]}")

    # Tune on full training split (small dataset — use all of it, 5-fold CV)
    X_tune, _, y_tune, _ = train_test_split(sanitize(X_all), y_all, test_size=0.20, random_state=RANDOM_STATE)
    sel_tune = VarianceThreshold(threshold=1e-6)
    X_tune = sel_tune.fit_transform(X_tune)
    print(f"\n  [Optuna Tuning]  ({X_tune.shape[0]} compounds, {X_tune.shape[1]} features, 5-fold CV)")

    for label, fn in [
        ("LGBM",     lambda: tune_lgbm(X_tune, y_tune, n_trials=50, cv=5)),
        ("XGBoost",  lambda: tune_xgb(X_tune, y_tune, n_trials=50, cv=5)),
        ("CatBoost", lambda: tune_catboost(X_tune, y_tune, n_trials=50, cv=5)),
    ]:
        print(f"    Tuning {label:10s}...", end=" ", flush=True)
        t0 = time.time()
        result = fn()
        print(f"[{time.time()-t0:.0f}s]")
        if label == "LGBM":     lgbm_p = result
        elif label == "XGBoost": xgb_p = result
        else:                    cat_p  = result

    del X_tune, sel_tune; gc.collect()

    # Final stacking model
    print(f"\n  [Final Stacking Model]")
    X_tr, X_cal, X_te, y_tr, y_cal, y_te = split_data(sanitize(X_all), y_all)
    X_tr, X_cal, X_te, sel = apply_var_filter(X_tr, X_cal, X_te)

    t0 = time.time()
    model = make_stacking_regressor_v3(lgbm_params=lgbm_p, xgb_params=xgb_p, cat_params=cat_p)
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    m_tr = regression_metrics(y_tr, model.predict(X_tr))
    m_te = regression_metrics(y_te, model.predict(X_te))
    cov  = coverage_table(y_cal, model.predict(X_cal), y_te, model.predict(X_te))

    print(f"    Stacking-v3-fix  R²_tr={m_tr['R2']:.4f}  R²_te={m_te['R2']:.4f}  "
          f"RMSE={m_te['RMSE']:.4f}  MAE={m_te['MAE']:.4f}  [{elapsed:.0f}s]")
    print(f"\n  Paper (XGBoost):  R²=0.71")
    print(f"  Delta:            {m_te['R2'] - paper['R2_test']:+.4f}")

    # Individual models
    print(f"\n  [Individual Models]")
    X_tr2, X_cal2, X_te2, y_tr2, y_cal2, y_te2 = split_data(sanitize(X_all), y_all)
    X_tr2, X_cal2, X_te2, sel2 = apply_var_filter(X_tr2, X_cal2, X_te2)
    for label, factory in [
        ("LightGBM",  lambda: lgb.LGBMRegressor(**lgbm_p)),
        ("XGBoost",   lambda: xgb.XGBRegressor(**xgb_p)),
        ("CatBoost",  lambda: CatBoostRegressor(**cat_p)),
        ("RandomForest", make_rf_regressor),
    ]:
        t0 = time.time()
        m = factory(); m.fit(X_tr2, y_tr2)
        mt = regression_metrics(y_te2, m.predict(X_te2))
        print(f"    {label:20s}  R²_te={mt['R2']:.4f}  RMSE={mt['RMSE']:.4f}  MAE={mt['MAE']:.4f}  [{time.time()-t0:.0f}s]")
        del m; gc.collect()

    # Save
    cal_residuals = np.abs(y_cal - model.predict(X_cal))
    joblib.dump(
        {"model": model, "selector": sel, "cal_residuals": cal_residuals,
         "features": FEATURES, "lgbm_params": lgbm_p, "xgb_params": xgb_p, "cat_params": cat_p},
        os.path.join(MODELS_DIR, "LogPapp_stacking_v3_fix.pkl"), compress=3
    )
    print(f"\n  Saved: saved_models/v3/LogPapp_stacking_v3_fix.pkl")

    # Update table2 CSV
    row = {
        "Endpoint": "LogPapp_fix", "Paper_Model": paper["model"],
        "Paper_R2_test": paper["R2_test"], "Paper_RMSE": paper["RMSE"], "Paper_MAE": paper["MAE"],
        "Ours_R2_train": m_tr["R2"], "Ours_R2_test": m_te["R2"],
        "Ours_RMSE_test": m_te["RMSE"], "Ours_MAE_test": m_te["MAE"],
        "Delta_R2_test": round(m_te["R2"] - paper["R2_test"], 4),
        "Delta_MAE_test": round(m_te["MAE"] - paper["MAE"], 4),
        "N_train": len(df), "Features": "+".join(FEATURES),
    }
    pd.DataFrame([row]).to_csv(
        f"{RESULTS_DIR}/table2_v3_papp_fix.csv", index=False)
    print(f"  Results saved to results/v3/table2_v3_papp_fix.csv")

    banner("Coverage (90% nominal)")
    cov90 = cov[cov["Nominal (%)"] == 90.0]
    print(cov90[["Nominal (%)", "Empirical (%)", "Δ (pp)", "Mean Width"]].to_string(index=False))


if __name__ == "__main__":
    main()
