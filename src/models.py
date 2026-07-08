"""
Model definitions: individual base learners + stacking ensemble.
Includes regressor and classifier variants.
"""

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingRegressor, StackingClassifier, ExtraTreesRegressor
from sklearn.linear_model import Ridge, LogisticRegression, ElasticNetCV
from sklearn.multioutput import MultiOutputClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


# ── Shared hyperparameter blocks ──────────────────────────────────────────────

_LGBM_REG_DEFAULTS = dict(
    n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    min_child_samples=10, random_state=42, n_jobs=-1, verbose=-1,
)

_XGB_REG_DEFAULTS = dict(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1, verbosity=0,
)

_RF_REG_DEFAULTS = dict(
    n_estimators=300, max_features="sqrt", min_samples_leaf=2,
    random_state=42, n_jobs=-1,
)

_LGBM_CLF_DEFAULTS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=6, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
    random_state=42, n_jobs=-1, verbose=-1,
)

_XGB_CLF_DEFAULTS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss",
)


_CATBOOST_REG_DEFAULTS = dict(
    iterations=500, learning_rate=0.05, depth=6,
    l2_leaf_reg=3.0, subsample=0.8, colsample_bylevel=0.8,
    random_seed=42, verbose=0, thread_count=-1,
    allow_writing_files=False,
)

# ── Individual regressors ─────────────────────────────────────────────────────

def make_lgbm_regressor(**kwargs) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**{**_LGBM_REG_DEFAULTS, **kwargs})


def make_xgb_regressor(**kwargs) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(**{**_XGB_REG_DEFAULTS, **kwargs})


def make_rf_regressor(**kwargs) -> RandomForestRegressor:
    return RandomForestRegressor(**{**_RF_REG_DEFAULTS, **kwargs})


def make_catboost_regressor(**kwargs) -> CatBoostRegressor:
    return CatBoostRegressor(**{**_CATBOOST_REG_DEFAULTS, **kwargs})


# ── Stacking regressor ────────────────────────────────────────────────────────

def make_stacking_regressor(**kwargs) -> StackingRegressor:
    estimators = [
        ("lgbm", make_lgbm_regressor()),
        ("xgb",  make_xgb_regressor()),
        ("rf",   make_rf_regressor()),
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5, passthrough=False, n_jobs=1,
    )


def make_stacking_regressor_v3(lgbm_params=None, xgb_params=None,
                                cat_params=None) -> StackingRegressor:
    """Enhanced stacking: LGBM + XGB + CatBoost + RF, ElasticNetCV meta-learner."""
    estimators = [
        ("lgbm", make_lgbm_regressor(**(lgbm_params or {}))),
        ("xgb",  make_xgb_regressor(**(xgb_params or {}))),
        ("cat",  make_catboost_regressor(**(cat_params or {}))),
        ("rf",   make_rf_regressor()),
    ]
    return StackingRegressor(
        estimators=estimators,
        final_estimator=ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 1.0], cv=5,
            max_iter=10000, random_state=42
        ),
        cv=5, passthrough=False, n_jobs=1,
    )


# ── Individual classifiers ────────────────────────────────────────────────────

def make_lgbm_classifier(**kwargs) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**{**_LGBM_CLF_DEFAULTS, **kwargs})


def make_xgb_classifier(**kwargs) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**{**_XGB_CLF_DEFAULTS, **kwargs})


def make_rf_classifier(**kwargs) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        max_features="sqrt", random_state=42, n_jobs=-1, **kwargs
    )


# ── Stacking classifier ───────────────────────────────────────────────────────

def make_stacking_classifier(**kwargs) -> StackingClassifier:
    estimators = [
        ("lgbm", make_lgbm_classifier()),
        ("xgb",  make_xgb_classifier()),
        ("rf",   make_rf_classifier()),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=1000, random_state=42
        ),
        cv=5, passthrough=False, n_jobs=1,
    )


# ── Multi-label classifier ────────────────────────────────────────────────────

def make_multilabel_classifier() -> MultiOutputClassifier:
    return MultiOutputClassifier(make_lgbm_classifier(n_estimators=200))


# ── Model registry for ablation ───────────────────────────────────────────────

REGRESSOR_CONFIGS = {
    "LightGBM":     make_lgbm_regressor,
    "XGBoost":      make_xgb_regressor,
    "RandomForest": make_rf_regressor,
    "Stacking":     make_stacking_regressor,
}
