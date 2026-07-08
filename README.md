# BCS-StackNet

**Stacking Ensemble Prediction of BCS Properties with Conformal Uncertainty Quantification**

Stalin Arulsamy, Rajesh Kumar M, Vanktesh Kumar — Lovely Professional University

---

## Overview

BCS-StackNet is a machine learning framework for in silico prediction of Biopharmaceutics Classification System (BCS) properties. It addresses key limitations of the FormulationBCS platform (Wu et al., Mol. Pharmaceutics 2025) by introducing:

- **Stacking ensemble** (LightGBM + XGBoost + CatBoost + Random Forest; ElasticNet meta-learner) with Bayesian hyperparameter optimisation (Optuna, 50 trials per model)
- **Split-conformal prediction intervals** with finite-sample coverage guarantees (70–95%)
- **Tanimoto nearest-neighbour applicability domain** analysis
- **Multi-label BCS classification** retaining all 294 drugs including ≈30% ambiguous dual-class compounds
- **SHAP TreeExplainer** feature importance for global and per-compound interpretation

## Performance (held-out test set)

| Endpoint | BCS-StackNet R² | FormulationBCS R² | Δ |
|----------|----------------|-------------------|---|
| Log S    | 0.8388 | 0.84 (LightGBM)  | ≈ tied |
| Log P    | 0.9525 | 0.96 (AttentiveFP) | −0.008 |
| Log D    | 0.7683 | 0.76 (AttentiveFP) | **+0.008 ✅** |
| Log Papp | 0.7470 | 0.71 (XGBoost)   | **+0.037 ✅** |

## Repository Structure

```
src/
  data_loading.py      # Dataset loading utilities
  featurizer.py        # Morgan ECFP + RDKit 2D + Mordred featurization
  models.py            # Stacking ensemble, classifiers
  evaluation.py        # Metrics, conformal prediction, applicability domain

pipeline_step0_fetch_data.py      # Data augmentation (ChEMBL)
pipeline_step1_features.py        # Precompute and cache all features
pipeline_step2_train_ablation.py  # Ablation training (v2)
pipeline_step2_v3.py              # Enhanced training with CatBoost + Optuna
pipeline_step2_v3_papp_fix.py     # LogPapp-specific fix (Wang 2016 only)
pipeline_step3_bcs_and_ad.py      # BCS classification + applicability domain

app/app.py           # Streamlit web application
results/             # Generated CSV tables and metrics
data/                # Curated datasets
```

## Datasets

| Endpoint | N | Source |
|----------|---|--------|
| Log S    | 19,528 | AqSolDB + Cui et al. 2020 |
| Log P    | 14,133 | OpenChem / PHYSPROP |
| Log D    | 4,200  | MoleculeNet (pH 7.4) |
| Log Papp | 2,337  | Wang et al. 2016 |
| BCS validation | 294 drugs | FDA/WHO biowaiver reports |

## Installation

```bash
pip install lightgbm xgboost catboost scikit-learn rdkit mordred optuna shap streamlit
```

## Usage

```bash
# Step 1: Precompute features (run once, ~5 min)
python pipeline_step1_features.py

# Step 2: Train models
python pipeline_step2_v3.py

# Step 3: BCS classification + applicability domain
python pipeline_step3_bcs_and_ad.py

# Launch web app
streamlit run app/app.py
```

## Citation

If you use BCS-StackNet, please cite:

> Arulsamy S, Kumar MR, Kumar V. BCS-StackNet: Stacking Ensemble Prediction of BCS Properties with Conformal Uncertainty Quantification. *[Journal]* 2026.

## License

MIT License
