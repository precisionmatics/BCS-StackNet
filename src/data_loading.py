"""Data loading and SMILES preprocessing utilities."""

import warnings
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover

RDLogger.DisableLog("rdApp.*")

warnings.filterwarnings("ignore")
_remover = SaltRemover.SaltRemover()


def _canonicalize(smi: str) -> str | None:
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        mol = _remover.StripMol(mol, dontRemoveEverything=True)
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        # Reject inorganics (no carbon atoms)
        if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def load_endpoint(path: str, smiles_col: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={smiles_col: "smiles_raw", target_col: "target"})
    df = df[["smiles_raw", "target"]].dropna()
    df["smiles_clean"] = df["smiles_raw"].map(_canonicalize)
    df = df.dropna(subset=["smiles_clean", "target"]).reset_index(drop=True)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    # Remove exact duplicates — average conflicting measurements
    df = df.groupby("smiles_clean", as_index=False)["target"].mean()
    return df


def load_bcs_drugs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"Drug_Name": "name", "SMILES": "smiles_raw",
                             "BCS_category": "bcs_primary"})
    df["smiles_clean"] = df["smiles_raw"].map(_canonicalize)
    df = df.dropna(subset=["smiles_clean"]).reset_index(drop=True)

    # Parse primary class (may be "1", "1/3", "2/4", etc.)
    def _parse_primary(v):
        s = str(v).strip()
        parts = [p.strip() for p in s.replace("/", ",").split(",")]
        try:
            return int(float(parts[0]))
        except Exception:
            return np.nan

    def _is_ambiguous(v):
        s = str(v).strip()
        return "," in s or "/" in s

    def _multilabel(v):
        s = str(v).strip()
        vec = [0, 0, 0, 0]
        for part in s.replace("/", ",").split(","):
            try:
                idx = int(float(part.strip())) - 1
                if 0 <= idx < 4:
                    vec[idx] = 1
            except Exception:
                pass
        return np.array(vec, dtype=np.int8)

    # Re-read raw to get original category strings
    raw_df = pd.read_csv(path)
    raw_df = raw_df.rename(columns={"BCS_category": "bcs_raw"})
    raw_df["smiles_clean"] = raw_df["SMILES"].map(_canonicalize)
    raw_df = raw_df.dropna(subset=["smiles_clean"]).reset_index(drop=True)

    ambig_map     = dict(zip(raw_df["smiles_clean"], raw_df["bcs_raw"].map(_is_ambiguous)))
    multilabel_map = dict(zip(raw_df["smiles_clean"], raw_df["bcs_raw"].map(_multilabel)))
    primary_map   = dict(zip(raw_df["smiles_clean"], raw_df["bcs_raw"].map(_parse_primary)))

    df["bcs_primary"]   = df["smiles_clean"].map(lambda s: primary_map.get(s, np.nan))
    df["is_ambiguous"]  = df["smiles_clean"].map(lambda s: ambig_map.get(s, False))
    df["bcs_multilabel"] = df["smiles_clean"].map(
        lambda s: multilabel_map.get(s, np.array([0, 0, 0, 0], dtype=np.int8))
    )
    return df


# ── Endpoint registry ────────────────────────────────────────────────────────
DATA_DIR = "/home/stalin/Desktop/BCS/OLD/data/raw"

ENDPOINTS = {
    "LogS":    (f"{DATA_DIR}/Log_S.csv",    "SMILES",  "Log S"),
    "LogP":    (f"{DATA_DIR}/Log_P.csv",    "SMILES",  "Log P"),
    "LogD":    (f"{DATA_DIR}/Log_D.csv",    "SMILES",  "Log D"),
    "LogPapp": (f"{DATA_DIR}/Log_Papp.csv", "SMILES",  "Log  Papp"),
}

BCS_PATH = f"{DATA_DIR}/bcs_drugs.csv"

PAPER_BENCHMARKS = {
    "LogS":    {"model": "LightGBM",    "R2_test": 0.84, "RMSE": 0.88, "MAE": 0.59},
    "LogP":    {"model": "AttentiveFP", "R2_test": 0.96, "RMSE": 0.36, "MAE": 0.25},
    "LogD":    {"model": "AttentiveFP", "R2_test": 0.76, "RMSE": 0.60, "MAE": 0.43},
    "LogPapp": {"model": "XGBoost",     "R2_test": 0.71, "RMSE": 0.42, "MAE": 0.33},
}
