"""
Molecular feature computation: Morgan ECFP, RDKit 2D, Mordred.
Each descriptor family is computed and cached separately to support ablation.
"""

import os
import warnings
import hashlib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs, MACCSkeys
from sklearn.feature_selection import VarianceThreshold

warnings.filterwarnings("ignore")

CACHE_DIR = "/home/stalin/Desktop/BCS/cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Molecular validation ──────────────────────────────────────────────────────

def smiles_to_mols(smiles_list: list[str]) -> tuple[list, list[int]]:
    mols, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        try:
            m = Chem.MolFromSmiles(str(smi))
            if m is not None and m.GetNumAtoms() > 0:
                mols.append(m)
                valid_idx.append(i)
        except Exception:
            pass
    return mols, valid_idx


# ── Morgan ECFP ───────────────────────────────────────────────────────────────

def compute_morgan(mols: list, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
        def _fp(m):
            return gen.GetFingerprint(m)
    except ImportError:
        def _fp(m):
            return AllChem.GetMorganFingerprintAsBitVect(m, radius, n_bits)

    X = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        arr = np.zeros(n_bits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(_fp(m), arr)
        X[i] = arr
    return X


# ── RDKit 2D Descriptors ──────────────────────────────────────────────────────

def compute_rdkit_2d(mols: list) -> tuple[np.ndarray, list[str]]:
    desc_fns = [(name, fn) for name, fn in Descriptors.descList]
    names = [n for n, _ in desc_fns]
    X = np.zeros((len(mols), len(desc_fns)), dtype=np.float32)
    for i, mol in enumerate(mols):
        for j, (_, fn) in enumerate(desc_fns):
            try:
                v = float(fn(mol))
                X[i, j] = 0.0 if (np.isnan(v) or np.isinf(v)) else v
            except Exception:
                X[i, j] = 0.0
    return X, names


# ── MACCS Keys (167-bit) ─────────────────────────────────────────────────────

def compute_maccs(mols: list) -> tuple[np.ndarray, list[str]]:
    X = np.zeros((len(mols), 167), dtype=np.float32)
    for i, mol in enumerate(mols):
        arr = np.zeros(167, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr)
        X[i] = arr
    names = [f"MACCS_{i}" for i in range(167)]
    return X, names


# ── Mordred Descriptors ───────────────────────────────────────────────────────

def compute_mordred(mols: list) -> tuple[np.ndarray, list[str]]:
    try:
        from mordred import Calculator, descriptors as mordred_descs
    except ImportError:
        try:
            from mordredcommunity import Calculator, descriptors as mordred_descs
        except ImportError:
            raise ImportError("mordred or mordredcommunity must be installed.")

    calc = Calculator(mordred_descs, ignore_3D=True)
    df = calc.pandas(mols, quiet=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    names = df.columns.tolist()
    X = df.values.astype(np.float64)
    X = np.where(np.isfinite(X), X, 0.0)
    X = np.clip(X, -1e9, 1e9).astype(np.float32)
    return X, names


def compute_mordred_3d(mols: list) -> tuple[np.ndarray, list[str]]:
    """Mordred with 3D descriptors — generates ETKDG conformers first."""
    try:
        from mordred import Calculator, descriptors as mordred_descs
    except ImportError:
        from mordredcommunity import Calculator, descriptors as mordred_descs

    mols_3d = []
    for mol in mols:
        try:
            mol_h = Chem.AddHs(mol)
            ok = AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
            if ok == 0:
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
                mols_3d.append(mol_h)
            else:
                # Fallback: use 2D mol (3D descriptors will be NaN → 0)
                mols_3d.append(mol)
        except Exception:
            mols_3d.append(mol)

    calc = Calculator(mordred_descs, ignore_3D=False)
    df = calc.pandas(mols_3d, quiet=True)
    df = df.apply(pd.to_numeric, errors="coerce")
    names = df.columns.tolist()
    X = df.values.astype(np.float64)
    X = np.where(np.isfinite(X), X, 0.0)
    X = np.clip(X, -1e9, 1e9).astype(np.float32)
    return X, names


# ── Variance + Missing-value filter ──────────────────────────────────────────

def filter_features(X: np.ndarray, missing_thresh: float = 0.10,
                    var_thresh: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    # Missing-value filter (treat 0 as "missing" only if originally NaN-imputed;
    # here we check for constant-zero columns as surrogate for bad features)
    nan_frac = np.mean(X == 0.0, axis=0)  # conservative: remove all-zero features
    mask_missing = nan_frac < 1.0          # keep any column with at least one nonzero
    X = X[:, mask_missing]

    # Zero-variance filter
    selector = VarianceThreshold(threshold=var_thresh)
    try:
        X = selector.fit_transform(X)
    except Exception:
        pass
    return X, selector.get_support() if hasattr(selector, 'get_support') else np.ones(X.shape[1], dtype=bool)


# ── Cache utilities ───────────────────────────────────────────────────────────

def _cache_key(smiles_list: list[str], desc_type: str) -> str:
    raw = "".join(smiles_list).encode()
    h = hashlib.md5(raw).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"{desc_type}_{h}.npz")


def _save_cache(path: str, X: np.ndarray, names: list[str]):
    np.savez_compressed(path, X=X, names=np.array(names, dtype=object))


def _load_cache(path: str) -> tuple[np.ndarray, list[str]] | None:
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=True)
        return d["X"], d["names"].tolist()
    except Exception:
        return None


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_features(smiles_list: list[str],
                     descriptor_types: list[str] = ("morgan", "rdkit", "mordred"),
                     n_bits: int = 2048,
                     radius: int = 2,
                     use_cache: bool = True
                     ) -> tuple[dict[str, np.ndarray], dict[str, list[str]], list[int]]:
    """
    Compute molecular descriptors for a list of SMILES.

    Returns:
        feature_arrays : dict[desc_type -> np.ndarray shape (N_valid, D)]
        feature_names  : dict[desc_type -> list[str]]
        valid_idx      : list of indices into smiles_list for which mol was valid
    """
    mols, valid_idx = smiles_to_mols(smiles_list)
    if not mols:
        return {}, {}, []

    feature_arrays: dict[str, np.ndarray] = {}
    feature_names: dict[str, list[str]] = {}

    for dtype in descriptor_types:
        cache_path = _cache_key([smiles_list[i] for i in valid_idx], dtype) if use_cache else None

        cached = _load_cache(cache_path) if cache_path else None
        if cached is not None:
            X, names = cached
        else:
            if dtype == "morgan":
                X = compute_morgan(mols, n_bits=n_bits, radius=radius)
                names = [f"Morgan_{i}" for i in range(n_bits)]
            elif dtype == "rdkit":
                X, names = compute_rdkit_2d(mols)
            elif dtype == "mordred":
                X, names = compute_mordred(mols)
            elif dtype == "maccs":
                X, names = compute_maccs(mols)
            elif dtype == "mordred_3d":
                X, names = compute_mordred_3d(mols)
            else:
                raise ValueError(f"Unknown descriptor type: {dtype}")

            if cache_path:
                _save_cache(cache_path, X, names)

        feature_arrays[dtype] = X
        feature_names[dtype] = names

    return feature_arrays, feature_names, valid_idx


def assemble(feature_arrays: dict[str, np.ndarray],
             descriptor_types: list[str]) -> np.ndarray:
    """Horizontally concatenate the requested descriptor arrays."""
    return np.hstack([feature_arrays[dt] for dt in descriptor_types if dt in feature_arrays])


# ── Descriptor set registry for ablation ─────────────────────────────────────

DESCRIPTOR_SETS = {
    "Morgan":           ["morgan"],
    "RDKit-2D":         ["rdkit"],
    "Mordred":          ["mordred"],
    "Morgan+RDKit":     ["morgan", "rdkit"],
    "Morgan+Mordred":   ["morgan", "mordred"],
    "RDKit+Mordred":    ["rdkit", "mordred"],
    "All (M+R+Mo)":     ["morgan", "rdkit", "mordred"],
}
