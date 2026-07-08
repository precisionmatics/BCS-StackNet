"""
Step 1: Precompute and cache all molecular features for every endpoint.
Must be run before any training or ablation step.
Caches Morgan, RDKit-2D, and Mordred arrays to disk so they are
computed only once (Mordred is expensive: ~10-30 min per large dataset).
"""

import os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loading import load_endpoint, load_bcs_drugs, ENDPOINTS, BCS_PATH
from src.featurizer import compute_features

DESC_TYPES = ["morgan", "rdkit", "mordred"]


def banner(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}")


def precompute(name, smiles_list):
    banner(f"Featurizing: {name}  (N={len(smiles_list)})")
    for dt in DESC_TYPES:
        t0 = time.time()
        feat, feat_names, valid_idx = compute_features(
            smiles_list, descriptor_types=[dt], use_cache=True
        )
        elapsed = time.time() - t0
        print(f"  [{dt:10s}]  shape={feat[dt].shape}  "
              f"valid={len(valid_idx)}/{len(smiles_list)}  "
              f"time={elapsed:.1f}s  (cached if <1s)")
    return valid_idx


if __name__ == "__main__":
    # ── Property endpoints ────────────────────────────────────────────────────
    valid_idx_map = {}
    for ep_name, (path, smiles_col, target_col) in ENDPOINTS.items():
        df = load_endpoint(path, smiles_col, target_col)
        smiles = df["smiles_clean"].tolist()
        valid_idx_map[ep_name] = precompute(ep_name, smiles)

    # ── BCS validation set ────────────────────────────────────────────────────
    bcs_df = load_bcs_drugs(BCS_PATH)
    smiles_bcs = bcs_df["smiles_clean"].tolist()
    valid_idx_map["BCS"] = precompute("BCS_drugs", smiles_bcs)

    # ── Summary ──────────────────────────────────────────────────────────────
    banner("Done — all features cached")
    for k, idx in valid_idx_map.items():
        print(f"  {k:12s}: {len(idx)} valid molecules")
    print(f"\nCache directory: /home/stalin/Desktop/BCS/cache/")
    print(f"Cache files:     {len(os.listdir('/home/stalin/Desktop/BCS/cache/'))} files")
