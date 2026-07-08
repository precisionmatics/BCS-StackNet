"""
Step 0 (run once): Expand training datasets from public sources.

LogPapp: Wang 2016 (2,337) + ChEMBL Caco-2 → data/augmented/Log_Papp_expanded.csv
LogD   : MoleculeNet (4,200) + nanxstats/logd74 (1,130) → data/augmented/Log_D_expanded.csv
"""

import os, warnings, requests, io
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToSmiles, MolFromSmiles

warnings.filterwarnings("ignore")

OUT_DIR  = "/home/stalin/Desktop/BCS/data/augmented"
DATA_DIR = "/home/stalin/Desktop/BCS/OLD/data/raw"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Utilities ─────────────────────────────────────────────────────────────────

def canonical(smi):
    try:
        m = MolFromSmiles(str(smi))
        return MolToSmiles(m) if m else None
    except Exception:
        return None


def banner(msg):
    print(f"\n{'='*65}\n  {msg}\n{'='*65}")


# ── Unit conversion for Papp ──────────────────────────────────────────────────

# Map ChEMBL unit strings → multiply by this factor to get cm/s
_PAPP_UNIT_FACTORS = {
    "10'-6 cm/s":   1e-6,
    "10^-6 cm/s":   1e-6,
    "10'6cm/s":     1e-6,
    "10^6cm/s":     1e-6,
    "cm/s * 10e6":  1e-6,
    "ucm/s":        1e-6,
    "nm/s":         1e-7,   # 1 nm/s = 10^-7 cm/s
    "cm s-1":       1.0,
    "cm/s":         1.0,
}

def papp_to_log_cms(value, units):
    if units is None or value is None:
        return None
    try:
        v = float(value)
        if v <= 0:
            return None
        u = str(units).strip()
        factor = _PAPP_UNIT_FACTORS.get(u)
        if factor is None:
            return None
        log_v = np.log10(v * factor)
        # Valid range: -8 to -3 (10^-8 to 10^-3 cm/s)
        if -8.0 <= log_v <= -3.0:
            return log_v
        return None
    except Exception:
        return None


# ── Fetch ChEMBL Caco-2 ───────────────────────────────────────────────────────

def fetch_chembl_caco2():
    banner("Fetching Caco-2 Papp from ChEMBL")
    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        print("  chembl_webresource_client not installed — skipping")
        return pd.DataFrame()

    act = new_client.activity
    res = act.filter(
        standard_type="Papp",
        assay_type="A",
    ).only(["molecule_chembl_id", "standard_value", "standard_units",
            "canonical_smiles", "standard_relation"])

    rows = []
    for r in res:
        # Only exact measurements (=), not > or <
        if r.get("standard_relation") not in ["=", None, "'"]:
            continue
        smi  = r.get("canonical_smiles")
        val  = r.get("standard_value")
        unit = r.get("standard_units")
        if not smi or not val:
            continue
        log_papp = papp_to_log_cms(val, unit)
        if log_papp is None:
            continue
        canon = canonical(smi)
        if canon:
            rows.append({"SMILES": canon, "Log  Papp": round(log_papp, 4)})

    df = pd.DataFrame(rows)
    print(f"  Raw ChEMBL records: {len(df)}")

    if df.empty:
        return df

    # Deduplicate: keep median if multiple measurements per compound
    df = df.groupby("SMILES")["Log  Papp"].agg(
        lambda x: round(np.median(x), 4)
    ).reset_index()
    # Remove high-variance compounds (SD > 0.5 log units if multiple entries)
    print(f"  After dedup: {len(df)} unique compounds")
    return df


def merge_papp(existing_path, chembl_df):
    banner("Merging LogPapp datasets")
    base = pd.read_csv(existing_path)
    base["SMILES"] = base["SMILES"].apply(canonical)
    base = base.dropna(subset=["SMILES"])
    print(f"  Wang 2016 (existing): {len(base)}")

    if chembl_df.empty:
        print("  No ChEMBL data — keeping original")
        out_path = os.path.join(OUT_DIR, "Log_Papp_expanded.csv")
        base.to_csv(out_path, index=False)
        return out_path

    # Remove ChEMBL duplicates already in Wang 2016
    existing_smiles = set(base["SMILES"])
    new_rows = chembl_df[~chembl_df["SMILES"].isin(existing_smiles)]
    print(f"  New from ChEMBL (after dedup): {len(new_rows)}")

    merged = pd.concat([base, new_rows], ignore_index=True)
    merged = merged.dropna(subset=["SMILES", "Log  Papp"])
    print(f"  Total merged: {len(merged)}")

    out_path = os.path.join(OUT_DIR, "Log_Papp_expanded.csv")
    merged.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return out_path


# ── Fetch nanxstats LogD ──────────────────────────────────────────────────────

def fetch_logd_github():
    banner("Fetching LogD7.4 from nanxstats/logd74 (GitHub)")
    url = "https://raw.githubusercontent.com/nanxstats/logd74/main/data-raw/logd74.csv"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        print(f"  Downloaded {len(df)} rows, columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"  Failed to fetch LogD data: {e}")
        return pd.DataFrame()


def merge_logd(existing_path, github_df):
    banner("Merging LogD datasets")
    base = pd.read_csv(existing_path)
    base["SMILES"] = base["SMILES"].apply(canonical)
    base = base.dropna(subset=["SMILES"])
    print(f"  MoleculeNet (existing): {len(base)}")

    if github_df.empty:
        print("  No GitHub data — keeping original")
        out_path = os.path.join(OUT_DIR, "Log_D_expanded.csv")
        base.to_csv(out_path, index=False)
        return out_path

    # Find the SMILES and logD columns
    smi_col = next((c for c in github_df.columns if "smil" in c.lower()), None)
    logd_col = next((c for c in github_df.columns if "logd" in c.lower() or "log_d" in c.lower() or c.lower() in ["logd", "logd7", "logd74"]), None)

    if not smi_col or not logd_col:
        print(f"  Could not identify SMILES/LogD columns: {github_df.columns.tolist()}")
        out_path = os.path.join(OUT_DIR, "Log_D_expanded.csv")
        base.to_csv(out_path, index=False)
        return out_path

    print(f"  Using columns: SMILES={smi_col}, LogD={logd_col}")
    github_df = github_df[[smi_col, logd_col]].copy()
    github_df.columns = ["SMILES", "Log D"]
    github_df["SMILES"] = github_df["SMILES"].apply(canonical)
    github_df = github_df.dropna(subset=["SMILES", "Log D"])
    github_df["Log D"] = pd.to_numeric(github_df["Log D"], errors="coerce")
    github_df = github_df.dropna(subset=["Log D"])

    existing_smiles = set(base["SMILES"])
    new_rows = github_df[~github_df["SMILES"].isin(existing_smiles)]
    print(f"  New from GitHub (after dedup): {len(new_rows)}")

    merged = pd.concat([base, new_rows], ignore_index=True)
    merged = merged.dropna(subset=["SMILES", "Log D"])
    print(f"  Total merged: {len(merged)}")

    out_path = os.path.join(OUT_DIR, "Log_D_expanded.csv")
    merged.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # LogPapp
    chembl_df = fetch_chembl_caco2()
    papp_path = merge_papp(
        os.path.join(DATA_DIR, "Log_Papp.csv"), chembl_df
    )

    # LogD
    logd_github = fetch_logd_github()
    logd_path = merge_logd(
        os.path.join(DATA_DIR, "Log_D.csv"), logd_github
    )

    banner("Summary")
    if os.path.exists(papp_path):
        n = len(pd.read_csv(papp_path))
        print(f"  LogPapp expanded: {n} compounds → {papp_path}")
    if os.path.exists(logd_path):
        n = len(pd.read_csv(logd_path))
        print(f"  LogD expanded:    {n} compounds → {logd_path}")
    print("\n  Ready for pipeline_step2_v3.py")


if __name__ == "__main__":
    main()
