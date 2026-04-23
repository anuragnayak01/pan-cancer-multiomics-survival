import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PATCH 1 — Ensembl ID → Gene Symbol mapping
# =============================================================================
def map_ensembl_to_symbol(mrna_df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Ensembl gene IDs (ENSG00000141510) to HGNC gene symbols (TP53).
    Required for ESTIMATE to find its signature genes.

    Strategy (tries in order, stops at first success):
      1. mygene.info REST API (pip install mygene, free, ~2 min for 12940 genes)
      2. biomart via pybiomart (pip install pybiomart, ~3 min)
      3. Local mapping file: 'ensembl_to_symbol.csv' with cols [ensembl_id, symbol]
         (download from: https://www.genenames.org/download/custom/)
      4. Heuristic: strip version suffix (ENSG00000141510.14 → ENSG00000141510),
         return as-is if already looks like a gene symbol

    Args:
        mrna_df: DataFrame with Ensembl IDs as column names

    Returns:
        mrna_df with columns renamed to gene symbols where mapping succeeded.
        Unmapped columns keep their original Ensembl ID.
    """
    cols = list(mrna_df.columns)

    # Check if already gene symbols (not Ensembl)
    n_ensembl = sum(1 for c in cols if str(c).startswith("ENSG"))
    if n_ensembl < len(cols) * 0.10:
        print(f"  [Ensembl map] Columns appear to already be gene symbols "
              f"({n_ensembl}/{len(cols)} Ensembl) — skipping mapping")
        return mrna_df

    print(f"  [Ensembl map] {n_ensembl}/{len(cols)} columns are Ensembl IDs")
    print(f"  [Ensembl map] Mapping to gene symbols for ESTIMATE purity scoring...")

    # Strip version suffix (ENSG00000141510.14 → ENSG00000141510)
    clean_ids = [str(c).split('.')[0] for c in cols]

    mapping = {}  # ensembl_id → symbol

    # ── Try 1: local mapping file ─────────────────────────────────────────────
    import os
    for path in ["ensembl_to_symbol.csv", "data/ensembl_to_symbol.csv",
                 "~/data/ensembl_to_symbol.csv"]:
        path = os.path.expanduser(path)
        if os.path.exists(path):
            try:
                df_map = pd.read_csv(path)
                # Auto-detect column names
                ens_col = next((c for c in df_map.columns
                                if 'ensembl' in c.lower() or 'ensg' in c.lower()), None)
                sym_col = next((c for c in df_map.columns
                                if 'symbol' in c.lower() or 'name' in c.lower()
                                or 'gene' in c.lower()), None)
                if ens_col and sym_col:
                    mapping = dict(zip(df_map[ens_col].astype(str),
                                       df_map[sym_col].astype(str)))
                    print(f"  [Ensembl map] Loaded {len(mapping)} mappings from {path}")
                    break
            except Exception as e:
                print(f"  [Ensembl map] Could not load {path}: {e}")

    # ── Try 2: mygene.info API ────────────────────────────────────────────────
    if not mapping:
        try:
            import mygene
            mg     = mygene.MyGeneInfo()
            # Query in batches of 1000
            batch_size = 1000
            print(f"  [Ensembl map] Querying mygene.info ({len(clean_ids)} genes, "
                  f"~{len(clean_ids)//batch_size + 1} batches)...", flush=True)
            for i in range(0, len(clean_ids), batch_size):
                batch  = clean_ids[i:i + batch_size]
                result = mg.querymany(batch, scopes='ensembl.gene',
                                      fields='symbol', species='human',
                                      verbose=False)
                for r in result:
                    if 'symbol' in r and 'query' in r:
                        mapping[r['query']] = r['symbol']
                if (i // batch_size + 1) % 5 == 0:
                    print(f"    {i + batch_size}/{len(clean_ids)} processed...",
                          flush=True)
            print(f"  [Ensembl map] mygene.info: {len(mapping)} genes mapped")
        except ImportError:
            print("  [Ensembl map] mygene not installed "
                  "(pip install mygene) — trying biomart...")
        except Exception as e:
            print(f"  [Ensembl map] mygene.info failed: {e}")

    # ── Try 3: pybiomart ─────────────────────────────────────────────────────
    if not mapping:
        try:
            from pybiomart import Server
            print("  [Ensembl map] Querying BioMart...", flush=True)
            server  = Server(host='http://www.ensembl.org')
            dataset = server.marts['ENSEMBL_MART_ENSEMBL'].datasets['hsapiens_gene_ensembl']
            result  = dataset.query(
                attributes=['ensembl_gene_id', 'hgnc_symbol'],
                filters={'ensembl_gene_id': clean_ids}
            )
            for _, row in result.iterrows():
                if pd.notna(row['HGNC symbol']) and row['HGNC symbol'] != '':
                    mapping[row['Gene stable ID']] = row['HGNC symbol']
            print(f"  [Ensembl map] BioMart: {len(mapping)} genes mapped")
        except ImportError:
            print("  [Ensembl map] pybiomart not installed "
                  "(pip install pybiomart) — using Ensembl IDs for ESTIMATE")
        except Exception as e:
            print(f"  [Ensembl map] BioMart failed: {e}")

    # ── Apply mapping ─────────────────────────────────────────────────────────
    if mapping:
        new_cols  = []
        n_mapped  = 0
        seen_syms = {}  # handle duplicate symbols
        for orig, clean in zip(cols, clean_ids):
            sym = mapping.get(clean)
            if sym and sym not in ('nan', '', 'None'):
                # Handle duplicate symbols by appending suffix
                if sym in seen_syms:
                    seen_syms[sym] += 1
                    sym = f"{sym}.{seen_syms[sym]}"
                else:
                    seen_syms[sym] = 0
                new_cols.append(sym)
                n_mapped += 1
            else:
                new_cols.append(orig)  # keep original Ensembl ID

        mrna_df.columns = new_cols
        pct = 100 * n_mapped / len(cols)
        print(f"  [Ensembl map] ✅ Mapped {n_mapped}/{len(cols)} ({pct:.1f}%) "
              f"Ensembl IDs → gene symbols")
        if pct < 50:
            print(f"  [Ensembl map] ⚠️  < 50% mapping rate. "
                  f"ESTIMATE purity will use {n_mapped} genes.")
    else:
        print("  [Ensembl map] ⚠️  No mapping source available. "
              "ESTIMATE will use fallback purity = 0.80.")
        print("  To fix: run  pip install mygene  then re-run Phase 1.")
        print("  Or download ensembl_to_symbol.csv from:")
        print("    https://www.genenames.org/download/custom/")
        print("    (fields: Ensembl gene ID + Approved symbol, human only)")

    return mrna_df


# =============================================================================
# PATCH 2 — Merge small batch groups for ComBat stability
# =============================================================================
def merge_small_batches(batch_series: pd.Series,
                         min_batch_size: int = 10,
                         other_label: str = "OTHER") -> pd.Series:
    """
    Merge batch groups with < min_batch_size patients into a single 'OTHER' group.

    Why: ComBat requires at least 2 patients per batch for parameter estimation,
    and becomes numerically unstable with many single-patient batches.
    625 TSS groups with ~11 patients each on average means many groups have
    1–3 patients, which will cause ComBat to fail or produce garbage corrections.

    Strategy:
      1. Count patients per batch group
      2. Identify batches with < min_batch_size patients
      3. Relabel those patients as 'OTHER'
      4. Result: ~20–30 stable batch groups with ≥ 10 patients each

    Args:
        batch_series  : pd.Series from extract_batch_labels()
        min_batch_size: minimum patients per batch group (default: 10)
        other_label   : label for merged small batches

    Returns:
        pd.Series with merged batch labels
    """
    counts   = batch_series.value_counts()
    small    = counts[counts < min_batch_size].index
    n_small  = len(small)
    n_patients_affected = counts[small].sum()

    if n_small == 0:
        print(f"  [Batch merge] All {counts.nunique()} groups have ≥ {min_batch_size} "
              f"patients — no merging needed")
        return batch_series

    merged = batch_series.copy()
    merged[batch_series.isin(small)] = other_label

    n_after = merged.nunique()
    print(f"  [Batch merge] {n_small} small batches (< {min_batch_size} patients) "
          f"merged into '{other_label}'")
    print(f"  [Batch merge] Patients affected: {n_patients_affected} "
          f"({100*n_patients_affected/len(batch_series):.1f}%)")
    print(f"  [Batch merge] Batch groups: {len(counts)} → {n_after}")
    print(f"  [Batch merge] Group size range after merge: "
          f"{merged.value_counts().min()} – {merged.value_counts().max()}")

    if n_after > 50:
        print(f"  [Batch merge] ⚠️  Still {n_after} batches. "
              f"ComBat may be slow. Consider increasing min_batch_size.")
    elif n_after < 2:
        print(f"  [Batch merge] ⚠️  Only {n_after} batch. "
              f"ComBat will be skipped in Phase 2 (needs ≥ 2).")
    else:
        print(f"  [Batch merge] ✅ {n_after} stable batches ready for ComBat")

    return merged


# =============================================================================
# PATCH 3 — Hard cap rounding fix
# =============================================================================
def fixed_max_remove(n_patients: int, contamination: float = 0.01) -> int:
    """
    Use ceiling instead of floor so cap = ceil(n * contamination).
    For n=7038, contamination=0.01:
      int(7038 * 0.01) = int(70.38) = 70  (floor, old behavior)
      ceil(7038 * 0.01) = ceil(70.38) = 71 (correct)
    """
    return int(np.ceil(n_patients * contamination))


# =============================================================================
# INTEGRATION FUNCTION
# Call this in run_phase1() to apply all three patches
# =============================================================================
def apply_patches(data: dict, patients: list,
                  batch_labels: pd.Series) -> tuple:
    """
    Apply all three patches in sequence.

    Usage in run_phase1():
        # After load + intersection + survival filter + outlier removal:
        data["mrna"], batch_labels_fixed, purity_scores = apply_patches(
            data, valid, batch_labels)

    Returns:
        (data_dict_with_mapped_mrna, merged_batch_labels, purity_scores)
    """
    print("\n" + "─" * 55)
    print("  Applying Phase 1 patches...")
    print("─" * 55)

    # Patch 1: Map Ensembl IDs → gene symbols (enables ESTIMATE)
    data_patched = dict(data)
    data_patched["mrna"] = map_ensembl_to_symbol(data["mrna"].copy())

    # Patch 2: Merge small batch groups (enables ComBat)
    batch_merged = merge_small_batches(batch_labels, min_batch_size=10)

    # Now compute purity with gene-symbol-mapped mRNA
    # (imported from phase1_qc to avoid circular imports)
    try:
        from phase1_qc import compute_purity_scores
        purity_scores = compute_purity_scores(data_patched["mrna"])
        purity_scores = purity_scores.reindex(patients)
    except Exception as e:
        print(f"  ⚠️  Purity recomputation failed ({e}) — using 0.80 fallback")
        purity_scores = pd.Series(0.80, index=patients)

    return data_patched, batch_merged, purity_scores


# =============================================================================
# STANDALONE: patch just the mRNA gene names in an existing parquet
# Run once to create a new mRNA.parquet with gene symbols
# =============================================================================
def patch_mrna_parquet(input_path: str = "mRNA.parquet",
                        output_path: str = "mRNA_symbols.parquet"):
    """
    One-time utility: load mRNA.parquet, map Ensembl IDs to gene symbols,
    save as a new parquet file. Run once, then update FILES dict to use
    'mRNA_symbols.parquet' instead.

    Usage:
        python -c "from phase1_patches import patch_mrna_parquet; patch_mrna_parquet()"
    """
    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path).astype(np.float32)
    print(f"  Shape: {df.shape}")

    print("Mapping Ensembl IDs → gene symbols...")
    df_mapped = map_ensembl_to_symbol(df)

    print(f"Saving to {output_path}...")
    df_mapped.to_parquet(output_path)
    print(f"✅ Saved {output_path}  ({df_mapped.shape})")
    print(f"\nNext step: in phase1_qc.py, change:")
    print(f'    "mrna": "mRNA.parquet"')
    print(f'  to:')
    print(f'    "mrna": "mRNA_symbols.parquet"')


if __name__ == "__main__":
    # Quick test of batch merging logic
    import pandas as pd
    import numpy as np
    np.random.seed(42)

    # Simulate 625 TSS batches with ~11 patients each
    n_patients = 6968
    batches = np.random.choice([f"TSS_{i:03d}" for i in range(625)],
                                size=n_patients, replace=True)
    test_series = pd.Series(batches, index=[f"TCGA-{i:05d}" for i in range(n_patients)])

    print(f"Before: {test_series.nunique()} batch groups")
    print(f"Size range: {test_series.value_counts().min()} – "
          f"{test_series.value_counts().max()}")

    merged = merge_small_batches(test_series, min_batch_size=10)

    print(f"\nAfter merging (min_size=10): {merged.nunique()} batch groups")
    print(f"Size range: {merged.value_counts().min()} – "
          f"{merged.value_counts().max()}")

    # Test cap rounding
    n = 7038
    old_cap = int(n * 0.01)
    new_cap = fixed_max_remove(n)
    print(f"\nCap rounding: int({n}*0.01)={old_cap}  ceil({n}*0.01)={new_cap}")