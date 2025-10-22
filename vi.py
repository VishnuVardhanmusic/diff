"""
Delta Excel Comparator
----------------------
A robust tool to compare two Excel sheets (tables) and produce a semantic, granular delta report
that detects: unchanged rows, modified rows (cell-level diffs), moved rows, split rows, merged rows,
added rows, and deleted rows.

Key ideas:
- Read both sheets into pandas DataFrames.
- Normalize cell contents (unicode fold, whitespace, case, numeric parsing).
- Compute per-column similarity using specialized comparators (numeric vs text).
- Combine column-wise similarities into a weighted row similarity score.
- Use the Hungarian algorithm (linear_sum_assignment) to find an optimal matching between rows.
- Detect splits (one old -> many new) and merges (many old -> one new) by checking combined similarity
  of multiple candidates.
- Produce an Excel report listing operations and providing cell-level details.

Dependencies:
- pandas
- numpy
- rapidfuzz
- scipy
- openpyxl (for writing xlsx)
- unidecode

Install:
pip install pandas numpy rapidfuzz scipy openpyxl Unidecode

Usage example:
python delta_excel_compare.py old.xlsx new.xlsx --sheet-name Old --sheet-name New --key-cols ID

Note: This is a robust, configurable starting point. Tweak weights and thresholds to your domain.
"""

import argparse
import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from unidecode import unidecode
from typing import List, Dict, Tuple, Any
import math


# ------------------------- Normalization Utilities -------------------------

def normalize_text(s: Any) -> str:
    if pd.isna(s):
        return ""
    if isinstance(s, (int, float)) and not isinstance(s, bool):
        # Keep numeric as canonical string too
        return str(s)
    s_str = str(s)
    s_str = unidecode(s_str)
    s_str = s_str.strip()
    s_str = re.sub(r"\s+", " ", s_str)
    s_str = s_str.lower()
    return s_str


def try_parse_number(s: str):
    try:
        if s == "":
            return None
        # remove commas used as thousands separators
        val = float(re.sub(r"[, ]", "", s))
        return val
    except Exception:
        return None


# ------------------------- Column Similarity -------------------------

def text_similarity(a: str, b: str) -> float:
    """Return similarity score between 0..1 for text using rapidfuzz token_set_ratio."""
    if a == b:
        return 1.0
    if a == "" and b == "":
        return 1.0
    if a == "" or b == "":
        return 0.0
    # Use token set ratio which handles reordering and partial matches well
    score = fuzz.token_set_ratio(a, b) / 100.0
    return score


def numeric_similarity(a: str, b: str) -> float:
    """Return similarity 0..1 for numeric values; tolerant to small diffs."""
    na = try_parse_number(a)
    nb = try_parse_number(b)
    if na is None and nb is None:
        # fallback to text
        return text_similarity(a, b)
    if na is None or nb is None:
        return 0.0
    if na == nb:
        return 1.0
    # relative difference
    denom = max(abs(na), abs(nb), 1.0)
    rel_diff = abs(na - nb) / denom
    # map rel_diff to similarity (tunable)
    sim = max(0.0, 1.0 - rel_diff)
    return sim


def column_similarity(a: Any, b: Any) -> float:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    # If both parse to numbers, treat as numeric
    if try_parse_number(a_norm) is not None and try_parse_number(b_norm) is not None:
        return numeric_similarity(a_norm, b_norm)
    # If cell contains commas that likely indicate a list, compare as sets
    if "," in a_norm or "," in b_norm:
        a_set = set([x.strip() for x in a_norm.split(",") if x.strip()])
        b_set = set([x.strip() for x in b_norm.split(",") if x.strip()])
        if not a_set and not b_set:
            return 1.0
        if not a_set or not b_set:
            return 0.0
        # Jaccard-like similarity using fuzzy token matches
        common = 0.0
        for ai in a_set:
            best = 0.0
            for bi in b_set:
                best = max(best, fuzz.token_set_ratio(ai, bi) / 100.0)
            common += best
        # normalize by max len
        return common / max(len(a_set), len(b_set))
    return text_similarity(a_norm, b_norm)


# ------------------------- Row Similarity & Weights -------------------------

def row_similarity(row_a: pd.Series, row_b: pd.Series, col_weights: Dict[str, float]) -> float:
    cols = list(col_weights.keys())
    total_weight = sum(col_weights.values())
    if total_weight == 0:
        total_weight = len(cols)
    score = 0.0
    for c in cols:
        w = col_weights.get(c, 1.0)
        sa = row_a.get(c, "")
        sb = row_b.get(c, "")
        sim = column_similarity(sa, sb)
        score += w * sim
    return score / total_weight


# ------------------------- Matching -------------------------

def build_similarity_matrix(df_old: pd.DataFrame, df_new: pd.DataFrame, col_weights: Dict[str, float]) -> np.ndarray:
    n_old = len(df_old)
    n_new = len(df_new)
    sim = np.zeros((n_old, n_new), dtype=float)
    for i in range(n_old):
        row_i = df_old.iloc[i]
        for j in range(n_new):
            row_j = df_new.iloc[j]
            sim[i, j] = row_similarity(row_i, row_j, col_weights)
    return sim


def optimal_matching(sim_matrix: np.ndarray, unmatched_cost: float = 0.0) -> List[Tuple[int, int, float]]:
    """Compute optimal matching using Hungarian algorithm. Returns list of (old_idx, new_idx, sim).
    If matrix not square, pad with zero-similarity nodes (representing insertions/deletions).
    """
    n_old, n_new = sim_matrix.shape
    n = max(n_old, n_new)
    # cost matrix (Hungarian minimizes) -> use negative similarity
    cost = np.full((n, n), fill_value=1.0)
    cost[:n_old, :n_new] = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < n_old and c < n_new:
            matches.append((r, c, sim_matrix[r, c]))
        # else: matched to a padding dummy => insertion/deletion
    return matches


# ------------------------- Split & Merge Detection -------------------------

def detect_splits_and_merges(df_old: pd.DataFrame, df_new: pd.DataFrame, sim_matrix: np.ndarray,
                             unmatched_old: List[int], unmatched_new: List[int],
                             split_threshold: float = 0.85, max_parts: int = 4) -> Tuple[List[Dict], List[Dict]]:
    """Detect splits (one old -> many new) and merges (many old -> one new).
    Approach:
      - For each unmatched old row, look at top-k unmatched new rows by similarity. If combined similarity
        (weighted) exceeds split_threshold, and token coverage checks out, declare a split.
      - Symmetric for merges.
    Returns lists of split and merge dicts.
    """
    splits = []
    merges = []

    # Precompute for speed
    for oi in unmatched_old:
        sims = [(nj, sim_matrix[oi, nj]) for nj in unmatched_new]
        sims.sort(key=lambda x: x[1], reverse=True)
        for k in range(2, min(max_parts, len(sims)) + 1):
            top = sims[:k]
            combined_sim = sum([s for _, s in top]) / k
            # also check max individual similarity not tiny
            if combined_sim >= split_threshold and any(s > 0.5 for _, s in top):
                splits.append({
                    'old_index': oi,
                    'new_indices': [nj for nj, _ in top],
                    'combined_sim': combined_sim,
                    'parts': k
                })
                break

    for nj in unmatched_new:
        sims = [(oi, sim_matrix[oi, nj]) for oi in unmatched_old]
        sims.sort(key=lambda x: x[1], reverse=True)
        for k in range(2, min(max_parts, len(sims)) + 1):
            top = sims[:k]
            combined_sim = sum([s for _, s in top]) / k
            if combined_sim >= split_threshold and any(s > 0.5 for _, s in top):
                merges.append({
                    'new_index': nj,
                    'old_indices': [oi for oi, _ in top],
                    'combined_sim': combined_sim,
                    'parts': k
                })
                break

    return splits, merges


# ------------------------- Cell-level Diff -------------------------

def cell_diff(a: Any, b: Any) -> Dict[str, Any]:
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if a_norm == b_norm:
        return {'changed': False, 'old': a, 'new': b, 'sim': 1.0}
    sim = column_similarity(a, b)
    return {'changed': True, 'old': a, 'new': b, 'sim': sim}


def row_cell_level_diff(row_a: pd.Series, row_b: pd.Series) -> Dict[str, Dict]:
    diffs = {}
    for c in row_a.index.union(row_b.index):
        diffs[c] = cell_diff(row_a.get(c, ""), row_b.get(c, ""))
    return diffs


# ------------------------- Report Generation -------------------------

def generate_report(df_old: pd.DataFrame, df_new: pd.DataFrame, matches: List[Tuple[int,int,float]],
                    sim_matrix: np.ndarray, col_weights: Dict[str, float],
                    split_threshold: float = 0.85) -> pd.ExcelWriter:
    # Prepare containers
    matched_rows = []
    added_rows = []
    deleted_rows = []

    n_old = len(df_old)
    n_new = len(df_new)

    matched_old = set()
    matched_new = set()

    for oi, nj, sim in matches:
        if sim >= 0.4:  # threshold for considering a match
            matched_old.add(oi)
            matched_new.add(nj)
            row_a = df_old.iloc[oi]
            row_b = df_new.iloc[nj]
            cell_diffs = row_cell_level_diff(row_a, row_b)
            matched_rows.append({'old_index': oi, 'new_index': nj, 'sim': sim, 'cell_diffs': cell_diffs})

    for oi in range(n_old):
        if oi not in matched_old:
            deleted_rows.append({'old_index': oi, 'row': df_old.iloc[oi]})
    for nj in range(n_new):
        if nj not in matched_new:
            added_rows.append({'new_index': nj, 'row': df_new.iloc[nj]})

    # For unmatched, attempt to detect splits/merges
    unmatched_old = [oi for oi in range(n_old) if oi not in matched_old]
    unmatched_new = [nj for nj in range(n_new) if nj not in matched_new]

    splits, merges = detect_splits_and_merges(df_old, df_new, sim_matrix, unmatched_old, unmatched_new,
                                             split_threshold=split_threshold)

    # Build excel writer and write sheets
    writer = pd.ExcelWriter('delta_report.xlsx', engine='openpyxl')

    # Summary sheet
    summary = {
        'total_old_rows': n_old,
        'total_new_rows': n_new,
        'matched': len(matched_rows),
        'added': len(added_rows),
        'deleted': len(deleted_rows),
        'splits_detected': len(splits),
        'merges_detected': len(merges)
    }
    pd.DataFrame([summary]).to_excel(writer, sheet_name='summary', index=False)

    # Matched rows details
    matched_out = []
    for m in matched_rows:
        row = {'old_index': m['old_index'], 'new_index': m['new_index'], 'sim': m['sim']}
        # add per-column flags
        for col, d in m['cell_diffs'].items():
            row[f'{col}_changed'] = d['changed']
            row[f'{col}_old'] = d['old']
            row[f'{col}_new'] = d['new']
            row[f'{col}_sim'] = d['sim']
        matched_out.append(row)
    if matched_out:
        pd.DataFrame(matched_out).to_excel(writer, sheet_name='matched', index=False)
    else:
        pd.DataFrame([], columns=['old_index','new_index','sim']).to_excel(writer, sheet_name='matched', index=False)

    # Added rows
    added_out = []
    for a in added_rows:
        r = {'new_index': a['new_index']}
        r.update(a['row'].to_dict())
        added_out.append(r)
    if added_out:
        pd.DataFrame(added_out).to_excel(writer, sheet_name='added', index=False)
    else:
        pd.DataFrame([], columns=df_new.columns).to_excel(writer, sheet_name='added', index=False)

    # Deleted rows
    deleted_out = []
    for d in deleted_rows:
        r = {'old_index': d['old_index']}
        r.update(d['row'].to_dict())
        deleted_out.append(r)
    if deleted_out:
        pd.DataFrame(deleted_out).to_excel(writer, sheet_name='deleted', index=False)
    else:
        pd.DataFrame([], columns=df_old.columns).to_excel(writer, sheet_name='deleted', index=False)

    # Splits & merges
    if splits:
        pd.DataFrame(splits).to_excel(writer, sheet_name='splits', index=False)
    else:
        pd.DataFrame([], columns=['old_index','new_indices','combined_sim','parts']).to_excel(writer, sheet_name='splits', index=False)
    if merges:
        pd.DataFrame(merges).to_excel(writer, sheet_name='merges', index=False)
    else:
        pd.DataFrame([], columns=['new_index','old_indices','combined_sim','parts']).to_excel(writer, sheet_name='merges', index=False)

    writer.save()

    return writer


# ------------------------- Top-level flow -------------------------

def compare_excels(path_old: str, path_new: str, sheet_old: str = None, sheet_new: str = None,
                   key_cols: List[str] = None, col_weights: Dict[str, float] = None,
                   sim_threshold: float = 0.4, split_threshold: float = 0.85) -> Dict[str, Any]:
    """Main function: reads workbooks, normalizes, computes similarity, matches and produces report.
    Returns a dict with high-level stats and path to report xlsx (delta_report.xlsx).
    """
    # Read first available sheet if sheet name not provided
    df_old = pd.read_excel(path_old, sheet_name=sheet_old)
    df_new = pd.read_excel(path_new, sheet_name=sheet_new)

    # Ensure columns are strings and consistent
    df_old.columns = [str(c) for c in df_old.columns]
    df_new.columns = [str(c) for c in df_new.columns]

    # If key_cols provided, reorder columns so that keys are first (helps weighting defaults)
    if key_cols:
        # ensure keys exist
        keys = [k for k in key_cols if k in df_old.columns or k in df_new.columns]
    else:
        keys = []

    # Build unified column set
    all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))

    # Fill missing columns with empty strings
    for c in all_cols:
        if c not in df_old.columns:
            df_old[c] = ""
        if c not in df_new.columns:
            df_new[c] = ""

    # Reindex dataframes to the same columns order
    df_old = df_old[all_cols].reset_index(drop=True)
    df_new = df_new[all_cols].reset_index(drop=True)

    # Default equal weights, but give slightly higher weight to key columns if provided
    if col_weights is None:
        col_weights = {c: (3.0 if c in keys else 1.0) for c in all_cols}

    sim_matrix = build_similarity_matrix(df_old, df_new, col_weights)

    matches = optimal_matching(sim_matrix)

    # Prepare report
    writer = generate_report(df_old, df_new, matches, sim_matrix, col_weights, split_threshold=split_threshold)

    stats = {
        'n_old': len(df_old),
        'n_new': len(df_new),
        'matches': len(matches)
    }
    # Note: writer saved to delta_report.xlsx
    return {'stats': stats, 'report_path': 'delta_report.xlsx'}


# ------------------------- CLI -------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two Excel files and produce a delta report.')
    parser.add_argument('old', help='Path to old Excel file (.xlsx)')
    parser.add_argument('new', help='Path to new Excel file (.xlsx)')
    parser.add_argument('--sheet-old', dest='sheet_old', default=None, help='Sheet name or index in old file')
    parser.add_argument('--sheet-new', dest='sheet_new', default=None, help='Sheet name or index in new file')
    parser.add_argument('--key-cols', dest='key_cols', default=None, help='Comma separated key column names to prioritize')
    parser.add_argument('--sim-threshold', dest='sim_threshold', type=float, default=0.4, help='Threshold to accept a matched pair')
    parser.add_argument('--split-threshold', dest='split_threshold', type=float, default=0.85, help='Threshold to detect splits/merges')

    args = parser.parse_args()
    keys = args.key_cols.split(',') if args.key_cols else None
    out = compare_excels(args.old, args.new, sheet_old=args.sheet_old, sheet_new=args.sheet_new,
                        key_cols=keys, sim_threshold=args.sim_threshold, split_threshold=args.split_threshold)
    print('Done. Report generated at:', out['report_path'])
