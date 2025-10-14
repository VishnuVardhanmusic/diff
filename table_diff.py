

import pandas as pd
import numpy as np


# -------------------------- OCR PARSING --------------------------

def load_ocr_csv(csv_path: str) -> pd.DataFrame:
    """
    Load EasyOCR output CSV and compute right/bottom coordinates.
    """
    df = pd.read_csv(csv_path)
    df['right'] = df['left'] + df['width']
    df['bottom'] = df['top'] + df['height']
    df = df.sort_values(by=['top', 'left']).reset_index(drop=True)
    return df


# -------------------------- TABLE RECONSTRUCTION --------------------------

def cluster_rows(ocr_df: pd.DataFrame, threshold: int = 40) -> list:
    """
    Cluster bounding boxes into row groups based on Y-axis proximity.
    """
    rows, current_row = [], []
    last_y = None

    for _, r in ocr_df.iterrows():
        if last_y is None or abs(r['top'] - last_y) < threshold:
            current_row.append(r)
        else:
            rows.append(current_row)
            current_row = [r]
        last_y = r['top']

    if current_row:
        rows.append(current_row)
    return rows


def get_column_boundaries(header_row: list) -> list:
    """
    Infer column boundaries using left positions of header cells.
    """
    col_positions = sorted([r['left'] for r in header_row])
    boundaries = []
    for i in range(len(col_positions)):
        start = col_positions[i] - 20
        end = col_positions[i+1] - 20 if i+1 < len(col_positions) else col_positions[i] + 400
        boundaries.append((start, end))
    return boundaries


def assign_column(x: float, col_boundaries: list) -> int:
    """
    Find which column a given x-coordinate belongs to.
    """
    for i, (start, end) in enumerate(col_boundaries):
        if start <= x <= end:
            return i
    return None


def reconstruct_table(ocr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert EasyOCR bounding boxes into structured pandas DataFrame.
    """
    rows = cluster_rows(ocr_df)
    header_row = next((row for row in rows if any("column" in str(x['text']).lower() for x in row)), rows[0])
    col_boundaries = get_column_boundaries(header_row)

    structured_rows = []
    for row in rows:
        row_text = [""] * len(col_boundaries)
        for _, cell in pd.DataFrame(row).iterrows():
            cidx = assign_column(cell['left'], col_boundaries)
            if cidx is not None:
                existing = row_text[cidx]
                new_text = str(cell['text']).strip()
                row_text[cidx] = (existing + " " + new_text).strip()
        structured_rows.append(row_text)

    df = pd.DataFrame(structured_rows)
    df.columns = [f"Column {i+1}" for i in range(len(df.columns))]
    return df


# -------------------------- DIFF COMPUTATION --------------------------

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize all text entries (lowercase, strip spaces).
    """
    return df.applymap(lambda x: str(x).strip().lower() if pd.notna(x) else "")


def compute_diff(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Δ between two tables of equal or unequal size.
    """
    max_rows = max(len(df1), len(df2))
    max_cols = max(df1.shape[1], df2.shape[1])

    df1 = df1.reindex(range(max_rows), fill_value="")
    df2 = df2.reindex(range(max_rows), fill_value="")

    df1 = df1.reindex(columns=range(max_cols), fill_value="")
    df2 = df2.reindex(columns=range(max_cols), fill_value="")

    diff_matrix = np.full((max_rows, max_cols), "", dtype=object)
    for i in range(max_rows):
        for j in range(max_cols):
            if df1.iat[i, j] != df2.iat[i, j]:
                if df1.iat[i, j] == "":
                    diff_matrix[i, j] = "+ (added)"
                elif df2.iat[i, j] == "":
                    diff_matrix[i, j] = "- (removed)"
                else:
                    diff_matrix[i, j] = "Δ (modified)"
    diff_df = pd.DataFrame(diff_matrix, columns=[f"Column {i+1}" for i in range(max_cols)])
    return diff_df


def highlight_diff(val):
    """
    Color code Δ table for visualization.
    """
    if "Δ" in val:
        return 'background-color: yellow'
    elif "+" in val:
        return 'background-color: lightgreen'
    elif "-" in val:
        return 'background-color: lightcoral'
    else:
        return ''


def generate_html_diff(diff_df: pd.DataFrame, output_path: str = "diff_visual.html"):
    """
    Save highlighted Δ table as HTML for visual inspection.
    """
    styled = diff_df.style.applymap(highlight_diff)
    styled.to_html(output_path)
    print(f"✅ HTML diff visualization saved to: {output_path}")


# -------------------------- MAIN EXECUTION --------------------------

if __name__ == "__main__":
    # Example usage: Replace paths with your OCR CSV results
    ocr_csv_1 = "data/output/extracted_tables/input01_page3_easyocr.csv"
    ocr_csv_2 = "data/output/extracted_tables/input02_page3_easyocr.csv"

    print("📥 Loading OCR results...")
    ocr_df1 = load_ocr_csv(ocr_csv_1)
    ocr_df2 = load_ocr_csv(ocr_csv_2)

    print("🧩 Reconstructing tables...")
    table1 = reconstruct_table(ocr_df1)
    table2 = reconstruct_table(ocr_df2)

    print("📊 Normalizing data...")
    table1n = normalize_df(table1)
    table2n = normalize_df(table2)

    print("⚙️ Computing Δ (diff)...")
    diff = compute_diff(table1n, table2n)

    print("\n=== Δ (Diff) Matrix ===\n")
    #print(diff)

    # Optional: HTML visualization
    #generate_html_diff(diff)
    print(table1)
    print(table2)
