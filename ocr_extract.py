# snippet: use_easyocr.py (can be merged into stage3_ocr_extract.py)
import os
from pathlib import Path
import cv2
import pandas as pd
import easyocr
from rich.console import Console
from utils.logger import get_logger

console = Console()
logger = get_logger(__name__)

# Create the EasyOCR reader (languages can be changed; 'en' for English)
_reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have GPU + CUDA

def preprocess_image_for_ocr(image_path: str):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold may help; keep an un-thresholded copy for EasyOCR too
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY,11,2)
    # return both original-color (or gray) and preprocessed for debugging
    return img, th

def easyocr_extract_table(image_path: str, keyword: str = "ABC TITLE"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    logger.info(f"Starting EasyOCR on {image_path}")
    orig_img, proc_img = preprocess_image_for_ocr(image_path)

    # EasyOCR accepts color or grayscale images (as numpy arrays) or file path
    # We'll pass the preprocessed image to help recognition
    results = _reader.readtext(proc_img, detail=1, paragraph=False)  # detail=1 -> (bbox, text, conf)

    # results is a list like: [ (bbox, text, conf), ... ] where bbox is 4 points
    rows = []
    for bbox, text, conf in results:
        # bbox example: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]; compute bounding rect
        xs = [int(p[0]) for p in bbox]
        ys = [int(p[1]) for p in bbox]
        left, top = min(xs), min(ys)
        width, height = max(xs) - left, max(ys) - top
        rows.append({
            "left": left,
            "top": top,
            "width": width,
            "height": height,
            "text": text.strip(),
            "conf": float(conf)
        })

    df = pd.DataFrame(rows).sort_values(by=["top", "left"]).reset_index(drop=True)

    # If keyword provided, optionally filter to lines below the keyword occurrence
    if keyword:
        matches = df[df["text"].str.lower().str.contains(keyword.lower(), na=False)]
        if not matches.empty:
            start_y = int(matches.iloc[0]["top"])
            df = df[df["top"] > start_y + 10]  # capture below header

    # Save CSV
    output_dir = Path("data/output/extracted_tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{Path(image_path).stem}_easyocr.csv"
    df.to_csv(out_csv, index=False)
    console.log(f"[bold green]✅ EasyOCR extraction saved:[/bold green] {out_csv}")
    logger.info(f"EasyOCR wrote {len(df)} text blocks to {out_csv}")

    return df

if __name__ == "__main__":
    img_path = "data/output/page_images/input02_page3.png"
    df = easyocr_extract_table(img_path, keyword="ABC TITLE")
    print(df.head(30))
