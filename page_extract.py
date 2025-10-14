import os
import fitz  # PyMuPDF
from pathlib import Path
from rich.console import Console
from utils.logger import get_logger

console = Console()
logger = get_logger(__name__)

def extract_pdf_page(pdf_path: str, page_number: int, dpi: int = 400) -> str:
    """
    Extracts a given page from a PDF and saves it as a high-resolution PNG image.

    Args:
        pdf_path (str): Path to the input PDF.
        page_number (int): Page number to extract (1-based index).
        dpi (int): Resolution for rendering.

    Returns:
        str: Path to the saved page image.
    """
    # Validate file
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Open PDF
    doc = fitz.open(pdf_path)
    if page_number < 1 or page_number > len(doc):
        raise ValueError(f"Invalid page number: {page_number}. PDF has {len(doc)} pages.")

    # Prepare output directory
    output_dir = Path("data/output/page_images")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Render page to image
    page = doc[page_number - 1]
    pix = page.get_pixmap(dpi=dpi)
    image_name = f"{Path(pdf_path).stem}_page{page_number}.png"
    image_path = output_dir / image_name
    pix.save(str(image_path))

    logger.info(f"Rendered page {page_number} from {pdf_path} → {image_path}")
    console.log(f"[bold green]✅ Saved Page {page_number} as image → {image_path}[/bold green]")
    return str(image_path)


if __name__ == "__main__":
    # Example usage
    pdf_file = "data/input/input02.pdf"
    output_image = extract_pdf_page(pdf_file, page_number=3, dpi=400)
    console.log(f"Output image path: {output_image}")
    