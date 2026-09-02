"""Document preprocessing using PyPDF - no torch/OCR dependency."""
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DoclingProcessor:
    """Process PDF documents using PyPDF (no torch/OCR required)."""

    def __init__(self):
        try:
            import pypdf
            self.use_pypdf = True
            logger.info("DoclingProcessor initialized with PyPDF backend")
        except ImportError:
            raise ImportError("pypdf not installed. Run: pip install pypdf")

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF and return structured content."""
        import pypdf
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Processing PDF: {path.name}")

        text_parts = []
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            num_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"## Page {i+1}\n{page_text}")

        full_text = "\n\n".join(text_parts)
        metadata = {
            "filename": path.name,
            "file_path": str(path.absolute()),
            "source": path.name,
            "num_pages": num_pages,
        }

        logger.info(f"Processed {path.name}: {len(full_text)} chars, {num_pages} pages")
        return {
            "text": full_text,
            "metadata": metadata,
            "filename": path.name,
        }

    def process_directory(self, data_dir: str) -> List[Dict[str, Any]]:
        """Process all PDF files in a directory."""
        data_path = Path(data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        pdf_files = list(data_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {data_dir}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files")
        documents = []
        for pdf_file in pdf_files:
            try:
                doc = self.process_pdf(str(pdf_file))
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                continue

        logger.info(f"Successfully processed {len(documents)} documents")
        return documents
