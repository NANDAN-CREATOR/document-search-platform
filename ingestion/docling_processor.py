"""Document preprocessing using Docling."""
import logging
from pathlib import Path
from typing import List, Dict, Any

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

logger = logging.getLogger(__name__)


class DoclingProcessor:
    """Process PDF documents using Docling."""

    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF and return structured content."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Processing PDF: {path.name}")
        result = self.converter.convert(str(path))
        doc = result.document

        # Extract full markdown text
        markdown_text = doc.export_to_markdown()

        # Extract metadata
        metadata = {
            "filename": path.name,
            "file_path": str(path.absolute()),
            "num_pages": len(doc.pages) if hasattr(doc, 'pages') else 0,
            "source": path.name,
        }

        logger.info(f"Processed {path.name}: {len(markdown_text)} chars")

        return {
            "text": markdown_text,
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
