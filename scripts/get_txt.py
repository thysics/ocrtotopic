from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from pathlib import Path
from tqdm import tqdm
import logging
import torch
import os

class PDFTextExtractor:
    """
    A robust class to extract text from PDFs using doctr and save as txt files.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the OCR processor with the most reliable model configuration.
        
        Args:
            device (str): Device to run the model on ('cpu' or 'cuda')
        """
        # Initialize with more robust model configuration
        self.model = ocr_predictor(
            det_arch='db_resnet50',
            reco_arch='crnn_vgg16_bn',
            pretrained=True
        ).to(device)
        
        self.setup_logging()
        
    def setup_logging(self):
        """Set up detailed logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a single PDF file with enhanced error checking.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            self.logger.info(f"Processing: {pdf_path}")
            
            # Load and check the PDF
            doc = DocumentFile.from_pdf(str(pdf_path))
            if not doc:
                self.logger.error(f"Failed to load PDF: {pdf_path}")
                return ""
            
            self.logger.info(f"Successfully loaded PDF: {pdf_path}")
            
            # Process with OCR using error handling
            try:
                with torch.no_grad():  # Prevent memory leaks
                    result = self.model(doc)
                    self.logger.info(f"OCR processing completed for: {pdf_path}")
            except Exception as e:
                self.logger.error(f"OCR processing failed: {str(e)}")
                return ""
            
            # Extract text with detailed logging
            text_content = []
            total_words = 0
            
            for page_idx, page in enumerate(result.pages):
                page_text = []
                page_words = 0
                
                for block in page.blocks:
                    for line in block.lines:
                        # Filter out low confidence words
                        words = [word.value for word in line.words 
                               if word.confidence > 0.3]  # Adjust confidence threshold as needed
                        
                        if words:
                            line_text = ' '.join(words)
                            page_text.append(line_text)
                            page_words += len(words)
                
                if page_text:
                    text_content.append('\n'.join(page_text))
                    total_words += page_words
                    self.logger.info(f"Page {page_idx + 1}: Found {page_words} words")
                else:
                    self.logger.warning(f"No text found on page {page_idx + 1}")
            
            if not text_content:
                self.logger.warning("No text was extracted from any page")
                return ""
            
            self.logger.info(f"Total words extracted: {total_words}")
            return '\n\n'.join(text_content)
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {str(e)}")
            return ""
        finally:
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """
        Process all PDFs in a directory and save their text content to txt files.
        
        Args:
            input_dir (str): Directory containing PDF files
            output_dir (str): Directory to save text files
        """
        try:
            # Convert to absolute paths
            input_path = Path(input_dir).resolve()
            output_path = Path(output_dir).resolve()
            
            # Validate input directory
            if not input_path.exists():
                self.logger.error(f"Input directory does not exist: {input_dir}")
                return
            
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Find all PDFs
            pdf_files = list(input_path.rglob("*.pdf"))
            
            if not pdf_files:
                self.logger.warning(f"No PDF files found in {input_dir}")
                return
            
            self.logger.info(f"Found {len(pdf_files)} PDF files in {input_dir}")
            
            # Process each PDF
            successful = 0
            failed = 0
            
            for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
                try:
                    # Extract text
                    self.logger.info(f"Processing: {pdf_file}")
                    text_content = self.extract_text_from_pdf(pdf_file)
                    
                    if not text_content or text_content.isspace():
                        self.logger.warning(f"No text content extracted from {pdf_file}")
                        failed += 1
                        continue
                    
                    # Create output file path
                    rel_path = pdf_file.relative_to(input_path)
                    output_file = output_path / f"{rel_path.stem}.txt"
                    
                    # Ensure output directory exists
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write content to file
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(text_content)
                        successful += 1
                        self.logger.info(f"Successfully saved text to: {output_file}")
                        
                        # Verify file was created and has content
                        if not output_file.exists():
                            raise FileNotFoundError(f"Output file was not created: {output_file}")
                        if output_file.stat().st_size == 0:
                            raise ValueError(f"Output file is empty: {output_file}")
                            
                    except Exception as e:
                        self.logger.error(f"Error saving {output_file}: {str(e)}")
                        failed += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing {pdf_file}: {str(e)}")
                    failed += 1
                    continue
            
            # Final summary
            self.logger.info(
                f"Processing complete:\n"
                f"- Total PDFs found: {len(pdf_files)}\n"
                f"- Successfully processed: {successful}\n"
                f"- Failed: {failed}"
            )
            
            # If no files were processed successfully, raise an error
            if successful == 0:
                raise RuntimeError(
                    f"Failed to process any PDFs successfully out of {len(pdf_files)} files"
                )
                
        except Exception as e:
            self.logger.error(f"Error in process_directory: {str(e)}")
            raise


    def process_single_pdf(self, pdf_path: str, output_path: str = None) -> str:
        """
        Process a single PDF file and save to file.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_path (str): Path to save the text file
            
        Returns:
            str: Extracted text content
        """
        try:
            # Convert to Path objects and resolve paths
            pdf_path = Path(pdf_path).resolve()
            
            # Check if PDF exists
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            # Extract text content
            self.logger.info(f"Processing PDF: {pdf_path}")
            text_content = self.extract_text_from_pdf(pdf_path)
            
            # Handle output file
            if output_path:
                output_path = Path(output_path).resolve()
                
                # Create output directory if it doesn't exist
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write content to file
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    self.logger.info(f"Saved text content to: {output_path}")
                except Exception as e:
                    self.logger.error(f"Error saving text file: {str(e)}")
                    raise
            
            return text_content
            
        except Exception as e:
            self.logger.error(f"Error in process_single_pdf: {str(e)}")
            raise