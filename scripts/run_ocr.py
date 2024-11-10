import argparse
from pathlib import Path
import concurrent.futures
import multiprocessing
import os
from typing import List, Tuple
import logging
from tqdm import tqdm

# Import the PDFTextExtractor class here
# Assuming the class is in pdf_extractor.py
from scripts.get_txt import PDFTextExtractor

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def process_single_file(args: Tuple[Path, Path, PDFTextExtractor]) -> bool:
    """
    Process a single PDF file and save as text.
    Returns True if successful, False otherwise.
    """
    pdf_file, output_dir, extractor = args
    try:
        # Replace spaces with underscores in the output filename
        output_file = output_dir / f"{pdf_file.stem.replace(' ', '_')}.txt"
        
        # Create output subdirectories if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract text and save
        text_content = extractor.extract_text_from_pdf(pdf_file)
        if text_content and not text_content.isspace():
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            return True
        return False
    except Exception as e:
        logging.error(f"Error processing {pdf_file}: {str(e)}")
        return False

def get_pdf_files(input_dir: Path) -> List[Path]:
    """Get all PDF files from input directory recursively."""
    return list(input_dir.rglob("*.pdf"))

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract text from PDFs using multiple threads')
    parser.add_argument('--input_directory', required=True, help='Directory containing PDF files')
    parser.add_argument('--output_directory', required=True, help='Directory to save text files')
    
    # For M1 Pro, default to number of performance cores (usually 8)
    # but allow override through command line
    default_threads = min(8, multiprocessing.cpu_count())
    parser.add_argument('--num_threads', type=int, default=default_threads,
                       help=f'Number of threads to use (default: {default_threads})')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Convert to Path objects
    input_dir = Path(args.input_directory).resolve()
    output_dir = Path(args.output_directory).resolve()
    
    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of PDF files
    pdf_files = get_pdf_files(input_dir)
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    logger.info(f"Using {args.num_threads} threads")
    
    # Initialize the extractor
    extractor = PDFTextExtractor(device='mps')  # Use Metal Performance Shaders for M1 Mac
    
    # Create arguments for the worker function
    work_items = [(pdf_file, output_dir, extractor) for pdf_file in pdf_files]
    
    # Process files using thread pool
    successful = 0
    failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # Submit all tasks and wrap with tqdm for progress bar
        futures = list(tqdm(
            executor.map(process_single_file, work_items),
            total=len(work_items),
            desc="Processing PDFs"
        ))
        
        # Count successes and failures
        successful = sum(1 for result in futures if result)
        failed = len(futures) - successful
    
    # Final summary
    logger.info(
        f"Processing complete:\n"
        f"- Total PDFs found: {len(pdf_files)}\n"
        f"- Successfully processed: {successful}\n"
        f"- Failed: {failed}"
    )

if __name__ == "__main__":
    main()