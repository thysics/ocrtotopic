from pathlib import Path
import spacy
from collections import defaultdict
import re
import json
import yaml
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union
from tqdm import tqdm

class WordEncoder:
    """Process text files to encode words and phrases with their frequencies."""
    
    DEFAULT_CONFIG_DIR = Path("config")
    
    def __init__(self, max_workers=4, config_file: Union[str, Path] = None):
        """
        Initialize the encoder with spaCy model and thread pool.
        
        Args:
            max_workers (int): Number of worker threads
            config_file (str or Path, optional): Path to configuration file (JSON or YAML)
                                               If not provided, looks for config.yaml in DEFAULT_CONFIG_DIR
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(self.nlp.Defaults.stop_words)
        self.max_workers = max_workers
        self.phrases = set()
        self.setup_logging()
        
        # Regular expressions for cleaning
        self.cleanup_patterns = [
            (r'\s+', ' '),           # Multiple spaces to single space
            (r'[\n\r\t]', ' '),      # Newlines and tabs to space
            (r'[^\w\s-]', ''),       # Remove special characters except hyphens
        ]
        
        # Load configuration
        if config_file:
            self.load_config(config_file)
        else:
            # Try to load from default location
            default_config = self.DEFAULT_CONFIG_DIR / "config.json"
            if default_config.exists():
                self.load_config(default_config)
            else:
                self.logger.info(f"No configuration file found at {default_config}")

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_file: Union[str, Path]):
        """
        Load configuration from JSON or YAML file.
        
        Args:
            config_file (str or Path): Path to configuration file
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            # Determine file type from extension
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with config_path.open('r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with config_path.open('r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                raise ValueError("Configuration file must be JSON or YAML")
            
            # Load phrases from config
            if 'phrases' in config:
                self.add_phrases(config['phrases'])
            
            # Load any other configuration parameters
            if 'stop_words' in config:
                self.stop_words.update(config['stop_words'])
                
            self.logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise

    @classmethod
    def create_default_config(cls, output_file: Union[str, Path] = None, format: str = 'yaml'):
        """
        Create a default configuration file template.
        
        Args:
            output_file (str or Path, optional): Path to save the configuration file.
                                               If not provided, saves to DEFAULT_CONFIG_DIR/config.{format}
            format (str): Format of the configuration file ('yaml' or 'json')
        """
        if output_file is None:
            # Use default location
            output_file = cls.DEFAULT_CONFIG_DIR / f"config.{format.lower()}"
        
        default_config = {
            'phrases': [
                'new york',
                'san francisco',
                'machine learning',
                'artificial intelligence',
                'neural network',
                'deep learning',
                'computer vision',
                'natural language processing'
            ],
            'stop_words': [
                'etc',
                'ie',
                'eg',
                'vs',
                'viz'
            ]
        }
        
        output_path = Path(output_file)
        
        try:
            # Create config directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'yaml':
                with output_path.open('w', encoding='utf-8') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
            else:  # json
                with output_path.open('w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                    
            print(f"Created default configuration file: {output_file}")
            
        except Exception as e:
            print(f"Error creating configuration file: {str(e)}")

    def add_phrases(self, phrase_list: List[str]):
        """
        Add multi-word phrases to be counted as single units.
        Accumulates phrases instead of replacing them.
        
        Args:
            phrase_list (List[str]): List of phrases (e.g., ["new york", "van gogh"])
        """
        # Initialize phrases set if it doesn't exist
        if not hasattr(self, 'phrases'):
            self.phrases = set()
        
        # Add new phrases to existing set
        for phrase in phrase_list:
            cleaned_phrase = self.clean_text(phrase)
            if cleaned_phrase:
                self.phrases.add(cleaned_phrase)
        
        if not self.phrases:
            self.phrase_pattern = None
            return

        # Sort phrases by length (descending) to ensure longer phrases are matched first
        self.sorted_phrases = sorted(self.phrases, key=lambda x: (-len(x), x))
        
        # Create pattern string with exact word boundaries
        pattern_parts = []
        for phrase in self.sorted_phrases:
            # Split phrase into words and escape each word
            words = [re.escape(word) for word in phrase.split()]
            # Join words with flexible whitespace matching
            phrase_pattern = r'\s+'.join(words)
            pattern_parts.append(f'({phrase_pattern})')
        
        # Join all patterns with OR operator and add word boundaries
        full_pattern = '|'.join(pattern_parts)
        self.phrase_pattern = re.compile(fr'\b(?:{full_pattern})\b', re.IGNORECASE)

        # Log the current phrases for debugging
        print(f"Current phrases: {sorted(self.phrases)}")

    def clean_text(self, text: str) -> str:
        """Clean text before processing."""
        text = text.lower()
        for pattern, replacement in self.cleanup_patterns:
            text = re.sub(pattern, replacement, text)
        return text.strip()

    def count_vocabulary(self, text: str) -> Dict[str, int]:
        """
        Count word frequencies in text, treating specified phrases as single units.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, int]: Dictionary of word frequencies
        """
        clean_text = self.clean_text(text)
        word_freq = defaultdict(int)
        
        # Track positions of matched phrases to avoid double counting
        matched_positions = set()
        print(self.phrases)
        if hasattr(self, 'phrases') and self.phrases:
            # Sort phrases by length (descending) to match longer phrases first
            for phrase in sorted(self.phrases, key=len, reverse=True):
                # Create pattern for this specific phrase
                pattern = r'\b' + re.escape(phrase) + r'\b'
                matches = list(re.finditer(pattern, clean_text, re.IGNORECASE))
                
                # If phrase is found, add to frequency and mark positions
                for match in matches:
                    start, end = match.span()
                    
                    # Check if this position has already been matched
                    if not any(start >= pos[0] and end <= pos[1] for pos in matched_positions):
                        word_freq[phrase] += 1
                        matched_positions.add((start, end))
        
        # Now process remaining words that aren't part of phrases
        doc = self.nlp(clean_text)
        
        for token in doc:
            # Skip unwanted tokens
            if (token.is_punct or token.is_space or 
                token.text.lower() in self.stop_words or 
                len(token.text) < 2):
                continue
                
            # Get token position in the original text
            start = token.idx
            end = start + len(token.text)
            
            # Only count if token isn't part of a matched phrase
            if not any(start >= pos[0] and end <= pos[1] for pos in matched_positions):
                word_freq[token.text.lower()] += 1
        
        return dict(word_freq)

    def preprocess_text_with_phrases(self, text: str) -> str:
        """
        Replace multi-word phrases with single tokens.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with phrases replaced by underscore-connected tokens
        """
        if not hasattr(self, 'phrase_pattern') or not self.phrase_pattern:
            return self.clean_text(text)

        cleaned_text = self.clean_text(text)
        
        # Function to handle each match
        def replace_phrase(match):
            matched_text = match.group(0)
            # Clean the matched text and replace spaces with underscores
            cleaned_match = ' '.join(matched_text.lower().split())
            return cleaned_match.replace(' ', '_')
        
        return self.phrase_pattern.sub(replace_phrase, cleaned_text)

    def process_file(self, file_path: Path) -> Dict:
        """Process a single text file."""
        try:
            text = file_path.read_text(encoding='utf-8')
            if not text.strip():
                self.logger.warning(f"Empty file: {file_path}")
                return None
            
            vocab_counts = self.count_vocabulary(text)
            return {
                "document": file_path.stem,
                "vocabulary": vocab_counts
            }
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return None

    def write_jsonl(self, data: Dict, output_file: Path):
        """Write a single result to JSONL file."""
        if data:
            try:
                with output_file.open('a', encoding='utf-8') as f:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
            except Exception as e:
                self.logger.error(f"Error writing to {output_file}: {str(e)}")

    def process_directory(self, input_dir: str, output_file: str):
        """Process all text files in directory and subdirectories."""
        input_path = Path(input_dir)
        output_path = Path(output_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        # Create or clear output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text('', encoding='utf-8')
        
        # Find all text files recursively
        txt_files = list(input_path.rglob("*.txt"))
        
        if not txt_files:
            self.logger.warning(f"No text files found in {input_dir}")
            return
        
        self.logger.info(f"Found {len(txt_files)} text files")
        
        # Process files using thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for txt_file in txt_files:
                future = executor.submit(self.process_file, txt_file)
                futures.append(future)
            
            for future in tqdm(futures, desc="Processing files"):
                result = future.result()
                if result:
                    self.write_jsonl(result, output_path)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process text files to count word and phrase frequencies.')
    parser.add_argument('--input-dir', help='Input directory containing text files')
    parser.add_argument('--output-file', help='Output JSONL file path')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--config', type=str, help='Path to configuration file (JSON or YAML)')
    parser.add_argument('--create-config', action='store_true', help='Create default configuration file in ../config')
    parser.add_argument('--config-format', type=str, choices=['json', 'yaml'], default='yaml',
                      help='Format for the configuration file when using --create-config')
    
    args = parser.parse_args()
    
    if args.create_config:
        WordEncoder.create_default_config(format=args.config_format)
        return
    
    encoder = WordEncoder(max_workers=args.workers, config_file=args.config)
    encoder.process_directory(args.input_dir, args.output_file)

if __name__ == '__main__':
    main()
