import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
import re
import argparse
import logging
from tqdm import tqdm

class VocabularyMerger:
    """Process vocabulary dictionaries to merge similar words and correct OCR errors."""
    
    DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / 'config'
    DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / 'config.json'
    
    def __init__(self, config_file: str = None):
        """
        Initialize merger with configuration file.
        
        Args:
            config_file (str, optional): Path to JSON config file.
                                       If not provided, uses default config from config/config.json
        """
        self.setup_logging()
        
        # If no config file provided, use default
        if config_file is None:
            config_file = self.DEFAULT_CONFIG_FILE
            
        self.load_config(config_file)
    
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_file: str):
        """
        Load configuration from JSON file.
        
        Args:
            config_file (str): Path to JSON configuration file
        """
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_file}\n"
                f"Expected location: {self.DEFAULT_CONFIG_FILE}"
            )
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Store all configuration parameters
            self.phrases = set(config.get('phrases', []))
            self.stop_words = set(config.get('stop_words', []))
            self.merge_groups = config.get('merge_groups', {})
            
            # For each phrase in phrases, create a merge group if it doesn't exist
            for phrase in self.phrases:
                if phrase not in self.merge_groups:
                    # Create variations of the phrase
                    variations = [
                        phrase,
                        phrase.replace(' ', ''),
                        phrase.replace(' ', '-'),
                        phrase.replace(' ', '_')
                    ]
                    self.merge_groups[phrase] = variations
            
            self.logger.info(f"Loaded configuration with {len(self.merge_groups)} merge groups")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise

    def clean_word(self, word: str) -> str:
        """Clean individual word by removing OCR artifacts and standardizing format."""
        word = word.lower().strip()
        
        # Remove non-alphanumeric characters except spaces and hyphens
        word = re.sub(r'[^\w\s-]', '', word)
        
        return word.strip()

    def find_merge_group(self, word: str) -> str:
        """Find the correct merge group for a word if it exists."""
        cleaned_word = self.clean_word(word)
        
        # Check predefined merge groups
        for group_key, variations in self.merge_groups.items():
            if cleaned_word in [self.clean_word(v) for v in variations]:
                return group_key
        
        return cleaned_word

    def process_vocabulary(self, vocab_dict: Dict[str, int]) -> Dict[str, int]:
        """
        Process single vocabulary dictionary.
        
        Args:
            vocab_dict (Dict[str, int]): Original vocabulary dictionary
            
        Returns:
            Dict[str, int]: Cleaned and merged vocabulary dictionary
        """
        merged_vocab = defaultdict(int)
        processed_words = set()
        
        # First pass: handle predefined merge groups and phrases
        for word, count in vocab_dict.items():
            if word in self.stop_words:
                continue
                
            merged_key = self.find_merge_group(word)
            merged_vocab[merged_key] += count
            processed_words.add(word)
        
        # Filter out very short words and obvious artifacts
        final_vocab = {k: v for k, v in merged_vocab.items() 
                      if len(k) > 1 and not k.isdigit()}
        
        return dict(final_vocab)

    def process_jsonl(self, input_file: str, output_file: str):
        """
        Process JSONL file containing vocabulary dictionaries.
        
        Args:
            input_file (str): Path to input JSONL file
            output_file (str): Path to output JSONL file
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process each line
        processed_docs = []
        total_lines = sum(1 for _ in open(input_path, 'r'))
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Processing documents"):
                try:
                    # Parse JSON line
                    doc = json.loads(line)
                    
                    # Process vocabulary if present
                    if 'vocabulary' in doc:
                        doc['vocabulary'] = self.process_vocabulary(doc['vocabulary'])
                    
                    processed_docs.append(doc)
                    
                except Exception as e:
                    self.logger.error(f"Error processing line: {str(e)}")
        
        # Write all processed documents to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in processed_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Processed {len(processed_docs)} documents")
        
    @staticmethod
    def create_default_config():
        """Create default configuration file in the config directory."""
        default_config = {
            "phrases": [
                "new york",
                "museum of modern art",
                "los angeles",
                "van gough",
                "machine learning",
                "deep learning",
                "artificial intelligence",
                "data science",
                "neural network",
                "united states",
                "san francisco",
                "max weber"
            ],
            "stop_words": [
                "etc",
                "ie",
                "eg",
                "vs",
                "ie"
            ],
            "merge_groups": {
                "new york": [
                    "new york",
                    "newyork",
                    "new-york",
                    "new_york"
                ],
                "museum of modern art": [
                    "museum of modern art",
                    "museum of modern arts",
                    "modern art museum",
                    "moma"
                ],
                "acknowledgments": [
                    "acknowledgments",
                    "acknowiedgments",
                    "acknowledgements"
                ],
                "collection": [
                    "collection",
                    "collections",
                    "collectiona"
                ],
                "painting": [
                    "painting",
                    "paintings",
                    "paintingsby",
                    "paintingsare"
                ]
            }
        }
        
        config_dir = Path(__file__).parent / 'config'
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = config_dir / 'config.json'
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"Created default configuration file: {config_path}")
        return config_path

def main():
    parser = argparse.ArgumentParser(description='Process vocabulary dictionaries to merge similar words.')
    parser.add_argument('--input-file', help='Input JSONL file path')
    parser.add_argument('--output-file', help='Output JSONL file path')
    parser.add_argument('--config', help='Path to configuration file (JSON)')
    parser.add_argument('--create-config', action='store_true', help='Create default configuration file')
    
    args = parser.parse_args()
    
    if args.create_config:
        config_path = VocabularyMerger.create_default_config()
        print(f"You can now edit the configuration file at: {config_path}")
        return
    
    # Initialize merger with provided config or let it use default
    merger = VocabularyMerger(args.config)
    merger.process_jsonl(args.input_file, args.output_file)

if __name__ == '__main__':
    main()