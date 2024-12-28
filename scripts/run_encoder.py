from pathlib import Path
import spacy
from collections import defaultdict
import re
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union, Optional
from tqdm import tqdm
import importlib


def _load_workflow_config(config_file: Optional[str] = None):
    """Load workflow configuration from .py file"""
    if config_file:
        config_file_path = Path(config_file)
        if config_file_path.is_file():
            spec = importlib.util.spec_from_file_location("default_config", config_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.default_config
        else:
            return {}
    else:
        return {}

class WordEncoder:
    """Process text files to encode words and phrases with their frequencies."""
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
        self.phrases = set(config_file["phrases"])
        self.setup_logging()

        # Regular expressions for cleaning
        self.cleanup_patterns = [
            (r"\s+", " "),  # Multiple spaces to single space
            (r"[\n\r\t]", " "),  # Newlines and tabs to space
            (r"[^\w\s-]", ""),  # Remove special characters except hyphens
        ]

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def add_phrases(self, phrase_list: List[str]):
        """
        Add multi-word phrases to be counted as single units.
        Accumulates phrases instead of replacing them.

        Args:
            phrase_list (List[str]): List of phrases (e.g., ["new york", "van gogh"])
        """
        # Initialize phrases set if it doesn't exist
        if not hasattr(self, "phrases"):
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
            phrase_pattern = r"\s+".join(words)
            pattern_parts.append(f"({phrase_pattern})")

        # Join all patterns with OR operator and add word boundaries
        full_pattern = "|".join(pattern_parts)
        self.phrase_pattern = re.compile(rf"\b(?:{full_pattern})\b", re.IGNORECASE)

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
        if hasattr(self, "phrases") and self.phrases:
            # Sort phrases by length (descending) to match longer phrases first
            for phrase in sorted(self.phrases, key=len, reverse=True):
                # Create pattern for this specific phrase
                pattern = r"\b" + re.escape(phrase) + r"\b"
                matches = list(re.finditer(pattern, clean_text, re.IGNORECASE))

                # If phrase is found, add to frequency and mark positions
                for match in matches:
                    start, end = match.span()

                    # Check if this position has already been matched
                    if not any(
                        start >= pos[0] and end <= pos[1] for pos in matched_positions
                    ):
                        word_freq[phrase] += 1
                        matched_positions.add((start, end))

        # Now process remaining words that aren't part of phrases
        doc = self.nlp(clean_text)

        for token in doc:
            # Skip unwanted tokens
            if (
                token.is_punct
                or token.is_space
                or token.text.lower() in self.stop_words
                or len(token.text) < 2
            ):
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
        if not hasattr(self, "phrase_pattern") or not self.phrase_pattern:
            return self.clean_text(text)

        cleaned_text = self.clean_text(text)

        # Function to handle each match
        def replace_phrase(match):
            matched_text = match.group(0)
            # Clean the matched text and replace spaces with underscores
            cleaned_match = " ".join(matched_text.lower().split())
            return cleaned_match.replace(" ", "_")

        return self.phrase_pattern.sub(replace_phrase, cleaned_text)

    def process_file(self, file_path: Path) -> Dict:
        """Process a single text file."""
        try:
            text = file_path.read_text(encoding="utf-8")
            if not text.strip():
                self.logger.warning(f"Empty file: {file_path}")
                return None

            vocab_counts = self.count_vocabulary(text)
            return {"document": file_path.stem, "vocabulary": vocab_counts}

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return None

    def write_jsonl(self, data: Dict, output_file: Path):
        """Write a single result to JSONL file."""
        if data:
            try:
                with output_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
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
        output_path.write_text("", encoding="utf-8")

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

    parser = argparse.ArgumentParser(
        description="Process text files to count word and phrase frequencies."
    )
    parser.add_argument("--input-dir", help="Input directory containing text files")
    parser.add_argument("--output-file", help="Output JSONL file path")
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/workflow_configs/default_config.py",
        help="Path to the configuration file",
    )

    args = parser.parse_args()

    config_file = _load_workflow_config(args.config)

    encoder = WordEncoder(max_workers=args.workers, config_file=config_file)
    encoder.process_directory(args.input_dir, args.output_file)



if __name__ == "__main__":
    main()


