import unittest
import sys
from pathlib import Path

# Add parent directory to Python path to access scripts directory
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Now we can import from scripts directory
from scripts.run_encoder import WordEncoder


class TestWordEncoder(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.encoder = WordEncoder()

        # Sample text from the art catalog
        self.sample_text = """LYONEL FEININGER
        Born in New York, 1870
        To Hamburg 1886. Worked as caricaturist and comic strip artist for
        Chicago Tribune and German and French papers. First paintings,
        1907, influenced by impressionism, van Gogh, and, later, cubism. Ex-
        hibited Salon des Indépendents, I9II. Exhibited with Marc, Kandin-
        sky, and Paul Klee in first Autumn Salon, Berlin I913. Is now profes-
        sor of painting at the Bauhaus Academy at Dessau. Has recently been
        offered a studio by the town of Halle
        19 NIEDERGRONSTADT: ink and watercolor, 1912
        Collection J.I B. Neumann, New York
        20 SIDEWHEELER, 1913
        Collection Detroit Institute of Art
        21 IN THE VILLAGE, ink and watercolor, 1915
        Collection J. B. Neumann, New York
        22 FISHING SMACK, ink and watercolor, 1922
        Collection Dr. W.R. Valentiner, Detroit
        23 EICHELBORN, 1922
        Collection Dr. W.R. Valentiner, Detroit
        24 GATE TOWER I, ink and watercolor, 1923
        Collection Mrs. Fannie M. Pollak, New York
        25 SUMMER CLOUDS, ink and watercolor, 1927
        Private Collection, New York"""

    def test_clean_text(self):
        """Test the clean_text method."""
        # Test basic cleaning
        self.assertEqual(
            self.encoder.clean_text("LYONEL FEININGER"), "lyonel feininger"
        )

        # Test multiple spaces and special characters
        self.assertEqual(
            self.encoder.clean_text("Born in    New   York,  1870!"),
            "born in new york 1870",
        )

        # Test newlines and tabs
        self.assertEqual(
            self.encoder.clean_text("Chicago\tTribune\nand German"),
            "chicago tribune and german",
        )

        # Test hyphens preservation
        self.assertEqual(
            self.encoder.clean_text("Walraff-Richartz Museum"),
            "walraff-richartz museum",
        )

        # Test multiple punctuation and special characters
        self.assertEqual(
            self.encoder.clean_text("Collection J.I B. Neumann, New York!!!"),
            "collection ji b neumann new york",
        )

    def test_add_phrases(self):
        """Test the add_phrases method."""
        # Initialize phrases
        test_phrases = [
            "new york",
            "chicago tribune",
            "van gogh",
            "detroit institute of art",
            "summer clouds",
        ]

        self.encoder.add_phrases(test_phrases)

        # Process the sample text
        processed_text = self.encoder.preprocess_text_with_phrases(self.sample_text)

        # Test that phrases are properly connected with underscores
        self.assertIn("new_york", processed_text)
        self.assertIn("chicago_tribune", processed_text)
        self.assertIn("van_gogh", processed_text)
        self.assertIn("detroit_institute_of_art", processed_text)
        self.assertIn("summer_clouds", processed_text)

        # Test that phrases are case-insensitive
        test_text = "NEW YORK and New York and new york"
        processed_case_text = self.encoder.preprocess_text_with_phrases(test_text)
        self.assertEqual(processed_case_text, "new_york and new_york and new_york")

        # Test that longer phrases are matched first
        self.encoder.add_phrases(
            ["detroit", "detroit institute", "detroit institute of art"]
        )
        processed_multi = self.encoder.preprocess_text_with_phrases(
            "Detroit Institute of Art"
        )
        self.assertIn("detroit_institute_of_art", processed_multi)
        self.assertIn("detroit_institute", processed_multi)

        # Test vocabulary counting with phrases
        vocab_counts = self.encoder.count_vocabulary(self.sample_text)

        # Check that phrases are counted as single units
        self.assertIn("new york", vocab_counts)
        self.assertNotIn("new", vocab_counts)
        self.assertNotIn("york", vocab_counts)
        self.assertIn("detroit institute of art", vocab_counts)
        self.assertNotIn("detroit institute", vocab_counts)

        # Test matching whole words only
        test_text = "newspaper newyork new york"
        processed_boundary = self.encoder.preprocess_text_with_phrases(test_text)
        self.assertEqual(processed_boundary, "newspaper newyork new_york")

    def test_edge_cases(self):
        """Test edge cases for both methods."""
        # Empty text
        self.assertEqual(self.encoder.clean_text(""), "")

        # Text with only spaces
        self.assertEqual(self.encoder.clean_text("   "), "")

        # Empty phrases list
        self.encoder.add_phrases([])
        self.assertEqual(
            self.encoder.preprocess_text_with_phrases("some text"), "some text"
        )

        # Phrases with special characters
        self.encoder.add_phrases(["J.B. Neumann", "W.R. Valentiner"])
        processed = self.encoder.preprocess_text_with_phrases(
            "Collection J.B. Neumann and Dr. W.R. Valentiner"
        )
        self.assertIn("jb_neumann", processed)
        self.assertIn("wr_valentiner", processed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
