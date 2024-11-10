# OCRtoTopic

A Python package that processes PDF documents through OCR (Optical Character Recognition) and analyzes the extracted text to identify topics and key phrases. It supports batch processing of PDFs with multi-threading capabilities and provides tools for text analysis and vocabulary management.

## Installation

1. Create and activate a Conda environment:
```
conda create -n ocrtotopic python=3.8
conda activate ocrtotopic
```

2. Install PyTorch (see details in https://pytorch.org/get-started/locally/)

3. Clone and install the package:
```
git clone https://github.com/yourusername/ocrtotopic.git
cd ocrtotopic
pip install -e .
```

4. Install required language model:
```
python -m spacy download en_core_web_sm
```

That's it! The package is now ready to use. For usage examples and documentation, see the script headers in the `scripts/` directory.
