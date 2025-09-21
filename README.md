# RAG Challenge - 10-K Financial Document Analysis

A lightweight Retrieval-Augmented Generation (RAG) system for analyzing 10-K financial filings from major tech companies (Microsoft, Google/Alphabet, NVIDIA). The system extracts text from PDFs, builds semantic search indices, and provides intelligent querying capabilities for financial metrics and insights.

## 🚀 Features

- **Robust PDF Processing**: Handles FontBBox errors and other PDF parsing issues with multiple fallback methods
- **Semantic Search**: Uses sentence transformers for intelligent document retrieval
- **Financial Metric Extraction**: Automatically extracts revenue, margins, and other key financial data
- **Multi-Company Analysis**: Supports comparative analysis across Microsoft, Google, and NVIDIA
- **Interactive CLI**: Command-line interface for easy querying
- **Error Recovery**: Multiple PDF processing libraries (pdfplumber, PyMuPDF, PyPDF2) for maximum compatibility

## 📋 Requirements

- Python 3.8+
- Virtual environment (recommended)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hsh1304/rag_challenge.git
   cd rag_challenge
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add 10-K PDF files:**
   Place your 10-K PDF files in the `./data` directory with naming convention:
   ```
   data/
   ├── MSFT_2023.pdf
   ├── GOOGL_2024.pdf
   ├── NVDA_2022.pdf
   └── ...
   ```

## 🏃‍♂️ Quick Start

### Basic Usage

```python
from main import RAGIndex, answer_query

# Initialize the RAG system
rag = RAGIndex()

# Ingest PDFs from data directory
rag.ingest_folder("./data")

# Query the system
result = answer_query("What was NVIDIA's total revenue in 2024?", rag)
print(result)
```

### Interactive Mode

```bash
python main.py
```

Then choose:
- `cli` - Enter interactive command-line interface
- `sample` - Run predefined sample queries
- `quit` - Exit

### Sample Queries

The system comes with built-in sample queries:

```python
sample_queries = [
    "What was NVIDIA's total revenue in fiscal year 2024?",
    "Which company had the highest operating margin in 2023?",
    "How did NVIDIA's data center revenue grow from 2022 to 2023?",
    "What percentage of Google's 2023 revenue came from advertising?",
    "Compare AI investments mentioned by all three companies in their 2024 10-Ks"
]
```

## 📊 Supported Query Types

### 1. Single Company Metrics
```
"What was Microsoft's total revenue in 2023?"
"Show me NVIDIA's operating margin for 2024"
```

### 2. Comparative Analysis
```
"Which company had the highest revenue in 2023?"
"Compare operating margins across all three companies"
```

### 3. Growth Analysis
```
"How did Google's cloud revenue grow from 2022 to 2023?"
"Show me revenue growth trends for NVIDIA"
```

### 4. Specific Metrics
- Total Revenue
- Operating Margin
- Cloud Revenue
- Data Center Revenue
- R&D Investments
- AI Investments

## 🔧 Code Examples

### Custom Metric Extraction

```python
from main import extract_metric, RAGIndex

# Initialize RAG system
rag = RAGIndex()
rag.ingest_folder("./data")

# Extract specific metric
revenue_data = extract_metric("total revenue", "NVDA", "2024", rag)
print(f"Revenue: {revenue_data['value']}")
print(f"Source: {revenue_data['source']['text'][:200]}...")
```

### Building Custom Queries

```python
from main import answer_query, RAGIndex

rag = RAGIndex()
rag.ingest_folder("./data")

# Custom comparative query
query = "Compare cloud revenue between Microsoft and Google for 2023"
result = answer_query(query, rag)

print(f"Answer: {result['answer']}")
print(f"Reasoning: {result['reasoning']}")
print(f"Sources: {len(result['sources'])} documents found")
```

### Direct Text Retrieval

```python
from main import RAGIndex

rag = RAGIndex()
rag.ingest_folder("./data")

# Direct semantic search
chunks = rag.retrieve("artificial intelligence investments", top_k=5)
for chunk in chunks:
    print(f"Score: {chunk['_score']:.3f}")
    print(f"Company: {chunk['company']} ({chunk['year']})")
    print(f"Text: {chunk['text'][:200]}...")
    print("-" * 50)
```

## 🏗️ Architecture

### Core Components

1. **PDF Processing** (`extract_text_from_pdf`)
   - Handles FontBBox errors and malformed PDFs
   - Multiple fallback extraction methods
   - Page-by-page error recovery

2. **Text Chunking** (`chunk_text`)
   - Configurable chunk sizes (default: 200 words)
   - Overlapping chunks for better context
   - Page-aware chunking

3. **Semantic Indexing** (`RAGIndex`)
   - Sentence transformer embeddings
   - FAISS vector search
   - Normalized similarity scoring

4. **Query Processing** (`answer_query`)
   - Query decomposition for complex questions
   - Metric extraction using regex patterns
   - Comparative analysis logic

### Data Flow

```
PDF Files → Text Extraction → Chunking → Embeddings → FAISS Index
                                                           ↓
User Query → Query Processing → Semantic Search → Answer Synthesis
```

## ⚙️ Configuration

### Environment Variables

```bash
# Optional: Set custom data directory
export DATA_DIR="./custom_data"

# Optional: Set embedding model
export EMBED_MODEL="all-MiniLM-L6-v2"
```

### Customization Options

```python
# Custom chunk size and overlap
CHUNK_TOKEN_ESTIMATE = 200  # words per chunk
CHUNK_OVERLAP = 50         # words overlap

# Search parameters
TOP_K = 5                  # number of results to retrieve

# Embedding model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
```

## 🐛 Troubleshooting

### Common Issues

1. **FontBBox Errors**
   - The system automatically handles these with multiple fallback methods
   - Check console output for which extraction method was used

2. **Empty PDF Pages**
   - Some PDFs may have pages with no extractable text
   - The system will show warnings but continue processing

3. **Memory Issues**
   - Large PDFs may require more RAM
   - Consider reducing chunk size or processing fewer files

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
rag = RAGIndex()
rag.ingest_folder("./data")
```

## 📈 Performance Tips

1. **Use SSD storage** for faster PDF processing
2. **Increase chunk overlap** for better context (trade-off with memory)
3. **Pre-filter PDFs** to only include relevant years/companies
4. **Use GPU** for faster embedding computation (if available)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [pdfplumber](https://github.com/jsvine/pdfplumber) for PDF text extraction
- [sentence-transformers](https://www.sbert.net/) for semantic embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [PyMuPDF](https://pymupdf.readthedocs.io/) for robust PDF processing

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed description

---

**Happy analyzing! 📊✨**