# Quick Start Guide

## 5-Minute Setup

### 1. Install and Setup

```bash
cd markg-experiment-reproducible

# Option A: Automatic setup
bash setup.sh

# Option B: Manual setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Your Data

**Copy your data files to `MKG-RAG-BENCH-G/test/`:**

```
MKG-RAG-BENCH-G/test/
├── mm_queries.jsonl        # Multimodal queries
├── mm_corpus.jsonl         # Multimodal knowledge graph 
├── mm_qrels.tsv            # Multimodal relevance labels
├── text_queries.jsonl      # Text-only queries
├── text_corpus.jsonl       # Text-only corpus
└── text_qrels.tsv          # Text relevance labels
```

**Copy images to `images_subset_kg/`:**

```bash
# Images should be named after entities:
images_subset_kg/
├── entity_001.jpg
├── entity_002.jpg
└── ...
```

### 3. Run Experiments

#### Retrieval Evaluation Only

```bash
bash scripts/run_retrieval.sh

# Output: Results saved to results_retrieval/
```

#### RAG Generation (with OpenAI API)

```bash
export OPENAI_API_KEY="sk-..."
bash scripts/run_generation.sh

# Output: Results saved to rag_outputs/
```

## Common Commands

### Run on validation set instead of test
```bash
DATA_DIR=./MKG-RAG-BENCH-G/val bash scripts/run_retrieval.sh
```

### Evaluate only specific metrics
```bash
KS="10,50" bash scripts/run_retrieval.sh
```

### Use faster embedding model
```bash
MODEL_NAME="all-MiniLM-L6-v2" bash scripts/run_retrieval.sh
```

### Use specific retriever
Edit `scripts/run_retrieval.sh` and change:
```bash
retrievers=(
  "MMAnchorRetriever"    # Change this
)
```

Available options:
- `MMAnchorRetriever`
- `SimpleMultimodalRetriever`
- `SimpleTextRetriever`
- `CaptionRetriever`
- `RandomRetriever`

## Expected Output

### Retrieval Results
```
results_retrieval/
└── MMAnchorRetriever.metrics.json
```

Sample output:
```json
{
  "ndcg@5": 0.75,
  "ndcg@10": 0.68,
  "precision@5": 0.80,
  ...
}
```

### RAG Results
```
rag_outputs/
└── MMAnchorRetriever/
    ├── MMAnchorRetriever.retrieved.json
    └── MMAnchorRetriever.generated.json
```

## Troubleshooting

**Q: ImportError when running scripts**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate
```

**Q: CUDA out of memory**
```bash
# Reduce batch size
BATCH_SIZE=16 bash scripts/run_retrieval.sh
```

**Q: OpenAI API key not working**
```bash
# Verify your key is set
echo $OPENAI_API_KEY

# Test connection
python -c "from openai import OpenAI; print('OK')"
```

**Q: Image files not found**
```bash
# Check image directory
ls images_subset_kg/ | head

# Verify path configuration in the script
# Edit scripts/run_retrieval.sh, look for IMAGE_ROOT
```

## Next Steps

- See [README.md](README.md) for detailed configuration options
- Check individual Python files in `code/` for parameter details
- Review log files in `logs/` for debugging

---

**Need help?** Go back to README.md for comprehensive documentation.
