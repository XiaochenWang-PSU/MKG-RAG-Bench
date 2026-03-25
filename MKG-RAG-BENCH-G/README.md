# MarKG RAG Experiment - Reproducible Package

A self-contained, reproducible package for running MarKG Retrieval and RAG (Retrieval-Augmented Generation) experiments on multimodal knowledge graph data.

## Overview

This package contains the complete code for:
1. **Retrieval Evaluation** - Evaluating different retriever methods on multimodal knowledge graphs
2. **RAG Generation** - Generating answers using retrieved evidence with LLM integration

## Folder Structure

```
markg-experiment-reproducible/
├── code/                           # All Python modules
│   ├── MarKG_retrieval_v2.py       # Retriever implementations
│   ├── markg_main_v2.py            # Retrieval evaluation script
│   ├── MarKG_rag.py                # RAG generation script
│   ├── prompt_builder.py           # Prompt building utilities
│   ├── utils.py                    # Utility functions
│   ├── splitting.py                # Deterministic train/val/test splitting
│   └── embedding_cache.py          # Embedding cache management
├── scripts/                         # Shell scripts for running experiments
│   ├── run_retrieval.sh            # Run retrieval evaluation
│   └── run_generation.sh           # Run RAG generation
├── MKG-RAG-BENCH-G/               # Data directory (user provides data here)
│   ├── train/                      # Training data (optional)
│   ├── val/                        # Validation data (optional)
│   └── test/                       # Test data (required)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── cache_embeddings/               # Auto-created: embedding caches (optional)
```

## Setup Instructions

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

**Alternative**: Using conda
```bash
conda create -n markg python=3.9
conda activate markg
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your data in the `MKG-RAG-BENCH-G/` directory. The expected format for each split (train/val/test) includes:

#### Query Files (`.jsonl`)
- `mm_queries.jsonl` - Multimodal queries
- `text_queries.jsonl` - Text-only queries

**Format** (one JSON object per line):
```json
{
  "qid": 0,
  "question": "What is the capital of France?",
  "image_path": "/path/to/image.jpg"
}
```

#### Corpus Files (`.jsonl`)
- `mm_corpus.jsonl` - Multimodal knowledge graph
- `text_corpus.jsonl` - Text-only knowledge graph

**Format** (one JSON object per line):
```json
{
  "doc_id": 0,
  "head_id": "entity_1",
  "relation_id": "rel_type",
  "tail_id": "entity_2",
  "head_text": "Entity 1",
  "relation_text": "Relation Type",
  "tail_text": "Entity 2",
  "image_path": "/path/to/entity_1.jpg"
}
```

#### Relevance Labels (`.tsv`)
- `mm_qrels.tsv` - Ground truth for multimodal queries
- `text_qrels.tsv` - Ground truth for text queries

**Format** (tab-separated):
```
qid\tdoc_id\trelevance
0\t42\t1
0\t105\t1
```

### 3. Prepare Image Files (if using multimodal retrieval)

- Add images to a folder (e.g., `images_subset_kg/`)
- Images should be organized as:
  - `images_subset_kg/{entity_id}.jpg` for direct entity images
  - OR `images_subset_kg/{entity_id}/image1.jpg` for folder-based organization

## Running Experiments

### Run Retrieval Evaluation

```bash
cd /path/to/markg-experiment-reproducible

# Basic usage (test split)
bash scripts/run_retrieval.sh

# Custom configuration
DATA_DIR=./MKG-RAG-BENCH-G/val bash scripts/run_retrieval.sh

# With custom evaluation metrics
KS="5,10,20" bash scripts/run_retrieval.sh
```

**Environment Variables** for `run_retrieval.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `MKG-RAG-BENCH-G/test` | Path to data directory |
| `MODEL_NAME` | `clip-ViT-B-32` | Embedding model for retrievers |
| `BATCH_SIZE` | `32` | Batch size for encoding |
| `KS` | `5,10,20,50,100` | Evaluation metrics @K |
| `IMAGE_ROOT` | `images_subset_kg` | Path to knowledge graph images |
| `DO_SPLIT` | `0` | Enable deterministic split (1/0) |
| `EVAL_PARTITION` | `test` | Which partition to evaluate if DO_SPLIT=1 |

### Run RAG Generation

```bash
cd /path/to/markg-experiment-reproducible

# Basic usage (requires OpenAI API key)
export OPENAI_API_KEY="your-key-here"
bash scripts/run_generation.sh

# Custom configuration
DATA_DIR=./MKG-RAG-BENCH-G/val RAG_TOP_K=10 bash scripts/run_generation.sh

# With specific LLM model
LLM_MODEL="gpt-4o" bash scripts/run_generation.sh
```

**Environment Variables** for `run_generation.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `MKG-RAG-BENCH-G/test` | Path to data directory |
| `MODEL_NAME` | `clip-ViT-B-32` | Embedding model for retrievers |
| `RAG_TOP_K` | `5` | Number of documents to retrieve |
| `LLM_MODEL` | `gpt-5` | LLM model to use for generation |
| `MAX_OUTPUT_TOKENS` | `512` | Max tokens in generated response |
| `REASONING_EFFORT` | `minimal` | LLM reasoning effort (minimal/low/medium/high) |
| `GPU_IDS` | `0,1` | GPU IDs to use (comma-separated) |
| `GPU_JOBS_PER_GPU` | `3` | Max concurrent jobs per GPU |

### Available Retrievers

The following retriever methods are implemented:

- **MMAnchorRetriever** - Two-stage multimodal retriever
- **SimpleMultimodalRetriever** - Joint text-image retriever
- **SimpleTextRetriever** - Text-only retriever
- **CaptionRetriever** - Image caption-based retriever
- **RandomRetriever** - Random baseline

To use specific retrievers, modify the `retrievers=()` array in the shell scripts.

## Output Structure

### Retrieval Results

```
results_retrieval/
├── {retriever_name}.metrics.json  # NDCG@K, Precision@K, Recall@K
└── {retriever_name}.retrieved.json # Retrieved documents for each query
```

### RAG Results

```
rag_outputs/
├── {retriever_name}/
│   ├── {retriever_name}.retrieved.json  # Retrieved evidence
│   └── {retriever_name}.generated.json  # Generated answers + metrics
└── logs/
    └── {retriever_name}.log  # Detailed execution logs
```

## Configuration Examples

### Example 1: Evaluate on validation set with fewer metrics

```bash
DATA_DIR=./MKG-RAG-BENCH-G/val \
KS="10,50" \
BATCH_SIZE=64 \
bash scripts/run_retrieval.sh
```

### Example 2: RAG with GPT-4 and faster response

```bash
export OPENAI_API_KEY="sk-..."
LLM_MODEL="gpt-4o" \
MAX_OUTPUT_TOKENS=256 \
REASONING_EFFORT="minimal" \
bash scripts/run_generation.sh
```

### Example 3: Use deterministic split

```bash
DO_SPLIT=1 \
SPLIT_SEED="markg_v1" \
EVAL_PARTITION="val" \
bash scripts/run_retrieval.sh
```

## Customization

### Adding New Retrievers

Edit `code/MarKG_retrieval_v2.py` and add your retriever class, then add it to the `retrievers=()` array in the shell scripts.

### Modifying Prompt Format

Edit `code/prompt_builder.py` to customize how evidence is formatted in the LLM prompt.

### Different Encoding Models

Change `MODEL_NAME` environment variable to any model supported by sentence-transformers:
- `all-MiniLM-L6-v2` - Smaller, faster
- `all-mpnet-base-v2` - Larger, more accurate
- `clip-ViT-L-14` - Better image-text alignment

## Troubleshooting

### Issue: CUDA out of memory
**Solution**: Reduce `BATCH_SIZE` or `GPU_JOBS_PER_GPU`
```bash
BATCH_SIZE=16 GPU_JOBS_PER_GPU=1 bash scripts/run_generation.sh
```

### Issue: Missing embedding cache
**Solution**: First cache will be generated automatically on first run. For multimodal retrieval, ensure `IMAGE_ROOT` is correct.

### Issue: OpenAI API errors
**Solution**: Verify your API key and:
```bash
export OPENAI_API_KEY="your-key-here"
# Test connection
python -c "from openai import OpenAI; OpenAI().models.list()"
```

### Issue: Data files not found
**Solution**: Verify the exact file paths:
```bash
ls -la MKG-RAG-BENCH-G/test/
# Should show: mm_queries.jsonl, mm_corpus.jsonl, mm_qrels.tsv, etc.
```

## Performance Tips

1. **Batch Processing**: Increase `BATCH_SIZE` if GPU memory allows (8GB+ GPU: 128-256, 16GB+: 512+)
2. **Parallel Execution**: Adjust `GPU_JOBS_PER_GPU` based on available VRAM
3. **Model Selection**: Use smaller models (`all-MiniLM`) for speed, larger models for accuracy
4. **Caching**: Embeddings are cached automatically - first run is slower but subsequent runs reuse cache

## Citation

If you use this code, please cite:
```bibtex
[Add your citation here]
```

## License

See LICENSE file for details.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files in `logs/` directory
3. Examine output JSON files for detailed error messages

## Additional Notes

- All paths in scripts are relative to the project root for portability
- Commands should be run from the project root directory
- GPU parallelization uses semaphores to manage concurrent jobs per GPU
- Embeddings are cached in `cache_embeddings/` to avoid recomputation
