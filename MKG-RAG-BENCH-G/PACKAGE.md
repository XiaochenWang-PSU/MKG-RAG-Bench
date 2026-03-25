# MarKG RAG Experiment - Package Contents

## Overview

This is a self-contained, reproducible package for running **MarKG Retrieval and RAG (Retrieval-Augmented Generation) experiments** on multimodal knowledge graph data.

The package includes:
- ✅ All core Python modules (retrievers, evaluation, RAG generation)
- ✅ Two shell scripts for easy experiment execution
- ✅ Complete documentation and quick start guide
- ✅ Data directory structure for organizing datasets
- ✅ Dependency management with requirements.txt

## Quick Links

- **Getting Started**: See [QUICKSTART.md](QUICKSTART.md) for immediate setup (5 minutes)
- **Full Documentation**: See [README.md](README.md) for comprehensive guide
- **Setup Script**: Run `bash setup.sh` for automatic environment setup

## Package Structure

```
markg-experiment-reproducible/
│
├── 📁 code/                          # Python modules (DO NOT EDIT unless customizing)
│   ├── MarKG_retrieval_v2.py        # Retriever implementations
│   ├── markg_main_v2.py             # Retrieval evaluation entry point
│   ├── MarKG_rag.py                 # RAG generation entry point
│   ├── prompt_builder.py            # LLM prompt construction
│   ├── utils.py                     # Utility functions
│   ├── splitting.py                 # Train/val/test splitting
│   ├── embedding_cache.py           # Embedding cache management
│   └── __init__.py                  # Package marker
│
├── 📁 scripts/                       # Execution scripts
│   ├── run_retrieval.sh             # Run retrieval evaluation (MAIN SCRIPT)
│   └── run_generation.sh            # Run RAG generation (MAIN SCRIPT)
│
├── 📁 MKG-RAG-BENCH-G/              # DATA DIRECTORY (USER PROVIDES DATA)
│   ├── train/                       # Optional: training data
│   ├── val/                         # Optional: validation data
│   └── test/                        # Required: test data
│       ├── mm_queries.jsonl         # Multimodal queries
│       ├── mm_corpus.jsonl          # MM knowledge graph
│       ├── mm_qrels.tsv             # MM relevance labels
│       ├── text_queries.jsonl       # Text-only queries
│       ├── text_corpus.jsonl        # Text-only corpus
│       └── text_qrels.tsv           # Text relevance labels
│
├── 📄 setup.sh                      # Automatic setup script
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # Full documentation
├── 📄 QUICKSTART.md                 # Quick start guide
└── 📄 PACKAGE.md                    # This file
```

## What to Do With This Package

### Step 1: Setup Environment (5 minutes)
```bash
cd markg-experiment-reproducible
bash setup.sh
# OR manually: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

### Step 2: Prepare Your Data (1-2 hours)
Place your multimodal knowledge graph data in `MKG-RAG-BENCH-G/test/` (or train/val):
- 6 required files (see README.md for format details)
- Images in `images_subset_kg/`

### Step 3: Run Experiments

**Option A: Retrieval Only** (~30 min - 2 hours)
```bash
bash scripts/run_retrieval.sh
```
Evaluates retrieval performance of different methods.
Outputs metrics: NDCG@K, Precision@K, Recall@K, etc.

**Option B: RAG Generation** (~1-4 hours)
```bash
export OPENAI_API_KEY="sk-..."
bash scripts/run_generation.sh
```
Generates answers using retrieved evidence and OpenAI API.
Outputs: retrieved results + LLM-generated answers + metrics

### Step 4: View Results
Check:
- `results_retrieval/` for retrieval metrics
- `rag_outputs/` for generation results
- `logs/` for detailed execution logs

## File Descriptions

### Core Python Modules

| File | Purpose |
|------|---------|
| `MarKG_retrieval_v2.py` | Contains 5 retriever classes: MMAnchorRetriever, SimpleMultimodalRetriever, SimpleTextRetriever, CaptionRetriever, RandomRetriever |
| `markg_main_v2.py` | Evaluates retrieval performance; computes NDCG, Precision, Recall |
| `MarKG_rag.py` | Generates answers using retrieved evidence + LLM; computes EM, F1, Contains, BLEU |
| `prompt_builder.py` | Formats queries and evidence into LLM prompts |
| `utils.py` | Utility functions (text normalization, file I/O, etc.) |
| `splitting.py` | Deterministic train/val/test splitting |
| `embedding_cache.py` | Caches embeddings to avoid recomputation |

### Execution Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `run_retrieval.sh` | Evaluate different retrievers on test data | `results_retrieval/` metrics |
| `run_generation.sh` | Generate answers with retrieved evidence | `rag_outputs/` generation results |

### Documentation

| File | Purpose |
|------|---------|
| `QUICKSTART.md` | 5-minute setup and first experiment |
| `README.md` | Comprehensive documentation |
| `PACKAGE.md` | This file - package overview |

## Key Features

✅ **Self-Contained**: No external dependencies except those in requirements.txt
✅ **Reproducible**: Deterministic splitting, fixed random seeds
✅ **Modular**: Easy to add new retrievers or modify prompts
✅ **Well-Documented**: README, QUICKSTART, and inline comments
✅ **Parallelizable**: Efficient GPU-based parallelization
✅ **Cached**: Embeddings are cached to avoid recomputation

## Experiment Workflow

```
Prepare Data
    ↓
Setup Environment (setup.sh)
    ↓
Run Retrieval Evaluation (run_retrieval.sh)
    ├─ Stage 1: Load queries, corpus, qrels
    ├─ Stage 2: For each retriever:
    │   ├─ Build embeddings
    │   ├─ Retrieve top-K documents
    │   ├─ Compute metrics (NDCG, Precision, Recall)
    │   └─ Save results
    ↓
[Optional] Run RAG Generation (run_generation.sh)
    ├─ Stage 1: Load queries, corpus, qrels
    ├─ Stage 2: For each retriever:
    │   ├─ Retrieve top-K documents
    │   ├─ Format evidence into prompt
    │   ├─ Call OpenAI API for answer generation
    │   ├─ Compute metrics (EM, F1, Contains, BLEU)
    │   └─ Save results
    ↓
View Results (results_retrieval/ + rag_outputs/)
```

## Important Notes

1. **Data Format**: Ensure your data matches the expected JSONL/TSV formats (see README.md)
2. **Image Paths**: If using multimodal retrieval, images must be accessible at specified paths
3. **OpenAI API**: Required for RAG generation (set OPENAI_API_KEY environment variable)
4. **GPU Resources**: Recommended 8GB+ VRAM for efficient execution
5. **Relative Paths**: All scripts use relative paths for portability

## Customization Points

| Component | File | How to Customize |
|-----------|------|------------------|
| Retriever Methods | `code/MarKG_retrieval_v2.py` | Add new retriever class |
| LLM Prompts | `code/prompt_builder.py` | Modify prompt formatting |
| Evaluation Metrics | `code/markg_main_v2.py` + `code/MarKG_rag.py` | Add new metrics |
| Configuration | `scripts/run_*.sh` | Modify environment variables |

## Troubleshooting

- **ImportError**: Verify virtual environment is activated
- **Data not found**: Check file paths and data format
- **CUDA errors**: Reduce BATCH_SIZE or GPU_JOBS_PER_GPU
- **OpenAI errors**: Verify API key and network connectivity

See README.md for more troubleshooting tips.

## Support

1. Read QUICKSTART.md for immediate help
2. Check README.md for comprehensive documentation
3. Review logs/  directory for execution details
4. Examine code files for parameter documentation

## License

See LICENSE file (if present) or parent directory for licensing information.

---

**Created**: March 2026  
**Purpose**: Reproducible multimodal knowledge graph retrieval and RAG experiments  
**Status**: Ready for use
