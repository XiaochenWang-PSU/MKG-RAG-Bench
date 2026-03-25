# ✅ MarKG RAG Experiment - Package Creation Complete

## Summary

Your reproducible experiment package has been successfully created at:

```
/home/xmw5190/KG-MMRAG/MKG_Analogy/markg-experiment-reproducible/
```

## What Was Created

### 📦 Core Code (3,757 lines of Python)
- ✅ `code/MarKG_retrieval_v2.py` - 5 retriever implementations
- ✅ `code/markg_main_v2.py` - Retrieval evaluation engine
- ✅ `code/MarKG_rag.py` - RAG generation engine
- ✅ `code/prompt_builder.py` - LLM prompt formatting
- ✅ `code/utils.py` - Utility functions
- ✅ `code/splitting.py` - Train/val/test splitting
- ✅ `code/embedding_cache.py` - Embedding cache system

### 🚀 Execution Scripts
- ✅ `scripts/run_retrieval.sh` - Run retrieval evaluation
- ✅ `scripts/run_generation.sh` - Run RAG generation

### 📚 Documentation
- ✅ `QUICKSTART.md` - Get started in 5 minutes
- ✅ `README.md` - Comprehensive guide (9KB)
- ✅ `PACKAGE.md` - Package overview
- ✅ `setup.sh` - Automatic environment setup
- ✅ `requirements.txt` - All dependencies

### 📁 Data Directory
- ✅ `MKG-RAG-BENCH-G/train/` - Training data (optional)
- ✅ `MKG-RAG-BENCH-G/val/` - Validation data (optional)
- ✅ `MKG-RAG-BENCH-G/test/` - Test data (required)

**Total Size**: 224 KB (lightweight and portable)

## Three Ways to Get Started

### Option 1: Automatic Setup (Recommended)
```bash
cd markg-experiment-reproducible
bash setup.sh
```

### Option 2: Manual Setup
```bash
cd markg-experiment-reproducible
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option 3: Quick Reference
1. Read: `QUICKSTART.md` (5-minute intro)
2. Read: `README.md` (comprehensive guide)
3. Review: `scripts/run_retrieval.sh` (understand parameters)

## Key Features

✅ **Self-Contained** - Everything needed in one folder
✅ **Reproducible** - Deterministic, versioned dependencies
✅ **Modular** - Easy to customize retrievers or prompts
✅ **Well-Documented** - 3 levels of documentation
✅ **Production-Ready** - GPU parallelization, caching, error handling
✅ **Data-Driven** - Clear data format specifications
✅ **Portable** - Uses relative paths, works anywhere

## Next Steps

### 1. Prepare Your Data (Most Important!)
Place your multimodal knowledge graph data in `MKG-RAG-BENCH-G/test/`:

**Required files (per split):**
```
test/
├── mm_queries.jsonl       # Multimodal queries
├── mm_corpus.jsonl        # Multimodal knowledge graph
├── mm_qrels.tsv           # Multimodal relevance labels
├── text_queries.jsonl     # Text-only queries
├── text_corpus.jsonl      # Text-only corpus
└── text_qrels.tsv         # Text relevance labels
```

**Format specifications** are in [README.md](README.md#-prepare-your-data)

### 2. Setup Environment
```bash
cd markg-experiment-reproducible
bash setup.sh
```

### 3. Run Your First Experiment
**Retrieval Evaluation:**
```bash
bash scripts/run_retrieval.sh
```

**RAG Generation (requires OpenAI API key):**
```bash
export OPENAI_API_KEY="sk-..."
bash scripts/run_generation.sh
```

### 4. Examine Results
- Retrieval results: `results_retrieval/`
- Generation results: `rag_outputs/`
- Execution logs: `logs/`

## Five Retriever Methods Included

1. **MMAnchorRetriever** - Two-stage multimodal approach
2. **SimpleMultimodalRetriever** - Joint text-image embedding
3. **SimpleTextRetriever** - Text-only baseline
4. **CaptionRetriever** - Image caption-based retrieval
5. **RandomRetriever** - Random retrieval (baseline)

## Configuration Examples

### Change evaluation metrics
```bash
KS="10,20" bash scripts/run_retrieval.sh
```

### Use validation set
```bash
DATA_DIR=./MKG-RAG-BENCH-G/val bash scripts/run_retrieval.sh
```

### Use faster embedding model
```bash
MODEL_NAME="all-MiniLM-L6-v2" bash scripts/run_retrieval.sh
```

### Use different LLM for generation
```bash
LLM_MODEL="gpt-4o" bash scripts/run_generation.sh
```

See [README.md](README.md) for all configuration options.

## Troubleshooting Quick Links

### "ImportError: cannot import name..."
→ Make sure virtual environment is activated: `source venv/bin/activate`

### "No such file or directory: mm_queries.jsonl"
→ Check data is in `MKG-RAG-BENCH-G/test/` with correct filenames

### "CUDA out of memory"
→ Reduce batch size: `BATCH_SIZE=16 bash scripts/run_retrieval.sh`

### "OpenAI API Error"
→ Check API key: `echo $OPENAI_API_KEY` (should be non-empty)

See [README.md](README.md#troubleshooting) for more solutions.

## File Locations

```
markg-experiment-reproducible/
├── code/                      # Python implementation
├── scripts/                   # Execution scripts  
├── MKG-RAG-BENCH-G/          # DATA GOES HERE
├── QUICKSTART.md             # START HERE
├── README.md                 # Full documentation
├── PACKAGE.md                # This overview
├── setup.sh                  # Run this first
└── requirements.txt          # Dependencies
```

## Important Notes

1. **Data is Critical**: 90% of setup time is preparing data in correct format
2. **Relative Paths**: All scripts use relative paths for portability - run from project root
3. **GPU Optional**: Retrieval works on CPU, generation benefits from CUDA
4. **API Key Needed**: RAG generation requires OpenAI API account
5. **Caching Helps**: First run caches embeddings, subsequent runs are faster

## Architecture Flow

```
Your Data (MKG-RAG-BENCH-G/)
    ↓
Setup Environment (setup.sh / manual)
    ↓
RETRIEVAL PHASE (run_retrieval.sh)
    ├─ Load queries, corpus, qrels
    ├─ Build embeddings (cached)
    ├─ For each retriever method:
    │  ├─ Retrieve top-K documents
    │  ├─ Compute metrics (NDCG, P@K, R@K)
    │  └─ Save results
    ↓ (optional)
GENERATION PHASE (run_generation.sh)
    ├─ Load queries & retrieved documents
    ├─ Format into LLM prompt
    ├─ Call OpenAI API
    ├─ Compute metrics (EM, F1, Contains, BLEU)
    └─ Save results
    ↓
Results (results_retrieval/ + rag_outputs/)
```

## Support & Documentation

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md) (5 minutes)
- **Full Guide**: [README.md](README.md) (comprehensive)
- **Package Info**: [PACKAGE.md](PACKAGE.md) (this file)
- **Logs**: Check `logs/` directory for detailed execution info

## What's Included vs What You Provide

### ✅ Included in Package
- All Python code for retrieval & generation
- Shell scripts for easy execution
- Complete documentation
- Dependency specifications
- Data directory structure

### 📥 You Need to Provide
- Multimodal knowledge graph data
- Images (if using multimodal retrieval)
- OpenAI API key (for generation)
- Sufficient GPU/CPU resources

## Ready to Go! 🚀

The package is complete and ready for use. Start with:

```bash
cd markg-experiment-reproducible
cat QUICKSTART.md          # Read quick guide
bash setup.sh              # Setup environment
# ... prepare your data in MKG-RAG-BENCH-G/test/ ...
bash scripts/run_retrieval.sh  # Run first experiment
```

---

**Package Location**: `/home/xmw5190/KG-MMRAG/MKG_Analogy/markg-experiment-reproducible/`  
**Creation Date**: March 24, 2026  
**Status**: ✅ Ready for use
