#!/bin/bash
# setup.sh - Quick setup script for MarKG RAG experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

echo "==================================="
echo "MarKG RAG Experiment Setup"
echo "==================================="
echo

# Check Python version
echo "✓ Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "  Python version: ${python_version}"
echo

# Create virtual environment if not already activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "→ Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
else
    echo "✓ Using existing virtual environment: ${VIRTUAL_ENV}"
fi
echo

# Install requirements
echo "→ Installing requirements..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r "${PROJECT_ROOT}/requirements.txt"
echo "✓ Requirements installed"
echo

# Create necessary directories
echo "→ Creating necessary directories..."
mkdir -p "${PROJECT_ROOT}/cache_embeddings"
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/rag_outputs"
mkdir -p "${PROJECT_ROOT}/images_subset_kg"
mkdir -p "${PROJECT_ROOT}/images_subset_kg_infer"
echo "✓ Directories created"
echo

# Check if data exists
echo "→ Checking data structure..."
if [ -d "${PROJECT_ROOT}/MKG-RAG-BENCH-G/test" ]; then
    test_files=$(find "${PROJECT_ROOT}/MKG-RAG-BENCH-G/test" -type f | wc -l)
    if [ "$test_files" -gt 0 ]; then
        echo "✓ Test data found (${test_files} files)"
    else
        echo "⚠ Test data directory is empty"
        echo "  Place your data in: MKG-RAG-BENCH-G/{train,val,test}/"
    fi
else
    echo "⚠ No test data found"
    echo "  Expected: MKG-RAG-BENCH-G/test/"
    echo "  Place your data in: MKG-RAG-BENCH-G/{train,val,test}/"
fi
echo

# Setup complete
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo
echo "Next steps:"
echo "1. If not already done, place your data in: MKG-RAG-BENCH-G/{train,val,test}/"
echo "   Required files per split:"
echo "   - mm_queries.jsonl, mm_corpus.jsonl, mm_qrels.tsv"
echo "   - text_queries.jsonl, text_corpus.jsonl, text_qrels.tsv"
echo
echo "2. Place your images in: images_subset_kg/"
echo
echo "3. Run retrieval evaluation:"
echo "   bash scripts/run_retrieval.sh"
echo
echo "4. Run RAG generation (requires OpenAI API key):"
echo "   export OPENAI_API_KEY='sk-...'"
echo "   bash scripts/run_generation.sh"
echo
echo "For more details, see README.md"
echo
