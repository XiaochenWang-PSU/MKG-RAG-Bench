#!/usr/bin/env bash
set -euo pipefail

# MarKG Retrieval Evaluation Script (adapted for reproducible experiments)
# Usage: bash scripts/run_retrieval.sh [OPTIONS]
# Environment variables for configuration see below

export PYTHONNOUSERSITE=1

# Reduce CUDA allocator fragmentation + make caching less sticky
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True"
export CUDA_MODULE_LOADING=LAZY

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# ----------------------------
# Paths (edit if needed)
# ----------------------------
# Data directory containing *_queries.jsonl, *_corpus.jsonl, *_qrels.tsv files
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/MKG-RAG-BENCH-G/test}"

# These should be present in DATA_DIR
MM_QUERIES="${MM_QUERIES:-${DATA_DIR}/mm_queries.jsonl}"
MM_CORPUS="${MM_CORPUS:-${DATA_DIR}/mm_corpus.jsonl}"
MM_QRELS="${MM_QRELS:-${DATA_DIR}/mm_qrels.tsv}"

TEXT_QUERIES="${TEXT_QUERIES:-${DATA_DIR}/text_queries.jsonl}"
TEXT_CORPUS="${TEXT_CORPUS:-${DATA_DIR}/text_corpus.jsonl}"
TEXT_QRELS="${TEXT_QRELS:-${DATA_DIR}/text_qrels.tsv}"

# Image roots
IMAGE_ROOT="${IMAGE_ROOT:-${PROJECT_ROOT}/images_subset_kg}"
INFER_IMAGE_ROOT="${INFER_IMAGE_ROOT:-${PROJECT_ROOT}/images_subset_kg_infer}"

# Embedding caches
CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/cache_embeddings}"
CAPTION_CACHE_PATH="${CAPTION_CACHE_PATH:-${CACHE_DIR}/caption_cache_blip.json}"

# Eval Ks
KS="${KS:-5,10,20,50,100}"

# Encoder config (shared across baselines)
MODEL_NAME="${MODEL_NAME:-clip-ViT-B-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# For MMAnchorRetriever (only used when retriever=MMAnchorRetriever)
N_IMG="${N_IMG:-10}"
N_TEXT="${N_TEXT:-5}"

# Splitting (optional)
DO_SPLIT="${DO_SPLIT:-0}"                 # 1 to enable deterministic split
SPLIT_SEED="${SPLIT_SEED:-markg_v1}"
EVAL_PARTITION="${EVAL_PARTITION:-test}"  # train|val|test

# Cross-settings to run (all default ON)
RUN_TEXT_ON_HYBRID="${RUN_TEXT_ON_HYBRID:-1}"
RUN_MM_ON_TEXT="${RUN_MM_ON_TEXT:-1}"
RUN_MM_ON_HYBRID="${RUN_MM_ON_HYBRID:-1}"

# Retrievers to evaluate (match MarKG_retrieval_v2.py class names)
retrievers=(
  "MMAnchorRetriever"
  "SimpleMultimodalRetriever"
  "SimpleTextRetriever"
  "RandomRetriever"
  "CaptionRetriever"
)

# Log and output directories
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/retrieval_eval.log}"

# ----------------------------
# If this script is NOT already running under nohup, re-exec under nohup
# ----------------------------
if [[ -z "${_MARKG_NOHUP:-}" ]]; then
  export _MARKG_NOHUP=1
  echo "Launching under nohup -> ${LOG_FILE}"
  nohup bash "$0" "$@" >> "${LOG_FILE}" 2>&1 &
  echo "PID=$!"
  exit 0
fi

echo "=== MarKG cross retrieval eval started: $(date) ==="
echo "LOG_FILE=${LOG_FILE}"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "DATA_DIR=${DATA_DIR}"
echo "MODEL_NAME=${MODEL_NAME} BATCH_SIZE=${BATCH_SIZE} KS=${KS}"
echo "IMAGE_ROOT=${IMAGE_ROOT} INFER_IMAGE_ROOT=${INFER_IMAGE_ROOT}"
echo "CACHE_DIR=${CACHE_DIR} CAPTION_CACHE_PATH=${CAPTION_CACHE_PATH}"
echo "N_IMG=${N_IMG} N_TEXT=${N_TEXT}"
echo "DO_SPLIT=${DO_SPLIT} SPLIT_SEED=${SPLIT_SEED} EVAL_PARTITION=${EVAL_PARTITION}"
echo "RUN_TEXT_ON_HYBRID=${RUN_TEXT_ON_HYBRID} RUN_MM_ON_TEXT=${RUN_MM_ON_TEXT} RUN_MM_ON_HYBRID=${RUN_MM_ON_HYBRID}"
echo

sanitize_name() {
  echo "$1" | tr ' /' '__' | tr -cd '[:alnum:]_.-'
}

# Python script to run
PY_SCRIPT="${PY_SCRIPT:-${PROJECT_ROOT}/code/markg_main_v2.py}"

# Change to project root so relative paths work
cd "${PROJECT_ROOT}"

for r in "${retrievers[@]}"; do
  echo "=== retriever=${r} | start=$(date) ==="
  safe_r="$(sanitize_name "${r}")"

  args=(
    --retriever "${r}"
    --ks "${KS}"
    --model_name "${MODEL_NAME}"
    --batch_size "${BATCH_SIZE}"
    --image_root "${IMAGE_ROOT}"
    --inference_image_root "${INFER_IMAGE_ROOT}"
    --cache_dir "${CACHE_DIR}"
    --caption_cache_path "${CAPTION_CACHE_PATH}"
    --n_img "${N_IMG}"
    --n_text "${N_TEXT}"
    --mm_queries "${MM_QUERIES}"
    --mm_corpus "${MM_CORPUS}"
    --mm_qrels "${MM_QRELS}"
    --text_queries "${TEXT_QUERIES}"
    --text_corpus "${TEXT_CORPUS}"
    --text_qrels "${TEXT_QRELS}"
  )

  if [[ "${DO_SPLIT}" == "1" ]]; then
    args+=( --do_split --split_seed "${SPLIT_SEED}" --eval_partition "${EVAL_PARTITION}" )
  fi

  echo "[enabled] text_on_hybrid=${RUN_TEXT_ON_HYBRID} mm_on_text=${RUN_MM_ON_TEXT} mm_on_hybrid=${RUN_MM_ON_HYBRID}"

  python -s -u "${PY_SCRIPT}" "${args[@]}"

  echo "=== retriever=${r} finished: $(date) ==="
  echo
  sleep 1
done

echo "=== MarKG cross retrieval eval finished: $(date) ==="
