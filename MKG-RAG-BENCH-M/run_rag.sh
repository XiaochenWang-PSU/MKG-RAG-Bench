#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1

# Reduce CUDA allocator fragmentation + make caching less sticky
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128,expandable_segments:True"
export CUDA_MODULE_LOADING=LAZY

# ----------------------------
# Paths (edit if needed)
# ----------------------------
# DATA_DIR points to one split of the benchmark.
# Each split dir must contain: mm_queries.jsonl, mm_corpus.jsonl, mm_qrels.tsv
#                              text_queries.jsonl, text_corpus.jsonl, text_qrels.tsv
DATA_DIR="${DATA_DIR:-MKG-RAG-BENCH-M/test}"   # change to train/val/test as needed

MM_QUERIES="${MM_QUERIES:-${DATA_DIR}/mm_queries.jsonl}"
MM_CORPUS="${MM_CORPUS:-${DATA_DIR}/mm_corpus.jsonl}"
MM_QRELS="${MM_QRELS:-${DATA_DIR}/mm_qrels.tsv}"

TEXT_QUERIES="${TEXT_QUERIES:-${DATA_DIR}/text_queries.jsonl}"
TEXT_CORPUS="${TEXT_CORPUS:-${DATA_DIR}/text_corpus.jsonl}"
TEXT_QRELS="${TEXT_QRELS:-${DATA_DIR}/text_qrels.tsv}"

# Image mapping: IID -> absolute image path
# Place image_mapping.csv inside MKG-RAG-BENCH-M/ or override with IMAGE_MAP_PATH.
IMAGE_MAP_PATH="${IMAGE_MAP_PATH:-MKG-RAG-BENCH-M/image_mapping.csv}"
IMAGE_MAP_PREFIX="${IMAGE_MAP_PREFIX:-MKG-RAG-BENCH-M/}"

# Embedding caches (created automatically on first run)
CACHE_DIR="${CACHE_DIR:-cache_embeddings}"
CAPTION_CACHE_PATH="${CAPTION_CACHE_PATH:-${CACHE_DIR}/caption_cache_blip.json}"

# Encoder config (shared across baselines)
MODEL_NAME="${MODEL_NAME:-clip-ViT-B-32}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# For MMAnchorRetriever
N_IMG="${N_IMG:-10}"
N_TEXT="${N_TEXT:-5}"

# The benchmark is already split into train/val/test directories.
# DO_SPLIT=0 means we evaluate all queries in DATA_DIR (no in-script re-splitting).
DO_SPLIT="${DO_SPLIT:-0}"
SPLIT_SEED="${SPLIT_SEED:-markg_v1}"
EVAL_PARTITION="${EVAL_PARTITION:-test}"

# Choose which modality splits to evaluate
RUN_MM="${RUN_MM:-1}"
RUN_TEXT="${RUN_TEXT:-1}"

# RAG knobs
RAG_TOP_K="${RAG_TOP_K:-5}"
MAX_EVAL_QUERIES="${MAX_EVAL_QUERIES:-9999999}"

# LLM config
LLM_MODEL="${LLM_MODEL:-gpt-4o}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-512}"
REASONING_EFFORT="${REASONING_EFFORT:-minimal}"   # minimal|low|medium|high
VERBOSITY="${VERBOSITY:-low}"                     # low|medium|high

# Outputs
OUT_JSON_DIR="${OUT_JSON_DIR:-rag_outputs}"

# Retrievers to evaluate (match rag.py --retriever names)
retrievers=(
  # "None"
  # "RandomRetriever"
  # "SimpleTextRetriever"
  "MMAnchorRetriever"
  "SimpleMultimodalRetriever"
  "CaptionRetriever"
)

# ----------------------------
# One consolidated nohup log
# ----------------------------
LOG_FILE="${LOG_FILE:-logs/rag_out.log}"
mkdir -p "$(dirname "${LOG_FILE}")"

# ----------------------------
# If not already running under nohup, re-exec under nohup
# ----------------------------
if [[ -z "${_RAG_NOHUP:-}" ]]; then
  export _RAG_NOHUP=1
  echo "Launching under nohup -> ${LOG_FILE}"
  nohup bash "$0" "$@" >> "${LOG_FILE}" 2>&1 &
  echo "PID=$!"
  exit 0
fi

echo "=== MKG-RAG-BENCH-M RAG eval started: $(date) ==="
echo "LOG_FILE=${LOG_FILE}"
echo "DATA_DIR=${DATA_DIR}"
echo "RUN_MM=${RUN_MM} RUN_TEXT=${RUN_TEXT}"
echo "MODEL_NAME=${MODEL_NAME} BATCH_SIZE=${BATCH_SIZE}"
echo "CACHE_DIR=${CACHE_DIR} CAPTION_CACHE_PATH=${CAPTION_CACHE_PATH}"
echo "IMAGE_MAP_PATH=${IMAGE_MAP_PATH} IMAGE_MAP_PREFIX=${IMAGE_MAP_PREFIX}"
echo "N_IMG=${N_IMG} N_TEXT=${N_TEXT}"
echo "DO_SPLIT=${DO_SPLIT} SPLIT_SEED=${SPLIT_SEED} EVAL_PARTITION=${EVAL_PARTITION}"
echo "RAG_TOP_K=${RAG_TOP_K} MAX_EVAL_QUERIES=${MAX_EVAL_QUERIES}"
echo "LLM_MODEL=${LLM_MODEL} MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS} REASONING_EFFORT=${REASONING_EFFORT} VERBOSITY=${VERBOSITY}"
echo "OUT_JSON_DIR=${OUT_JSON_DIR}"
echo

sanitize_name() {
  echo "$1" | tr ' /' '__' | tr -cd '[:alnum:]_.-'
}

for r in "${retrievers[@]}"; do
  echo "=== retriever=${r} | model=${MODEL_NAME} | rag_top_k=${RAG_TOP_K} | llm=${LLM_MODEL} | start=$(date) ==="
  safe_r="$(sanitize_name "${r}")"

  # per-retriever output subdir to avoid overwrites
  run_out_dir="${OUT_JSON_DIR}/${safe_r}"
  mkdir -p "${run_out_dir}"

  args=(
    --retriever "${r}"
    --rag_top_k "${RAG_TOP_K}"
    --model_name "${MODEL_NAME}"
    --batch_size "${BATCH_SIZE}"
    --cache_dir "${CACHE_DIR}"
    --caption_cache_path "${CAPTION_CACHE_PATH}"
    --image_map_path "${IMAGE_MAP_PATH}"
    --image_map_prefix "${IMAGE_MAP_PREFIX}"
    --n_img "${N_IMG}"
    --n_text "${N_TEXT}"
    --max_eval_queries "${MAX_EVAL_QUERIES}"
    --llm_model "${LLM_MODEL}"
    --max_output_tokens "${MAX_OUTPUT_TOKENS}"
    --reasoning_effort "${REASONING_EFFORT}"
    --verbosity "${VERBOSITY}"
    --out_json_dir "${run_out_dir}"
  )

  # Optional in-script split (set DO_SPLIT=1 if DATA_DIR is a single flat split)
  if [[ "${DO_SPLIT}" == "1" ]]; then
    args+=( --do_split --split_seed "${SPLIT_SEED}" --eval_partition "${EVAL_PARTITION}" )
  fi

  if [[ "${RUN_MM}" == "1" ]]; then
    args+=( --mm_queries "${MM_QUERIES}" --mm_corpus "${MM_CORPUS}" --mm_qrels "${MM_QRELS}" )
  fi

  if [[ "${RUN_TEXT}" == "1" ]]; then
    args+=( --text_queries "${TEXT_QUERIES}" --text_corpus "${TEXT_CORPUS}" --text_qrels "${TEXT_QRELS}" )
  fi

  python -s -u rag.py "${args[@]}"

  echo "=== retriever=${r} finished: $(date) ==="
  echo

  sleep 1
done

echo "=== MKG-RAG-BENCH-M RAG eval finished: $(date) ==="
