#!/usr/bin/env bash
set -euo pipefail

# MarKG RAG Generation Script (adapted for reproducible experiments)
# Usage: bash scripts/run_generation.sh [OPTIONS]
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

MM_QUERIES="${MM_QUERIES:-${DATA_DIR}/mm_queries.jsonl}"
MM_CORPUS="${MM_CORPUS:-${DATA_DIR}/mm_corpus.jsonl}"
MM_QRELS="${MM_QRELS:-${DATA_DIR}/mm_qrels.tsv}"

TEXT_QUERIES="${TEXT_QUERIES:-${DATA_DIR}/text_queries.jsonl}"
TEXT_CORPUS="${TEXT_CORPUS:-${DATA_DIR}/text_corpus.jsonl}"
TEXT_QRELS="${TEXT_QRELS:-${DATA_DIR}/text_qrels.tsv}"

# Legacy image roots required by MarKG retrievers
IMAGE_ROOT="${IMAGE_ROOT:-${PROJECT_ROOT}/images_subset_kg}"
INFERENCE_IMAGE_ROOT="${INFERENCE_IMAGE_ROOT:-${PROJECT_ROOT}/images_subset_kg_infer}"
mkdir -p "${INFERENCE_IMAGE_ROOT}"

# Embedding caches
CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/cache_embeddings}"
CAPTION_CACHE_PATH="${CAPTION_CACHE_PATH:-${CACHE_DIR}/caption_cache_blip.json}"

# Encoder config
MODEL_NAME="${MODEL_NAME:-clip-ViT-B-32}"
BATCH_SIZE="${BATCH_SIZE:-512}"

# For MMAnchorRetriever
N_IMG="${N_IMG:-10}"
N_TEXT="${N_TEXT:-5}"

# Deterministic split controls (optional)
DO_SPLIT="${DO_SPLIT:-0}"                 # 1 to enable deterministic split
SPLIT_SEED="${SPLIT_SEED:-markg_v1}"
EVAL_PARTITION="${EVAL_PARTITION:-test}"  # train|val|test

# Choose which splits to evaluate
RUN_MM="${RUN_MM:-1}"
RUN_TEXT="${RUN_TEXT:-1}"

# RAG knobs
RAG_TOP_K="${RAG_TOP_K:-5}"
MAX_EVAL_QUERIES="${MAX_EVAL_QUERIES:-9999999}"

# LLM config
LLM_MODEL="${LLM_MODEL:-gpt-5}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-512}"
REASONING_EFFORT="${REASONING_EFFORT:-minimal}"   # minimal|low|medium|high
VERBOSITY="${VERBOSITY:-low}"                     # low|medium|high

# Outputs
OUT_JSON_DIR="${OUT_JSON_DIR:-${PROJECT_ROOT}/rag_outputs}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${OUT_JSON_DIR}" "${LOG_DIR}"

# ----------------------------
# Parallelism controls
# ----------------------------
JOBS="${JOBS:-999999}"
GPU_IDS="${GPU_IDS:-0,1}"
GPU_JOBS_PER_GPU="${GPU_JOBS_PER_GPU:-3}"

# Which retrievers use GPU?
GPU_RETRIEVERS=(
  "SimpleTextRetriever"
  "SimpleMultimodalRetriever"
  "CaptionRetriever"
  "MMAnchorRetriever"
)

# ----------------------------
# Retrievers to evaluate
# ----------------------------
retrievers=(
  "MMAnchorRetriever"
)

sanitize_name() {
  echo "$1" | tr ' /' '__' | tr -cd '[:alnum:]_.-'
}

is_gpu_retriever() {
  local r="$1"
  for g in "${GPU_RETRIEVERS[@]}"; do
    if [[ "$g" == "$r" ]]; then
      return 0
    fi
  done
  return 1
}

# ----------------------------
# Per-GPU semaphores (named pipes)
# ----------------------------
declare -a GPU_LIST=()
declare -A SEM_FD=()   # gpu_id -> fd
declare -A SEM_NAME=() # gpu_id -> fifo_name

parse_gpu_ids() {
  IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS}"
  if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
    echo "GPU_IDS is empty" >&2
    exit 1
  fi
}

sem_init_one_gpu() {
  local gpu="$1"
  local slots="$2"
  local fifo
  fifo="$(mktemp -u)"
  mkfifo "${fifo}"

  # pick a free fd >= 20
  local fd
  for fd in {20..99}; do
    if eval "exec ${fd}<>\"${fifo}\""; then
      break
    fi
  done
  rm -f "${fifo}"

  SEM_FD["${gpu}"]="${fd}"
  SEM_NAME["${gpu}"]="${fifo}"

  # preload tokens
  for ((i=0; i<slots; i++)); do
    eval "printf '.' >&${fd}"
  done
}

sem_acquire_gpu() {
  local gpu="$1"
  local fd="${SEM_FD[${gpu}]}"
  local _t
  eval "IFS= read -r -n 1 _t <&${fd}"
}

sem_release_gpu() {
  local gpu="$1"
  local fd="${SEM_FD[${gpu}]}"
  eval "printf '.' >&${fd}"
}

cleanup() {
  for gpu in "${GPU_LIST[@]}"; do
    local fd="${SEM_FD[${gpu}]-}"
    if [[ -n "${fd}" ]]; then
      eval "exec ${fd}>&-; exec ${fd}<&-" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

parse_gpu_ids
for gpu in "${GPU_LIST[@]}"; do
  sem_init_one_gpu "${gpu}" "${GPU_JOBS_PER_GPU}"
done

# Stable mapping of retriever -> GPU (evenly distributed, deterministic)
gpu_for_retriever() {
  local r="$1"
  local sum=0
  local i ch
  for ((i=0; i<${#r}; i++)); do
    ch=$(printf "%d" "'${r:i:1}")
    sum=$((sum + ch))
  done
  local idx=$((sum % ${#GPU_LIST[@]}))
  echo "${GPU_LIST[$idx]}"
}

echo "=== MarKG RAG parallel eval started: $(date) ==="
echo "OUT_JSON_DIR=${OUT_JSON_DIR}"
echo "LOG_DIR=${LOG_DIR}"
echo "JOBS=${JOBS}"
echo "GPU_IDS=${GPU_IDS} GPU_JOBS_PER_GPU=${GPU_JOBS_PER_GPU}"
echo "IMAGE_ROOT=${IMAGE_ROOT}"
echo "INFERENCE_IMAGE_ROOT=${INFERENCE_IMAGE_ROOT}"
echo

# Python script
PY_SCRIPT="${PY_SCRIPT:-${PROJECT_ROOT}/code/MarKG_rag.py}"

# Change to project root so relative paths work
cd "${PROJECT_ROOT}"

# track PIDs
pids=()

run_one() {
  local r="$1"
  local safe_r
  safe_r="$(sanitize_name "${r}")"

  local run_out_dir="${OUT_JSON_DIR}/${safe_r}"
  mkdir -p "${run_out_dir}"

  local log_file="${LOG_DIR}/${safe_r}.log"

  # Build args
  local args=(
    --retriever "${r}"
    --rag_top_k "${RAG_TOP_K}"
    --model_name "${MODEL_NAME}"
    --batch_size "${BATCH_SIZE}"
    --image_root "${IMAGE_ROOT}"
    --inference_image_root "${INFERENCE_IMAGE_ROOT}"
    --cache_dir "${CACHE_DIR}"
    --caption_cache_path "${CAPTION_CACHE_PATH}"
    --n_img "${N_IMG}"
    --n_text "${N_TEXT}"
    --max_eval_queries "${MAX_EVAL_QUERIES}"
    --out_json_dir "${run_out_dir}"
    --llm_model "${LLM_MODEL}"
    --max_output_tokens "${MAX_OUTPUT_TOKENS}"
    --reasoning_effort "${REASONING_EFFORT}"
    --verbosity "${VERBOSITY}"
  )

  if [[ "${DO_SPLIT}" == "1" ]]; then
    args+=( --do_split --split_seed "${SPLIT_SEED}" --eval_partition "${EVAL_PARTITION}" )
  fi

  if [[ "${RUN_MM}" == "1" ]]; then
    args+=( --mm_queries "${MM_QUERIES}" --mm_corpus "${MM_CORPUS}" --mm_qrels "${MM_QRELS}" )
  fi

  if [[ "${RUN_TEXT}" == "1" ]]; then
    args+=( --text_queries "${TEXT_QUERIES}" --text_corpus "${TEXT_CORPUS}" --text_qrels "${TEXT_QRELS}" )
  fi

  local need_gpu=0
  if is_gpu_retriever "${r}"; then
    need_gpu=1
  fi

  (
    echo "=== retriever=${r} start=$(date) ==="
    echo "log_file=${log_file}"
    echo "out_dir=${run_out_dir}"
    echo "need_gpu=${need_gpu}"
    if [[ "${need_gpu}" == "1" ]]; then
      echo "GPU_IDS=${GPU_IDS} GPU_JOBS_PER_GPU=${GPU_JOBS_PER_GPU}"
    fi
    echo

    if [[ "${need_gpu}" == "1" ]]; then
      local gpu_id
      gpu_id="$(gpu_for_retriever "${r}")"

      sem_acquire_gpu "${gpu_id}"
      echo "[${r}] acquired GPU${gpu_id} slot at $(date)"
      trap 'echo "['"${r}"'] releasing GPU'"${gpu_id}"' slot at $(date)"; sem_release_gpu "'"${gpu_id}"'"' EXIT

      CUDA_VISIBLE_DEVICES="${gpu_id}" python -s -u "${PY_SCRIPT}" "${args[@]}"
    else
      python -s -u "${PY_SCRIPT}" "${args[@]}"
    fi

    echo
    echo "=== retriever=${r} finished=$(date) ==="
  ) >"${log_file}" 2>&1
}

# launch in background up to JOBS
running=0
for r in "${retrievers[@]}"; do
  while [[ "${JOBS}" -ne 999999 && "${running}" -ge "${JOBS}" ]]; do
    if wait -n 2>/dev/null; then
      running=$((running - 1))
    fi
  done

  run_one "${r}" &
  pids+=("$!")
  running=$((running + 1))
  echo "Launched retriever=${r} pid=${pids[-1]}"
done

# wait all
fail=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    fail=1
  fi
done

echo "=== MarKG RAG parallel eval finished: $(date) ==="
if [[ "${fail}" == "1" ]]; then
  echo "Some jobs failed."
  exit 1
fi
