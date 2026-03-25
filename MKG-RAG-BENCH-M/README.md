# MKG-RAG-BENCH-M Evaluation Suite

Self-contained code for reproducing retrieval and RAG generation results on the **MKG-RAG-BENCH-M** benchmark.

---

## Folder structure

```
MKG-RAG-BENCH-M-eval/
├── MKG-RAG-BENCH-M/              # Benchmark data (place your splits here)
│   ├── train/
│   │   ├── mm_queries.jsonl
│   │   ├── mm_corpus.jsonl
│   │   ├── mm_qrels.tsv
│   │   ├── text_queries.jsonl
│   │   ├── text_corpus.jsonl
│   │   └── text_qrels.tsv
│   ├── val/
│   │   └── (same files as train/)
│   ├── test/
│   │   └── (same files as train/)
│   └── image_mapping.csv         # IID -> Image_Path (relative or absolute)
├── main.py                       # Retrieval-only evaluation
├── rag.py                        # RAG (retrieve + generate) evaluation
├── retrieval.py                  # Retriever implementations
├── embedding_cache.py            # Embedding cache utilities
├── prompt_builder.py             # Prompt construction for LLM calls
├── run_retr.sh                   # Shell script: run retrieval baselines
├── run_rag.sh                    # Shell script: run RAG baselines
├── cache_embeddings/             # Created automatically on first run
└── logs/                         # Created automatically on first run
```

---

## Data format

Each split directory must contain six files:

| File | Description |
|---|---|
| `mm_queries.jsonl` | Multimodal queries (one JSON object per line, fields: `qid`, `query`, `is_multimodal`, `masked_type`, `head_id`, …) |
| `mm_corpus.jsonl` | Multimodal corpus (fields: `doc_id`, `head_id`, `rel_id`, `tail_id`, `head_text`, `rel_text`, `tail_text`, `triplet_text`, `image_path`) |
| `mm_qrels.tsv` | Relevance judgements for MM queries: `qid  doc_id  relevance` |
| `text_queries.jsonl` | Text-only queries (same schema, `is_multimodal=false`) |
| `text_corpus.jsonl` | Text-only corpus (same schema, no `image_path` required) |
| `text_qrels.tsv` | Relevance judgements for text queries |

`image_mapping.csv` maps entity IDs (`IID`) to image paths (`Image_Path`):

```csv
IID,Image_Path
I50978146,images/I50978146.jpg
...
```

If `Image_Path` is relative, it will be joined with `IMAGE_MAP_PREFIX` (default `MKG-RAG-BENCH-M/`).

---

## Requirements

```bash
pip install torch sentence-transformers transformers pillow tqdm openai
```

For CaptionRetriever, BLIP weights are downloaded automatically from Hugging Face (`Salesforce/blip-image-captioning-base`).

For the RAG script, set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Reproducing retrieval results

Run all retrieval baselines on the **test** split (default):

```bash
bash run_retr.sh
```

Key environment variables (all optional — defaults shown in parentheses):

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `MKG-RAG-BENCH-M/test` | Path to the split to evaluate |
| `IMAGE_MAP_PATH` | `MKG-RAG-BENCH-M/image_mapping.csv` | Image mapping CSV |
| `IMAGE_MAP_PREFIX` | `MKG-RAG-BENCH-M/` | Prefix prepended to relative image paths |
| `MODEL_NAME` | `clip-ViT-B-32` | CLIP/SentenceTransformer encoder |
| `BATCH_SIZE` | `64` | Encoding batch size |
| `KS` | `5,10,20,50,100` | Evaluation K values |
| `RUN_MM` | `1` | Evaluate multimodal split |
| `RUN_TEXT` | `1` | Evaluate text-only split |
| `LOG_FILE` | `logs/retrieval_out.log` | Log file path |

Example — evaluate on the **val** split with a smaller batch size:

```bash
DATA_DIR=MKG-RAG-BENCH-M/val BATCH_SIZE=32 bash run_retr.sh
```

Metrics reported: **NDCG@K**, **Precision@K**, **Recall@K** for each K in `KS`, across three splits:
- `mm_only` — multimodal queries × multimodal corpus
- `text_only` — text queries × text corpus
- `hybrid` — combined mm + text queries and corpus

---

## Reproducing RAG generation results

Run all RAG baselines on the **test** split (default):

```bash
export OPENAI_API_KEY="sk-..."
bash run_rag.sh
```

Additional key variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `gpt-4o` | OpenAI model name |
| `RAG_TOP_K` | `5` | Number of retrieved items passed to the LLM |
| `MAX_OUTPUT_TOKENS` | `512` | Max tokens in the LLM response |
| `REASONING_EFFORT` | `minimal` | `minimal`/`low`/`medium`/`high` (o-series models) |
| `OUT_JSON_DIR` | `rag_outputs` | Directory for retrieved + generated JSON logs |

Metrics reported: **EM**, **Token F1**, **Contains@1**, **BLEU-4** (best over gold answers).

Output JSON logs are saved under `rag_outputs/<RetrieverName>/`:
- `<split>.<RetrieverName>.retrieved.json` — retrieved context per query
- `<split>.<RetrieverName>.generated.json` — generated answers + scores per query

---

## Retriever baselines

| Name | Description |
|---|---|
| `RandomRetriever` | Random baseline |
| `SimpleTextRetriever` | Text-only cosine similarity over triplet text |
| `SimpleMultimodalRetriever` | Avg(text, image) embedding cosine similarity |
| `MMAnchorRetriever` | Two-stage: image-head retrieval → text reranking |
| `CaptionRetriever` | Text retrieval with BLIP captions replacing image head text |

Enable/disable retrievers by editing the `retrievers=(...)` array in the shell scripts.

---

## Running individual Python scripts

**Retrieval only:**

```bash
python main.py \
  --data_dir MKG-RAG-BENCH-M/test \
  --retriever MMAnchorRetriever \
  --ks 5,10,20,50,100 \
  --model_name clip-ViT-B-32 \
  --image_map_path MKG-RAG-BENCH-M/image_mapping.csv \
  --image_map_prefix MKG-RAG-BENCH-M/
```

**RAG generation:**

```bash
python rag.py \
  --data_dir MKG-RAG-BENCH-M/test \
  --retriever MMAnchorRetriever \
  --rag_top_k 5 \
  --llm_model gpt-4o \
  --image_map_path MKG-RAG-BENCH-M/image_mapping.csv \
  --image_map_prefix MKG-RAG-BENCH-M/ \
  --out_json_dir rag_outputs/MMAnchorRetriever
```
