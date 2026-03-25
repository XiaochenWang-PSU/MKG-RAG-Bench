#!/usr/bin/env python3
"""
rag_eval_main.py

RAG (retrieve + generate) evaluation on the SAME retrieval_eval dataset format as your current
retrieval-only script:

Files per split:
  - mm_queries.jsonl,   mm_corpus.jsonl,   mm_qrels.tsv
  - text_queries.jsonl, text_corpus.jsonl, text_qrels.tsv
  - hybrid = merged (mm + text) with offsets to avoid ID collisions

Query format example (mm_queries.jsonl):
  {"qid": 0, "query": "[IMAGE] ...", "is_multimodal": true, "masked_type": "tail",
   "head_id":"I50978146", "rel_id":"POSITIVE", "tail_id": null, ...}

Corpus format example (mm_corpus.jsonl):
  {"doc_id":0, "head_id":"I50978146", "rel_id":"POSITIVE", "tail_id":"C0032285",
   "tail_text":"pneumonia (diagnosis)", "image_path":"/abs/...", "triplet_text":"..."}

Qrels format example (mm_qrels.tsv):
  qid  doc_id  rel

RAG logic:
  1) Build sample = {"question", "image_path", "is_multimodal", "answer"}.
     - question is open-ended (always).
     - image_path resolved via image_mapping.csv (IID -> Image_Path) if is_multimodal.
     - ground-truth answer is derived from the masked_type:
         masked_type == "tail" -> all relevant docs' tail_text
         masked_type == "rel"  -> all relevant docs' rel_text
         masked_type == "head" -> all relevant docs' head_text
     We keep multiple gold answers (one per positive doc) and score against the best match.

  2) Retrieve top-K items using your retriever.search(sample, k, ...).
     - We materialize retrieved items into dicts containing triplet_text/head_text/rel_text/tail_text/image_path.

  3) Build prompt using your existing prompt_builder functions (you said you'll put it in the same directory).
     - build_multimodal_input_for_sample_open(sample)
     - build_rag_prompt(retrieved_items, image_id_to_path)
     We prepend the rag_prompt to the user content (same as your reference script).

  4) Call LLM (OpenAI Responses API) to generate.
  5) Evaluate open-ended metrics (best over gold set):
       - EM (exact match after normalization)
       - Token F1 (SQuAD-style)
       - Contains@1 (prediction contains any gold span, normalized)

Outputs:
  - prints split metrics + LaTeX-friendly rows
  - saves retrieved + generated logs as JSON next to --out_json_dir
"""

from __future__ import annotations


import random
import io
from PIL import Image, ImageEnhance, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # optional but helpful


# ----------------------------
# Deterministic augmentation (no "swap")
# ----------------------------
_AUG_CHOICES_NO_SWAP = ["crop", "rotate", "jitter", "crop+jitter", "rotate+jitter"]


def _stable_u32(seed_text: str) -> int:
    h = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _to_rgb_jpeg_safe(im: Image.Image) -> Image.Image:
    if im.mode == "RGB":
        return im
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    return im.convert("RGB")


def _augment_crop(im: Image.Image, rng: random.Random, frac: float = 0.75) -> Image.Image:
    w, h = im.size
    if w < 4 or h < 4:
        return im
    new_w = max(1, int(w * frac))
    new_h = max(1, int(h * frac))
    if new_w >= w or new_h >= h:
        return im
    left = rng.randint(0, w - new_w)
    top = rng.randint(0, h - new_h)
    return im.crop((left, top, left + new_w, top + new_h))


def _augment_rotate(im: Image.Image, rng: random.Random, max_deg: float = 10.0) -> Image.Image:
    deg = rng.uniform(-max_deg, max_deg)
    return im.rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))


def _augment_jitter(im: Image.Image, rng: random.Random) -> Image.Image:
    b = rng.uniform(0.85, 1.15)
    c = rng.uniform(0.85, 1.15)
    s = rng.uniform(0.85, 1.15)
    out = ImageEnhance.Brightness(im).enhance(b)
    out = ImageEnhance.Contrast(out).enhance(c)
    out = ImageEnhance.Color(out).enhance(s)
    return out


def augment_query_image_deterministic(
    *,
    img_path: str,
    qid: int,
    question: str,
    head_id: str,
    out_dir: str,
) -> Optional[str]:
    if not img_path or not os.path.isfile(img_path):
        return None

    os.makedirs(out_dir, exist_ok=True)

    seed = _stable_u32(f"aug::{qid}::{head_id}::{img_path}::{question}")
    rng = random.Random(seed)
    choice = _AUG_CHOICES_NO_SWAP[rng.randrange(len(_AUG_CHOICES_NO_SWAP))]

    out_path = os.path.join(out_dir, f"qid_{qid}__{choice}__{seed:08x}.jpg")
    if os.path.isfile(out_path):
        return out_path

    try:
        with Image.open(img_path) as im0:
            im = _to_rgb_jpeg_safe(im0)

            if choice == "crop":
                im = _augment_crop(im, rng)
            elif choice == "rotate":
                im = _augment_rotate(im, rng)
            elif choice == "jitter":
                im = _augment_jitter(im, rng)
            elif choice == "crop+jitter":
                im = _augment_crop(im, rng)
                im = _augment_jitter(im, rng)
            elif choice == "rotate+jitter":
                im = _augment_rotate(im, rng)
                im = _augment_jitter(im, rng)

            im.save(out_path, format="JPEG", quality=92, optimize=True)
        return out_path
    except Exception:
        return None


import argparse
import csv
import hashlib
import json
import math
import os
import re
import string
import time
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from tqdm import tqdm

# ---- import your retrievers (same as retrieval-only script) ----
from retrieval import (  # noqa: E402
    MMAnchorRetriever,
    SimpleMultimodalRetriever,
    SimpleTextRetriever,
    RandomRetriever,
    CaptionRetriever,
)

# ---- prompt builder (lives in the same directory as this file) ----
from prompt_builder import (
    build_multimodal_input_for_sample_open,
    build_rag_prompt,
)

# ---- OpenAI client (same style as your reference) ----
from openai import OpenAI  # type: ignore

client = OpenAI()


# ----------------------------
# I/O: load jsonl / tsv qrels
# ----------------------------
def load_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path} at line {line_no}: {e}") from e
    return items


def load_queries(path: Path) -> Dict[int, dict]:
    data = load_jsonl(path)
    out: Dict[int, dict] = {}
    for obj in data:
        if "qid" not in obj:
            raise KeyError(f"Missing 'qid' in queries file: {path}")
        qid = int(obj["qid"])
        out[qid] = obj
    return out


def load_corpus(path: Path) -> Dict[int, dict]:
    data = load_jsonl(path)
    out: Dict[int, dict] = {}
    for obj in data:
        if "doc_id" not in obj:
            raise KeyError(f"Missing 'doc_id' in corpus file: {path}")
        doc_id = int(obj["doc_id"])
        out[doc_id] = obj
    return out


def load_qrels_tsv(path: Path) -> Dict[int, Dict[int, int]]:
    """
    Returns:
      qrels[qid][doc_id] = relevance (int)
    """
    qrels: Dict[int, Dict[int, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Bad qrels line in {path} at line {line_no}: '{line}'")
            qid = int(parts[0])
            doc_id = int(parts[1])
            rel = int(parts[2])
            qrels.setdefault(qid, {})[doc_id] = rel
    return qrels


# ----------------------------
# Image mapping: IID -> path
# ----------------------------
def load_image_mapping_csv(
    path: Path,
    *,
    iid_col: str = "IID",
    image_path_col: str = "Image_Path",
    prefix: str = "/data/xiaochen/",
) -> Dict[str, str]:
    """
    image_mapping.csv example:
      IID,Image_Path
      I53683003,physionet.org/files/...

    Returns:
      mapping[iid] = absolute_path
    """
    if not path.exists():
        raise FileNotFoundError(f"image_mapping.csv not found: {path}")

    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or iid_col not in reader.fieldnames or image_path_col not in reader.fieldnames:
            raise ValueError(
                f"image_mapping.csv must have columns {iid_col!r} and {image_path_col!r}. "
                f"Got: {reader.fieldnames}"
            )
        for row in reader:
            iid = str(row[iid_col]).strip()
            rel = str(row[image_path_col]).strip()
            if not iid or not rel:
                continue
            abs_path = rel if os.path.isabs(rel) else os.path.join(prefix, rel)
            mapping[iid] = abs_path
    return mapping


# ----------------------------
# Deterministic split (8/1/1)
# ----------------------------
def split_qids_deterministic(
    qids: List[int],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: str = "markg_v1",
) -> Tuple[List[int], List[int], List[int]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train/val/test ratios must sum to 1.0")

    def key(qid: int) -> str:
        s = f"{seed}::{qid}".encode("utf-8")
        return hashlib.sha1(s).hexdigest()

    qids_sorted = sorted(qids, key=key)
    n = len(qids_sorted)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = qids_sorted[:n_train]
    val = qids_sorted[n_train : n_train + n_val]
    test = qids_sorted[n_train + n_val :]
    return train, val, test


# ----------------------------
# Build retrieval dataset + vocab maps
# ----------------------------
def build_retrieval_inputs_from_corpus(
    corpus: Dict[int, dict],
) -> Tuple[List[Tuple[str, str, str]], List[int], Dict[str, str], Dict[str, str]]:
    """
    Returns:
      retrieval_dataset: list[(head_id, rel_id, tail_id)] in sorted doc_id order
      idx2docid: list[doc_id] aligned with retrieval_dataset indices
      entity2text: map entity_id -> entity_text
      relation2text: map rel_id -> rel_text
    """
    doc_ids_sorted = sorted(corpus.keys())
    retrieval_dataset: List[Tuple[str, str, str]] = []
    idx2docid: List[int] = []
    entity2text: Dict[str, str] = {}
    relation2text: Dict[str, str] = {}

    for doc_id in doc_ids_sorted:
        d = corpus[doc_id]
        h = str(d.get("head_id", ""))
        r = str(d.get("rel_id", ""))
        t = str(d.get("tail_id", ""))

        retrieval_dataset.append((h, r, t))
        idx2docid.append(doc_id)

        ht = d.get("head_text", None)
        rt = d.get("rel_text", None)
        tt = d.get("tail_text", None)
        if h and ht is not None and h not in entity2text:
            entity2text[h] = str(ht)
        if t and tt is not None and t not in entity2text:
            entity2text[t] = str(tt)
        if r and rt is not None and r not in relation2text:
            relation2text[r] = str(rt)

    return retrieval_dataset, idx2docid, entity2text, relation2text


# ----------------------------
# Hybrid merge (avoid qid/doc_id collisions)
# ----------------------------
def offset_queries(queries: Dict[int, dict], qid_offset: int) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for qid, q in queries.items():
        q2 = dict(q)
        q2["qid"] = int(qid) + qid_offset
        out[int(qid) + qid_offset] = q2
    return out


def offset_corpus(corpus: Dict[int, dict], doc_offset: int) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for doc_id, d in corpus.items():
        d2 = dict(d)
        d2["doc_id"] = int(doc_id) + doc_offset
        out[int(doc_id) + doc_offset] = d2
    return out


def offset_qrels(
    qrels: Dict[int, Dict[int, int]],
    qid_offset: int,
    doc_offset: int,
) -> Dict[int, Dict[int, int]]:
    out: Dict[int, Dict[int, int]] = {}
    for qid, rels in qrels.items():
        new_qid = int(qid) + qid_offset
        out[new_qid] = {int(doc_id) + doc_offset: int(rel) for doc_id, rel in rels.items()}
    return out


def merge_hybrid(
    *,
    mm_queries: Dict[int, dict],
    mm_corpus: Dict[int, dict],
    mm_qrels: Dict[int, Dict[int, int]],
    text_queries: Dict[int, dict],
    text_corpus: Dict[int, dict],
    text_qrels: Dict[int, Dict[int, int]],
) -> Tuple[Dict[int, dict], Dict[int, dict], Dict[int, Dict[int, int]]]:
    """
    Merge mm+text into one combined dataset with offsets:
      - keep mm as-is
      - offset text qids and doc_ids so there are no collisions
    """
    mm_max_qid = max(mm_queries.keys()) if mm_queries else -1
    mm_max_doc = max(mm_corpus.keys()) if mm_corpus else -1

    qid_offset = mm_max_qid + 1
    doc_offset = mm_max_doc + 1

    text_queries_off = offset_queries(text_queries, qid_offset)
    text_corpus_off = offset_corpus(text_corpus, doc_offset)
    text_qrels_off = offset_qrels(text_qrels, qid_offset, doc_offset)

    merged_queries = dict(mm_queries)
    merged_queries.update(text_queries_off)

    merged_corpus = dict(mm_corpus)
    merged_corpus.update(text_corpus_off)

    merged_qrels = dict(mm_qrels)
    merged_qrels.update(text_qrels_off)

    return merged_queries, merged_corpus, merged_qrels


# ----------------------------
# Retriever factory (same as before)
# ----------------------------
def build_retriever(
    name: str,
    *,
    retrieval_dataset: List[Tuple[str, str, str]],
    entity2text: Dict[str, str],
    relation2text: Dict[str, str],
    model_name: str,
    batch_size: int,
    cache_dir: str,
    caption_cache_path: str,
) -> object:
    if name == "MMAnchorRetriever":
        return MMAnchorRetriever(
            retrieval_dataset=retrieval_dataset,
            entity2text=entity2text,
            relation2text=relation2text,
            model_name=model_name,
            batch_size=batch_size,
            show_progress=False,
            cache_dir=cache_dir,
        )
    if name == "SimpleMultimodalRetriever":
        return SimpleMultimodalRetriever(
            retrieval_dataset=retrieval_dataset,
            entity2text=entity2text,
            relation2text=relation2text,
            model_name=model_name,
            batch_size=batch_size,
        )
    if name == "SimpleTextRetriever":
        return SimpleTextRetriever(
            retrieval_dataset=retrieval_dataset,
            entity2text=entity2text,
            relation2text=relation2text,
            model_name=model_name,
            batch_size=batch_size,
            cache_dir=cache_dir,
        )
    if name == "RandomRetriever":
        return RandomRetriever(retrieval_dataset=retrieval_dataset)
    if name == "CaptionRetriever":
        return CaptionRetriever(
            retrieval_dataset=retrieval_dataset,
            entity2text=entity2text,
            relation2text=relation2text,
            model_name=model_name,
            batch_size=batch_size,
            cache_dir=cache_dir,
            caption_cache_path=caption_cache_path,
        )
    raise ValueError(
        f"Unknown retriever '{name}'. Choose from: "
        "MMAnchorRetriever, SimpleMultimodalRetriever, SimpleTextRetriever, RandomRetriever, CaptionRetriever"
    )


def _call_retriever_search_items(
    retriever: object,
    *,
    sample: Dict[str, object],
    top_k: int,
    n_img: int,
    n_text: int,
) -> List[dict]:
    """
    Returns the raw list of items produced by retriever.search(...).
    Convention: each item contains at least {"index": int, ...}
    """
    cls = retriever.__class__.__name__
    if cls == "MMAnchorRetriever":
        return retriever.search(sample, top_k, n_img=n_img, n_text=n_text, return_unique=True)
    return retriever.search(sample, top_k)


def materialize_retrieved_items(
    raw_items: List[dict],
    *,
    idx2docid: List[int],
    corpus: Dict[int, dict],
) -> List[dict]:
    """
    Enrich retriever-returned items with corpus fields like triplet_text, image_path, etc.
    """
    out: List[dict] = []
    for it in raw_items:
        if "index" not in it:
            continue
        idx = int(it["index"])
        if not (0 <= idx < len(idx2docid)):
            continue
        doc_id = idx2docid[idx]
        doc = corpus.get(doc_id, {})
        merged = dict(it)
        merged["doc_id"] = doc_id
        # common fields
        for k in ["head_id", "rel_id", "tail_id", "head_text", "rel_text", "tail_text", "triplet_text", "image_path"]:
            if k in doc:
                merged[k] = doc[k]
        out.append(merged)
    return out


# ----------------------------
# LLM call
# ----------------------------
def get_llm_result(
    prompt_messages: Any,
    *,
    model: str,
    max_output_tokens: int,
    reasoning_effort: str,
    verbosity: str,
) -> Tuple[str, Any]:
    """
    Always returns (text, usage_or_error_like). Never raises.
    """
    try:
        resp = client.responses.create(
            model=model,
            input=prompt_messages,
            max_output_tokens=max_output_tokens,
            reasoning={"effort": reasoning_effort},  # minimal/low/medium/high
            text={"verbosity": verbosity},           # low/medium/high
        )
        text = (resp.output_text or "").strip()
        usage = resp.usage
        return text, usage
    except Exception as e:
        return "", {"exception": repr(e)}


# ----------------------------
# Open-ended scoring (best over gold set)
# ----------------------------
_ARTICLES = {"a", "an", "the"}


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _ARTICLES]
    return " ".join(tokens)


def _token_f1(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0
    from collections import Counter

    pc = Counter(pred_toks)
    gc = Counter(gold_toks)
    common = pc & gc
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / float(len(pred_toks))
    recall = num_same / float(len(gold_toks))
    return 2 * precision * recall / (precision + recall)


def _normalize(s: str) -> str:
    """Lightweight normalize for EM/Contains; keep consistent with your existing one if present."""
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _simple_tokens(s: str) -> List[str]:
    """Tokenize for BLEU/F1. Keep simple + stable; match your _token_f1 assumptions if possible."""
    s = _normalize(s)
    if not s:
        return []
    # word-ish tokens
    return re.findall(r"[a-z0-9]+", s)

def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens) - n + 1)))

def _bleu_score(pred: str, ref: str, max_n: int = 4, smooth: bool = True) -> float:
    """
    Sentence-level BLEU (BLEU-4 by default) with brevity penalty.
    Simple smoothing: add-1 to ngram matches + totals (when smooth=True).
    Returns BLEU in [0, 1].
    """
    pred_toks = _simple_tokens(pred)
    ref_toks = _simple_tokens(ref)

    if len(pred_toks) == 0 or len(ref_toks) == 0:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_counts = _ngram_counts(pred_toks, n)
        ref_counts = _ngram_counts(ref_toks, n)

        if len(pred_counts) == 0:
            precisions.append(0.0)
            continue

        # clipped matches
        match = 0
        total = 0
        for ng, c in pred_counts.items():
            total += c
            match += min(c, ref_counts.get(ng, 0))

        if smooth:
            # add-1 smoothing
            match += 1
            total += 1

        precisions.append(match / total)

    # geometric mean of precisions
    # (avoid log(0) by smoothing above; still guard just in case)
    if any(p <= 0 for p in precisions):
        geo_mean = 0.0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)

    # brevity penalty
    c = len(pred_toks)
    r = len(ref_toks)
    if c == 0:
        bp = 0.0
    elif c > r:
        bp = 1.0
    else:
        bp = math.exp(1.0 - (r / c))

    return bp * geo_mean


# -----------------------------
# Your function + BLEU (best match over golds)
# -----------------------------
def score_open_ended(pred: str, gold_list: List[str]) -> Dict[str, float]:
    """
    Best match over multiple gold answers.
    Returns: {"em":..., "f1":..., "contains":..., "bleu":...}

    BLEU here is sentence-level BLEU-4 (smoothed), best over gold answers.
    """
    pred_n = _normalize(pred)
    best_em = 0.0
    best_f1 = 0.0
    best_contains = 0.0
    best_bleu = 0.0

    for g in gold_list:
        g_n = _normalize(g)
        em = 1.0 if pred_n == g_n and g_n != "" else 0.0
        f1 = _token_f1(pred, g)  # assuming you already have this
        contains = 1.0 if (g_n != "" and g_n in pred_n) else 0.0
        bleu = _bleu_score(pred, g, max_n=4, smooth=True)

        if em > best_em:
            best_em = em
        if f1 > best_f1:
            best_f1 = f1
        if contains > best_contains:
            best_contains = contains
        if bleu > best_bleu:
            best_bleu = bleu

    return {"em": best_em, "f1": best_f1, "contains": best_contains, "bleu": best_bleu}

# ----------------------------
# Gold answer extraction from qrels+corpus
# ----------------------------
def get_gold_answers_for_query(
    qobj: dict,
    *,
    qrels_for_q: Dict[int, int],
    corpus: Dict[int, dict],
) -> List[str]:
    """
    masked_type controls which field is the ground truth:
      - tail -> tail_text
      - rel  -> rel_text
      - head -> head_text
    If missing, default to tail_text (your examples are tail-masked).
    """
    masked_type = str(qobj.get("masked_type", "tail")).strip().lower()

    field = "tail_text"
    if masked_type in ("rel", "relation"):
        field = "rel_text"
    elif masked_type in ("head",):
        field = "head_text"

    golds: List[str] = []
    for doc_id, rel in qrels_for_q.items():
        if int(rel) <= 0:
            continue
        doc = corpus.get(int(doc_id))
        if not doc:
            continue
        val = str(doc.get(field, "")).strip()
        if val:
            golds.append(val)

    # de-dup while preserving order
    seen = set()
    uniq: List[str] = []
    for g in golds:
        gn = _normalize(g)
        if gn in seen:
            continue
        seen.add(gn)
        uniq.append(g)
    return uniq


# ----------------------------
# Result container
# ----------------------------
@dataclass
class RagEvalResult:
    em: float
    f1: float
    contains: float
    bleu: float
    n: int
    
    
def format_latex_row_rag(res: RagEvalResult) -> str:
    # percent
    return f"{res.em*100:.2f} & {res.f1*100:.2f} & {res.contains*100:.2f} & {res.bleu*100:.2f}"

# ----------------------------
# RAG evaluation core
# ----------------------------
def evaluate_rag_dataset(
    *,
    name: str,
    queries: Dict[int, dict],
    corpus: Dict[int, dict],
    qrels: Dict[int, Dict[int, int]],
    image_id_to_path: Dict[str, str],
    retriever_name: str,
    rag_top_k: int,
    model_name: str,
    batch_size: int,
    cache_dir: str,
    caption_cache_path: str,
    n_img: int,
    n_text: int,
    do_split: bool,
    split_seed: str,
    eval_partition: str,
    max_eval_queries: Optional[int],
    llm_model: str,
    max_output_tokens: int,
    reasoning_effort: str,
    verbosity: str,
    out_json_dir: Path,
    query_aug_out_dir: str,   # <-- NEW
) -> RagEvalResult:
    t0 = time.time()
    print(f"[{name}] building retrieval inputs...")
    retrieval_dataset, idx2docid, entity2text, relation2text = build_retrieval_inputs_from_corpus(corpus)
    retrieval_dataset = retrieval_dataset# [:10]
    print(f"[{name}] done inputs in {time.time()-t0:.1f}s")

    t1 = time.time()
    print(f"[{name}] building retriever={retriever_name} ...")
    
    if retriever_name != 'None':
        retriever = build_retriever(
            retriever_name,
            retrieval_dataset=retrieval_dataset,
            entity2text=entity2text,
            relation2text=relation2text,
            model_name=model_name,
            batch_size=batch_size,
            cache_dir=cache_dir,
            caption_cache_path=caption_cache_path,
        )
        print(f"[{name}] done retriever in {time.time()-t1:.1f}s")

    all_qids = [qid for qid in sorted(queries.keys()) if qid in qrels]# [:10]
    if max_eval_queries is not None:
        all_qids = all_qids[: int(max_eval_queries)]

    if not all_qids:
        raise ValueError(f"[{name}] No overlapping qids between queries and qrels.")

    if do_split:
        tr, va, te = split_qids_deterministic(all_qids, seed=split_seed)
        part = {"train": tr, "val": va, "test": te}[eval_partition]
        eval_qids = part
        print(
            f"[{name}] split sizes: train={len(tr)} val={len(va)} test={len(te)} "
            f"| eval={eval_partition}={len(eval_qids)}"
        )
    else:
        eval_qids = all_qids

    if not eval_qids:
        raise ValueError(f"[{name}] eval_qids is empty.")

    # logs
    out_json_dir.mkdir(parents=True, exist_ok=True)
    retrieved_logs: List[dict] = []
    gen_logs: List[dict] = []

    em_sum = 0.0
    f1_sum = 0.0
    contains_sum = 0.0
    bleu_sum = 0.0

    iterator = tqdm(eval_qids, desc=f"[{name}] RAG queries", total=len(eval_qids), dynamic_ncols=True)

    for qid in iterator:
        qobj = queries[qid]
        qtext_raw = str(qobj.get("query", "")).strip()
        qtext = qtext_raw.replace("[IMAGE]", "").strip()

        is_mm = bool(qobj.get("is_multimodal", False))

        # resolve image
        img_path: Optional[str] = None
        if is_mm:
            head_id = str(qobj.get("head_id", "")).strip()
            p = image_id_to_path.get(head_id)
            if p and os.path.isfile(p):
                # create deterministic augmented query image
                aug = augment_query_image_deterministic(
                    img_path=p,
                    qid=qid,
                    question=qtext,
                    head_id=head_id,
                    out_dir=query_aug_out_dir,
                )
                img_path = aug if (aug and os.path.isfile(aug)) else p
        # gold answers from qrels+corpus
        qrels_for_q = qrels[qid]
        gold_list = get_gold_answers_for_query(qobj, qrels_for_q=qrels_for_q, corpus=corpus)
        if not gold_list:
            # if for some reason empty, keep a placeholder to avoid divide-by-zero weirdness
            gold_list = [""]

        # build sample expected by retriever + prompt_builder
        # IMPORTANT: treat all questions as open-ended.
        sample: Dict[str, object] = {
            "question": qtext,
            "is_multimodal": is_mm,
            "qid": qid,
            "image_path": img_path,            # prompt_builder may use this
            "answer": "; ".join(gold_list),    # only for display/logging; scoring uses gold_list
        }

        # retrieve
        if retriever_name != 'None':
            raw_items = _call_retriever_search_items(
                retriever,
                sample=sample,
                top_k=rag_top_k,
                n_img=n_img,
                n_text=n_text,
            )
            retrieved_items = materialize_retrieved_items(raw_items, idx2docid=idx2docid, corpus=corpus)
            rag_prompt = build_rag_prompt(retrieved_items, image_id_to_path)
        # build prompt via your existing prompt_builder
        prompt_messages = build_multimodal_input_for_sample_open(sample)
        s = json.dumps(prompt_messages)

        # prepend rag prompt to user content (same pattern as your reference)
        try:
            if isinstance(prompt_messages, list) and len(prompt_messages) >= 2 and isinstance(prompt_messages[1], dict):
                if  retriever_name == 'None':
                    prompt_messages[1]["content"] = str(prompt_messages[1]["content"])
                elif "content" in prompt_messages[1]:
                    prompt_messages[1]["content"] = str(rag_prompt) + str(prompt_messages[1]["content"])
                else:
                    prompt_messages[1]["content"] = str(rag_prompt)
            else:
                # fallback: wrap into a minimal messages format
                prompt_messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": str(rag_prompt) + "\n" + qtext},
                ]
        except Exception:
            prompt_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": str(rag_prompt) + "\n" + qtext},
            ]

        pred_text, usage = get_llm_result(
            prompt_messages,
            model=llm_model,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )

        # score open-ended (best over gold set)
        sc = score_open_ended(pred_text, gold_list)
        em_sum += sc["em"]
        f1_sum += sc["f1"]
        contains_sum += sc["contains"]
        bleu_sum += sc["bleu"]
        # logs
        if retriever_name!='None':
            retrieved_logs.append(
                {
                    "qid": qid,
                    "question": qtext,
                    "image_path": img_path,
                    "retrieved_items": retrieved_items,
                }
            )
        
        gen_logs.append(
            {
                "qid": qid,
                "question": qtext,
                "image_path": img_path,
                "gt_answers": gold_list,
                "pred_text": pred_text,
                "usage": usage,
                "scores": sc,
            }
        )

    n = float(len(eval_qids))
    result = RagEvalResult(
    em=em_sum / n,
    f1=f1_sum / n,
    contains=contains_sum / n,
    bleu=bleu_sum / n,
    n=len(eval_qids),
)

    # save logs
    retrieved_json_path = out_json_dir / f"{name}.{retriever_name}.retrieved.json"
    generated_json_path = out_json_dir / f"{name}.{retriever_name}.generated.json"

    def _json_fallback(o: Any) -> Any:
        try:
            import numpy as _np  # type: ignore

            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, (_np.ndarray,)):
                return o.tolist()
        except Exception:
            pass
        return str(o)

    with retrieved_json_path.open("w", encoding="utf-8") as f:
        json.dump(retrieved_logs, f, ensure_ascii=False, indent=2, default=_json_fallback)

    with generated_json_path.open("w", encoding="utf-8") as f:
        json.dump(gen_logs, f, ensure_ascii=False, indent=2, default=_json_fallback)

    print(f"[{name}] saved: {retrieved_json_path}")
    print(f"[{name}] saved: {generated_json_path}")

    return result


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    default_data_dir = Path("MKG-RAG-BENCH-M/test")
    p.add_argument("--data_dir", type=Path, default=default_data_dir)

    # optional overrides; filled after parse
    p.add_argument("--mm_queries", type=Path, default=None)
    p.add_argument("--mm_corpus", type=Path, default=None)
    p.add_argument("--mm_qrels", type=Path, default=None)

    p.add_argument("--text_queries", type=Path, default=None)
    p.add_argument("--text_corpus", type=Path, default=None)
    p.add_argument("--text_qrels", type=Path, default=None)

    p.add_argument(
        "--retriever",
        type=str,
        default="RandomRetriever",
        help="MMAnchorRetriever | SimpleMultimodalRetriever | SimpleTextRetriever | RandomRetriever | CaptionRetriever",
    )

    # RAG retrieval top-k
    p.add_argument("--rag_top_k", type=int, default=5)

    # retriever knobs
    p.add_argument("--model_name", type=str, default="clip-ViT-B-32")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--cache_dir", type=str, default="cache_embeddings")
    p.add_argument("--caption_cache_path", type=str, default="cache_embeddings/caption_cache_blip.json")
    p.add_argument("--n_img", type=int, default=10)
    p.add_argument("--n_text", type=int, default=5)

    # Query image mapping
    p.add_argument(
        "--image_map_path",
        type=Path,
        default=Path("MKG-RAG-BENCH-M/image_mapping.csv"),
        help="CSV with columns IID,Image_Path for resolving query head_id to image path.",
    )
    p.add_argument(
        "--image_map_prefix",
        type=str,
        default="MKG-RAG-BENCH-M/",
        help="Prefix to prepend to Image_Path when it is not absolute.",
    )

    # deterministic split
    p.add_argument("--do_split", action="store_true", default = True)
    p.add_argument("--split_seed", type=str, default="markg_v1")
    p.add_argument("--eval_partition", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--max_eval_queries", type=int, default=9999999)

    # LLM config
    p.add_argument("--llm_model", type=str, default="gpt-5")
    p.add_argument("--max_output_tokens", type=int, default=512)
    p.add_argument("--reasoning_effort", type=str, default="minimal", choices=["minimal", "low", "medium", "high"])
    p.add_argument("--verbosity", type=str, default="low", choices=["low", "medium", "high"])

    # outputs
    p.add_argument("--out_json_dir", type=Path, default=Path("rag_outputs"))
    p.add_argument(
        "--query_aug_out_dir",
        type=str,
        default="query_aug_images",
        help="Where to write deterministic augmented query images.",
    )
    args = p.parse_args()

    # fill defaults from --data_dir if not provided
    args.mm_queries = args.mm_queries or (args.data_dir / "mm_queries.jsonl")
    args.mm_corpus = args.mm_corpus or (args.data_dir / "mm_corpus.jsonl")
    args.mm_qrels = args.mm_qrels or (args.data_dir / "mm_qrels.tsv")

    args.text_queries = args.text_queries or (args.data_dir / "text_queries.jsonl")
    args.text_corpus = args.text_corpus or (args.data_dir / "text_corpus.jsonl")
    args.text_qrels = args.text_qrels or (args.data_dir / "text_qrels.tsv")

    return args


def main() -> None:
    args = parse_args()

    # Load both splits (they can exist independently)
    mm_exists = args.mm_queries.exists() and args.mm_corpus.exists() and args.mm_qrels.exists()
    text_exists = args.text_queries.exists() and args.text_corpus.exists() and args.text_qrels.exists()

    if not (mm_exists or text_exists):
        raise ValueError("No valid dataset files found to evaluate (mm/text). Check paths.")

    # Load query image map
    image_id_to_path = load_image_mapping_csv(args.image_map_path, prefix=args.image_map_prefix)

    results: Dict[str, RagEvalResult] = {}

    mm_queries = mm_corpus = mm_qrels = None
    text_queries = text_corpus = text_qrels = None

    if mm_exists:
        mm_queries = load_queries(args.mm_queries)
        mm_corpus = load_corpus(args.mm_corpus)
        mm_qrels = load_qrels_tsv(args.mm_qrels)

        results["mm_only"] = evaluate_rag_dataset(
            name="mm_only",
            queries=mm_queries,
            corpus=mm_corpus,
            qrels=mm_qrels,
            image_id_to_path=image_id_to_path,
            retriever_name=args.retriever,
            rag_top_k=args.rag_top_k,
            model_name=args.model_name,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            caption_cache_path=args.caption_cache_path,
            n_img=args.n_img,
            n_text=args.n_text,
            do_split=args.do_split,
            split_seed=args.split_seed,
            eval_partition=args.eval_partition,
            max_eval_queries=args.max_eval_queries,
            llm_model=args.llm_model,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            verbosity=args.verbosity,
            out_json_dir=args.out_json_dir,
            query_aug_out_dir=args.query_aug_out_dir,  # <-- NEW
        )

    if text_exists:
        text_queries = load_queries(args.text_queries)
        text_corpus = load_corpus(args.text_corpus)
        text_qrels = load_qrels_tsv(args.text_qrels)

        results["text_only"] = evaluate_rag_dataset(
            name="text_only",
            queries=text_queries,
            corpus=text_corpus,
            qrels=text_qrels,
            image_id_to_path=image_id_to_path,
            retriever_name=args.retriever,
            rag_top_k=args.rag_top_k,
            model_name=args.model_name,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            caption_cache_path=args.caption_cache_path,
            n_img=args.n_img,
            n_text=args.n_text,
            do_split=args.do_split,
            split_seed=args.split_seed,
            eval_partition=args.eval_partition,
            max_eval_queries=args.max_eval_queries,
            llm_model=args.llm_model,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            verbosity=args.verbosity,
            out_json_dir=args.out_json_dir,
            query_aug_out_dir=args.query_aug_out_dir,  # <-- NEW
        )

    # Hybrid only if both are present
    if mm_exists and text_exists:
        merged_queries, merged_corpus, merged_qrels = merge_hybrid(
            mm_queries=mm_queries,      # type: ignore[arg-type]
            mm_corpus=mm_corpus,        # type: ignore[arg-type]
            mm_qrels=mm_qrels,          # type: ignore[arg-type]
            text_queries=text_queries,  # type: ignore[arg-type]
            text_corpus=text_corpus,    # type: ignore[arg-type]
            text_qrels=text_qrels,      # type: ignore[arg-type]
        )

        results["hybrid"] = evaluate_rag_dataset(
            name="hybrid",
            queries=merged_queries,
            corpus=merged_corpus,
            qrels=merged_qrels,
            image_id_to_path=image_id_to_path,
            retriever_name=args.retriever,
            rag_top_k=args.rag_top_k,
            model_name=args.model_name,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            caption_cache_path=args.caption_cache_path,
            n_img=args.n_img,
            n_text=args.n_text,
            do_split=args.do_split,
            split_seed=args.split_seed,
            eval_partition=args.eval_partition,
            max_eval_queries=args.max_eval_queries,
            llm_model=args.llm_model,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            verbosity=args.verbosity,
            out_json_dir=args.out_json_dir,
            query_aug_out_dir=args.query_aug_out_dir,  # <-- NEW
        )

    print("\n=== RAG Evaluation (mean over queries) ===")
    for split_name, res in results.items():
        print(f"\n[{split_name}] retriever={args.retriever} | top_k={args.rag_top_k} | n={res.n}")
        print(
            f"  EM={res.em:.4f}  F1={res.f1:.4f}  Contains@1={res.contains:.4f}  BLEU={res.bleu:.4f}"
        )
    
    print("\n=== LaTeX-friendly rows (percent) ===")
    print("% split & EM & F1 & Contains@1 & BLEU \\\\")
    for split_name, res in results.items():
        print(f"{split_name} & {format_latex_row_rag(res)} \\\\")
    print("% ----------------------------------------")
    
    out = {
        split: {
            "n": res.n,
            "em": res.em,
            "f1": res.f1,
            "contains": res.contains,
            "bleu": res.bleu,
        }
        for split, res in results.items()
    }
    print("\n=== JSON (summary) ===")
    print(json.dumps(out, indent=2, ensure_ascii=False))



if __name__ == "__main__":
    main()
