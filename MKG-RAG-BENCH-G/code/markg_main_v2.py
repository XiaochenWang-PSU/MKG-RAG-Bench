#!/usr/bin/env python3
"""
MarKG_cross_eval.py

Cross-setting retrieval evaluation variants:

A) text queries   -> hybrid corpus   (text queries evaluated against corpus = text + mm)
B) mm queries     -> text corpus     (mm queries evaluated against text-only corpus)
C) mm queries     -> hybrid corpus   (mm queries evaluated against corpus = mm + text)

Important details / design choices:

1) Hybrid corpus with a single query split:
   - For (A), we KEEP text doc_ids unchanged so text_qrels stay valid, and OFFSET mm doc_ids
     to avoid doc_id collisions.
   - For (C), we KEEP mm doc_ids unchanged so mm_qrels stay valid, and OFFSET text doc_ids.

2) For (B) mm->text, the qrels doc_ids must refer to TEXT doc_ids.
   - We remap mm_qrels to text doc_ids by matching triplet keys (head_id, rel_id, tail_id)
     between mm_corpus and text_corpus.
   - Query images still come from the ORIGINAL mm_corpus/mm_qrels (since text docs may not
     have image_path). So we pass a query-image override into evaluation.

Retriever implementations are imported from:
  MarKG_retrieval_v2.py

Metrics:
  NDCG@K, Precision@K, Recall@K for K in {5,10,20,50,100} (configurable)

Multimodal query image rule:
  Query image is derived from the image_path of (one of) the relevant docs (first relevant doc_id).
  We also optionally apply deterministic augmentation to the query image and place it at:
      {inference_image_root}/qid_{qid}.jpg
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# ---- import your retrievers ----
from MarKG_retrieval_v2 import (  # noqa: E402
    MMAnchorRetriever,
    SimpleMultimodalRetriever,
    SimpleTextRetriever,
    RandomRetriever,
    CaptionRetriever,
)

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# ----------------------------
# Deterministic augmentation helpers
# ----------------------------
def _stable_int_seed(*parts: str) -> int:
    """Stable across runs/machines (unlike Python's built-in hash)."""
    s = "||".join(parts).encode("utf-8")
    return int(hashlib.sha1(s).hexdigest(), 16) % (2**31 - 1)


def _list_images_in_same_folder(img_path: str) -> List[str]:
    folder = os.path.dirname(img_path)
    if not os.path.isdir(folder):
        return []
    files: List[str] = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(_IMG_EXTS):
            files.append(os.path.join(folder, fn))
    return sorted(files)  # deterministic ordering


def _random_crop_resize(im: Image.Image, rng: random.Random, min_frac: float = 0.75) -> Image.Image:
    w, h = im.size
    frac = rng.uniform(min_frac, 0.95)
    new_w = max(2, int(w * frac))
    new_h = max(2, int(h * frac))

    left = rng.randint(0, max(0, w - new_w))
    top = rng.randint(0, max(0, h - new_h))
    crop = im.crop((left, top, left + new_w, top + new_h))
    return crop.resize((w, h), resample=Image.BICUBIC)


def _modest_rotate(im: Image.Image, rng: random.Random, max_deg: float = 10.0) -> Image.Image:
    w, h = im.size
    deg = rng.uniform(-max_deg, max_deg)
    rotated = im.rotate(deg, resample=Image.BICUBIC, expand=True)

    rw, rh = rotated.size
    left = max(0, (rw - w) // 2)
    top = max(0, (rh - h) // 2)
    cropped = rotated.crop((left, top, left + w, top + h))
    if cropped.size != (w, h):
        cropped = cropped.resize((w, h), resample=Image.BICUBIC)
    return cropped


def _color_jitter(im: Image.Image, rng: random.Random) -> Image.Image:
    b = rng.uniform(0.85, 1.15)
    c = rng.uniform(0.85, 1.15)
    s = rng.uniform(0.85, 1.15)
    sh = rng.uniform(0.90, 1.10)

    out = ImageEnhance.Brightness(im).enhance(b)
    out = ImageEnhance.Contrast(out).enhance(c)
    out = ImageEnhance.Color(out).enhance(s)
    out = ImageEnhance.Sharpness(out).enhance(sh)
    return out


def make_augmented_query_image(
    *,
    img_path: str,
    question_token: str,
    inference_image_root: str,
    seed_parts: Tuple[str, ...],  # e.g., (setting_name, split_name, str(qid))
) -> str:
    """
    Deterministic, equal-probability augmentation.
    Each call for the same seed_parts yields identical output file bytes.

    Aug choices (equal): ["swap", "crop", "rotate", "jitter", "crop+jitter", "rotate+jitter"]
    - swap: pick another image in same folder if exists; else fallback to jitter (deterministic)
    """
    os.makedirs(inference_image_root, exist_ok=True)
    dst = os.path.join(inference_image_root, f"{question_token}.jpg")

    rng_seed = _stable_int_seed(*seed_parts)
    rng = random.Random(rng_seed)

    aug_choices = ["swap", "crop", "rotate", "jitter", "crop+jitter", "rotate+jitter"]
    aug = rng.choice(aug_choices)

    chosen = img_path
    if aug == "swap":
        candidates = _list_images_in_same_folder(img_path)
        others = [p for p in candidates if os.path.abspath(p) != os.path.abspath(img_path)]
        if others:
            chosen = rng.choice(others)
        else:
            aug = "jitter"
            chosen = img_path

    with Image.open(chosen) as im:
        im = im.convert("RGB")

        if aug == "crop":
            im = _random_crop_resize(im, rng, min_frac=0.75)
        elif aug == "rotate":
            im = _modest_rotate(im, rng, max_deg=10.0)
        elif aug == "jitter":
            im = _color_jitter(im, rng)
        elif aug == "crop+jitter":
            im = _random_crop_resize(im, rng, min_frac=0.75)
            im = _color_jitter(im, rng)
        elif aug == "rotate+jitter":
            im = _modest_rotate(im, rng, max_deg=10.0)
            im = _color_jitter(im, rng)

        if rng.random() < 0.30:
            radius = rng.uniform(0.4, 1.0)
            im = im.filter(ImageFilter.GaussianBlur(radius=radius))

        jpg_quality = rng.randint(85, 95)
        im.save(dst, format="JPEG", quality=jpg_quality, optimize=True)

    return dst


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
# Corpus-only merge helpers (for cross settings)
# ----------------------------
def offset_corpus(corpus: Dict[int, dict], doc_offset: int) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for doc_id, d in corpus.items():
        d2 = dict(d)
        d2["doc_id"] = int(doc_id) + doc_offset
        out[int(doc_id) + doc_offset] = d2
    return out


def merge_corpus_keep_primary_ids(
    *,
    primary_corpus: Dict[int, dict],
    secondary_corpus: Dict[int, dict],
    secondary_offset: int,
) -> Dict[int, dict]:
    """
    Merge two corpora into one, keeping primary doc_ids unchanged,
    offsetting secondary doc_ids to avoid collisions.
    """
    merged = dict(primary_corpus)
    merged.update(offset_corpus(secondary_corpus, secondary_offset))
    return merged


# ----------------------------
# Qrels remapping: mm_qrels -> text doc_ids (by triplet key)
# ----------------------------
def _triplet_key_from_doc(doc: dict) -> Tuple[str, str, str]:
    return (str(doc.get("head_id", "")), str(doc.get("rel_id", "")), str(doc.get("tail_id", "")))
def _norm_field(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()

def _triplet_key_from_doc(doc: dict) -> Tuple[str, str, str]:
    return (
        _norm_field(doc.get("head_id")),
        _norm_field(doc.get("rel_id")),
        _norm_field(doc.get("tail_id")),
    )

def remap_qrels_by_triplet(
    *,
    source_qrels: Dict[int, Dict[int, int]],
    source_corpus: Dict[int, dict],
    target_corpus: Dict[int, dict],
) -> Dict[int, Dict[int, int]]:
    """
    Remap qrels doc_ids from source_corpus space -> target_corpus space by matching triplet keys.
    Keeps ALL matching target doc_ids (not just one), and preserves max relevance if duplicates.

    This is for: mm queries evaluated against text corpus.
    """
    # Build target index: triplet_key -> list[doc_id]
    key2tgt_docids: Dict[Tuple[str, str, str], List[int]] = {}
    for doc_id, d in target_corpus.items():
        key = _triplet_key_from_doc(d)
        if key == ("", "", ""):
            continue
        key2tgt_docids.setdefault(key, []).append(int(doc_id))

    out: Dict[int, Dict[int, int]] = {}
    for qid, rels in source_qrels.items():
        mapped: Dict[int, int] = {}
        for src_doc_id, rel in rels.items():
            rel = int(rel)
            if rel <= 0:
                continue
            src_doc = source_corpus.get(int(src_doc_id))
            if not src_doc:
                continue
            key = _triplet_key_from_doc(src_doc)
            tgt_ids = key2tgt_docids.get(key, [])
            for tgt_doc_id in tgt_ids:
                mapped[tgt_doc_id] = max(mapped.get(tgt_doc_id, 0), rel)
        out[int(qid)] = mapped
    return out

# ----------------------------
# Query image inference
# ----------------------------
def infer_query_image_from_qrels(
    qrels_for_q: Dict[int, int],
    corpus: Dict[int, dict],
) -> Optional[str]:
    rel_docs = [doc_id for doc_id, rel in qrels_for_q.items() if rel > 0]
    if not rel_docs:
        return None
    doc_id = sorted(rel_docs)[0]
    doc = corpus.get(doc_id)
    if not doc:
        return None
    img = doc.get("image_path")
    if not img:
        return None
    img = str(img)
    return img if os.path.isfile(img) else None


# ----------------------------
# Metrics
# ----------------------------
def dcg_at_k(relevances: List[int], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel <= 0:
            continue
        s += rel / math.log2(i + 1)
    return s


def ndcg_at_k(ranked_doc_ids: List[int], qrels_for_q: Dict[int, int], k: int) -> float:
    if k <= 0:
        return 0.0
    rels_ranked = [qrels_for_q.get(doc_id, 0) for doc_id in ranked_doc_ids[:k]]
    dcg = dcg_at_k(rels_ranked, k)
    ideal_rels = sorted(qrels_for_q.values(), reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def precision_at_k(ranked_doc_ids: List[int], qrels_for_q: Dict[int, int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = 0
    for doc_id in ranked_doc_ids[:k]:
        if qrels_for_q.get(doc_id, 0) > 0:
            hits += 1
    return hits / float(k)


def recall_at_k(ranked_doc_ids: List[int], qrels_for_q: Dict[int, int], k: int) -> float:
    total_rel = sum(1 for r in qrels_for_q.values() if r > 0)
    if total_rel == 0:
        return 0.0
    hits = 0
    for doc_id in ranked_doc_ids[:k]:
        if qrels_for_q.get(doc_id, 0) > 0:
            hits += 1
    return hits / float(total_rel)


@dataclass
class EvalResult:
    ndcg: Dict[int, float]
    precision: Dict[int, float]
    recall: Dict[int, float]


def format_latex_row(res: EvalResult, ks: List[int]) -> str:
    def vals(d: Dict[int, float]) -> str:
        return " & ".join(f"{d[k]*100:.2f}" for k in ks)

    return f"{vals(res.ndcg)} & {vals(res.precision)} & {vals(res.recall)}"


# ----------------------------
# Retriever factory + adapter
# ----------------------------
def build_retriever(
    name: str,
    *,
    retrieval_dataset: List[Tuple[str, str, str]],
    entity2text: Dict[str, str],
    relation2text: Dict[str, str],
    model_name: str,
    batch_size: int,
    image_root: str,
    inference_image_root: str,
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
            image_root=image_root,
            inference_image_root=inference_image_root,
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
            inference_image_root=inference_image_root,
            image_root=image_root,
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
            image_root=image_root,
            cache_dir=cache_dir,
            caption_cache_path=caption_cache_path,
        )
    raise ValueError(
        f"Unknown retriever '{name}'. Choose from: "
        "MMAnchorRetriever, SimpleMultimodalRetriever, SimpleTextRetriever, RandomRetriever, CaptionRetriever"
    )


def retriever_search(
    retriever: object,
    *,
    sample: Dict[str, Any],
    top_k: int,
    n_img: int,
    n_text: int,
) -> List[int]:
    if retriever.__class__.__name__ == "MMAnchorRetriever":
        out = retriever.search(sample, top_k, n_img=n_img, n_text=n_text, return_unique=True)
    else:
        out = retriever.search(sample, top_k)

    ranked_indices: List[int] = []
    for item in out:
        ranked_indices.append(int(item["index"]))
    return ranked_indices


# ----------------------------
# Evaluation core
# ----------------------------
def evaluate_dataset(
    *,
    setting_name: str,
    queries: Dict[int, dict],
    corpus: Dict[int, dict],
    qrels: Dict[int, Dict[int, int]],
    retriever_name: str,
    ks: List[int],
    model_name: str,
    batch_size: int,
    image_root: str,
    inference_image_root: str,
    cache_dir: str,
    caption_cache_path: str,
    n_img: int,
    n_text: int,
    do_split: bool,
    split_seed: str,
    eval_partition: str,
    # if provided, overrides splitting logic
    eval_qids_override: Optional[List[int]] = None,
    # overrides ONLY for query-image inference (useful for mm->text)
    query_image_corpus_override: Optional[Dict[int, dict]] = None,
    query_image_qrels_override: Optional[Dict[int, Dict[int, int]]] = None,
) -> EvalResult:
    retrieval_dataset, idx2docid, entity2text, relation2text = build_retrieval_inputs_from_corpus(corpus)
    retrieval_dataset = retrieval_dataset[:10]
    retriever = build_retriever(
        retriever_name,
        retrieval_dataset=retrieval_dataset,
        entity2text=entity2text,
        relation2text=relation2text,
        model_name=model_name,
        batch_size=batch_size,
        image_root=image_root,
        inference_image_root=inference_image_root,
        cache_dir=cache_dir,
        caption_cache_path=caption_cache_path,
    )

    # all_qids = [qid for qid in sorted(queries.keys()) if qid in qrels]# [:10]
    def _has_pos(q: Dict[int, int]) -> bool:
      return any(int(v) > 0 for v in q.values())

    all_qids = [
        qid for qid in sorted(queries.keys())
        if qid in qrels and _has_pos(qrels.get(qid, {}))
    ][:10]
    if not all_qids:
        raise ValueError(f"[{setting_name}] No overlapping qids between queries and qrels.")

    if eval_qids_override is not None:
        eval_qids = [qid for qid in eval_qids_override if qid in queries and qid in qrels]
    elif do_split:
        tr, va, te = split_qids_deterministic(all_qids, seed=split_seed)
        part = {"train": tr, "val": va, "test": te}[eval_partition]
        eval_qids = part
        print(
            f"[{setting_name}] split sizes: train={len(tr)} val={len(va)} test={len(te)} "
            f"| eval={eval_partition}={len(eval_qids)}"
        )
    else:
        eval_qids = all_qids

    if not eval_qids:
        raise ValueError(f"[{setting_name}] eval_qids is empty.")

    max_k = max(ks)
    ndcg_sum = {k: 0.0 for k in ks}
    prec_sum = {k: 0.0 for k in ks}
    rec_sum = {k: 0.0 for k in ks}

    iterator = tqdm(eval_qids, desc=f"[{setting_name}] queries", total=len(eval_qids), dynamic_ncols=True)

    # choose sources for query-image inference (defaults to current corpus/qrels)
    img_corpus = query_image_corpus_override if query_image_corpus_override is not None else corpus
    img_qrels = query_image_qrels_override if query_image_qrels_override is not None else qrels

    for qid in iterator:
        qobj = queries[qid]
        qtext = str(qobj.get("query", "")).strip()
        is_mm = bool(qobj.get("is_multimodal", False))

        question_token = f"qid_{qid}"
        sample: Dict[str, Any] = {
            "question": qtext,
            "image_path": None,
            "is_multimodal": is_mm,
            "qid": qid,
        }

        if is_mm:
            qrels_for_img = img_qrels.get(qid, {})
            img_path = infer_query_image_from_qrels(qrels_for_img, img_corpus)
            if img_path and os.path.isfile(img_path):
                aug_path = make_augmented_query_image(
                    img_path=img_path,
                    question_token=question_token,
                    inference_image_root=inference_image_root,
                    seed_parts=(setting_name, str(qid)),
                )
                sample["image_path"] = aug_path
            else:
                sample["is_multimodal"] = False  # fallback to text-only if no image

        ranked_indices = retriever_search(
            retriever,
            sample=sample,
            top_k=max_k,
            n_img=n_img,
            n_text=n_text,
        )

        ranked_doc_ids: List[int] = []
        for idx in ranked_indices:
            if 0 <= idx < len(idx2docid):
                ranked_doc_ids.append(idx2docid[idx])

        qrels_for_q = qrels.get(qid, {})
        for k in ks:
            ndcg_sum[k] += ndcg_at_k(ranked_doc_ids, qrels_for_q, k)
            prec_sum[k] += precision_at_k(ranked_doc_ids, qrels_for_q, k)
            rec_sum[k] += recall_at_k(ranked_doc_ids, qrels_for_q, k)

    n = float(len(eval_qids))
    return EvalResult(
        ndcg={k: ndcg_sum[k] / n for k in ks},
        precision={k: prec_sum[k] / n for k in ks},
        recall={k: rec_sum[k] / n for k in ks},
    )


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    default_data_dir = Path("out/group_retrieval")
    p.add_argument("--data_dir", type=Path, default=default_data_dir)

    p.add_argument("--mm_queries", type=Path, default=None)
    p.add_argument("--mm_corpus", type=Path, default=None)
    p.add_argument("--mm_qrels", type=Path, default=None)

    p.add_argument("--text_queries", type=Path, default=None)
    p.add_argument("--text_corpus", type=Path, default=None)
    p.add_argument("--text_qrels", type=Path, default=None)

    p.add_argument(
        "--retriever",
        type=str,
        default="MMAnchorRetriever",
        help="MMAnchorRetriever | SimpleMultimodalRetriever | SimpleTextRetriever | RandomRetriever | CaptionRetriever",
    )

    p.add_argument("--ks", type=str, default="5,10,20,50,100")
    p.add_argument("--model_name", type=str, default="clip-ViT-B-32")
    p.add_argument("--batch_size", type=int, default=512)

    p.add_argument("--image_root", type=str, default="images_subset_kg")
    p.add_argument("--inference_image_root", type=str, default="images_subset_kg")

    p.add_argument("--cache_dir", type=str, default="cache_embeddings")
    p.add_argument("--caption_cache_path", type=str, default="cache_embeddings/caption_cache_blip.json")

    # MMAnchorRetriever knobs
    p.add_argument("--n_img", type=int, default=10)
    p.add_argument("--n_text", type=int, default=5)

    # deterministic split
    p.add_argument("--do_split", action="store_true")
    p.add_argument("--split_seed", type=str, default="markg_v1")
    p.add_argument("--eval_partition", type=str, default="test", choices=["train", "val", "test"])

    args = p.parse_args()

    args.mm_queries = args.mm_queries or (args.data_dir / "mm_queries.jsonl")
    args.mm_corpus = args.mm_corpus or (args.data_dir / "mm_corpus.jsonl")
    args.mm_qrels = args.mm_qrels or (args.data_dir / "mm_qrels.tsv")

    args.text_queries = args.text_queries or (args.data_dir / "text_queries.jsonl")
    args.text_corpus = args.text_corpus or (args.data_dir / "text_corpus.jsonl")
    args.text_qrels = args.text_qrels or (args.data_dir / "text_qrels.tsv")

    return args


def main() -> None:
    args = parse_args()
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    if any(k <= 0 for k in ks):
        raise ValueError(f"All K must be > 0. Got: {ks}")

    mm_exists = args.mm_queries.exists() and args.mm_corpus.exists() and args.mm_qrels.exists()
    text_exists = args.text_queries.exists() and args.text_corpus.exists() and args.text_qrels.exists()

    if not (mm_exists and text_exists):
        raise ValueError(
            "This cross-eval script requires BOTH mm_* and text_* files to exist.\n"
            f"mm_exists={mm_exists} text_exists={text_exists}\n"
            "Check --data_dir and file names."
        )

    mm_queries = load_queries(args.mm_queries)
    mm_corpus = load_corpus(args.mm_corpus)
    mm_qrels = load_qrels_tsv(args.mm_qrels)

    text_queries = load_queries(args.text_queries)
    text_corpus = load_corpus(args.text_corpus)
    text_qrels = load_qrels_tsv(args.text_qrels)

    results: Dict[str, EvalResult] = {}

#    # ----------------------------
#    # (1) text-only query -> hybrid corpus
#    # Keep text doc_ids, offset mm doc_ids
#    # ----------------------------
#    text_max_doc = max(text_corpus.keys()) if text_corpus else -1
#    mm_offset_for_text_hybrid = text_max_doc + 1
#    hybrid_corpus_text_primary = merge_corpus_keep_primary_ids(
#        primary_corpus=text_corpus,
#        secondary_corpus=mm_corpus,
#        secondary_offset=mm_offset_for_text_hybrid,
#    )
#
#    results["text_on_hybrid"] = evaluate_dataset(
#        setting_name="text_on_hybrid",
#        queries=text_queries,
#        corpus=hybrid_corpus_text_primary,
#        qrels=text_qrels,  # still valid since text doc_ids unchanged
#        retriever_name=args.retriever,
#        ks=ks,
#        model_name=args.model_name,
#        batch_size=args.batch_size,
#        image_root=args.image_root,
#        inference_image_root=args.inference_image_root,
#        cache_dir=args.cache_dir,
#        caption_cache_path=args.caption_cache_path,
#        n_img=args.n_img,
#        n_text=args.n_text,
#        do_split=args.do_split,
#        split_seed=args.split_seed,
#        eval_partition=args.eval_partition,
#    )

    # ----------------------------
    # (2) multimodal query -> text corpus
    # Remap qrels: mm doc_ids -> text doc_ids by matching triplets
    # Query images still come from mm_corpus/mm_qrels (override)
    # ----------------------------
    mm_qrels_remapped_to_text = remap_qrels_by_triplet(
        source_qrels=mm_qrels,
        source_corpus=mm_corpus,
        target_corpus=text_corpus,
    )
    mm_total = len(mm_queries)
    mm_with_any = sum(1 for qid in mm_queries.keys() if qid in mm_qrels)
    mm_mapped_nonempty = sum(1 for qid, rels in mm_qrels_remapped_to_text.items() if any(v > 0 for v in rels.values()))
    print(f"[mm_on_text] qids: total_mm_queries={mm_total} | with_mm_qrels={mm_with_any} | mapped_nonempty={mm_mapped_nonempty}")
    results["mm_on_text"] = evaluate_dataset(
        setting_name="mm_on_text",
        queries=mm_queries,
        corpus=text_corpus,
        qrels=mm_qrels_remapped_to_text,
        retriever_name=args.retriever,
        ks=ks,
        model_name=args.model_name,
        batch_size=args.batch_size,
        image_root=args.image_root,
        inference_image_root=args.inference_image_root,
        cache_dir=args.cache_dir,
        caption_cache_path=args.caption_cache_path,
        n_img=args.n_img,
        n_text=args.n_text,
        do_split=args.do_split,
        split_seed=args.split_seed,
        eval_partition=args.eval_partition,
        query_image_corpus_override=mm_corpus,   # use real mm images
        query_image_qrels_override=mm_qrels,     # use real mm relevants for image inference
    )
#
#    # ----------------------------
#    # (3) multimodal query -> hybrid corpus
#    # Keep mm doc_ids, offset text doc_ids
#    # ----------------------------
#    mm_max_doc = max(mm_corpus.keys()) if mm_corpus else -1
#    text_offset_for_mm_hybrid = mm_max_doc + 1
#    hybrid_corpus_mm_primary = merge_corpus_keep_primary_ids(
#        primary_corpus=mm_corpus,
#        secondary_corpus=text_corpus,
#        secondary_offset=text_offset_for_mm_hybrid,
#    )
#
#    results["mm_on_hybrid"] = evaluate_dataset(
#        setting_name="mm_on_hybrid",
#        queries=mm_queries,
#        corpus=hybrid_corpus_mm_primary,
#        qrels=mm_qrels,  # still valid since mm doc_ids unchanged
#        retriever_name=args.retriever,
#        ks=ks,
#        model_name=args.model_name,
#        batch_size=args.batch_size,
#        image_root=args.image_root,
#        inference_image_root=args.inference_image_root,
#        cache_dir=args.cache_dir,
#        caption_cache_path=args.caption_cache_path,
#        n_img=args.n_img,
#        n_text=args.n_text,
#        do_split=args.do_split,
#        split_seed=args.split_seed,
#        eval_partition=args.eval_partition,
#    )

    # ----------------------------
    # Print
    # ----------------------------
    print("\n=== Cross Retrieval Evaluation (mean over queries) ===")
    for setting_name, res in results.items():
        print(f"\n[{setting_name}] retriever={args.retriever}")
        for k in ks:
            print(
                f"  K={k:<3d} | "
                f"NDCG={res.ndcg[k]:.4f}  "
                f"P={res.precision[k]:.4f}  "
                f"R={res.recall[k]:.4f}"
            )

    print("\n=== LaTeX-friendly rows (percent) ===")
    for setting_name, res in results.items():
        print(f"{setting_name} & {format_latex_row(res, ks)} \\\\")

    out = {
        setting: {
            "ndcg": {str(k): res.ndcg[k] for k in ks},
            "precision": {str(k): res.precision[k] for k in ks},
            "recall": {str(k): res.recall[k] for k in ks},
        }
        for setting, res in results.items()
    }
    print("\n=== JSON ===")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
