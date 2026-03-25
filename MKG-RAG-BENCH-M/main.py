#!/usr/bin/env python3
"""
main.py

Evaluate retrieval performance for:
  (1) text-only split
  (2) mm-only split
  (3) hybrid split = merged (text + mm) queries/corpus/qrels as one dataset

Files:
  - mm_queries.jsonl, mm_corpus.jsonl, mm_qrels.tsv
  - text_queries.jsonl, text_corpus.jsonl, text_qrels.tsv

Retriever implementations are imported from:
  retrieval.py

Metrics:
  NDCG@K, Precision@K, Recall@K for K in {5,10,20,50,100} (configurable)

New multimodal query image logic:
  - Queries provide head_id like "I58123117" when is_multimodal=True.
  - Query image path is resolved via image_mapping.csv (IID -> Image_Path),
    optionally with a prefix (e.g. "/data/xiaochen/").
  - Corpus docs may already contain absolute "image_path", but we do NOT need
    to symlink/copy anything into inference_image_root anymore.

IMPORTANT:
  - Retrievers are expected to support:
        retriever.search(sample: dict, k: int, **kwargs) -> List[dict]
    where sample can include:
        {"question": str, "image_path": Optional[str], "is_multimodal": bool, ...}
  - If your retrievers are still legacy (expecting a query_triplet), update them
    or add an adapter on their side.

"""

from __future__ import annotations


# --- add these imports near the top (with other imports) ---
import io
import random
from PIL import Image, ImageEnhance, ImageFile

# avoid PIL failing on truncated jpegs (optional but helpful)
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ----------------------------
# Deterministic augmentation (no "swap")
# ----------------------------
_AUG_CHOICES_NO_SWAP = ["crop", "rotate", "jitter", "crop+jitter", "rotate+jitter"]


def _stable_u32(seed_text: str) -> int:
    """Stable 32-bit seed from text (deterministic across runs)."""
    h = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _to_rgb_jpeg_safe(im: Image.Image) -> Image.Image:
    """Convert to RGB so we can always save as JPEG."""
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
    # keep full image; fill background with black (or 0)
    return im.rotate(deg, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))


def _augment_jitter(im: Image.Image, rng: random.Random) -> Image.Image:
    """
    Mild photometric jitter: brightness/contrast/saturation.
    (No hue to avoid weird shifts; keep mild for "same content".)
    """
    # factors in ~[0.85, 1.15]
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
    """
    Deterministically create an augmented query image (no swap; only crop/rotate/jitter combos).
    - Chooses one augmentation uniformly from _AUG_CHOICES_NO_SWAP.
    - Uses a stable RNG seed based on (qid, head_id, question, img_path) so re-runs match.
    - Saves to out_dir as JPEG and returns that path.

    If anything fails, returns None (caller can fallback to original img_path).
    """
    if not img_path or not os.path.isfile(img_path):
        return None

    os.makedirs(out_dir, exist_ok=True)

    # stable choice + params per sample
    seed = _stable_u32(f"aug::{qid}::{head_id}::{img_path}::{question}")
    rng = random.Random(seed)
    choice = _AUG_CHOICES_NO_SWAP[rng.randrange(len(_AUG_CHOICES_NO_SWAP))]

    # deterministic output filename (content-addressed by seed+choice)
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
            else:
                # should never happen
                pass

            # Save deterministically
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# ---- import your retrievers ----
from retrieval import (  # noqa: E402
    MMAnchorRetriever,
    SimpleMultimodalRetriever,
    SimpleTextRetriever,
    RandomRetriever,
    CaptionRetriever,
)


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
        if iid_col not in reader.fieldnames or image_path_col not in reader.fieldnames:
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
        return " & ".join(f"{d[k] * 100:.2f}" for k in ks)

    return f"{vals(res.ndcg)} & {vals(res.precision)} & {vals(res.recall)}"


# ----------------------------
# Retriever factory
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
    """
    NOTE: This main.py assumes retrievers no longer require image_root/inference_image_root
    for resolving query images. Query images are passed explicitly via sample["image_path"].
    """
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


def _call_retriever_search(
    retriever: object,
    *,
    sample: Dict[str, object],
    top_k: int,
    n_img: int,
    n_text: int,
) -> List[int]:
    """
    Unify output to ranked_indices: List[int] (indices into retrieval_dataset)

    Assumes retrievers return list[{"index": int, ...}]
    """
    cls = retriever.__class__.__name__
    if cls == "MMAnchorRetriever":
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
    name: str,
    queries: Dict[int, dict],
    corpus: Dict[int, dict],
    qrels: Dict[int, Dict[int, int]],
    image_id_to_path: Dict[str, str],
    retriever_name: str,
    ks: List[int],
    model_name: str,
    batch_size: int,
    cache_dir: str,
    caption_cache_path: str,
    n_img: int,
    n_text: int,
    do_split: bool,
    split_seed: str,
    eval_partition: str,
    eval_qids_override: Optional[List[int]] = None,
    max_eval_queries: Optional[int] = 9999999,  # keep your old behavior
        query_aug_out_dir: str,
) -> EvalResult:
    # retrieval_dataset, idx2docid, entity2text, relation2text = build_retrieval_inputs_from_corpus(corpus)


    import time

    t0 = time.time()
    print(f"[{name}] building retrieval inputs...")
    retrieval_dataset, idx2docid, entity2text, relation2text = build_retrieval_inputs_from_corpus(corpus)
    print(f"[{name}] done inputs in {time.time()-t0:.1f}s")
    retrieval_dataset = retrieval_dataset# 
    t1 = time.time()
    print(f"[{name}] building retriever={retriever_name} ...")
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
    
    



    all_qids = [qid for qid in sorted(queries.keys()) if qid in qrels]#[:100]
    if max_eval_queries is not None:
        all_qids = all_qids[: int(max_eval_queries)]

    if not all_qids:
        raise ValueError(f"[{name}] No overlapping qids between queries and qrels.")

    if eval_qids_override is not None:
        eval_qids = [qid for qid in eval_qids_override if qid in queries and qid in qrels]
    elif do_split:
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

    max_k = max(ks)

    ndcg_sum = {k: 0.0 for k in ks}
    prec_sum = {k: 0.0 for k in ks}
    rec_sum = {k: 0.0 for k in ks}

    iterator = tqdm(eval_qids, desc=f"[{name}] queries", total=len(eval_qids), dynamic_ncols=True)

    for qid in iterator:
        qobj = queries[qid]
        qtext_raw = str(qobj.get("query", "")).strip()

        # Remove any dataset-specific image token from text (optional but recommended)
        qtext = qtext_raw.replace("[IMAGE]", "").strip()

        is_mm = bool(qobj.get("is_multimodal", False))

        sample: Dict[str, object] = {
            "question": qtext,
            "is_multimodal": is_mm,
            "qid": qid,
        }

        if is_mm:
            head_id = str(qobj.get("head_id", "")).strip()
            img_path = image_id_to_path.get(head_id)

            chosen: Optional[str] = None
            if img_path and os.path.isfile(img_path):
                # deterministically create an augmented query image for this sample
                aug = augment_query_image_deterministic(
                    img_path=img_path,
                    qid=qid,
                    question=qtext,
                    head_id=head_id,
                    out_dir=query_aug_out_dir,   # e.g., "query_aug_images"
                )
                chosen = aug if (aug and os.path.isfile(aug)) else img_path

            sample["image_path"] = chosen  # may be None if no valid file

        ranked_indices = _call_retriever_search(
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

        qrels_for_q = qrels[qid]
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
        default="SimpleMultimodalRetriever",
        help="MMAnchorRetriever | SimpleMultimodalRetriever | SimpleTextRetriever | RandomRetriever | CaptionRetriever",
    )

    p.add_argument("--ks", type=str, default="5,10,20,50,100")
    p.add_argument("--model_name", type=str, default="clip-ViT-B-32")
    p.add_argument("--batch_size", type=int, default=32)

    p.add_argument("--cache_dir", type=str, default="cache_embeddings")
    p.add_argument("--caption_cache_path", type=str, default="cache_embeddings/caption_cache_blip.json")

    # Query image mapping (new)
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

    # MMAnchorRetriever knobs (passed through if supported)
    p.add_argument("--n_img", type=int, default=10)
    p.add_argument("--n_text", type=int, default=5)

    # deterministic split
    p.add_argument("--do_split", action="store_true")
    p.add_argument("--split_seed", type=str, default="markg_v1")
    p.add_argument("--eval_partition", type=str, default="test", choices=["train", "val", "test"])

    p.add_argument(
        "--query_aug_out_dir",
        type=str,
        default="query_aug_images",
        help="Where to write deterministic augmented query images.",
    )

    # evaluation cap (keep your old behavior default=100)
    p.add_argument("--max_eval_queries", type=int, default=9999999)

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
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    if any(k <= 0 for k in ks):
        raise ValueError(f"All K must be > 0. Got: {ks}")

    # Load both splits (they can exist independently)
    mm_exists = args.mm_queries.exists() and args.mm_corpus.exists() and args.mm_qrels.exists()
    text_exists = args.text_queries.exists() and args.text_corpus.exists() and args.text_qrels.exists()

    if not (mm_exists or text_exists):
        raise ValueError("No valid dataset files found to evaluate (mm/text). Check paths.")

    # Load query image map (only needed if you evaluate multimodal queries)
    image_id_to_path = load_image_mapping_csv(args.image_map_path, prefix=args.image_map_prefix)

    results: Dict[str, EvalResult] = {}

    mm_queries = mm_corpus = mm_qrels = None
    text_queries = text_corpus = text_qrels = None

    if mm_exists:
        mm_queries = load_queries(args.mm_queries)
        mm_corpus = load_corpus(args.mm_corpus)
        mm_qrels = load_qrels_tsv(args.mm_qrels)

        results["mm_only"] = evaluate_dataset(
            name="mm_only",
            queries=mm_queries,
            corpus=mm_corpus,
            qrels=mm_qrels,
            image_id_to_path=image_id_to_path,
            retriever_name=args.retriever,
            ks=ks,
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
             query_aug_out_dir=args.query_aug_out_dir,
        )

    if text_exists:
        text_queries = load_queries(args.text_queries)
        text_corpus = load_corpus(args.text_corpus)
        text_qrels = load_qrels_tsv(args.text_qrels)

        results["text_only"] = evaluate_dataset(
            name="text_only",
            queries=text_queries,
            corpus=text_corpus,
            qrels=text_qrels,
            image_id_to_path=image_id_to_path,
            retriever_name=args.retriever,
            ks=ks,
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
             query_aug_out_dir=args.query_aug_out_dir,
        )

    # Hybrid only if both are present
    if mm_exists and text_exists:
        merged_queries, merged_corpus, merged_qrels = merge_hybrid(
            mm_queries=mm_queries,   # type: ignore[arg-type]
            mm_corpus=mm_corpus,     # type: ignore[arg-type]
            mm_qrels=mm_qrels,       # type: ignore[arg-type]
            text_queries=text_queries,  # type: ignore[arg-type]
            text_corpus=text_corpus,    # type: ignore[arg-type]
            text_qrels=text_qrels,      # type: ignore[arg-type]
        )

        results["hybrid"] = evaluate_dataset(
            name="hybrid",
            queries=merged_queries,
            corpus=merged_corpus,
            qrels=merged_qrels,
            image_id_to_path=image_id_to_path,
            retriever_name=args.retriever,
            ks=ks,
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
             query_aug_out_dir=args.query_aug_out_dir,
        )
    
    print("\n=== Retrieval Evaluation (mean over queries) ===")
    for split_name, res in results.items():
        print(f"\n[{split_name}] retriever={args.retriever}")
        for k in ks:
            print(
                f"  K={k:<3d} | "
                f"NDCG={res.ndcg[k]:.4f}  "
                f"P={res.precision[k]:.4f}  "
                f"R={res.recall[k]:.4f}"
            )

    print("\n=== LaTeX-friendly rows (percent) ===")
    for split_name, res in results.items():
        print(f"{split_name} & {format_latex_row(res, ks)} \\\\")

    out = {
        split: {
            "ndcg": {str(k): res.ndcg[k] for k in ks},
            "precision": {str(k): res.precision[k] for k in ks},
            "recall": {str(k): res.recall[k] for k in ks},
        }
        for split, res in results.items()
    }
    print("\n=== JSON ===")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
