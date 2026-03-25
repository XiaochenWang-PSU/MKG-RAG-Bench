#!/usr/bin/env python3
"""
MarKG_rag.py

RAG evaluation on the SAME dataset layout as MarKG_main.py:
  (1) text-only split
  (2) mm-only split
  (3) hybrid split (merged mm+text with offsets)

Key constraints / differences vs the "new main.py" rag you used before:
- Retrievers are imported from MarKG_retrieval.py and are LEGACY:
    they expect query_triplet = [head, tail, question_token]
    and (for multimodal queries) they resolve the query image by
    first_jpg_path(question_token, inference_image_root)

Therefore for each query we do:
  - question_token = f"qid_{qid}"
  - entity2text[question_token] = query_text
  - if is_multimodal:
        infer img_path from qrels (first relevant doc's image_path)
        symlink/copy it to {inference_image_root}/{question_token}.jpg
  - query_triplet = [head, tail, question_token]
  - retriever.search(query_triplet, top_k, ...)

Then we build an OpenAI multimodal prompt:
  - always open-ended answers
  - for multimodal queries, include the query image (from inference_image_root)
  - for text-only queries, omit the image
  - prepend retrieved triplets as evidence (build_rag_prompt)
Finally compute simple open-ended metrics (Exact Match + token-F1) by default.

Outputs:
  - rag_outputs/<split>.<retriever>.retrieved.json
  - rag_outputs/<split>.<retriever>.generated.json
  - prints summary metrics

Note:
- This script assumes you already patched prompt_builder.py to support text-only
  (i.e., it does NOT crash when sample["image_path"] is missing/None).
"""

from __future__ import annotations

import random
from PIL import Image, ImageEnhance, ImageFilter



import string
from collections import Counter
import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from tqdm import tqdm

# ---- OpenAI (responses API) ----
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# ---- import your retrievers (LEGACY interface) ----
from MarKG_retrieval_v2 import (  # noqa: E402
    MMAnchorRetriever,
    SimpleMultimodalRetriever,
    SimpleTextRetriever,
    RandomRetriever,
    CaptionRetriever,
)

# ---- prompt builder (your existing file) ----
# IMPORTANT: use absolute import so "python3 MarKG_rag.py" works
from PIL import Image
import os, io, base64




_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _stable_int_seed(*parts: str) -> int:
    s = "||".join(parts).encode("utf-8")
    return int(hashlib.sha1(s).hexdigest(), 16) % (2**31 - 1)


def _list_images_in_same_folder(img_path: str) -> List[str]:
    folder = os.path.dirname(img_path)
    if not os.path.isdir(folder):
        return []
    files = []
    for fn in os.listdir(folder):
        if fn.lower().endswith(_IMG_EXTS):
            files.append(os.path.join(folder, fn))
    return sorted(files)


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


def make_augmented_query_image_legacy(
    *,
    img_path: str,
    question_token: str,
    inference_image_root: str,
    seed_parts: Tuple[str, ...],  # e.g. (split_name, str(qid))
) -> str:
    """
    Deterministic, equal-prob augmentation.
    Writes augmented query image to {inference_image_root}/{question_token}.jpg
    so legacy retrievers can discover it via first_jpg_path(question_token,...).
    """
    os.makedirs(inference_image_root, exist_ok=True)
    dst = os.path.join(inference_image_root, f"{question_token}.jpg")

    rng = random.Random(_stable_int_seed(*seed_parts))
    aug_choices = ["swap", "crop", "rotate", "jitter", "crop+jitter", "rotate+jitter"]
    aug = rng.choice(aug_choices)  # equal probability

    chosen = img_path
    if aug == "swap":
        candidates = _list_images_in_same_folder(img_path)
        others = [p for p in candidates if os.path.abspath(p) != os.path.abspath(img_path)]
        if others:
            chosen = rng.choice(others)
        else:
            aug = "jitter"  # deterministic fallback

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

        # Optional, deterministic blur
        if rng.random() < 0.30:
            radius = rng.uniform(0.4, 1.0)
            im = im.filter(ImageFilter.GaussianBlur(radius=radius))

        # Deterministic JPEG encoding params
        jpg_quality = rng.randint(85, 95)
        im.save(dst, format="JPEG", quality=jpg_quality, optimize=True)

    return dst
    
def first_jpg_path(
    entity_id: str,
    root: str,
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> Optional[str]:
    for ext in exts:
        p = os.path.join(root, f"{entity_id}{ext}")
        if os.path.isfile(p):
            return p

    d = os.path.join(root, str(entity_id))
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if os.path.splitext(fn.lower())[1] in exts:
                return os.path.join(d, fn)

    return None
    
def _to_jpeg_rgb(im: Image.Image) -> Image.Image:
    """
    Convert PIL image to something JPEG can save:
    - P (palette) -> RGB
    - RGBA/LA -> RGB (drop alpha onto white background)
    - L stays L (optional), but RGB is safest for consistent behavior
    """
    if im.mode == "RGB":
        return im

    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        return bg

    # P, CMYK, 1, I, F, etc.
    return im.convert("RGB")


def _path_to_data_url(path: str, max_side: int = 1024, jpeg_quality: int = 90) -> str:
    with Image.open(path) as im:
        im = _to_jpeg_rgb(im)
        im.thumbnail((max_side, max_side))

        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _img_to_data_url(img: Image.Image, max_side: int = 1024, jpeg_quality: int = 90) -> str:
    im = _to_jpeg_rgb(img.copy())
    im.thumbnail((max_side, max_side))

    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def build_multimodal_input_for_sample(sample: dict) -> list:
    """
    Open-ended VQA prompt.
    - If sample has a usable image (image_path or image), include it.
    - Otherwise, text-only prompt.

    Expected sample keys:
      - question: str
      - image_path: Optional[str]  (may be missing/None for text-only)
      - image: Optional[PIL.Image] (optional alternative to image_path)
    """
    question = str(sample.get("question", "")).strip()

    # Decide whether to attach an image
    img_url = None
    if sample.get("image_path"):
        p = str(sample["image_path"])
        if p and os.path.isfile(p):
            img_url = _path_to_data_url(p)
    elif sample.get("image") is not None:
        # PIL Image
        img_url = _img_to_data_url(sample["image"])

    system_text = (
        "You are a question answering model. "
        "Answer the question using a short, direct answer. "
        "Use as few words as possible (prefer 1-10 words). "
        "Do NOT provide reasoning, steps, or explanations. "
        "Do NOT add extra commentary. "
        "If the question is about relation between image and medical concept, "
    )

    user_blocks = []
    if img_url is not None:
        user_blocks.append({"type": "input_image", "image_url": img_url})
    user_blocks.append({"type": "input_text", "text": f"Question: {question}\nAnswer concisely."})

    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_blocks},
    ]


def build_rag_prompt(retrieved_items, image_root: str) -> list:
    """
    Build RAG evidence blocks from retrieved triplets.

    Robust to different retriever output schemas.

    Supports:
      A) item["item"] is an object (KGTripletLite / etc) with any of:
         - head_name / tail_name
         - head_text / tail_text
         - head / tail ids
         - relation / rel / rel_id / rel_text
      B) item["item"] is a dict with fields like:
         - head_text, rel_text, tail_text, head_id, rel_id, tail_id
      C) item itself is already such a dict.

    image_id_to_path:
      map from head_id (e.g., "I50978146") -> absolute image path
      If provided and exists, we will attach images for retrieved heads.
    """
    rag_blocks = [{
        "type": "input_text",
        "text": (
            "You can use the following knowledge-graph triples as evidence to solve the question. "
            "Images in the triplets (if present) are related/similar to the query image."
        ),
    }]

    def _get(obj, *names, default=""):
        # attribute first, then dict key
        for n in names:
            if obj is None:
                continue
            if hasattr(obj, n):
                v = getattr(obj, n)
                if v is not None and str(v).strip() != "":
                    return v
            if isinstance(obj, dict) and n in obj:
                v = obj.get(n)
                if v is not None and str(v).strip() != "":
                    return v
        return default

    for i, item in enumerate(retrieved_items or []):
        # typical schema: {"index":..., "score":..., "item": <triplet or dict>}
        triplet = item.get("item") if isinstance(item, dict) else None
        obj = triplet if triplet is not None else item
    
        head_name = _get(obj, "head_name", "head_text", "head", "head_id", default="")
        tail_name = _get(obj, "tail_name", "tail_text", "tail", "tail_id", default="")
        relation  = _get(obj, "relation", "rel", "rel_text", "rel_id", default="")
    
        # For image lookup we want the head entity ID (e.g., Q8047 or Ixxxx)
        head_id = _get(obj, "head_id", "head", default="")
    
        # 1) Always write the textual triplet evidence
        rag_blocks.append({
            "type": "input_text",
            "text": f"Triplet {i+1}: (head, relation, tail) = ({head_name}, {relation}, {tail_name})"
        })
    
#        # 2) Attach head image iff it actually exists under image_root
#        if head_id:
#            p = first_jpg_path(str(head_id), image_root)
#            if p and os.path.isfile(p):
#                with Image.open(p) as im:
#                    rag_blocks.append({"type": "input_text", "text": f"Image for head of triplet {i+1}"})
#                    rag_blocks.append({"type": "input_image", "image_url": _img_to_data_url(im)})
    return rag_blocks
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
    qrels: Dict[int, Dict[int, int]], qid_offset: int, doc_offset: int
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
# Query image prep (legacy)
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


def ensure_query_image_available(
    *,
    question_token: str,
    source_image_path: str,
    inference_image_root: str,
) -> str:
    os.makedirs(inference_image_root, exist_ok=True)
    dst = os.path.join(inference_image_root, f"{question_token}.jpg")
    if os.path.isfile(dst):
        return dst
    try:
        os.symlink(source_image_path, dst)
    except Exception:
        shutil.copy2(source_image_path, dst)
    return dst



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
        gn = _normalize_for_tokens(g)
        if gn in seen:
            continue
        seen.add(gn)
        uniq.append(g)
    return uniq


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
# Open-ended scoring (best over gold set): EM / F1 / Contains / BLEU
# ----------------------------
_ARTICLES = {"a", "an", "the"}

def _norm_for_em_contains(s: str) -> str:
    """Light normalize for EM/Contains (no truncation; stable)."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _normalize_for_tokens(text: str) -> str:
    """Normalize for token-level metrics (F1/BLEU)."""
    if text is None:
        return ""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _ARTICLES]
    return " ".join(tokens)
    


def _simple_tokens(s: str) -> List[str]:
    s = _normalize_for_tokens(s)
    if not s:
        return []
    return s.split()
def get_llm_client() -> Any:
    if OpenAI is None:
        raise ImportError("openai is not installed in this environment.")
    return OpenAI()


def call_llm_openai(
    client: Any,
    *,
    model: str,
    messages: List[dict],
    max_output_tokens: int,
    reasoning_effort: str,
    verbosity: str,
) -> Tuple[str, Any]:
    """
    Uses Responses API (works with GPT-5 style).
    """
    resp = client.responses.create(
        model=model,
        input=messages,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": reasoning_effort},
        text={"verbosity": verbosity},
    )
    text = (resp.output_text or "").strip()
    return text, resp.usage

def _token_f1(pred: str, gold: str) -> float:
    pred_toks = _simple_tokens(pred)
    gold_toks = _simple_tokens(gold)
    if len(pred_toks) == 0 and len(gold_toks) == 0:
        return 1.0
    if len(pred_toks) == 0 or len(gold_toks) == 0:
        return 0.0

    pc = Counter(pred_toks)
    gc = Counter(gold_toks)
    common = pc & gc
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / float(len(pred_toks))
    recall = num_same / float(len(gold_toks))
    return 2 * precision * recall / (precision + recall + 1e-12)

def _ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens) - n + 1)))

def _bleu_score(pred: str, ref: str, max_n: int = 4, smooth: bool = True) -> float:
    """
    Sentence-level BLEU (BLEU-4 by default) with brevity penalty.
    Simple add-1 smoothing on each n-gram precision when smooth=True.
    Returns BLEU in [0, 1].
    """
    pred_toks = _simple_tokens(pred)
    ref_toks = _simple_tokens(ref)

    if len(pred_toks) == 0 or len(ref_toks) == 0:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_n + 1):
        pred_counts = _ngram_counts(pred_toks, n)
        ref_counts = _ngram_counts(ref_toks, n)

        if len(pred_counts) == 0:
            precisions.append(0.0)
            continue

        match = 0
        total = 0
        for ng, c in pred_counts.items():
            total += c
            match += min(c, ref_counts.get(ng, 0))

        if smooth:
            match += 1
            total += 1

        precisions.append(match / float(total))

    # geometric mean
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
        bp = math.exp(1.0 - (r / float(c)))

    return bp * geo_mean

def score_open_ended(pred: str, gold_list: List[str]) -> Dict[str, float]:
    """
    Best match over multiple gold answers.
    Returns: {"em":..., "f1":..., "contains":..., "bleu":...}
    """
    pred_em = _norm_for_em_contains(pred)
    best_em = 0.0
    best_f1 = 0.0
    best_contains = 0.0
    best_bleu = 0.0

    # avoid empty list edge-case
    if not gold_list:
        gold_list = [""]

    for g in gold_list:
        g_em = _norm_for_em_contains(g)

        em = 1.0 if (g_em != "" and pred_em == g_em) else 0.0
        f1 = _token_f1(pred, g)
        contains = 1.0 if (g_em != "" and g_em in pred_em) else 0.0
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


@dataclass
class RagMetrics:
    exact_match: float
    f1: float
    contains: float
    bleu: float



# ----------------------------
# RAG evaluation core
# ----------------------------
def evaluate_rag_dataset(
    *,
    name: str,
    queries: Dict[int, dict],
    corpus: Dict[int, dict],
    qrels: Dict[int, Dict[int, int]],
    retriever_name: str,
    rag_top_k: int,
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
    out_json_dir: str,
    # LLM config
    llm_model: str,
    max_output_tokens: int,
    reasoning_effort: str,
    verbosity: str,
    # caps
    max_eval_queries: Optional[int] = 9999999,
) -> RagMetrics:
    os.makedirs(out_json_dir, exist_ok=True)

    retrieval_dataset, idx2docid, entity2text, relation2text = build_retrieval_inputs_from_corpus(corpus)
    retrieval_dataset = retrieval_dataset# [:10]
    if retriever_name != "None":
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

    # pick eval qids
    all_qids = [qid for qid in sorted(queries.keys()) if qid in qrels]# [:10]
    if max_eval_queries is not None:
        all_qids = all_qids[: int(max_eval_queries)]

    if not all_qids:
        raise ValueError(f"[{name}] No overlapping qids between queries and qrels.")

    if do_split:
        tr, va, te = split_qids_deterministic(all_qids, seed=split_seed)
        eval_qids = {"train": tr, "val": va, "test": te}[eval_partition]
        print(
            f"[{name}] split sizes: train={len(tr)} val={len(va)} test={len(te)} "
            f"| eval={eval_partition}={len(eval_qids)}"
        )
    else:
        eval_qids = all_qids

    if not eval_qids:
        raise ValueError(f"[{name}] eval_qids is empty.")

    client = get_llm_client()

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

        # For your dataset, ground-truth answer is in masked field(s)
        # We try common fields, fallback to empty.
        qrels_for_q = qrels[qid]
        gold_list = get_gold_answers_for_query(qobj, qrels_for_q=qrels_for_q, corpus=corpus)
        
        # Keep a display-friendly gt_answer string if you still want one
        gt_answer = "\n".join(gold_list) if gold_list else ""

        head_id = qobj.get("head_id")
        tail_id = qobj.get("tail_id")
        head = str(head_id) if head_id is not None else ""
        tail = str(tail_id) if tail_id is not None else ""

        is_mm = bool(qobj.get("is_multimodal", False))


                # Legacy query-token trick
        question_token = f"qid_{qid}"
        entity2text[question_token] = qtext

        # Build retrieval sample (your retrievers in this script already accept sample dict)
        sample_retrieval: Dict[str, Any] = {
            "question": qtext,
            "image_path": None,
            "is_multimodal": is_mm,
            "qid": qid,
        }

        # Build/query image path (AUGMENTED + deterministic)
        query_image_path: Optional[str] = None
        if is_mm:
            src = infer_query_image_from_qrels(qrels[qid], corpus)
            if src and os.path.isfile(src):
                query_image_path = make_augmented_query_image_legacy(
                    img_path=src,
                    question_token=question_token,
                    inference_image_root=inference_image_root,
                    seed_parts=(name, str(qid)),  # deterministic per split+qid
                )
                sample_retrieval["image_path"] = query_image_path
            else:
                # degrade gracefully if missing image
                sample_retrieval["is_multimodal"] = False
                is_mm = False

        if retriever_name != "None":
            ranked_indices = retriever_search(
                retriever,
                sample=sample_retrieval,
                top_k=rag_top_k,
                n_img=n_img,
                n_text=n_text,
            )
    
            # map indices -> docs + construct retrieved_items in the format build_rag_prompt expects
            retrieved_items: List[dict] = []
            ranked_doc_ids: List[int] = []
            for idx in ranked_indices:
                if 0 <= idx < len(idx2docid):
                    doc_id = idx2docid[idx]
                    ranked_doc_ids.append(doc_id)
                    d = corpus[doc_id]
                    # keep a lightweight dict; prompt_builder.build_rag_prompt handles it
                    retrieved_items.append(
                        {
                            "index": int(idx),
                            "score": float(0.0),
                            "item": {
                                "head_id": d.get("head_id", ""),
                                "rel_id": d.get("rel_id", ""),
                                "tail_id": d.get("tail_id", ""),
                                "head_text": d.get("head_text", ""),
                                "rel_text": d.get("rel_text", ""),
                                "tail_text": d.get("tail_text", ""),
                            },
                        }
                    )

        # Build LLM messages:
        # - base prompt = build_multimodal_input_for_sample_open(sample)
        # - prepend RAG prompt blocks into user content
        sample: Dict[str, Any] = {
            "question": qtext,
            "is_multimodal": is_mm,
        }
        if query_image_path:
            sample["image_path"] = query_image_path  # prompt_builder will attach the image
        # (text-only => omit image_path key)

        prompt_messages = build_multimodal_input_for_sample(sample)

        
        
        
        if retriever_name != "None":
            rag_blocks = build_rag_prompt(retrieved_items, image_root=image_root)
            # prompt_messages[1]["content"] is a list of blocks; prepend rag blocks
            prompt_messages[1]["content"] = rag_blocks + prompt_messages[1]["content"]
            retrieved_logs.append(
            {
                "qid": qid,
                "question": qtext,
                "is_multimodal": is_mm,
                "query_image_path": query_image_path,
                "retrieved_items": retrieved_items,
            }
        )

        else:
            prompt_messages[1]["content"] = prompt_messages[1]["content"]

        # log retrieved

        # call llm
        try:
            pred_text, usage = call_llm_openai(
                client,
                model=llm_model,
                messages=prompt_messages,
                max_output_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
            )
        except Exception as e:
            pred_text, usage = "", {"exception": repr(e)}

        gen_logs.append(
            {
                "qid": qid,
                "question": qtext,
                "is_multimodal": is_mm,
                "query_image_path": query_image_path,
                "gt_answer": gt_answer,
                "pred_text": pred_text,
                "usage": usage,
            }
        )

        gold_list = [gt_answer]

        sc = score_open_ended(pred_text, gold_list)
        em_sum += sc["em"]
        f1_sum += sc["f1"]
        contains_sum += sc["contains"]
        bleu_sum += sc["bleu"]

    n = float(len(eval_qids))
    metrics = RagMetrics(
        exact_match=em_sum / n,
        f1=f1_sum / n,
        contains=contains_sum / n,
        bleu=bleu_sum / n,
    )

    # save logs
    safe_r = re.sub(r"[^A-Za-z0-9_.-]+", "_", retriever_name)
    retrieved_json_path = os.path.join(out_json_dir, f"{name}.{safe_r}.retrieved.json")
    generated_json_path = os.path.join(out_json_dir, f"{name}.{safe_r}.generated.json")

    def _json_fallback(o):
        # last resort: string repr
        return str(o)

    with open(retrieved_json_path, "w", encoding="utf-8") as f:
        json.dump(retrieved_logs, f, ensure_ascii=False, indent=2, default=_json_fallback)
    with open(generated_json_path, "w", encoding="utf-8") as f:
        json.dump(gen_logs, f, ensure_ascii=False, indent=2, default=_json_fallback)

    print(f"[{name}] saved: {retrieved_json_path}")
    print(f"[{name}] saved: {generated_json_path}")

    return metrics


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    default_data_dir = Path("out/group_retrieval")
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

    # RAG top-k
    p.add_argument("--rag_top_k", type=int, default=5)

    # Encoder config
    p.add_argument("--model_name", type=str, default="clip-ViT-B-32")
    p.add_argument("--batch_size", type=int, default=512)

    # Legacy image roots (needed by MarKG_retrieval retrievers)
    p.add_argument("--image_root", type=str, default="images_subset_kg")
    p.add_argument("--inference_image_root", type=str, default="images_subset_kg")

    # caches
    p.add_argument("--cache_dir", type=str, default="cache_embeddings")
    p.add_argument("--caption_cache_path", type=str, default="cache_embeddings/caption_cache_blip.json")

    # MMAnchorRetriever knobs
    p.add_argument("--n_img", type=int, default=10)
    p.add_argument("--n_text", type=int, default=5)

    # deterministic split
    p.add_argument("--do_split", action="store_true", default = True)
    p.add_argument("--split_seed", type=str, default="markg_v1")
    p.add_argument("--eval_partition", type=str, default="test", choices=["train", "val", "test"])

    # caps
    p.add_argument("--max_eval_queries", type=int, default=9999999)

    # outputs
    p.add_argument("--out_json_dir", type=str, default="rag_outputs")

    # LLM settings
    p.add_argument("--llm_model", type=str, default="gpt-5")
    p.add_argument("--max_output_tokens", type=int, default=512)
    p.add_argument("--reasoning_effort", type=str, default="minimal")
    p.add_argument("--verbosity", type=str, default="low")

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

    results: Dict[str, RagMetrics] = {}

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
            retriever_name=args.retriever,
            rag_top_k=args.rag_top_k,
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
            out_json_dir=args.out_json_dir,
            llm_model=args.llm_model,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            verbosity=args.verbosity,
            max_eval_queries=args.max_eval_queries,
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
            retriever_name=args.retriever,
            rag_top_k=args.rag_top_k,
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
            out_json_dir=args.out_json_dir,
            llm_model=args.llm_model,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            verbosity=args.verbosity,
            max_eval_queries=args.max_eval_queries,
        )

    if mm_exists and text_exists:
        merged_queries, merged_corpus, merged_qrels = merge_hybrid(
            mm_queries=mm_queries,  # type: ignore[arg-type]
            mm_corpus=mm_corpus,  # type: ignore[arg-type]
            mm_qrels=mm_qrels,  # type: ignore[arg-type]
            text_queries=text_queries,  # type: ignore[arg-type]
            text_corpus=text_corpus,  # type: ignore[arg-type]
            text_qrels=text_qrels,  # type: ignore[arg-type]
        )

        results["hybrid"] = evaluate_rag_dataset(
            name="hybrid",
            queries=merged_queries,
            corpus=merged_corpus,
            qrels=merged_qrels,
            retriever_name=args.retriever,
            rag_top_k=args.rag_top_k,
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
            out_json_dir=args.out_json_dir,
            llm_model=args.llm_model,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            verbosity=args.verbosity,
            max_eval_queries=args.max_eval_queries,
        )

    print("\n=== RAG Evaluation (mean over queries) ===")
    for split_name, m in results.items():
        print(f"\n[{split_name}] retriever={args.retriever} rag_top_k={args.rag_top_k}")
        print(f"  EM        ={m.exact_match:.4f}")
        print(f"  F1        ={m.f1:.4f}")
        print(f"  Contains@1={m.contains:.4f}")
        print(f"  BLEU-4    ={m.bleu:.4f}")

    # JSON summary
    out = {
    k: {
        "exact_match": v.exact_match,
        "f1": v.f1,
        "contains@1": v.contains,
        "bleu": v.bleu,
    }
    for k, v in results.items()
}
    print("\n=== JSON ===")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
