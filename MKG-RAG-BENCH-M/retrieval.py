#!/usr/bin/env python3
"""
retrieval.py

New-style retrievers for MarKG retrieval_eval.

Key changes vs legacy:
- NO fixed image_root / inference_image_root assumptions.
- Query image is passed explicitly as sample["image_path"] (optional).
- Corpus head images are resolved via image_mapping.csv using head_id (IID -> Image_Path),
  because retrieval_dataset/corpus triplets only carry head_id, not guaranteed absolute paths.

Interface contract (used by main.py):
- Every retriever implements:
    search(sample: dict, k: int, **kwargs) -> List[dict]
  and returns a list of dicts with at least:
    {"rank": int, "index": int, "item": KGTriplet, "score": float}

- For MMAnchorRetriever, main.py calls:
    search(sample, k, n_img=..., n_text=..., return_unique=True)

Sample schema (main.py):
  sample = {
    "question": str,
    "image_path": Optional[str],   # for mm queries; may be None
    "is_multimodal": bool,
    "qid": int,
  }

Dependencies:
- torch
- pillow
- sentence-transformers
- transformers (for BLIP captioning in CaptionRetriever)
"""

from __future__ import annotations

import json
import os
import random
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# BLIP captioning (CaptionRetriever)
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
except Exception:
    BlipProcessor = None
    BlipForConditionalGeneration = None


# ----------------------------
# Utilities
# ----------------------------
def l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def _safe_open_rgb(path: str) -> Image.Image:
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:
        # keep alignment stable
        return Image.new("RGB", (224, 224), color=(0, 0, 0))


def _clip_fallback_truncate_words(text: str, max_words: int = 60) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _maybe_strip_image_token(text: str) -> str:
    # dataset-specific token
    return text.replace("[IMAGE]", "").strip()


def _load_image_mapping_csv(
    image_map_path: str,
    *,
    iid_col: str = "IID",
    image_path_col: str = "Image_Path",
    prefix: str = "/data/xiaochen/",
) -> Dict[str, str]:
    """
    image_mapping.csv:
      IID,Image_Path
      I53683003,physionet.org/files/...

    Returns iid -> absolute path
    """
    p = Path(image_map_path)
    if not p.exists():
        raise FileNotFoundError(f"image_mapping.csv not found: {image_map_path}")

    mapping: Dict[str, str] = {}
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV header in: {image_map_path}")
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
# Data structure
# ----------------------------
@dataclass
class KGTriplet:
    head: str
    relation: str
    tail: str
    head_text: str
    relation_text: str
    tail_text: str

    @property
    def triplet_text(self) -> str:
        # Lowercase for consistency
        return f"{self.head_text} {self.relation_text} {self.tail_text}".lower().strip()


# ----------------------------
# BaseRetriever
# ----------------------------
class BaseRetriever:
    """
    Base class that:
    - stores triplets (KGTriplet list)
    - optionally loads image_mapping.csv for resolving head images
    - provides CLIP-safe truncation utility
    """

    def __init__(
        self,
        *,
        retrieval_dataset: List[Tuple[str, str, str]],
        entity2text: Dict[str, str],
        relation2text: Dict[str, str],
        image_map_path: str = "/data/xiaochen/MedMKG_huggingface/image_mapping.csv",
        image_map_prefix: str = "/data/xiaochen/",
        device: Optional[str] = None,
        clip_max_len: int = 77,
        model_name: str = "clip-ViT-B-32",
    ):
        self.entity2text = entity2text
        self.relation2text = relation2text

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_max_len = int(clip_max_len)

        # SentenceTransformer CLIP-style backbone
        self.model = SentenceTransformer(model_name, device=self.device)

        # Try to use underlying tokenizer for token-level truncation
        self._clip_tokenizer = None
        try:
            fm = self.model._first_module()
            if hasattr(fm, "processor") and hasattr(fm.processor, "tokenizer"):
                self._clip_tokenizer = fm.processor.tokenizer
        except Exception:
            self._clip_tokenizer = None

        # Load image mapping (IID -> path)
        self.image_id_to_path = _load_image_mapping_csv(
            image_map_path, prefix=image_map_prefix
        )

        # Build KGTriplet list aligned with retrieval_dataset indices
        self.triplets: List[KGTriplet] = []
        for (h, r, t) in retrieval_dataset:
            head_text = str(entity2text.get(str(h), str(h)))
            rel_text = str(relation2text.get(str(r), str(r)))
            tail_text = str(entity2text.get(str(t), str(t)))

            self.triplets.append(
                KGTriplet(
                    head=str(h),
                    relation=str(r),
                    tail=str(t),
                    head_text=head_text,
                    relation_text=rel_text,
                    tail_text=tail_text,
                )
            )

    # ------------------------------------------
    # CLIP-safe truncation (token-level if possible)
    # ------------------------------------------
    def _truncate_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text
        if self._clip_tokenizer is None:
            return _clip_fallback_truncate_words(text, max_words=60)

        enc = self._clip_tokenizer(
            text,
            truncation=True,
            max_length=self.clip_max_len,
            add_special_tokens=True,
            return_tensors=None,
        )
        return self._clip_tokenizer.decode(enc["input_ids"], skip_special_tokens=True)

    # ------------------------------------------
    # Resolve head image path (for corpus heads)
    # ------------------------------------------
    def _head_image_path(self, head_id: str) -> Optional[str]:
        p = self.image_id_to_path.get(str(head_id))
        if p and os.path.isfile(p):
            return p
        return None


#from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

@dataclass
class KGTripletLite:
    head: str
    relation: str
    tail: str

class RandomRetriever:
    """
    Random baseline. Does NOT require entity2text/relation2text/image_map.
    Matches the return format used by other retrievers.
    """
    def __init__(self, retrieval_dataset: List[Tuple[str, str, str]]):
        self.triplets = [
            KGTripletLite(head=str(h), relation=str(r), tail=str(t))
            for (h, r, t) in retrieval_dataset
        ]

    def search(self, sample: Dict[str, Any], k: int) -> List[dict]:
        import random

        n = len(self.triplets)
        k = min(int(k), n)
        if k <= 0:
            return []

        idxs = random.sample(range(n), k)

        out = []
        for rank, i in enumerate(idxs, start=1):
            out.append(
                {"rank": rank, "index": i, "item": self.triplets[i], "score": float(k - rank + 1)}
            )
        return out
# ----------------------------
# SimpleTextRetriever
# ----------------------------
class SimpleTextRetriever(BaseRetriever):
    """
    Text-only retriever over triplet_text, excluding image heads by default.
    Uses SentenceTransformer to encode triplet texts and cosine similarity.
    """

    def __init__(
        self,
        *,
        retrieval_dataset: List[Tuple[str, str, str]],
        entity2text: Dict[str, str],
        relation2text: Dict[str, str],
        model_name: str = "clip-ViT-B-32",
        batch_size: int = 32,
        cache_dir: str = "cache_embeddings",
        exclude_image_heads: bool = True,
        **kwargs,
    ):
        super().__init__(
            retrieval_dataset=retrieval_dataset,
            entity2text=entity2text,
            relation2text=relation2text,
            model_name=model_name,
            image_map_path=kwargs.get("image_map_path", "/data/xiaochen/MedMKG_huggingface/image_mapping.csv"),
            image_map_prefix=kwargs.get("image_map_prefix", "/data/xiaochen/"),
            device=kwargs.get("device"),
            clip_max_len=kwargs.get("clip_max_len", 77),
        )
        self.batch_size = int(batch_size)
        self.exclude_image_heads = bool(exclude_image_heads)

        # Build texts and a mapping back to triplet indices
        self.texts: List[str] = []
        self.text_idx2triplet_idx: List[int] = []

        for idx, tri in enumerate(self.triplets):
            head_name = tri.head_text.lower()
            if self.exclude_image_heads and "image_" in head_name:
                continue
            t = tri.triplet_text
            t = self._truncate_text(t)
            self.texts.append(t)
            self.text_idx2triplet_idx.append(idx)

        if len(self.texts) == 0:
            self.retrieval_embeddings = torch.empty((0, 1), dtype=torch.float32, device="cpu")
            return

        emb = self.model.encode(
            self.texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=True,
        ).to(self.device)
        emb = l2norm(emb)
        self.retrieval_embeddings = emb.detach().cpu()

    @torch.no_grad()
    def search(self, sample: Dict[str, Any], k: int) -> List[dict]:
        query_text = _maybe_strip_image_token(str(sample.get("question", ""))).lower().strip()
        query_text = self._truncate_text(query_text)

        if self.retrieval_embeddings.shape[0] == 0:
            return []

        q = self.model.encode([query_text], convert_to_tensor=True, show_progress_bar=False).to(self.device)
        q = l2norm(q).detach().cpu()  # (1, D)

        scores = (self.retrieval_embeddings @ q.T).flatten()  # (N,)
        k = min(int(k), int(scores.shape[0]))
        if k <= 0:
            return []

        vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)
        out: List[dict] = []
        for rank, (local_i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            triplet_idx = self.text_idx2triplet_idx[local_i]
            out.append({"rank": rank, "index": triplet_idx, "item": self.triplets[triplet_idx], "score": float(s)})
        return out


# ----------------------------
# SimpleMultimodalRetriever
# ----------------------------
class SimpleMultimodalRetriever(BaseRetriever):
    """
    Multimodal retriever:
      - Triplet embedding = avg(text_emb, head_image_emb_if_exists)
      - Query embedding   = avg(question_text_emb, query_image_emb) if image provided
                          else text_emb only.
      - Retrieval by cosine similarity (dot product after L2 norm)

    Memory-safe:
      - Stores final retrieval embeddings on CPU.
      - Encodes head images in streaming batches.
    """

    def __init__(
        self,
        *,
        retrieval_dataset: List[Tuple[str, str, str]],
        entity2text: Dict[str, str],
        relation2text: Dict[str, str],
        model_name: str = "clip-ViT-B-32",
        batch_size: int = 256,
        image_batch_size: Optional[int] = None,
        exclude_image_heads_in_text: bool = False,
        show_progress: bool = True,
        **kwargs,
    ):
        super().__init__(
            retrieval_dataset=retrieval_dataset,
            entity2text=entity2text,
            relation2text=relation2text,
            model_name=model_name,
            image_map_path=kwargs.get("image_map_path", "/data/xiaochen/MedMKG_huggingface/image_mapping.csv"),
            image_map_prefix=kwargs.get("image_map_prefix", "/data/xiaochen/"),
            device=kwargs.get("device"),
            clip_max_len=kwargs.get("clip_max_len", 77),
        )
        self.batch_size = int(batch_size)
        self.image_batch_size = int(image_batch_size) if image_batch_size else int(batch_size)
        self.exclude_image_heads_in_text = bool(exclude_image_heads_in_text)

        # Build triplet text list
        triplet_texts: List[str] = []
        it: Iterable[KGTriplet] = self.triplets
        if show_progress:
            it = tqdm(it, desc="Building triplet texts")

        for tri in it:
            head = tri.head_text.lower()
            rel = tri.relation_text.lower()
            tail = tri.tail_text.lower()

            parts = []
            if not (self.exclude_image_heads_in_text and "image_" in head):
                parts.append(head)
            parts.append(rel)
            parts.append(tail)

            t = " ".join(parts).strip()
            t = self._truncate_text(t)
            triplet_texts.append(t)

        # Encode text embeddings (N, D)
        text_emb = self.model.encode(
            triplet_texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        ).to(self.device)
        text_emb = l2norm(text_emb)  # (N, D)
        fused = text_emb.clone()

        # Collect head image paths per triplet index
        img_pairs: List[Tuple[int, str]] = []
        it2 = enumerate(self.triplets)
        if show_progress:
            it2 = tqdm(it2, total=len(self.triplets), desc="Collecting head image paths")

        for idx, tri in it2:
            p = self._head_image_path(tri.head)
            if p:
                img_pairs.append((idx, p))

        # Stream-encode images and fuse
        if len(img_pairs) > 0:
            ranges = range(0, len(img_pairs), self.image_batch_size)
            if show_progress:
                ranges = tqdm(ranges, desc="Encoding head images (streaming)")

            for start in ranges:
                batch = img_pairs[start : start + self.image_batch_size]
                batch_indices = [x[0] for x in batch]
                batch_paths = [x[1] for x in batch]

                pil_images: List[Image.Image] = []
                valid_indices: List[int] = []
                for i, pth in zip(batch_indices, batch_paths):
                    try:
                        pil_images.append(_safe_open_rgb(pth))
                        valid_indices.append(i)
                    except Exception:
                        continue

                if not pil_images:
                    continue

                img_emb = self.model.encode(
                    pil_images,
                    batch_size=len(pil_images),
                    convert_to_tensor=True,
                    show_progress_bar=show_progress,
                ).to(self.device)
                img_emb = l2norm(img_emb)  # (B, D)

                idx_tensor = torch.tensor(valid_indices, device=self.device, dtype=torch.long)
                fused[idx_tensor] = l2norm((text_emb[idx_tensor] + img_emb) / 2.0)

                # cleanup
                del pil_images, img_emb, idx_tensor
                if "cuda" in str(self.device):
                    torch.cuda.empty_cache()

        self.retrieval_embeddings = fused.detach().cpu()  # (N, D)

    @torch.no_grad()
    def search(self, sample: Dict[str, Any], k: int) -> List[dict]:
        q_text = _maybe_strip_image_token(str(sample.get("question", ""))).lower().strip()
        q_text = self._truncate_text(q_text)

        q_text_emb = self.model.encode([q_text], convert_to_tensor=True, show_progress_bar=False).to(self.device)
        q_text_emb = l2norm(q_text_emb)  # (1, D)

        q_emb = q_text_emb

        img_path = sample.get("image_path", None)
        if img_path and isinstance(img_path, str) and os.path.isfile(img_path):
            q_img = _safe_open_rgb(img_path)
            q_img_emb = self.model.encode([q_img], convert_to_tensor=True, show_progress_bar=False).to(self.device)
            q_img_emb = l2norm(q_img_emb)
            q_emb = l2norm((q_text_emb + q_img_emb) / 2.0)

        scores = (self.retrieval_embeddings @ q_emb.detach().cpu().T).flatten()
        k = min(int(k), int(scores.shape[0]))
        if k <= 0:
            return []

        vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)
        out: List[dict] = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            out.append({"rank": rank, "index": int(i), "item": self.triplets[int(i)], "score": float(s)})
        return out


# ----------------------------
# MMAnchorRetriever
# ----------------------------
class MMAnchorRetriever(BaseRetriever):
    """
    Two-stage retriever:
      Stage 1: query image -> retrieve top n_img image-head IDs (image similarity)
      Stage 2: candidates = triplets whose head in those image IDs
               rerank candidates by query text similarity

    If query has no valid image_path, fallback to global text-only retrieval.

    Notes:
    - Requires image mapping for corpus heads.
    - Stores image head embedding index on CPU (memory-safe).
    """

    def __init__(
        self,
        *,
        retrieval_dataset: List[Tuple[str, str, str]],
        entity2text: Dict[str, str],
        relation2text: Dict[str, str],
        model_name: str = "clip-ViT-B-32",
        batch_size: int = 256,
        image_batch_size: int = 256,
        show_progress: bool = True,
        cache_dir: str = "cache_embeddings",
        clip_max_len: int = 77,
        store_image_emb_on_cpu: bool = True,
        **kwargs,
    ):
        super().__init__(
            retrieval_dataset=retrieval_dataset,
            entity2text=entity2text,
            relation2text=relation2text,
            model_name=model_name,
            image_map_path=kwargs.get("image_map_path", "/data/xiaochen/MedMKG_huggingface/image_mapping.csv"),
            image_map_prefix=kwargs.get("image_map_prefix", "/data/xiaochen/"),
            device=kwargs.get("device"),
            clip_max_len=clip_max_len,
        )
        self.batch_size = int(batch_size)
        self.image_batch_size = int(image_batch_size)
        self.store_image_emb_on_cpu = bool(store_image_emb_on_cpu)

        # head -> triplet indices
        self.head2triplet_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, tri in enumerate(self.triplets):
            self.head2triplet_indices[str(tri.head)].append(idx)

        # Build triplet text embeddings (N, D) on device
        triplet_texts: List[str] = []
        it: Iterable[KGTriplet] = self.triplets
        if show_progress:
            it = tqdm(it, desc="Building triplet texts")

        for tri in it:
            t = tri.triplet_text
            t = self._truncate_text(t)
            triplet_texts.append(t)

        text_emb = self.model.encode(
            triplet_texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=show_progress,
        ).to(self.device)
        self.text_emb = l2norm(text_emb)  # (N, D)

        # Collect unique image heads + paths
        self.image_ids: List[str] = []
        self.image_paths: List[str] = []

        heads = list(self.head2triplet_indices.keys())
        if show_progress:
            heads = tqdm(heads, desc="Collecting image-head paths")  # type: ignore[assignment]

        for hid in heads:
            p = self._head_image_path(hid)
            if p:
                self.image_ids.append(hid)
                self.image_paths.append(p)

        # Build image embedding index (M, D) on CPU by streaming encoding
        if len(self.image_paths) == 0:
            self.image_emb = None
        else:
            self.image_emb = self._encode_images_streaming(self.image_paths, show_progress=show_progress)

    @torch.no_grad()
    def _encode_images_streaming(self, paths: List[str], *, show_progress: bool) -> torch.Tensor:
        D = int(self.text_emb.shape[1])
        M = len(paths)

        image_emb_cpu = torch.empty((M, D), dtype=torch.float32, device="cpu")

        rng = range(0, M, self.image_batch_size)
        if show_progress:
            rng = tqdm(rng, desc="Encoding image heads (streaming)")

        for start in rng:
            batch_paths = paths[start : start + self.image_batch_size]
            imgs = [_safe_open_rgb(p) for p in batch_paths]

            emb = self.model.encode(
                imgs,
                batch_size=len(imgs),
                convert_to_tensor=True,
                show_progress_bar=False,
            ).to(self.device)
            emb = l2norm(emb).detach().cpu()

            image_emb_cpu[start : start + len(imgs)] = emb

            # free
            del imgs, emb
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()

        return image_emb_cpu

    @torch.no_grad()
    def search(
        self,
        sample: Dict[str, Any],
        k: int,
        *,
        n_img: int = 10,
        n_text: int = 5,          # kept for compatibility; not required in this design
        return_unique: bool = True,
    ) -> List[dict]:
        k = int(k)
        if k <= 0:
            return []

        q_text = _maybe_strip_image_token(str(sample.get("question", ""))).lower().strip()
        q_text = self._truncate_text(q_text)

        q_text_emb = self.model.encode([q_text], convert_to_tensor=True, show_progress_bar=False).to(self.device)
        q_text_emb = l2norm(q_text_emb)  # (1, D)

        # If no image index or query has no image -> fallback to global text-only
        img_path = sample.get("image_path", None)
        has_query_img = isinstance(img_path, str) and img_path and os.path.isfile(img_path)

        if self.image_emb is None or len(self.image_ids) == 0 or not has_query_img:
            scores = (self.text_emb @ q_text_emb.T).flatten()
            keep = min(k, int(scores.shape[0]))
            vals, idxs = torch.topk(scores, k=keep, largest=True, sorted=True)
            out = []
            for rank, (ti, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
                out.append({"rank": rank, "index": int(ti), "item": self.triplets[int(ti)], "score": float(s)})
            return out

        # Stage 1: query image embedding -> retrieve top n_img heads
        q_img = _safe_open_rgb(str(img_path))
        q_img_emb = self.model.encode([q_img], convert_to_tensor=True, show_progress_bar=False).to(self.device)
        q_img_emb = l2norm(q_img_emb).detach().cpu()  # (1, D) on CPU for scoring against CPU index

        s_img = (self.image_emb @ q_img_emb.T).flatten()  # (M,)
        keep_m = min(int(n_img), int(s_img.shape[0]))
        if keep_m <= 0:
            return []

        top_img_local = torch.topk(s_img, k=keep_m, largest=True, sorted=True).indices.tolist()
        retrieved_image_ids = [self.image_ids[i] for i in top_img_local]

        # Stage 2: candidates linked to retrieved images
        candidate_indices: List[int] = []
        seen = set()
        for img_id in retrieved_image_ids:
            for ti in self.head2triplet_indices.get(str(img_id), []):
                if return_unique:
                    if ti in seen:
                        continue
                    seen.add(ti)
                candidate_indices.append(ti)

        if not candidate_indices:
            return []

        cand_emb = self.text_emb[candidate_indices]  # (C, D) on device
        cand_scores = (cand_emb @ q_text_emb.T).flatten()  # (C,)
        keep = min(k, int(cand_scores.shape[0]))
        vals, local_idxs = torch.topk(cand_scores, k=keep, largest=True, sorted=True)

        out = []
        for rank, (j, s) in enumerate(zip(local_idxs.tolist(), vals.tolist()), start=1):
            ti = candidate_indices[int(j)]
            out.append({"rank": rank, "index": int(ti), "item": self.triplets[int(ti)], "score": float(s)})

        return out


# ----------------------------
# CaptionRetriever
# ----------------------------
class CaptionRetriever(BaseRetriever):
    """
    Caption-aware text retriever:
      - If head is an image entity, replace head_text with an auto-caption (BLIP).
      - Then do text-only retrieval with SentenceTransformer.

    Uses a JSON cache (caption_cache_path) to avoid recaptioning.

    NOTE:
      - Captioning is expensive; run once and cache.
      - Requires transformers + BLIP weights.
    """

    def __init__(
        self,
        *,
        retrieval_dataset: List[Tuple[str, str, str]],
        entity2text: Dict[str, str],
        relation2text: Dict[str, str],
        model_name: str = "clip-ViT-B-32",
        batch_size: int = 32,
        cache_dir: str = "cache_embeddings",
        caption_cache_path: str = "cache_embeddings/caption_cache_blip.json",
        blip_model_name: str = "Salesforce/blip-image-captioning-base",
        show_progress: bool = False,
        **kwargs,
    ):
        super().__init__(
            retrieval_dataset=retrieval_dataset,
            entity2text=entity2text,
            relation2text=relation2text,
            model_name=model_name,
            image_map_path=kwargs.get("image_map_path", "/data/xiaochen/MedMKG_huggingface/image_mapping.csv"),
            image_map_prefix=kwargs.get("image_map_prefix", "/data/xiaochen/"),
            device=kwargs.get("device"),
            clip_max_len=kwargs.get("clip_max_len", 77),
        )
        self.batch_size = int(batch_size)
        self.caption_cache_path = str(caption_cache_path)

        if BlipProcessor is None or BlipForConditionalGeneration is None:
            raise ImportError(
                "CaptionRetriever requires transformers with BLIP. "
                "Please install/upgrade transformers."
            )

        # Load / init caption cache
        self._caption_cache: Dict[str, str] = {}
        cap_p = Path(self.caption_cache_path)
        cap_p.parent.mkdir(parents=True, exist_ok=True)
        if cap_p.exists():
            try:
                self._caption_cache = json.loads(cap_p.read_text(encoding="utf-8"))
            except Exception:
                self._caption_cache = {}

        # BLIP runs fine on CPU but is slow; use GPU if available
        self._blip_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._blip_processor = BlipProcessor.from_pretrained(blip_model_name)
        self._blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name).to(self._blip_device)
        self._blip_model.eval()

        # Build texts (with captions where applicable)
        texts: List[str] = []
        self.text_idx2triplet_idx: List[int] = []

        it = enumerate(self.triplets)
        if show_progress:
            it = tqdm(it, total=len(self.triplets), desc="Building caption texts")

        for idx, tri in it:
            head_name = tri.head_text.lower()
            rel = tri.relation_text.lower()
            tail = tri.tail_text.lower()

            if "image_" in head_name or str(tri.head).startswith("I"):
                # Use caption if head image exists
                cap = self._get_or_make_caption(str(tri.head))
                head_part = cap.lower().strip() if cap else head_name
            else:
                head_part = head_name

            t = f"{head_part} {rel} {tail}".strip()
            t = self._truncate_text(t)
            texts.append(t)
            self.text_idx2triplet_idx.append(idx)

        # Save updated cache
        try:
            cap_p.write_text(json.dumps(self._caption_cache, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        # Encode retrieval embeddings
        if len(texts) == 0:
            self.retrieval_embeddings = torch.empty((0, 1), dtype=torch.float32, device="cpu")
            return

        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        ).to(self.device)
        emb = l2norm(emb)
        self.retrieval_embeddings = emb.detach().cpu()

    @torch.no_grad()
    def _get_or_make_caption(self, head_id: str) -> str:
        # Cached?
        if head_id in self._caption_cache:
            return self._caption_cache[head_id]

        p = self._head_image_path(head_id)
        if not p:
            self._caption_cache[head_id] = ""
            return ""

        img = _safe_open_rgb(p)
        inputs = self._blip_processor(images=img, return_tensors="pt").to(self._blip_device)
        out = self._blip_model.generate(**inputs, max_new_tokens=40)
        cap = self._blip_processor.decode(out[0], skip_special_tokens=True).strip()

        self._caption_cache[head_id] = cap
        return cap

    @torch.no_grad()
    def search(self, sample: Dict[str, Any], k: int) -> List[dict]:
        query_text = _maybe_strip_image_token(str(sample.get("question", ""))).lower().strip()
        query_text = self._truncate_text(query_text)

        if self.retrieval_embeddings.shape[0] == 0:
            return []

        q = self.model.encode([query_text], convert_to_tensor=True, show_progress_bar=False).to(self.device)
        q = l2norm(q).detach().cpu()

        scores = (self.retrieval_embeddings @ q.T).flatten()
        k = min(int(k), int(scores.shape[0]))
        if k <= 0:
            return []

        vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)
        out: List[dict] = []
        for rank, (local_i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            triplet_idx = self.text_idx2triplet_idx[int(local_i)]
            out.append({"rank": rank, "index": int(triplet_idx), "item": self.triplets[int(triplet_idx)], "score": float(s)})
        return out
