import os
from typing import List, Tuple, Dict, Optional, Iterable, Any
from PIL import Image
from utils import *
import torch
import random
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from tqdm import tqdm

from embedding_cache import EmbeddingCache, CacheConfig, l2norm
from transformers import BlipProcessor, BlipForConditionalGeneration

import hashlib

CACHE_DIR = "cache_embeddings"
cache = EmbeddingCache(CacheConfig(cache_dir=CACHE_DIR))

TAG_TRIPLET_TEXT = "triplet_text_concat_v1"
TAG_KG_ENTITY_IMAGES = "kg_entity_images_v1"


def first_jpg_path(
    entity_id: str,
    root: str,
    exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> Optional[str]:
    """
    Resolve an image path for an entity id:
      1) {root}/{entity_id}.jpg|jpeg|png|webp
      2) {root}/{entity_id}/<first image file>
    """
    # direct file
    for ext in exts:
        p = os.path.join(root, f"{entity_id}{ext}")
        if os.path.isfile(p):
            return p

    # folder with files
    d = os.path.join(root, str(entity_id))
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if os.path.splitext(fn.lower())[1] in exts:
                return os.path.join(d, fn)

    return None


# ----------------------------
# Helpers for "new-style" sample dict
# ----------------------------
def _sample_question_text(sample: Dict[str, Any]) -> str:
    # Key requirement: use sample["question"] as query text; no lowercase, no truncation.
    return str(sample.get("question", "")).strip()


def _sample_image_path(sample: Dict[str, Any]) -> Optional[str]:
    # Prefer explicit image_path if caller provides it (matches the "second" script).
    p = sample.get("image_path", None)
    if isinstance(p, str) and p and os.path.isfile(p):
        return p
    return None


class MMAnchorRetriever:
    """
    Two-stage strategy aligned with the "second" script:

      Stage 1: query image -> retrieve top-n_img HEAD entities by image similarity
               (image index is per-head entity, NOT per-triplet; uses head images only)
      Stage 2: candidates = triplets whose head in those retrieved heads
               rerank candidates by text similarity using query text = sample["question"].

    Fallback:
      If query has no valid image (or no indexed head images), do global text-only retrieval.
    """

    def __init__(
        self,
        retrieval_dataset: Iterable[Tuple[str, str, str]],
        entity2text: Dict[str, str],
        relation2text: Dict[str, str],
        model_name: str = "clip-ViT-B-32",
        batch_size: int = 32,
        image_root: str = "images_subset_kg",
        inference_image_root: str = "images_subset_inference",
        show_progress: bool = False,
        device: Optional[str] = None,
        cache_dir: str = "cache_embeddings",
    ):
        self.retrieval_dataset = list(retrieval_dataset)
        self.entity2text = entity2text
        self.relation2text = relation2text
        self.image_root = image_root
        self.inference_image_root = inference_image_root
        self.batch_size = batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)

        self.cache = EmbeddingCache(CacheConfig(cache_dir=cache_dir))

        # -----------------------
        # 1) Triplet text strings + cached embeddings (N, D)
        # -----------------------
        self.triplet_text: List[str] = []
        for (h, r, t) in self.retrieval_dataset:
            parts = [
                self.entity2text.get(h, h),
                self.relation2text.get(r, r),
                self.entity2text.get(t, t),
            ]
            self.triplet_text.append(" ".join([p for p in parts if p]).strip())

        self.text_emb = self.cache.get_text_embeddings(
            model_name=self.model_name,
            texts=self.triplet_text,
            encoder=self.model,
            batch_size=self.batch_size,
            device=self.device,
            normalize=True,
            tag=TAG_TRIPLET_TEXT,
        )  # (N, D)

        # -----------------------
        # 2) Build head -> triplet indices mapping
        # -----------------------
        self.head2triplet_indices: Dict[str, List[int]] = {}
        for idx, (h, _, _) in enumerate(self.retrieval_dataset):
            self.head2triplet_indices.setdefault(h, []).append(idx)

        # -----------------------
        # 3) Cache HEAD entity image embeddings only (KG head images)
        #    Build image index over unique heads that have an image.
        # -----------------------
        uniq_heads = sorted({h for (h, _, _) in self.retrieval_dataset})

        def _resolve_kg_head(eid: str) -> Optional[str]:
            return first_jpg_path(eid, self.image_root)

        head_img_emb_all, head2row = self.cache.get_entity_image_embeddings(
            model_name=self.model_name,
            entity_ids=uniq_heads,
            resolve_path_fn=_resolve_kg_head,
            encoder=self.model,
            batch_size=self.batch_size,
            device=self.device,
            normalize=True,
            tag=TAG_KG_ENTITY_IMAGES,
        )  # (M_all, D), mapping in head2row

        # Only keep heads that resolved to an embedding
        self.image_heads: List[str] = []
        self.image_emb: Optional[torch.Tensor] = None

        if len(head2row) > 0:
            # Keep a stable ordering aligned with image_emb rows
            self.image_heads = [h for h in uniq_heads if h in head2row]
            if self.image_heads:
                rows = [head2row[h] for h in self.image_heads]
                self.image_emb = head_img_emb_all[rows]  # (M, D)

    @torch.no_grad()
    def search(
        self,
        sample: Dict[str, Any],
        k: int = 10,
        *,
        n_img: int = 10,
        n_text: int = 5,  # kept for API compatibility; not required in this design
        return_unique: bool = True,
    ) -> List[dict]:
        """
        sample schema (expected):
          {
            "question": str,
            "image_path": Optional[str],  # preferred if provided
            ... (other fields ignored)
          }

        Returns list[dict] with {"rank","index","item","score"}.
        score is text similarity (stage-2 text rerank).
        """
        device = self.text_emb.device
        N = len(self.retrieval_dataset)
        k = min(int(k), N)
        if k <= 0:
            return []

        # -----------------------
        # Query text: sample["question"]
        # -----------------------
        q_text = _sample_question_text(sample)
        q_text_emb = self.model.encode(
            [q_text], convert_to_tensor=True, show_progress_bar=False
        ).to(device)
        q_text_emb = F.normalize(q_text_emb, p=2, dim=1).to(self.text_emb.dtype)  # (1, D)

        # -----------------------
        # Determine query image path
        # Prefer sample["image_path"], else keep old "question-as-id" fallback without changing logic.
        # -----------------------
        qp = _sample_image_path(sample)
        if qp is None:
            # Fallback: preserve old behavior if caller uses "question" as an ID (legacy pipeline)
            # (No changes to finding logic; just a fallback.)
            qid_like = str(sample.get("question", "")).strip()
            fp = first_jpg_path(qid_like, self.inference_image_root)
            if fp and os.path.isfile(fp):
                qp = fp

        has_query_img = qp is not None and os.path.isfile(qp)
        has_img_index = self.image_emb is not None and self.image_emb.numel() > 0 and len(self.image_heads) > 0

        # -----------------------
        # Fallback: global text-only retrieval (question-only)
        # -----------------------
        if (not has_query_img) or (not has_img_index):
            scores = (self.text_emb @ q_text_emb.T).flatten()  # (N,)
            vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)
            out: List[dict] = []
            for rank, (ti, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
                out.append(
                    {
                        "rank": rank,
                        "index": int(ti),
                        "item": self.retrieval_dataset[int(ti)],
                        "score": float(s),
                    }
                )
            return out

        # -----------------------
        # Stage 1: query image -> retrieve top-n_img heads
        # -----------------------
        with Image.open(qp) as im:
            q_img_emb = self.model.encode(
                [im.convert("RGB")], convert_to_tensor=True, show_progress_bar=False
            ).to(device)
        q_img_emb = F.normalize(q_img_emb, p=2, dim=1)  # (1, D)
        q_img_emb = q_img_emb.to(dtype=self.image_emb.dtype, device=self.image_emb.device)

        s_img = (self.image_emb @ q_img_emb.T).flatten()  # (M,)
        keep_m = min(int(n_img), int(s_img.shape[0]))
        top_img_local = torch.topk(s_img, k=keep_m, largest=True, sorted=True).indices.tolist()
        retrieved_heads = [self.image_heads[i] for i in top_img_local]

        # -----------------------
        # Stage 2: candidates = triplets whose head in retrieved_heads, rerank by text similarity
        # -----------------------
        candidate_indices: List[int] = []
        seen = set()
        for h in retrieved_heads:
            for ti in self.head2triplet_indices.get(h, []):
                if return_unique:
                    if ti in seen:
                        continue
                    seen.add(ti)
                candidate_indices.append(ti)

        if not candidate_indices:
            return []

        cand_emb = self.text_emb[candidate_indices]  # (C, D)
        cand_scores = (cand_emb @ q_text_emb.T).flatten()  # (C,)

        keep = min(k, int(cand_scores.shape[0]))
        vals, local_idxs = torch.topk(cand_scores, k=keep, largest=True, sorted=True)

        out: List[dict] = []
        for rank, (j, s) in enumerate(zip(local_idxs.tolist(), vals.tolist()), start=1):
            ti = candidate_indices[int(j)]
            out.append(
                {
                    "rank": rank,
                    "index": int(ti),
                    "item": self.retrieval_dataset[int(ti)],
                    "score": float(s),
                }
            )
        return out


class SimpleMultimodalRetriever:
    """
    Revised to match key points:
      - Query text is sample["question"] only.
      - Triplet fusion uses head image only (no tail image).
      - Keep existing image finding logic for KG heads (image_root) and query (sample["image_path"] preferred,
        fallback to first_jpg_path(question_like, inference_image_root)).
      - No truncation / no lowercasing.
    """

    def __init__(
        self,
        retrieval_dataset,
        entity2text,
        relation2text,
        model_name="clip-ViT-B-32",
        batch_size=32,
        inference_image_root: str = "images_subset_inference",
        image_root: str = "images_subset_kg",
    ):
        self.retrieval_dataset = list(retrieval_dataset)
        self.entity2text = entity2text
        self.relation2text = relation2text
        self.inference_image_root = inference_image_root
        self.image_root = image_root

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

        cache = EmbeddingCache(CacheConfig(cache_dir="cache_embeddings"))

        # 1) Build triplet text strings
        self.text = []
        for (head, relation, tail) in self.retrieval_dataset:
            t = (
                f"{self.entity2text.get(head, head)} "
                f"{self.relation2text.get(relation, relation)} "
                f"{self.entity2text.get(tail, tail)}"
            ).strip()
            self.text.append(t)

        # Cache triplet text embeddings: (N, D)
        text_emb = cache.get_text_embeddings(
            model_name=model_name,
            texts=self.text,
            encoder=self.model,
            batch_size=batch_size,
            device=self.device,
            normalize=True,
            tag=TAG_TRIPLET_TEXT,
        )

        # 2) Cache HEAD entity image embeddings only (KG head images)
        uniq_heads = sorted({h for (h, _, _) in self.retrieval_dataset})

        def _resolve_kg_head(eid: str):
            return first_jpg_path(eid, self.image_root)

        head_img_emb, head2row = cache.get_entity_image_embeddings(
            model_name=model_name,
            entity_ids=uniq_heads,
            resolve_path_fn=_resolve_kg_head,
            encoder=self.model,
            batch_size=batch_size,
            device=self.device,
            normalize=True,
            tag=TAG_KG_ENTITY_IMAGES,
        )

        # 3) Compose per-triplet embedding by averaging [triplet_text, head_img] when available
        N = len(self.retrieval_dataset)
        D = text_emb.shape[1]
        triplet_vecs = torch.zeros((N, D), device=self.device)
        counts = torch.zeros((N,), device=self.device)

        triplet_vecs += text_emb
        counts += 1.0

        for i, (h, _, _t) in enumerate(self.retrieval_dataset):
            if h in head2row:
                triplet_vecs[i] += head_img_emb[head2row[h]]
                counts[i] += 1.0

        triplet_vecs = triplet_vecs / counts.unsqueeze(1).clamp_min(1.0)
        self.retrieval_embeddings = torch.nn.functional.normalize(triplet_vecs, p=2, dim=1)

        print("self.retrieval_embeddings:", self.retrieval_embeddings.shape)

    @torch.no_grad()
    def search(self, sample: Dict[str, Any], k: int):
        """
        sample: {"question": str, "image_path": Optional[str], ...}

        Unified behavior (aligned with second script):
          - Encode question text (only).
          - If query image exists, average with image embedding.
        """
        k = min(int(k), len(self.retrieval_dataset))
        if k <= 0:
            return []

        def _encode_text(s: str) -> torch.Tensor:
            emb = self.model.encode([s], convert_to_tensor=True, show_progress_bar=False).to(self.device)
            return emb  # (1, D)

        def _encode_image(path: str) -> Optional[torch.Tensor]:
            if not path or (not os.path.isfile(path)):
                return None
            with Image.open(path) as im:
                im = im.convert("RGB")
                emb = self.model.encode([im], convert_to_tensor=True, show_progress_bar=False).to(self.device)
            return emb  # (1, D)

        q_text = _sample_question_text(sample)
        q_text_emb = _encode_text(q_text)

        # Prefer sample["image_path"], fallback to first_jpg_path(question_like, inference_image_root)
        q_img_path = _sample_image_path(sample)
        if q_img_path is None:
            qid_like = str(sample.get("question", "")).strip()
            fp = first_jpg_path(qid_like, self.inference_image_root)
            if fp and os.path.isfile(fp):
                q_img_path = fp

        query_embs: List[torch.Tensor] = [q_text_emb]
        img_emb = _encode_image(q_img_path) if q_img_path else None
        if img_emb is not None:
            query_embs.append(img_emb)

        query_embedding = torch.cat(query_embs, dim=0).mean(dim=0, keepdim=True)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

        scores = (self.retrieval_embeddings @ query_embedding.T).flatten()
        vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)

        out = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            out.append(
                {
                    "rank": rank,
                    "index": i,
                    "item": self.retrieval_dataset[i],
                    "score": float(s),
                }
            )
        return out


class SimpleTextRetriever:
    """
    Text-only retriever.

    Revised per key point:
      - Query text is sample["question"] only.
    """

    def __init__(
        self,
        retrieval_dataset: Iterable[Tuple[str, str, str]],
        entity2text: Dict[str, str],
        relation2text: Dict[str, str],
        model_name: str = "clip-ViT-B-32",
        batch_size: int = 32,
        device: Optional[str] = None,
        cache_dir: str = "cache_embeddings",
        cache_tag: str = "simpletext_triplet_text_concat",
    ):
        self.retrieval_dataset = list(retrieval_dataset)
        self.entity2text = entity2text
        self.relation2text = relation2text
        self.batch_size = batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)

        self.triplet_texts: List[str] = []
        for (h, r, t) in self.retrieval_dataset:
            parts = [
                self.entity2text.get(h, h),
                self.relation2text.get(r, r),
                self.entity2text.get(t, t),
            ]
            self.triplet_texts.append(" ".join([p for p in parts if p]).strip())

        self.cache = EmbeddingCache(CacheConfig(cache_dir=cache_dir))
        self.retrieval_embeddings = self.cache.get_text_embeddings(
            model_name=self.model_name,
            texts=self.triplet_texts,
            encoder=self.model,
            batch_size=self.batch_size,
            device=self.device,
            normalize=True,
            tag=TAG_TRIPLET_TEXT,
        )

    @torch.no_grad()
    def search(self, sample: Dict[str, Any], k: int):
        """
        sample: {"question": str, ...}
        Returns: list[dict] with rank/index/item/score
        """
        k = min(int(k), len(self.retrieval_dataset))
        if k <= 0:
            return []

        q_text = _sample_question_text(sample)

        q_emb = self.model.encode([q_text], convert_to_tensor=True, show_progress_bar=False).to(self.device)
        q_emb = l2norm(q_emb).to(self.retrieval_embeddings.dtype)

        scores = (self.retrieval_embeddings @ q_emb.T).flatten()
        vals, idxs = torch.topk(scores, k=min(k, scores.numel()), largest=True, sorted=True)

        out = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            out.append(
                {
                    "rank": rank,
                    "index": i,
                    "item": self.retrieval_dataset[i],
                    "score": float(s),
                }
            )
        return out


class RandomRetriever:
    # Random baseline
    def __init__(self, retrieval_dataset):
        self.retrieval_dataset = list(retrieval_dataset)

    def search(self, sample: Dict[str, Any], k: int):
        k = min(int(k), len(self.retrieval_dataset))
        if k <= 0:
            return []

        random_idx = random.sample([i for i in range(len(self.retrieval_dataset))], k)

        out = []
        for rank, i in enumerate(random_idx, start=1):
            out.append(
                {
                    "rank": rank,
                    "index": i,
                    "item": self.retrieval_dataset[i],
                    "score": float(k - rank + 1),  # arbitrary
                }
            )
        return out


def _file_sig(path: str) -> str:
    """
    A lightweight signature to invalidate cached captions if the file changes.
    """
    st = os.stat(path)
    return f"{os.path.basename(path)}::{st.st_size}::{st.st_mtime_ns}"


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


class CaptionRetriever:
    """
    Caption-based text retriever.

    Revised per key points:
      - Query text is sample["question"] only.
      - Only use images of HEAD entities (no tail image captioning).
      - Keep caption cache + BLIP logic + first_jpg_path resolution unchanged.
      - No truncation / no lowercasing added.
    """

    def __init__(
        self,
        retrieval_dataset: Iterable[Tuple[str, str, str]],
        entity2text: Dict[str, str],
        relation2text: Dict[str, str],
        model_name: str = "clip-ViT-B-32",
        batch_size: int = 32,
        device: Optional[str] = None,
        image_root: str = "images_subset_kg",
        cache_dir: str = "cache_embeddings",
        caption_cache_path: str = "cache_embeddings/caption_cache_blip.json",
        cache_tag: str = "captionretriever_triplet_caption_text",
        blip_model_id: str = "Salesforce/blip-image-captioning-base",
    ):
        self.retrieval_dataset = list(retrieval_dataset)
        self.entity2text = entity2text
        self.relation2text = relation2text
        self.image_root = image_root
        self.batch_size = batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)

        self.cache = EmbeddingCache(CacheConfig(cache_dir=cache_dir))

        os.makedirs(os.path.dirname(caption_cache_path), exist_ok=True)
        self.caption_cache_path = caption_cache_path
        self._caption_cache: Dict[str, str] = {}
        if os.path.isfile(self.caption_cache_path):
            import json

            with open(self.caption_cache_path, "r", encoding="utf-8") as f:
                try:
                    self._caption_cache = json.load(f)
                except Exception:
                    self._caption_cache = {}

        self._blip_processor = BlipProcessor.from_pretrained(blip_model_id)
        self._blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_id)
        self._blip_model.to("cpu")
        self._blip_model.eval()

        self.texts: List[str] = []
        for (head, relation, tail) in self.retrieval_dataset:
            # HEAD may be captioned; TAIL uses text only (no tail image)
            head_text = self._head_caption_or_text(head)
            rel_text = self.relation2text.get(relation, relation)
            tail_text = self.entity2text.get(tail, tail)
            self.texts.append(" ".join([p for p in [head_text, rel_text, tail_text] if p]).strip())

        self._flush_caption_cache()

        self.retrieval_embeddings = self.cache.get_text_embeddings(
            model_name=self.model_name,
            texts=self.texts,
            encoder=self.model,
            batch_size=self.batch_size,
            device=self.device,
            normalize=True,
            tag=cache_tag,
        )

    def _flush_caption_cache(self):
        import json

        tmp = self.caption_cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._caption_cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.caption_cache_path)

    @torch.no_grad()
    def _caption_image(self, img_path: str) -> str:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        inputs = self._blip_processor(im, return_tensors="pt")
        out = self._blip_model.generate(**inputs)
        cap = self._blip_processor.decode(out[0], skip_special_tokens=True)
        return str(cap).strip()

    def _head_caption_or_text(self, entity_id: str) -> str:
        img_path = first_jpg_path(entity_id, self.image_root)
        if not img_path or not os.path.isfile(img_path):
            return self.entity2text.get(entity_id, entity_id)

        sig = _file_sig(img_path)
        key = f"{entity_id}::{sig}"

        if key in self._caption_cache:
            return self._caption_cache[key]

        cap = self._caption_image(img_path)
        self._caption_cache[key] = cap
        return cap

    @torch.no_grad()
    def search(self, sample: Dict[str, Any], k: int):
        """
        sample: {"question": str, ...}
        Returns: list[dict] with rank/index/item/score
        """
        k = min(int(k), len(self.retrieval_dataset))
        if k <= 0:
            return []

        q_text = _sample_question_text(sample)

        q_emb = self.model.encode([q_text], convert_to_tensor=True, show_progress_bar=False).to(self.device)
        q_emb = l2norm(q_emb)
        q_emb = q_emb.to(dtype=self.retrieval_embeddings.dtype, device=self.retrieval_embeddings.device)

        scores = (self.retrieval_embeddings @ q_emb.T).flatten()
        vals, idxs = torch.topk(scores, k=min(k, scores.numel()), largest=True, sorted=True)

        out = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            out.append(
                {
                    "rank": rank,
                    "index": i,
                    "item": self.retrieval_dataset[i],
                    "score": float(s),
                }
            )
        return out
