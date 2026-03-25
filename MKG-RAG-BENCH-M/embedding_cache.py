# embedding_cache.py
import os
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import PreTrainedTokenizerBase

def _truncate_texts_for_clip(encoder, texts, max_len: int = 77):
    """
    Truncate texts to CLIP max token length using the encoder's tokenizer.
    Returns decoded truncated texts so we can still call encoder.encode(list[str]).
    """
    # SentenceTransformer usually exposes tokenizer
    tok = getattr(encoder, "tokenizer", None)
    if tok is None:
        # fallback: no tokenizer; return original
        return texts

    # Tokenize with truncation
    enc = tok(
        texts,
        padding=False,
        truncation=True,
        max_length=max_len,
        return_tensors=None,
    )

    # Convert back to text (drop special tokens)
    # Note: batch_decode exists on HF tokenizers
    if hasattr(tok, "batch_decode"):
        return tok.batch_decode(enc["input_ids"], skip_special_tokens=True)

    return texts


def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _sha1_json(obj) -> str:
    return _sha1_bytes(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8"))


def _sanitize(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in s)


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def _file_sig(path: str) -> Tuple[str, int, int]:
    """
    Lightweight invalidation signature:
      (basename, mtime_ns, size)
    """
    st = os.stat(path)
    return (os.path.basename(path), int(st.st_mtime_ns), int(st.st_size))


@dataclass
class CacheConfig:
    cache_dir: str = "cache_embeddings"
    # Optional: store as fp16 to save disk; set None to store original dtype
    save_dtype: Optional[torch.dtype] = torch.float16

    def ensure(self):
        os.makedirs(self.cache_dir, exist_ok=True)

def _get_hf_text_tokenizer(encoder):
    """
    SentenceTransformer CLIP wrappers sometimes store a CLIPProcessor in encoder.tokenizer.
    We need the underlying *text* tokenizer (CLIPTokenizer/CLIPTokenizerFast).
    """
    tok = getattr(encoder, "tokenizer", None)

    # Case 1: already a real HF tokenizer
    if isinstance(tok, PreTrainedTokenizerBase):
        return tok

    # Case 2: it's a Processor (e.g., CLIPProcessor) that contains a .tokenizer
    inner = getattr(tok, "tokenizer", None)
    if isinstance(inner, PreTrainedTokenizerBase):
        return inner

    # Case 3: SentenceTransformer module sometimes has tokenizer on first module
    try:
        fm = encoder._first_module()
        tok2 = getattr(fm, "tokenizer", None)
        if isinstance(tok2, PreTrainedTokenizerBase):
            return tok2
        inner2 = getattr(tok2, "tokenizer", None)
        if isinstance(inner2, PreTrainedTokenizerBase):
            return inner2
    except Exception:
        pass

    return None


def _truncate_texts_for_clip(encoder, texts: List[str], max_len: int = 77) -> List[str]:
    hf_tok = _get_hf_text_tokenizer(encoder)
    if hf_tok is None:
        # Can't safely truncate; return original (but CLIP may crash if too long)
        return texts

    enc = hf_tok(
        texts,
        padding=False,
        truncation=True,
        max_length=max_len,
        return_tensors=None,
    )
    return hf_tok.batch_decode(enc["input_ids"], skip_special_tokens=True)

class EmbeddingCache:
    """
    Caches:
      - Text embeddings for an ordered list of strings
      - Image embeddings for an ordered list of image paths
      - Entity image embeddings (entity_id -> embedding) given a resolver
    """

    def __init__(self, cfg: CacheConfig):
        self.cfg = cfg
        self.cfg.ensure()

    def _save(self, path: str, payload: dict):
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def _load(self, path: str) -> Optional[dict]:
        if not os.path.isfile(path):
            return None
        return torch.load(path, map_location="cpu")

    def get_text_embeddings(
        self,
        *,
        model_name: str,
        texts: List[str],
        encoder,  # SentenceTransformer
        batch_size: int,
        device: str,
        normalize: bool = True,
        tag: str = "triplet_text",
    ) -> torch.Tensor:
        """
        Returns (N, D) tensor on `device`.
    
        Notes:
        - For CLIP-based SentenceTransformers (e.g., "clip-ViT-B-32"), the text encoder
          hard-limits inputs to 77 tokens. We therefore truncate using the encoder's
          tokenizer *before* encoding.
        - Cache key is computed from the *actual encoded texts* (post-truncation),
          so cache files stay consistent and avoid mismatches.
        """
        def _is_clip_model() -> bool:
            mn = (model_name or "").lower()
            if "clip" in mn:
                return True
            # heuristic fallback
            cname = encoder.__class__.__name__.lower()
            return "clip" in cname
    

    
        # 1) Preprocess texts for CLIP (truncate to 77 tokens)
        texts_to_encode = texts
        if "clip" in (model_name or "").lower():
            texts_to_encode = _truncate_texts_for_clip(encoder, texts, max_len=77)
            
        # 2) Cache key based on *post-truncation* texts (order-sensitive)
        key = _sha1_json({"model": model_name, "tag": tag, "texts": texts_to_encode})
        fn = f"text__{_sanitize(model_name)}__{tag}__{key}.pt"
        path = os.path.join(self.cfg.cache_dir, fn)
    
        cached = self._load(path)
        if cached is not None:
            emb = cached["emb"].to(device)
            return l2norm(emb) if normalize else emb
    
        # 3) Encode
        emb = encoder.encode(
            texts_to_encode,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        ).to(device)
    
        if normalize:
            emb = l2norm(emb)
    
        # 4) Save
        save_emb = emb.detach().cpu()
        if self.cfg.save_dtype is not None:
            save_emb = save_emb.to(self.cfg.save_dtype)
    
        self._save(path, {"emb": save_emb})
        return emb


    def get_image_embeddings_from_paths(
        self,
        *,
        model_name: str,
        image_paths: List[str],
        encoder,  # SentenceTransformer
        batch_size: int,
        device: str,
        normalize: bool = True,
        tag: str = "image_paths",
    ) -> torch.Tensor:
        """
        Returns (M, D) tensor for the ordered image_paths list.
        Cache key depends on:
          - model_name
          - image_paths list
          - each file's (basename, mtime, size) to invalidate on changes
        """
        sig = [_file_sig(p) for p in image_paths]
        key = _sha1_json({"model": model_name, "tag": tag, "paths": image_paths, "sig": sig})
        fn = f"img__{_sanitize(model_name)}__{tag}__{key}.pt"
        path = os.path.join(self.cfg.cache_dir, fn)

        cached = self._load(path)
        if cached is not None:
            emb = cached["emb"].to(device)
            return l2norm(emb) if normalize else emb

        imgs: List[Image.Image] = []
        for p in image_paths:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))

        emb = encoder.encode(
            imgs,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        ).to(device)

        if normalize:
            emb = l2norm(emb)

        save_emb = emb.detach().cpu()
        if self.cfg.save_dtype is not None:
            save_emb = save_emb.to(self.cfg.save_dtype)

        self._save(path, {"emb": save_emb, "paths": image_paths})
        return emb

    def get_entity_image_embeddings(
        self,
        *,
        model_name: str,
        entity_ids: List[str],
        resolve_path_fn,  # (entity_id) -> Optional[path]
        encoder,  # SentenceTransformer
        batch_size: int,
        device: str,
        normalize: bool = True,
        tag: str,
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Encodes entities that have an image (resolved by resolve_path_fn).
        Returns:
          - emb: (M, D) tensor (only entities with images)
          - ent2row: mapping entity_id -> row in emb
        Cache key depends on:
          - model_name
          - list of (entity_id, resolved_path, file_sig)
        """
        items = []
        for eid in entity_ids:
            p = resolve_path_fn(eid)
            if p and os.path.isfile(p):
                items.append((eid, p, _file_sig(p)))

        # stable ordering for key + output
        items.sort(key=lambda x: x[0])
        key = _sha1_json({"model": model_name, "tag": tag, "items": items})

        fn = f"entimg__{_sanitize(model_name)}__{tag}__{key}.pt"
        path = os.path.join(self.cfg.cache_dir, fn)

        cached = self._load(path)
        if cached is not None:
            emb = cached["emb"].to(device)
            ent2row = cached["ent2row"]
            if normalize:
                emb = l2norm(emb)
            return emb, ent2row

        if len(items) == 0:
            # no images
            emb = torch.empty((0, 1), device=device)
            return emb, {}

        img_paths = [p for (_, p, _) in items]
        embs = self.get_image_embeddings_from_paths(
            model_name=model_name,
            image_paths=img_paths,
            encoder=encoder,
            batch_size=batch_size,
            device=device,
            normalize=normalize,
            tag=tag + "__paths",
        )

        ent2row = {eid: i for i, (eid, _, _) in enumerate(items)}

        save_emb = embs.detach().cpu()
        if self.cfg.save_dtype is not None:
            save_emb = save_emb.to(self.cfg.save_dtype)

        self._save(path, {"emb": save_emb, "ent2row": ent2row, "items": items})
        return embs, ent2row
