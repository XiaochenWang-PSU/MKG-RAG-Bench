import os
from typing import List, Dict, Any, Tuple
from PIL import Image
from utils import *
import torch
import numpy as np
import numpy
import torch
import random
from sentence_transformers import SentenceTransformer
from typing import *
from transformers import BlipProcessor, BlipForConditionalGeneration

def l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)



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
        for fn in os.listdir(d):
            if os.path.splitext(fn.lower())[1] in exts:
                return os.path.join(d, fn)

    return None

class MMAnchorRetriever:
    """
    Late-fusion (v2: image -> text)
      - Triplet text -> text embedding
      - Triplet images (head/tail) -> image embedding (avg if both exist)
      - Search: top n images first, then top n texts per selected image
      - Output: ONLY text strings
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
    ):
        self.retrieval_dataset = list(retrieval_dataset)
        self.entity2text = entity2text
        self.relation2text = relation2text
        self.image_root = image_root
        self.inference_image_root = inference_image_root
        self.batch_size = batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)

        # --------
        # Build per-triplet text strings
        # --------
        self.triplet_text: List[str] = []
        for (h, r, t) in self.retrieval_dataset:
            parts = []
            if h in entity2text:
                parts.append(entity2text[h])
            if r in relation2text:
                parts.append(relation2text[r])
            if t in entity2text:
                parts.append(entity2text[t])
            self.triplet_text.append(" ".join(parts).strip())

        # Encode text embeddings (N, D)
        text_emb = self.model.encode(
            self.triplet_text,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        self.text_emb = l2norm(text_emb.to(self.device))

        # --------
        # Build per-triplet image embedding (avg of head/tail if exist)
        # --------
        triplet_img_emb: List[Optional[torch.Tensor]] = []
        has_image: List[bool] = []

        it = self.retrieval_dataset
        if show_progress:
            it = tqdm(it, desc="Indexing triplet images")

        for (h, _, t) in it:
            imgs = []
            hp = first_jpg_path(h, self.image_root)
            tp = first_jpg_path(t, self.image_root)

            if hp:
                with Image.open(hp) as im:
                    imgs.append(im.convert("RGB"))
            if tp:
                with Image.open(tp) as im:
                    imgs.append(im.convert("RGB"))

            if len(imgs) == 0:
                triplet_img_emb.append(None)
                has_image.append(False)
            else:
                emb = self.model.encode(
                    imgs,
                    batch_size=batch_size,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                ).to(self.device)
                emb = l2norm(emb)
                triplet_img_emb.append(emb.mean(dim=0, keepdim=True))  # (1, D)
                has_image.append(True)

        # Pack image embeddings into dense matrix for fast matmul
        self.has_image = torch.tensor(has_image, device=self.device, dtype=torch.bool)
        if self.has_image.any():
            self.img_indices = torch.nonzero(self.has_image).flatten()
            self.img_emb = torch.cat([triplet_img_emb[i] for i in self.img_indices.tolist()], dim=0)  # (M, D)
        else:
            self.img_indices = None
            self.img_emb = None
    
    @torch.no_grad()
    def search(
        self,
        query: List[str],
        k: int = 10,  # kept for compatibility; not used in v2 strategy except as an upper bound
        mode: int = 0,
        *,
        n_img: int = 10,
        n_text: int = 5,
        return_unique: bool = True,
    ) -> List[dict]:
        """
        query: [head, tail, question]
        mode:
          0: (T1, T2) -> (I1, ?)   => text from head/tail/question-text, image from question-image
          1: (I1, I2) -> (T1, ?)   => text from question-text, image from head/tail images (avg)
          2: (I1, T1) -> (I2, ?)   => text from tail-text, image from head/question images (avg)
    
        Strategy (unchanged from your v2):
          - Build q_text -> rank ALL triplets by text similarity (global_text_order)
          - Build q_img  -> pick top-n_img triplets-by-image (if possible)
          - For each selected image triplet, emit the next top-n_text items from global_text_order
          - Optionally dedup by text string across the whole output
    
        Returns:
          list[dict] with {"rank","index","item","score"} (same schema as your text-only retriever),
          where:
            - index: triplet index in retrieval_dataset
            - item : retrieval_dataset[index]
            - score: text similarity score (since stage-2 ranking is purely text-based in v2)
        """
        head, tail, question = query
        device = self.text_emb.device
        N = len(self.retrieval_dataset)
    
        # -----------------------
        # Build query text (same as your original organization)
        # -----------------------
        if mode == 0:
            q_text = (
                f"{self.entity2text.get(head, head)} "
                f"{self.entity2text.get(tail, tail)} "
                f"{self.entity2text.get(question, question)}"
            )
        elif mode == 1:
            q_text = f"{self.entity2text.get(question, question)}"
        else:  # mode == 2
            q_text = f"{self.entity2text.get(tail, tail)}"
    
        q_text_emb = self.model.encode([q_text], convert_to_tensor=True, show_progress_bar=False).to(device)
        q_text_emb = F.normalize(q_text_emb, p=2, dim=1)  # (1, D)
    
        # Text scores over ALL triplets (N,)
        s_text_all = (self.text_emb @ q_text_emb.T).flatten()  # (N,)
        text_sorted_idxs = torch.argsort(s_text_all, descending=True)  # (N,)
        global_text_order = text_sorted_idxs.tolist()
    
        # -----------------------
        # Build query image embedding
        # -----------------------
        def _encode_image(path: str) -> torch.Tensor:
            with Image.open(path) as im:
                emb = self.model.encode([im.convert("RGB")], convert_to_tensor=True, show_progress_bar=False).to(device)
            return F.normalize(emb, p=2, dim=1)  # (1, D)
    
        def _avg_img_emb(paths: List[Optional[str]]) -> Optional[torch.Tensor]:
            embs = []
            for p in paths:
                if p and os.path.isfile(p):
                    embs.append(_encode_image(p))
            if not embs:
                return None
            e = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)
            return F.normalize(e, p=2, dim=1)
    
        q_img_emb = None
        if mode == 0:
            qp = first_jpg_path(question, self.inference_image_root)
            if qp and os.path.isfile(qp):
                q_img_emb = _encode_image(qp)
        elif mode == 1:
            q_img_emb = _avg_img_emb([
                first_jpg_path(head, self.inference_image_root),
                first_jpg_path(tail, self.inference_image_root),
            ])
        else:  # mode == 2
            q_img_emb = _avg_img_emb([
                first_jpg_path(head, self.inference_image_root),
                first_jpg_path(question, self.inference_image_root),
            ])
    
        # -----------------------
        # If no query image (or no indexed images), fallback to text-only formatted output
        # -----------------------
        # We keep v2 behavior (return the top n_text by text) but format as list[dict].
        if q_img_emb is None or self.img_emb is None or not getattr(self, "has_image", torch.tensor([])).any():
            keep = min(n_text, N)  # v2 fallback behavior
            idxs = text_sorted_idxs[:keep].tolist()
            out: List[dict] = []
            for rank, i in enumerate(idxs, start=1):
                out.append({
                    "rank": rank,
                    "index": i,
                    "item": self.retrieval_dataset[i],
                    "score": float(s_text_all[i].item()),
                })
            return out
    
        # -----------------------
        # Stage 1: top-n by image similarity (same as v2)
        # -----------------------
        s_img = (self.img_emb @ q_img_emb.T).flatten()  # (M,)
        keep_m = min(n_img, int(s_img.shape[0]))
        top_img_local = torch.topk(s_img, k=keep_m, largest=True).indices  # local indices into img_emb
        top_img_global = self.img_indices[top_img_local].tolist()          # triplet indices in [0..N)
    
        # -----------------------
        # Stage 2: for each selected image, emit top-n_text from global text order (same as v2)
        # but return list[dict] with rank/index/item/score
        # -----------------------
        results: List[dict] = []
        seen_text: set = set()      # for return_unique=True (by text string, matching your v2 semantics)
        seen_index: set = set()     # avoid emitting identical (index,item) twice if return_unique=False still
                                    # (harmless, but keeps ranks stable)
    
        max_total = n_img * n_text
        if isinstance(k, int) and k > 0:
            # keep k as an optional ceiling for compatibility
            max_total = min(max_total, k)
    
        rank = 1
        for _img_triplet_idx in top_img_global:
            added = 0
            for ti in global_text_order:
                if len(results) >= max_total:
                    return results
    
                # v2 dedup key was the *text string* (triplet_text[ti])
                txt = self.triplet_text[ti]
    
                if return_unique:
                    if txt in seen_text:
                        continue
                    seen_text.add(txt)
                else:
                    # even if not deduping by text, don't output the exact same index repeatedly
                    if ti in seen_index:
                        continue
                    seen_index.add(ti)
    
                results.append({
                    "rank": rank,
                    "index": ti,
                    "item": self.retrieval_dataset[ti],
                    "score": float(s_text_all[ti].item()),  # text-driven ordering in stage-2
                })
                rank += 1
                added += 1
    
                if added >= n_text:
                    break
    
        return results

class MMWeightedRetriver:
    """
    Late-fusion retriever:
      - Triplet text -> text embedding
      - Triplet images (head/tail) -> image embedding (avg if both exist)
      - Query may have text and/or image; scores fused with weights
    """

    def __init__(
        self,
        retrieval_dataset,
        entity2text,
        relation2text,
        model_name="clip-ViT-B-32",
        batch_size=32,
        image_root="images_subset_kg",
    ):
        self.retrieval_dataset = retrieval_dataset
        self.entity2text = entity2text
        self.relation2text = relation2text
        self.image_root = image_root

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

        # --------
        # Build per-triplet text strings
        # --------
        self.triplet_text = []
        for (h, r, t) in self.retrieval_dataset:
            parts = []
            if h in entity2text: parts.append(entity2text[h])
            if r in relation2text: parts.append(relation2text[r])
            if t in entity2text: parts.append(entity2text[t])
            self.triplet_text.append(" ".join(parts))

        # Encode text embeddings (N, D)
        text_emb = self.model.encode(
            self.triplet_text,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        self.text_emb = l2norm(text_emb)

        # --------
        # Build per-triplet image embedding (avg of head/tail if exist)
        # --------
        triplet_img_emb = []
        has_image = []

        for (h, _, t) in tqdm(self.retrieval_dataset):
            imgs = []
            hp = first_jpg_path(h, self.image_root)
            tp = first_jpg_path(t, self.image_root)

            if hp:
                with Image.open(hp) as im:
                    imgs.append(im.convert("RGB"))
            if tp:
                with Image.open(tp) as im:
                    imgs.append(im.convert("RGB"))

            if len(imgs) == 0:
                triplet_img_emb.append(None)
                has_image.append(False)
            else:
                emb = self.model.encode(imgs, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
                emb = l2norm(emb)
                triplet_img_emb.append(emb.mean(dim=0, keepdim=True))  # (1, D)
                has_image.append(True)

        # pack image embeddings into a dense matrix for fast matmul
        self.has_image = torch.tensor(has_image, device=self.device)
        if self.has_image.any():
            self.img_indices = torch.nonzero(self.has_image).flatten()
            self.img_emb = torch.cat([triplet_img_emb[i] for i in self.img_indices.tolist()], dim=0)  # (M, D)
        else:
            self.img_indices = None
            self.img_emb = None

    @torch.no_grad()
    def search(self, query, k, mode, *, w_img=0.75, w_text=0.25, agreement=0.0):
        """
        query: [head, tail, question]
        mode:
          0: (T1, T2) -> (I1, ?)   => text from head/tail/question-text, image from question-image
          1: (I1, I2) -> (T1, ?)   => text from question-text, image from head/tail images (avg)
          2: (I1, T1) -> (I2, ?)   => text from tail-text, image from head/question images (avg)

        Returns: list[dict] with rank/index/item/score (same format as your text-only retriever)
        """

        head, tail, question = query
        device = self.text_emb.device
        N = len(self.retrieval_dataset)

        # -----------------------
        # Build query text
        # -----------------------
        # (keep it close to your text-only retriever: concatenate whatever text fields are available)
        if mode == 0:
            q_text = f"{self.entity2text[head]} {self.entity2text[tail]} {self.entity2text[question]}"
        elif mode == 1:
            q_text = f"{self.entity2text[question]}"
        else:  # mode == 2
            q_text = f"{self.entity2text[tail]}"

        q_text_emb = self.model.encode([q_text], convert_to_tensor=True, show_progress_bar=False).to(device)
        q_text_emb = F.normalize(q_text_emb, p=2, dim=1)  # (1, D)

        s_text = (self.text_emb @ q_text_emb.T).flatten()  # (N,)

        # -----------------------
        # Build query image embedding (may be None)
        # -----------------------
        q_img_emb = None

        def _encode_image(path):
            with Image.open(path) as im:
                emb = self.model.encode([im.convert("RGB")], convert_to_tensor=True, show_progress_bar=False).to(device)
            return F.normalize(emb, p=2, dim=1)  # (1, D)

        def _avg_img_emb(paths):
            embs = []
            for p in paths:
                if p:
                    embs.append(_encode_image(p))
            if not embs:
                return None
            e = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)
            return F.normalize(e, p=2, dim=1)

        if mode == 0:
            # image comes from the "question" (as your original code did)
            q_img_emb = _encode_image(first_jpg_path(question, "images_subset_inference"))
        elif mode == 1:
            # images from head & tail (avg)
            q_img_emb = _avg_img_emb([
                first_jpg_path(head, "images_subset_inference"),
                first_jpg_path(tail, "images_subset_inference"),
            ])
        else:  # mode == 2
            # images from head & question (avg)
            q_img_emb = _avg_img_emb([
                first_jpg_path(head, "images_subset_inference"),
                first_jpg_path(question, "images_subset_inference"),
            ])

        # -----------------------
        # Score fusion (image-heavy)
        # -----------------------
        scores = w_text * s_text

        if q_img_emb is not None and self.img_emb is not None:
            # compute image similarity only for triplets that have images (M,)
            s_img = (self.img_emb @ q_img_emb.T).flatten()  # (M,)

            # scatter into full (N,) vector
            s_img_full = torch.zeros(N, device=device)
            s_img_full[self.img_indices] = s_img

            scores = scores + (w_img * s_img_full)

            # optional agreement bonus: reward triplets that match both modalities
            if agreement != 0.0:
                scores = scores + agreement * (s_text * s_img_full)

        # -----------------------
        # Top-k and output (same schema as your text-only retriever)
        # -----------------------
        vals, idxs = torch.topk(scores, k=min(k, N), largest=True, sorted=True)

        out = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            out.append({
                "rank": rank,
                "index": i,
                "item": self.retrieval_dataset[i],
                "score": float(s),
            })
        return out
        
class SimpleMultimodalRetriever:
    # In this retrieval, for every triplet (head, relation, tail), we will take average of embedding
    def __init__(self, retrieval_dataset, entity2text, relation2text, model_name="clip-ViT-B-32", batch_size=32):
        # retrieval_dataset = [[h1, r1, t1], [h2, r2, t2], ...]

        # Initialization
        self.retrieval_dataset = retrieval_dataset
        
        self.text = []
        self.image = []
        
        self.text2retrieval = {}
        self.image2retrieval = {}
        
        self.entity2text = entity2text

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

        for idx, triplet in enumerate(self.retrieval_dataset):
            head, relation, tail = triplet
            t = ""

            if head in entity2text:
                t += entity2text[head]
                t += " "

            if relation in relation2text:
                t += relation2text[relation]
                t += " "

            if tail in entity2text:
                t += entity2text[tail]

            self.text2retrieval[len(self.text)]= idx
            self.text.append(t)

            if first_jpg_path(head, "images_subset_kg"):
                self.image2retrieval[len(self.image)] = idx 
                with Image.open(first_jpg_path(head, "images_subset_kg")) as im:    
                    self.image.append(im.convert("RGB"))

            if first_jpg_path(tail, "images_subset_kg"):
                self.image2retrieval[len(self.image)] = idx
                with Image.open(first_jpg_path(tail, "images_subset_kg")) as im:
                    self.image.append(im.convert("RGB"))
        
        # Encode text and image to embedding
        self.image_emb = self.model.encode(self.image, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        self.text_emb = self.model.encode(self.text, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)

        # Take Average
        self.retrieval_embeddings = [[] for i in range(len(retrieval_dataset))]
        
        for i in range(len(self.image_emb)):
            self.retrieval_embeddings[self.image2retrieval[i]].append(self.image_emb[i])
        
        for i in range(len(self.text_emb)):
            self.retrieval_embeddings[self.text2retrieval[i]].append(self.text_emb[i])

        for i in range(len(retrieval_dataset)):
            self.retrieval_embeddings[i] = torch.stack(self.retrieval_embeddings[i], dim = 0).mean(dim = 0)

        self.retrieval_embeddings = torch.stack(self.retrieval_embeddings, dim = 0)

        # Normalize for cosine similarity via dot product
        self.retrieval_embeddings = torch.nn.functional.normalize(self.retrieval_embeddings, p=2, dim=1)

    def search(self, query, k, mode):
        # query: [head, tail, question]. Do not have relation since MARS do not allow to provide relation to model.
        # mode is defined as follow
        #   mode 0: (T1, T2) -> (I1, ?)
        #   mode 1: (I1, I2) -> (T1, ?)
        #   mode 2: (I1, T1) -> (I2, ?)

        query_embeddings = []

        head, tail, question = query

        if mode == 0:
            query_embeddings.append(self.model.encode([self.entity2text[head]], convert_to_tensor=True, show_progress_bar=False))
            query_embeddings.append(self.model.encode([self.entity2text[tail]], convert_to_tensor=True, show_progress_bar=False))
            with Image.open(first_jpg_path(question, "images_subset_inference")) as im:
                query_embeddings.append(self.model.encode([im.convert("RGB")], convert_to_tensor=True, show_progress_bar=False))

        if mode == 1:
            with Image.open(first_jpg_path(head, "images_subset_inference")) as im:
                query_embeddings.append(self.model.encode([im.convert("RGB")], convert_to_tensor=True, show_progress_bar=False))
            with Image.open(first_jpg_path(tail, "images_subset_inference")) as im:
                query_embeddings.append(self.model.encode([im.convert("RGB")], convert_to_tensor=True, show_progress_bar=False))
            query_embeddings.append(self.model.encode([self.entity2text[question]], convert_to_tensor=True, show_progress_bar=False))

        if mode == 2:
            with Image.open(first_jpg_path(head, "images_subset_inference")) as im:
                query_embeddings.append(self.model.encode([im.convert("RGB")], convert_to_tensor=True, show_progress_bar=False))
            query_embeddings.append(self.model.encode([self.entity2text[tail]], convert_to_tensor=True, show_progress_bar=False))
            with Image.open(first_jpg_path(question, "images_subset_inference")) as im:
                query_embeddings.append(self.model.encode([im.convert("RGB")], convert_to_tensor=True, show_progress_bar=False))

        query_embedding = torch.stack(query_embeddings, dim = 0).mean(dim = 0)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)  
        scores = (self.retrieval_embeddings @ query_embedding.T).flatten()
        vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)

        out = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            out.append({
                "rank": rank,
                "index": i,
                "item": self.retrieval_dataset[i],
                "score": float(s),
            })
        
        return out

class SimpleTextRetriever:
    # In this retrieval, for every triplet (head, relation, tail), we will text embedding of string head + relation + tail
    def __init__(self, retrieval_dataset, entity2text, relation2text, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
        # retrieval_dataset = [[h1, r1, t1], [h2, r2, t2], ...]

        # Initialization
        self.retrieval_dataset = retrieval_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.entity2text = entity2text

        # Get texts from triplet
        self.texts = []
        for idx, triplet in enumerate(self.retrieval_dataset):
            head, relation, tail = triplet

            text = ""

            if head in entity2text:
                text += entity2text[head]
                text += " "
            
            if relation in relation2text:
                text += relation2text[relation]
                text += " "
            
            if tail in entity2text:
                text += entity2text[tail]
            
            self.texts.append(text)

        # Get text embedding
        self.retrieval_embeddings = self.model.encode(self.texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)

        # Normalize for cosine similarity via dot product
        self.retrieval_embeddings = torch.nn.functional.normalize(self.retrieval_embeddings, p=2, dim=1)

    def search(self, query, k, mode):
        # query: [head, tail, question]. Do not have relation since MARS do not allow to provide relation to model.
        # mode is defined as follow
        #   mode 0: (T1, T2) -> (I1, ?)
        #   mode 1: (I1, I2) -> (T1, ?)
        #   mode 2: (I1, T1) -> (I2, ?)

        query_embeddings = []
        head, tail, question = query
        text = self.entity2text[head] + " " + self.entity2text[tail] + " " + self.entity2text[question]
        
        query_embedding = self.model.encode([text], convert_to_tensor=True, show_progress_bar=False)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)  
        scores = (self.retrieval_embeddings @ query_embedding.T).flatten()
        vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)

        out = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            out.append({
                "rank": rank,
                "index": i,
                "item": self.retrieval_dataset[i],
                "score": float(s),
            })
        
        return out

class RandomRetriever:
    # In this retrieval, for every triplet (head, relation, tail), we will take random
    def __init__(self, retrieval_dataset):
        # retrieval_dataset = [[h1, r1, t1], [h2, r2, t2], ...]

        # Initialization
        self.retrieval_dataset = retrieval_dataset

    def search(self, query, k, mode):
        # query: [head, tail, question]. Do not have relation since MARS do not allow to provide relation to model.
        # mode is defined as follow
        #   mode 0: (T1, T2) -> (I1, ?)
        #   mode 1: (I1, I2) -> (T1, ?)
        #   mode 2: (I1, T1) -> (I2, ?)
        random_idx = random.sample([i for i in range(len(self.retrieval_dataset))], k)
        
        out = []
        for rank, i in enumerate(random_idx):
            out.append({
                "rank": rank + 1,
                "index": i,
                "item": self.retrieval_dataset[i],
                "score": rank + 1,
            })
        
        return out

class CaptionRetriever:
    # In this retrieval, for every triplet (head, relation, tail), we will text embedding of string head + relation + tail
    def __init__(self, retrieval_dataset, entity2text, relation2text, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
        # Initialization
        self.retrieval_dataset = retrieval_dataset
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.entity2text = entity2text

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # Get texts from triplet
        self.texts = []
        for idx, triplet in enumerate(self.retrieval_dataset):
            head, relation, tail = triplet

            text = ""
            if first_jpg_path(head, "images_subset_kg"):
                inputs = processor(Image.open(first_jpg_path(head, "images_subset_kg")).convert("RGB"), return_tensors="pt")
                out = model.generate(**inputs)
                text += str(processor.decode(out[0], skip_special_tokens=True))
                text += " "
            else:
                text += entity2text[head]
                text += " "
                
            if relation in relation2text:
                text += relation2text[relation]
                text += " "
            
            if first_jpg_path(tail, "images_subset_kg"):
                inputs = processor(Image.open(first_jpg_path(tail, "images_subset_kg")).convert("RGB"), return_tensors="pt")
                out = model.generate(**inputs)
                text += str(processor.decode(out[0], skip_special_tokens=True))
            else:
                text += entity2text[tail]
            
            self.texts.append(text)

        # Get text embedding
        self.retrieval_embeddings = self.model.encode(self.texts, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)

        # Normalize for cosine similarity via dot product
        self.retrieval_embeddings = torch.nn.functional.normalize(self.retrieval_embeddings, p=2, dim=1)

    def search(self, query, k, mode):
        # query: [head, tail, question]. Do not have relation since MARS do not allow to provide relation to model.
        # mode is defined as follow
        #   mode 0: (T1, T2) -> (I1, ?)
        #   mode 1: (I1, I2) -> (T1, ?)
        #   mode 2: (I1, T1) -> (I2, ?)

        query_embeddings = []
        head, tail, question = query
        text = self.entity2text[head] + " " + self.entity2text[tail] + " " + self.entity2text[question]
        
        query_embedding = self.model.encode([text], convert_to_tensor=True, show_progress_bar=False)
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)  
        scores = (self.retrieval_embeddings @ query_embedding.T).flatten()
        vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)

        out = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            out.append({
                "rank": rank,
                "index": i,
                "item": self.retrieval_dataset[i],
                "score": float(s),
            })
        
        return out