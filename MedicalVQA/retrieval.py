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
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer
import open_clip
from tqdm import tqdm

@dataclass
class KGTriplet:
    head: str  # Head ID
    head_name: str  # Head name/description
    relation: str  # type of relation
    tail: str  # Tail ID
    tail_name: str  # Tail name/description




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

class BaseRetriever:
    def __init__(self, kg_path: str, image_map_path: str = "image_mapping.csv"):    
        # Load image mapping
        df_image_map = pd.read_csv(image_map_path)
        self.image_id_to_path = dict(zip(df_image_map['IID'], 
                                df_image_map['Image_Path'].apply(
                                    lambda path: "/data/xiaochen/"+path
                                )))
        
        # Load KG and create mappings
        self.triplets = self._load_kg(kg_path)#[:100]
        self._create_mappings()
       
    def _load_kg(self, kg_path: str) -> List[KGTriplet]:
        if not Path(kg_path).exists():
            raise FileNotFoundError(f"Knowledge graph file not found: {kg_path}")
            
        df = pd.read_csv(kg_path)
        return [
            KGTriplet(
                head=str(row['Head']),
                head_name=str(row['Head_Name']),
                relation=str(row['Relation']),
                tail=str(row['Tail']),
                tail_name=str(row['Tail_Name'])
            )
            for _, row in df.iterrows()
            # if row['Head'].startswith('I')
        ]

    def _create_mappings(self):
        self.id_to_name = {}
        self.name_to_ids = {}
        self.tail_to_heads = {}
        self.head_to_tails = {}
        
        for triplet in self.triplets:
            # ID-name mappings
            self.id_to_name[triplet.head] = triplet.head_name
            self.id_to_name[triplet.tail] = triplet.tail_name
            
            for name, id_ in [(triplet.head_name, triplet.head), 
                            (triplet.tail_name, triplet.tail)]:
                if name not in self.name_to_ids:
                    self.name_to_ids[name] = set()
                self.name_to_ids[name].add(id_)
            
            # Relation mappings
            for source, target, mapping in [
                (triplet.tail, triplet.head, self.tail_to_heads),
                (triplet.head, triplet.tail, self.head_to_tails)
            ]:
                if source not in mapping:
                    mapping[source] = {}
                if triplet.relation not in mapping[source]:
                    mapping[source][triplet.relation] = set()
                mapping[source][triplet.relation].add(target)


class SimpleMultimodalRetriever(BaseRetriever):
    def __init__(
        self,
        kg_path: str,
        image_map_path: str = "image_mapping.csv",
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",  # or "laion2b_s34b_b79k"
        batch_size: int = 256,  # <-- smaller default to be safer on GPU
    ):
        super().__init__(kg_path, image_map_path)

        # ---- device ----
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.device = "cuda:1"
            else:
                self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.batch_size = batch_size

        # ---- open_clip model + preprocess + tokenizer ----
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model,
            pretrained=clip_pretrained,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(clip_model)

        # Store texts and *paths* instead of PIL images
        self.texts: List[str] = []
        self.image_paths: List[str] = []          # <--- changed
        self.image2idx: Dict[int, int] = {}

        # -------- build texts & image paths from triplets --------
        for idx, triplet in enumerate(tqdm(self.triplets, desc="Building texts & image paths")):
            # build text string
            text = ""
            if "image_" not in triplet.head_name.lower():
                text += triplet.head_name.lower() + " "
            text += triplet.relation.lower() + " " + triplet.tail_name.lower()
            self.texts.append(text)

            # attach image path if exists
            if (
                triplet.head in self.image_id_to_path
                and os.path.exists(self.image_id_to_path[triplet.head])
            ):
                self.image2idx[len(self.image_paths)] = idx
                self.image_paths.append(self.image_id_to_path[triplet.head])

        # -------- encode all texts & images with open_clip --------
        self.text_emb = self._encode_texts(self.texts)  # (N_nodes, D)

        if len(self.image_paths) > 0:
            self.image_emb = self._encode_images(self.image_paths)  # (N_images, D)
        else:
            # empty tensor with correct dim
            self.image_emb = torch.empty(
                0, self.text_emb.size(1), dtype=self.text_emb.dtype
            )

        # -------- build multimodal retrieval embeddings (avg text+image) --------
        self.retrieval_embeddings: List[torch.Tensor] = [[] for _ in range(len(self.texts))]

        # add image embeddings per triplet (if any)
        for i in range(len(self.image_emb)):
            node_idx = self.image2idx[i]
            self.retrieval_embeddings[node_idx].append(self.image_emb[i])

        # add text embeddings (always one per node)
        for i in range(len(self.text_emb)):
            self.retrieval_embeddings[i].append(self.text_emb[i])

        # average across modalities and normalize
        for i in range(len(self.retrieval_embeddings)):
            # each entry is a list of 1 or 2 vectors (text, [image])
            stacked = torch.stack(self.retrieval_embeddings[i], dim=0).mean(dim=0)
            self.retrieval_embeddings[i] = stacked

        self.retrieval_embeddings = torch.stack(self.retrieval_embeddings, dim=0)  # (N_nodes, D)
        self.retrieval_embeddings = torch.nn.functional.normalize(
            self.retrieval_embeddings, p=2, dim=1
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of texts with open_clip in batches.
        Returns embeddings on CPU.
        """
        all_embs = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]
            tokens = self.tokenizer(batch_texts).to(self.device)
            text_features = self.model.encode_text(tokens)
            all_embs.append(text_features.cpu())
        return torch.cat(all_embs, dim=0)

    @torch.no_grad()
    def _encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        Encode a list of image paths with open_clip in batches.
        Images are loaded on the fly to avoid holding them all in RAM.
        Returns embeddings on CPU.
        """
        all_embs = []
        for start in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[start:start + self.batch_size]

            pil_images = []
            for p in batch_paths:
                with Image.open(p) as im:
                    pil_images.append(im.convert("RGB"))

            # preprocess to tensors and stack
            img_tensors = [self.preprocess(im) for im in pil_images]
            img_batch = torch.stack(img_tensors, dim=0).to(self.device)

            img_features = self.model.encode_image(img_batch)
            all_embs.append(img_features.cpu())

            # free GPU memory between big batches if needed
            del img_batch, img_features
            torch.cuda.empty_cache()

        return torch.cat(all_embs, dim=0)

    # ----------------- public search API -----------------

    def search(self, sample: Dict[str, Any], k: int):
        # image
        if "image_path" in sample:
            with Image.open(sample["image_path"]) as im:
                query_image = im.convert("RGB")
        else:
            query_image = sample["image"]

        # text
        query_text = sample["question"].lower()

        with torch.no_grad():
            # encode query text
            tok = self.tokenizer([query_text]).to(self.device)        # (1,77)
            q_text_feat = self.model.encode_text(tok)                 # (1,D)
            q_text_feat = q_text_feat / q_text_feat.norm(dim=-1, keepdim=True)

            # encode query image
            q_img_tensor = self.preprocess(query_image).unsqueeze(0).to(self.device)
            q_img_feat = self.model.encode_image(q_img_tensor)        # (1,D)
            q_img_feat = q_img_feat / q_img_feat.norm(dim=-1, keepdim=True)

            # average text + image & normalize
            q_feat = (q_text_feat + q_img_feat) / 2.0                 # (1,D)
            q_feat = q_feat / q_feat.norm(dim=-1, keepdim=True)       # (1,D)

        # compute cosine similarity via dot product (already L2-normalized)
        # retrieval_embeddings is on CPU; move query to same device
        q_feat_cpu = q_feat.cpu()
        scores = (self.retrieval_embeddings @ q_feat_cpu.T).flatten()  # (N_nodes,)

        vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)

        out = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            out.append(
                {
                    "rank": rank,
                    "index": i,
                    "item": self.triplets[i],
                    "score": float(s),
                }
            )
        return out
class SimpleTextRetriever(BaseRetriever):
    def __init__(self, kg_path: str, image_map_path: str = "image_mapping.csv",
                 model_name="clip-ViT-B-32", batch_size=32):
        super().__init__(kg_path, image_map_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

        self.texts = []
        # mapping: index in self.texts / self.retrieval_embeddings -> index in self.triplets
        self.text_idx2triplet_idx = []

        for idx, triplet in enumerate(self.triplets):
            # deliberately exclude "image_" heads from ranking
            if "image_" in triplet.head_name.lower():
                continue

            text = (
                triplet.head_name.lower()
                + " "
                + triplet.relation.lower()
                + " "
                + triplet.tail_name.lower()
            )
            self.texts.append(text)
            self.text_idx2triplet_idx.append(idx)

        self.retrieval_embeddings = self.model.encode(
            self.texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        print(self.retrieval_embeddings.shape)

        # Normalize for cosine similarity via dot product
        self.retrieval_embeddings = torch.nn.functional.normalize(
            self.retrieval_embeddings, p=2, dim=1
        )

    def search(self, sample, k):
        query_text = sample["question"].lower()

        query_embedding = self.model.encode(
            [query_text],
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        query_embedding = torch.nn.functional.normalize(
            query_embedding, p=2, dim=1
        )

        scores = (self.retrieval_embeddings @ query_embedding.T).flatten()

        # guard in case k > #candidates
        k = min(k, scores.shape[0])

        vals, idxs = torch.topk(scores, k=k, largest=True, sorted=True)

        out = []
        for rank, (i, s) in enumerate(zip(idxs.tolist(), vals.tolist()), start=1):
            # i indexes into self.texts / self.retrieval_embeddings
            triplet_idx = self.text_idx2triplet_idx[i]  # map back to original triplet index
            triplet = self.triplets[triplet_idx]

            out.append(
                {
                    "rank": rank,
                    "index": triplet_idx,  # index in the original triplet list
                    "item": triplet,
                    "score": float(s),
                }
            )

        return out



class RandomRetriever(BaseRetriever):
    def search(self, sample, k):
        random_idx = random.sample([i for i in range(len(self.triplets))], k)

        out = []
        for rank, i in enumerate(random_idx):
            out.append({
                "rank": rank + 1,
                "index": i,
                "item": self.triplets[i],
                "score": rank + 1,
            })
        
        return out
