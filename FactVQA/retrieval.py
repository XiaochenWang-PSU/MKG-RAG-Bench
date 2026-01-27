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

    def search(self, question, k):
        query_embeddings = []
        with Image.open("new_dataset_release/new_dataset_release/images/"+question['img_file']) as im:
            query_embeddings.append(self.model.encode([im.convert("RGB")], convert_to_tensor=True, show_progress_bar=False))
        query_embeddings.append(self.model.encode([question['question']], convert_to_tensor=True, show_progress_bar=False))
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

    def search(self, question, k):
        query_embedding = self.model.encode([question['question']], convert_to_tensor=True, show_progress_bar=False)
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

    def search(self, question, k):
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
        # retrieval_dataset = [[h1, r1, t1], [h2, r2, t2], ...]

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

    def search(self, question, k):
        query_embedding = self.model.encode([question['question']], convert_to_tensor=True, show_progress_bar=False)
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