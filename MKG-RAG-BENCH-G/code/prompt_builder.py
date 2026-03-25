import json 
import random
import os, io,  base64, glob, random
from PIL import Image
from utils import *
from typing import List, Dict, Optional

# Transform path to base64 for Open API prompt
def img_to_data_url(path: str) -> str:
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Image path not found: {path}")
    with Image.open(path) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# Build Prompt
def _need_image(eid: str, root: str) -> Optional[str]:
        p = first_jpg_path(eid, root)
        if p and os.path.isfile(p):
            return p
        return None
def build_multimodal_input_for_sample(sample, entity2text):
    head, tail = sample["example"][0], sample["example"][1]
    question = sample["question"]
    mode = int(sample["mode"])
    content = []

    description = (
        "You are solving a knowledge-graph analogy with one exemplar and one question.\n"
        "Interpret (T) as text-only, (I) as image-only.\n"
        "You have to infer the relation hinted by the exemplar to get the relation between question and answer"
    )
    content.append({"type": "input_text", "text": description})

    img_root = "images_subset_inference"

    if mode == 0:
        head_txt = entity2text.get(head, head)
        tail_txt = entity2text.get(tail, tail)

        q_img = _need_image(question, img_root)
        if q_img is None:
            return None  # <-- skip

        content.append({"type": "input_text", "text": f"Exemplar (T1, T2): head = {head_txt} and tail = {tail_txt}"})
        content.append({"type": "input_text", "text": "Question (I1, ?): head = "})
        content.append({"type": "input_image", "image_url": img_to_data_url(q_img)})
        content.append({"type": "input_text", "text": " and tail = ?"})

    elif mode == 1:
        question_txt = entity2text.get(question, question)

        h_img = _need_image(head, img_root)
        t_img = _need_image(tail, img_root)
        if h_img is None or t_img is None:
            return None  # <-- skip

        content.append({"type": "input_text", "text": "Exemplar (I1, I2): head = "})
        content.append({"type": "input_image", "image_url": img_to_data_url(h_img)})
        content.append({"type": "input_text", "text": " and tail = "})
        content.append({"type": "input_image", "image_url": img_to_data_url(t_img)})
        content.append({"type": "input_text", "text": f"Question (T1, ?): head = {question_txt} and tail = ?"})

    else:
        tail_txt = entity2text.get(tail, tail)

        h_img = _need_image(head, img_root)
        q_img = _need_image(question, img_root)
        if h_img is None or q_img is None:
            return None  # <-- skip

        content.append({"type": "input_text", "text": "Exemplar (I1, T1): head = "})
        content.append({"type": "input_image", "image_url": img_to_data_url(h_img)})
        content.append({"type": "input_text", "text": f" and tail = {tail_txt}"})
        content.append({"type": "input_text", "text": "Question (I2, ?): head = "})
        content.append({"type": "input_image", "image_url": img_to_data_url(q_img)})
        content.append({"type": "input_text", "text": " and tail = ?"})

    return content

# Build Prompt for Retrieved Item
def build_rag_prompt(retrieved_items, entity2text, relation2text):

    
    
    rag_prompt = []
    rag_prompt.append({"type": "input_text", "text": f"You can use the following knowledge-graph triples as evidence to solve the following question"})
    for (i, item) in enumerate(retrieved_items):
        head, relation, tail = item["item"]
        head_txt = entity2text[head] if head in entity2text else ""
        tail_txt = entity2text[tail] if tail in entity2text else ""
        relation_txt = relation2text[relation] if relation in relation2text else ""
        # print(f"Triplet {i+1}: (head, relation, tail) = ({head_txt}, {relation_txt}, {tail_txt})")
        rag_prompt.append({"type": "input_text", "text": f"Triplet {i+1}: (head, relation, tail) = ({head_txt}, {relation_txt}, {tail_txt})"})

#        if first_jpg_path(head, "images_subset_kg"):
#            with Image.open(first_jpg_path(head, "images_subset_kg")) as im:    
#                rag_prompt.append({"type": "input_text", "text": f"Image for head of triplet {i+1}"})
#                rag_prompt.append({"type": "input_image", "image_url": img_to_data_url(first_jpg_path(head, "images_subset_kg"))})
#
#        if first_jpg_path(tail, "images_subset_kg"):
#            with Image.open(first_jpg_path(tail, "images_subset_kg")) as im:    
#                rag_prompt.append({"type": "input_text", "text": f"Image for tail of triplet {i+1}"})
#                rag_prompt.append({"type": "input_image", "image_url": img_to_data_url(first_jpg_path(tail, "images_subset_kg"))})

    return rag_prompt
