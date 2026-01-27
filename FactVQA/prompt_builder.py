from PIL import Image
import os, io,  base64, glob, random
from matplotlib import pyplot as plt
from utils import *

# Transform path to base64 for Open API prompt
def path_to_data_url(path):
    """Load image, (optionally) downscale, and return data URL for OpenAI vision input."""
    with Image.open(path) as im:
        # (Optional) downscale very large images to save tokens:
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im = im.convert("RGB")
        im.save(buf, format="JPEG")

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# Transform path to base64 for Open API prompt
def img_to_data_url(path):
    """Load image, (optionally) downscale, and return data URL for OpenAI vision input."""
    with Image.open(path) as im:
        # (Optional) downscale very large images to save tokens:
        im.thumbnail((1024, 1024))
        buf = io.BytesIO()
        im = im.convert("RGB")
        im.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# Build Prompt
def build_multimodal_input_for_sample(question):

    img_url = path_to_data_url("new_dataset_release/new_dataset_release/images/"+question['img_file'])

    return [
        {
            "role": "system",
            "content": (
                "You are a fact-based visual question answering model."
                "You MUST answer with a short phrase, and nothing else."
                "No punctuation, no explanation."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Now, given an image, please answer the question.",
                },
                {
                    "type": "input_image",
                    "image_url": img_url,
                },
                {
                    "type": "input_text",
                    "text": f"Question: {question['question']}",
                },
            ],
        },
    ]

# Build Prompt for Retrieved Item
def build_rag_prompt(retrieved_items, entity2text, relation2text):
    rag_prompt = []
    rag_prompt.append({"type": "input_text", "text": f"You can use the following knowledge-graph triples as evidence to solve the following question"})
    for (i, item) in enumerate(retrieved_items):
        head, relation, tail = item["item"]
        head_txt = entity2text[head] if head in entity2text else ""
        tail_txt = entity2text[tail] if tail in entity2text else ""
        relation_txt = relation2text[relation] if relation in relation2text else ""
        rag_prompt.append({"type": "input_text", "text": f"Triplet {i+1}: (head, relation, tail) = ({head_txt}, {relation_txt}, {tail_txt})"})

        if first_jpg_path(head, "images_subset_kg"):
            with Image.open(first_jpg_path(head, "images_subset_kg")) as im:    
                rag_prompt.append({"type": "input_text", "text": f"Image for head of triplet {i+1}"})
                rag_prompt.append({"type": "input_image", "image_url": img_to_data_url(first_jpg_path(head, "images_subset_kg"))})

        if first_jpg_path(tail, "images_subset_kg"):
            with Image.open(first_jpg_path(tail, "images_subset_kg")) as im:    
                rag_prompt.append({"type": "input_text", "text": f"Image for tail of triplet {i+1}"})
                rag_prompt.append({"type": "input_image", "image_url": img_to_data_url(first_jpg_path(tail, "images_subset_kg"))})

    return rag_prompt
