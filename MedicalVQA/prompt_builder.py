from PIL import Image
import os, io,  base64, glob, random
from matplotlib import pyplot as plt

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

# Transform image to base64 for Open API prompt
def img_to_data_url(img):
    im = img.copy()
    im.thumbnail((1024, 1024))
    buf = io.BytesIO()
    im.save(buf, format="JPEG")

    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# Build Prompt for close-ended question
def build_multimodal_input_for_sample_close(sample):

    if "image_path" in sample:
        img_url = path_to_data_url(sample["image_path"])
    else:
        img_url = img_to_data_url(sample["image"])

    return [
        {
            "role": "system",
            "content": (
                "You are a medical visual question answering model. "
                "You MUST answer using exactly one character: "
                "'1' if the correct answer is yes/true, or '0' if the correct answer is no/false. "
                "No words, no punctuation, no explanation."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": img_url,
                },
                {
                    "type": "input_text",
                    "text": f"Question: {sample['question']}\nAnswer with 1 for yes/true or 0 for no/false.",
                },
            ],
        },
    ]

# Build Prompt for open-ened question
def build_multimodal_input_for_sample_open(sample):

    if "image_path" in sample:
        img_url = path_to_data_url(sample["image_path"])
    else:
        img_url = img_to_data_url(sample["image"])

    return [
        {
            "role": "system",
            "content": (
                "You are a medical visual question answering model. "
                "Answer the question using a short, direct answer. "
                "Use as few words as possible (prefer 1-10 words). "
                "Do NOT provide reasoning, steps, or explanations. "
                "Do NOT add extra commentary. "
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Now, given an image, please answer the question.",
                },
                {"type": "input_image", "image_url": img_url},
                {
                    "type": "input_text",
                    "text": f"Question: {sample['question']}\nAnswer concisely.",
                },
            ],
        },
    ]

# Build Prompt for Retrieved Item
def build_rag_prompt(retrieved_items, image_id_to_path):
    rag_prompt = []
    rag_prompt.append({"type": "input_text", "text": f"You can use the following knowledge-graph triples as evidence to solve the following question"})
    for (i, item) in enumerate(retrieved_items):
        triplet = item["item"]
        head, relation, tail = triplet.head_name, triplet.relation, triplet.tail_name
        rag_prompt.append({"type": "input_text", "text": f"Triplet {i+1}: (head, relation, tail) = ({head}, {relation}, {tail})"})

        if triplet.head in image_id_to_path and os.path.exists(image_id_to_path[triplet.head]):
            with Image.open(image_id_to_path[triplet.head]) as im:     
                rag_prompt.append({"type": "input_text", "text": f"Image for head of triplet {i+1}"})
                rag_prompt.append({"type": "input_image", "image_url": img_to_data_url(im)})

    return rag_prompt