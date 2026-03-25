from PIL import Image
import os, io,  base64, glob, random
from matplotlib import pyplot as plt

# Transform path to base64 for Open API prompt
import base64, io
from PIL import Image, ImageOps

def pool_downscale(im: Image.Image, *, max_side: int = 512, pool_factor: int = None) -> Image.Image:
    """
    Downscale using a pooling-like (area/box) filter to reduce vision token cost.
    - If pool_factor is set (e.g., 2,4), it shrinks by that integer factor.
    - Else it shrinks so the longest side <= max_side.
    """
    im = ImageOps.exif_transpose(im).convert("RGB")  # fix rotation + ensure 3 channels

    w, h = im.size
    if pool_factor is not None and pool_factor > 1:
        new_w = max(1, w // pool_factor)
        new_h = max(1, h // pool_factor)
        return im.resize((new_w, new_h), resample=Image.Resampling.BOX)

    # scale to max_side (keep aspect ratio), pooling-like filter
    if max(w, h) <= max_side:
        return im
    scale = max_side / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return im.resize((new_w, new_h), resample=Image.Resampling.BOX)

def img_to_data_url(img: Image.Image, *, max_side: int = 512, pool_factor: int = None,
                    jpeg_quality: int = 85) -> str:
    im = pool_downscale(img, max_side=max_side, pool_factor=pool_factor)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def path_to_data_url(path: str, *, max_side: int = 512, pool_factor: int = None,
                     jpeg_quality: int = 85) -> str:
    with Image.open(path) as im:
        return img_to_data_url(im, max_side=max_side, pool_factor=pool_factor, jpeg_quality=jpeg_quality)

def build_multimodal_input_for_sample_open(sample, *, max_side: int = 256, pool_factor: int = None):
    img_url = None
    if sample.get("image_path"):
        img_url = path_to_data_url(sample["image_path"], max_side=max_side, pool_factor=pool_factor)
    elif sample.get("image") is not None:
        img_url = img_to_data_url(sample["image"], max_side=max_side, pool_factor=pool_factor)

    user_content = []
    if img_url is not None:
            user_content.append({
        "type": "input_image",
        "image_url": img_url,
        "detail": "low",   
    })


    user_content.append({
        "type": "input_text",
        "text": f"Question: {sample['question']}\nAnswer concisely.",
    })
    return [
        {
            "role": "system",
            "content": (
                "You are a medical visual question answering model. "
                "Answer the question using a short, direct answer. "
                "Use as few words as possible (prefer 1-10 words). "
                "Do NOT provide reasoning, steps, or explanations. "
                "Do NOT add extra commentary. "
                "If the question is answering about relation between image and medical concept, "
                "then only three relations are possible: Positive, Negative, and Uncertain."
            ),
        },
        {"role": "user", "content": user_content},
    ]

def build_multimodal_input_for_sample_close(sample):
    """
    Supports both multimodal (image available) and text-only (no image).
    """
    img_url = None

    # Prefer image_path if it exists and is valid
    if sample.get("image_path"):
        img_url = path_to_data_url(sample["image_path"])
    # Otherwise allow passing a PIL image object in sample["image"]
    elif sample.get("image") is not None:
        img_url = img_to_data_url(sample["image"])

    user_content = []
    if img_url is not None:
        user_content.append({"type": "input_image", "image_url": img_url})

    user_content.append({
        "type": "input_text",
        "text": f"Question: {sample['question']}\nAnswer with 1 for yes/true or 0 for no/false.",
    })

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
            "content": user_content,
        },
    ]


## Build Prompt for open-ended question
#def build_multimodal_input_for_sample_open(sample):
#    """
#    Supports both multimodal (image available) and text-only (no image).
#    For text-only, it simply omits the image input.
#    """
#    img_url = None
#
#    if sample.get("image_path"):
#        img_url = path_to_data_url(sample["image_path"])
#    elif sample.get("image") is not None:
#        img_url = img_to_data_url(sample["image"])
#
#    user_content = []
#    if img_url is not None:
#        user_content.append({"type": "input_image", "image_url": img_url})
#
#    user_content.append({
#        "type": "input_text",
#        "text": f"Question: {sample['question']}\nAnswer concisely.",
#    })
#
#    return [
#        {
#            "role": "system",
#            "content": (
#                "You are a medical visual question answering model. "
#                "Answer the question using a short, direct answer. "
#                "Use as few words as possible (prefer 1-10 words). "
#                "Do NOT provide reasoning, steps, or explanations. "
#                "Do NOT add extra commentary. "
#                "If the question is answering about relation between image and medical concept, "
#                "then only three relations are possible: Positive, Negative, and Uncertain."
#            ),
#        },
#        {
#            "role": "user",
#            "content": user_content,
#        },
#    ]


# Build Prompt for Retrieved Item
def build_rag_prompt(retrieved_items, image_id_to_path):
    """
    Robust to different retriever output schemas.

    Supports:
      A) item["item"] is an object (KGTripletLite) with any of:
         - head_name / tail_name
         - head_text / tail_text
         - head / tail ids
         - relation / rel / rel_id / rel_text
      B) item itself is a dict with fields like:
         - head_text, rel_text, tail_text, head_id, rel_id, tail_id
    """
    rag_prompt = []
    rag_prompt.append({
        "type": "input_text",
        "text": (
            "You can use the following knowledge-graph triples as evidence to solve the following question. "
            "Images provided in the triplets are those similar to the image in the question."
        )
    })

    def _get(obj, *names, default=""):
        # Try attribute first, then dict key
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

    for i, item in enumerate(retrieved_items):
        triplet = item.get("item") if isinstance(item, dict) else None
        obj = triplet if triplet is not None else item  # fall back to dict itself

        # Prefer readable names/text; fall back to IDs
        head_name = _get(obj, "head_name", "head_text", "head", "head_id", default="")
        tail_name = _get(obj, "tail_name", "tail_text", "tail", "tail_id", default="")

        # relation field: could be relation / rel / rel_id / rel_text
        relation = _get(obj, "relation", "rel", "rel_text", "rel_id", default="")

        # For image lookup we want the head IID (e.g., I50978146)
        head_id = _get(obj, "head", "head_id", default="")

        # Some pipelines store the literal "Image_Ixxxx" as head_text; keep prompt clean
        head_str = str(head_name)
        if head_str.startswith("Image"):
            head_str = "Image"

        rag_prompt.append({
            "type": "input_text",
            "text": f"Triplet {i+1}: (head, relation, tail) = ({head_str}, {relation}, {tail_name})"
        })

#        # attach head image if available
#        if head_id and head_id in image_id_to_path:
#            p = image_id_to_path[head_id]
#            if p and os.path.exists(p):
#                with Image.open(p) as im:
#                    rag_prompt.append({"type": "input_text", "text": f"Image for head of triplet {i+1}"})
#                    rag_prompt.append({"type": "input_image", "image_url": img_to_data_url(im)})

    return rag_prompt