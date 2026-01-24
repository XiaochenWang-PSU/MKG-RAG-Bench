import argparse
from data import MedicalVQADataset
from prompt_builder import *
from openai import OpenAI
from utils import *
from retrieval import *

client = OpenAI()

def get_gpt_result(prompt):
    resp = client.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=0,
        max_output_tokens=512,
    )
    return resp.output_text.strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # None, "SimpleMultimodalRetriever", "SimpleTextRetriever", "RandomRetriever", "CaptionRetriever"
    parser.add_argument('--retriever', type=str, default="CaptionRetriever", help='Retriever') 
    # 'slake', 'vqa_rad', 'pathvqa'
    parser.add_argument("--dataset", type=str, default='slake')
    # "open", "close"
    parser.add_argument("--type", type=str, default='open')

    args = parser.parse_args()

    vqa_data = MedicalVQADataset(args.dataset, split="test", is_close = (1 if args.type == "close" else 0))

    if args.retriever == "SimpleMultimodalRetriever":
        retriever = SimpleMultimodalRetriever(kg_path="MedMKG_huggingface/MedMKG.csv", image_map_path="MedMKG_huggingface/image_mapping.csv", model_name="clip-ViT-B-32")
    elif args.retriever == "RandomRetriever":
        retriever = RandomRetriever(kg_path="MedMKG_huggingface/MedMKG.csv", image_map_path="MedMKG_huggingface/image_mapping.csv")
    elif args.retriever == "SimpleTextRetriever":
        retriever = SimpleTextRetriever(kg_path="MedMKG_huggingface/MedMKG.csv", image_map_path="MedMKG_huggingface/image_mapping.csv", model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif args.retriever == "CaptionRetriever":
        retriever = CaptionRetriever(kg_path="MedMKG_huggingface/MedMKG.csv", image_map_path="MedMKG_huggingface/image_mapping.csv", model_name="sentence-transformers/all-MiniLM-L6-v2")

    outputs = []
    answers = []

    for sample in vqa_data.samples[:10]:
        if args.type == "close":
            prompt = build_multimodal_input_for_sample_close(sample)
        else:
            prompt = build_multimodal_input_for_sample_open(sample)
        if args.retriever:
            retrieved_items = retriever.search(sample, 3)
            rag_prompt = build_rag_prompt(retrieved_items, retriever.image_id_to_path)
            prompt[1]["content"] = rag_prompt + prompt[1]["content"]
        outputs.append(get_gpt_result(prompt))
        answers.append(sample["answer"])

    if args.type == "close":
        print(compute_metrics_close(answers, outputs))
    else:
        print(compute_metrics_open(answers, outputs))