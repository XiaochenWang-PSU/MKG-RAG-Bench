import json 
import random
import os, io,  base64, glob, random
from PIL import Image
from openai import OpenAI
from utils import *
from retrieval import *
from prompt_builder import *
import argparse
import random
random.seed(42)

client = OpenAI()

# Get Metric
def get_metrics(rankings, answers):
    hits1 = 0
    hits3 = 0
    hits5 = 0
    hits10 = 0
    mrr = 0
   
    for i in range(len(rankings)):
        if answers[i] in rankings[i]:
            rank = rankings[i].index(answers[i]) + 1
            hits1 += int(rank <= 1)
            hits3 += int(rank <= 3)
            hits5 += int(rank <= 5)
            hits10 += int(rank <= 10)
            mrr += 1.0/rank
    
    return hits1/len(rankings), hits3/len(rankings), hits5/len(rankings), hits10/len(rankings), mrr/len(rankings)

def rank_candidates_with_gpt(content, candidates):
    """Ask GPT to return a strict ranking of candidate IDs (most->least likely)."""
    rules = f"""
You solve analogies with a single exemplar and a question in one of 3 modes:
- mode 0: (T1, T2) -> (I1, ?)
- mode 1: (I1, I2) -> (T1, ?)
- mode 2: (I1, T1) -> (I2, ?)

CANDIDATE ENTITIES (choose ALL and rank them):
{json.dumps(candidates, ensure_ascii=False)}

Task:
- Rank ALL candidates from most likely to least likely tail for the QUESTION.
- Use the exemplar and mode semantics above to infer the relation.
- Return STRICT JSON ONLY:
{{
  "ranking": ["entity_most_likely", "entity_2", ..., "entity_least_likely"]
}}
- Include each candidate exactly once; no extra items.
    """.strip()

    schema = {
        "format":{
            "type": "json_schema",
            "name": "analogy_ranking",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["ranking"],
                "properties": {
                    "ranking": {
                        "type": "array",
                        "items": {"type": "string", "enum": candidates},
                        "minItems": 10,
                        "maxItems": 15,
                    }
                },
                "required": ["ranking"],
            },
            "strict": True,
        }
    }

    resp = client.responses.create(
        model="gpt-4o-2024-08-06",
        input=[
            
            {
                "role": "system",
                "content": rules,
            },
            {"role": "user", "content": content},
        
        ],
        temperature=0,
        max_output_tokens=512,
        text = schema
    )
    return json.loads(resp.output_text.strip())
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # "SimpleMultimodalRetriever" or "SimpleTextRetriever" or "RandomRetriever" or None
    parser.add_argument('--retriever', type=str, default="SimpleMultimodalRetriever", help='Retriever') 
    
    args = parser.parse_args()

    # Inference Data Initialization
    with open("dataset/MARS/test.json", "r", encoding="utf-8") as f:
        lines = f.readlines()
        test_samples = [json.loads(line) for line in lines]
        
    entity2text = read_txt("dataset/MarKG/entity2text.txt")
    relation2text = read_txt("dataset/MarKG/relation2text.txt")
    all_candidates = []
    for sample in test_samples:
        all_candidates.append(entity2text[sample["answer"]])
    
    
    test_samples = random.sample(test_samples, 5)
    
    # Retriever Build
    triplets = load_triplets("dataset/MarKG/wiki_tuple_ids.txt")
    triplets = random.sample(triplets, 10)

    if args.retriever == "SimpleMultimodalRetriever":
        retriever = SimpleMultimodalRetriever(triplets, entity2text, relation2text, "clip-ViT-B-32")
    elif args.retriever == "SimpleTextRetriever":
        retriever = SimpleTextRetriever(triplets, entity2text, relation2text, "sentence-transformers/all-MiniLM-L6-v2")
    elif args.retriever == "RandomRetriever":
        retriever = RandomRetriever(triplets)
        
    rankings = []
    answers = []

    for sample in test_samples:
        content = build_multimodal_input_for_sample(sample, entity2text)
        if args.retriever:
            retrieved_items = retriever.search([sample["example"][0], sample["example"][1], sample["question"]], 3, sample["mode"])
            rag_prompt = build_rag_prompt(retrieved_items, entity2text, relation2text)
            content = rag_prompt + content
        
        # Ranking Sample
        all_candidates.remove(entity2text[sample["answer"]])
        candidates = random.sample(all_candidates, 19)
        candidates.append(entity2text[sample["answer"]])
        random.shuffle(candidates)
        all_candidates.append(entity2text[sample["answer"]])
        
        rankings.append(rank_candidates_with_gpt(content, candidates)["ranking"])
        answers.append(entity2text[sample["answer"]])
        print(answers[-1], rankings[-1])

    metrics = get_metrics(rankings, answers)
    print("Hits@1", metrics[0])
    print("Hits@3", metrics[1])
    print("Hits@5", metrics[2])
    print("Hits@10", metrics[3])
    print("MRR", metrics[4])
