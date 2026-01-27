import argparse
from prompt_builder import *
from openai import OpenAI
import json 
from retrieval import *

client = OpenAI()

def get_gpt_result(prompt):
    resp = client.responses.create(
        model="gpt-4o",
        input=prompt,
        temperature=0,
        max_output_tokens=512,
        timeout=60,
    )
    return resp.output_text.strip()

def compute_accuracy(outputs, answers):
    count_correct = 0
    
    for i in range(len(outputs)):
        if outputs[i].replace("a ","").replace("an ","").replace("the ","") == \
            answers[i].replace("a ","").replace("an ","").replace("the ",""):
            count_correct += 1
    
    return {"ACC": 100*count_correct/len(outputs)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # "SimpleMultimodalRetriever" or "SimpleTextRetriever" or "RandomRetriever" or "CaptionRetriever" or None
    parser.add_argument('--retriever', type=str, default="RandomRetriever", help='Retriever') 
    
    args = parser.parse_args()

    # MarKG Retriever Build
    triplets = load_triplets("dataset/MarKG/wiki_tuple_ids.txt")
    triplets = random.sample(triplets, 10)

    entity2text = read_txt("dataset/MarKG/entity2text.txt")
    relation2text = read_txt("dataset/MarKG/relation2text.txt")

    if args.retriever == "SimpleMultimodalRetriever":
        retriever = SimpleMultimodalRetriever(triplets, entity2text, relation2text, "clip-ViT-B-32")
    elif args.retriever == "SimpleTextRetriever":
        retriever = SimpleTextRetriever(triplets, entity2text, relation2text, "sentence-transformers/all-MiniLM-L6-v2")
    elif args.retriever == "RandomRetriever":
        retriever = RandomRetriever(triplets)
    elif args.retriever == "CaptionRetriever":
        retriever = CaptionRetriever(triplets, entity2text, relation2text, "sentence-transformers/all-MiniLM-L6-v2")

    with open("new_dataset_release/new_dataset_release/all_qs_dict_release.json", 'r', encoding='utf-8') as file:
        questions = json.load(file)
        
    outputs = []
    answers = []

    for question_id in questions:
        prompt = build_multimodal_input_for_sample(questions[question_id])

        if args.retriever:
            retrieved_items = retriever.search(questions[question_id], 3)
            rag_prompt = build_rag_prompt(retrieved_items, entity2text, relation2text)
            prompt[1]['content'] = rag_prompt + prompt[1]['content']

        outputs.append(get_gpt_result(prompt).lower().strip())
        answers.append(questions[question_id]['answer'].lower().strip())
        break
    print(compute_accuracy(outputs, answers))