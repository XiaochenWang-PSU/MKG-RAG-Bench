import os
import re
import json
import argparse
import random
from tqdm import tqdm
from prompt_builder import *
from utils import *
from retrieval import *
from datasets import load_dataset
import openai
from openai import OpenAI

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def load_data(args):
    train = load_dataset('derek-thomas/ScienceQA', split='train') # choose the test set
    test = load_dataset('derek-thomas/ScienceQA', split='test')
    # val = load_dataset('derek-thomas/ScienceQA', split='validation')

    test = test.select(range(args.test_number)) if args.test_number > 0 else qids

    # pick up shot examples from the training set
    shots = train.shuffle(args.seed).select(range(args.shot_number)) # random sample
    
    return test, shots


def get_gpt_result(prompt, base64_images, args):
    content = [{ "type": "input_text", "text": f"{prompt}" }]
    for base64_image in base64_images:
        content.append({"type": "input_image","image_url": f"data:image/jpeg;base64,{base64_image}"})

    response = client.responses.create(
        model=args.model,
        input=[
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,  # note the new name
        top_p=args.top_p,
    )
    output = response.output_text.strip()

    # extract the answer
    pattern = re.compile(r'The answer is ([A-Z]).')
    res = pattern.findall(output)
    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED"

    return answer, output

def get_gpt_result_interleaved(segments, args):
    response = client.responses.create(
        model=args.model,  # e.g., "gpt-4o"
        input=[{"role": "user", "content": segments}],
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
        top_p=args.top_p,
    )
    output = response.output_text.strip()

    # a slightly more forgiving pattern
    m = re.findall(r'\b(?:The answer is|Answer:)\s*([A-Z])\b', output)
    answer = m[0] if len(m) == 1 else "FAILED"
    return answer, output

def get_qwen_result_interleaved(qwen_model, segments, args):
    output = qwen_model.generate(segments, max_new_tokens=args.max_tokens,
                                 temperature=args.temperature, top_p=args.top_p).strip()
    m = re.findall(r'\b(?:The answer is|Answer:)\s*([A-Z])\b', output)
    prediction = m[0] if len(m) == 1 else "FAILED"
    return prediction, output

def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))


def get_result_file(args):
    result_file = "{}/{}/{}_{}_{}_{}_seed_{}_multimodal.json".format(args.output_root, args.model, args.label, args.test_split,
                                                          args.prompt_format, args.shot_number, args.seed)

    return result_file


def save_results(result_file, acc, correct, count, args, results, outputs):
    data = {}
    data['acc'] = acc
    data['correct'] = correct
    data['count'] = count
    data['args'] = vars(args)
    data['results'] = results
    data['outputs'] = outputs

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/scienceqa')
    parser.add_argument('--output_root', type=str, default='results')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--model', type=str, default='gpt-4o-mini')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    # user options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--test_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_number', type=int, default=10, help='GPT-3 is expensive. -1 for whole val/test set')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--save_every', type=int, default=3, help='Save the result with every n examples.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--prompt_format',
                        type=str,
                        default='CQM-A',
                        choices=[
                            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
                            'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
                            'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCML-AE'
                        ],
                        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=5, help='Number of n-shot training examples.')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    # GPT-3 settings
    parser.add_argument('--engine', type=str, default='text-davinci-002')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=512,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--llm', type=str, default="gpt") # "gpt" or "qwen"
    
    # Retrieval
    # "SimpleMultimodalRetriever" or "SimpleTextRetriever" or "RandomRetriever" or None
    parser.add_argument('--retriever', type=str, default="SimpleMultimodalRetriever", help='Retriever') 
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    print("====Input Arguments====")
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    random.seed(args.seed)

    # NEW ARGUMENT
    if not hasattr(args, "llm"):
        args.llm = "gpt"   # default

    test, shots = load_data(args)
    result_file = get_result_file(args)

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

    # Load checkpoint
    if os.path.exists(result_file):
        print("# The result file exists! We will load the check point!!!")
        check_point = json.load(open(result_file))
        acc = check_point["acc"]
        correct = check_point["correct"]
        results = check_point["results"]
        outputs = check_point["outputs"]
        print(f"{len(results)}/{len(test)}, correct: {correct}, acc: {round(acc,2)}%")
    else:
        correct = 0
        results, outputs = {}, {}

    # If using Qwen, load it once here
    qwen_model = None
    if args.llm.lower() == "qwen":
        print("Loading Qwen-VL model locally ...")
        qwen_model = QwenLocal("Qwen/Qwen2-VL-2B-Instruct")

    for qid, problem in enumerate(test):
        if qid in results:
            continue

        choices = problem["choices"]
        answer = problem["answer"]
        label = args.options[answer]

        segments = build_segments_multimodal(shots, problem, args)

        if args.retriever:
            retrieved_items = retriever.search(problem, 3)
            rag_prompt = build_rag_prompt(retrieved_items, entity2text, relation2text)
            segments = rag_prompt + segments

        if args.llm.lower() == "gpt":
            prediction, output = get_gpt_result_interleaved(segments, args)
        elif args.llm.lower() == "qwen":
            prediction, output = get_qwen_result_interleaved(qwen_model, segments, args)
        else:
            raise ValueError(f"Unknown LLM type: {args.llm}")

        pred_idx = get_pred_idx(prediction, choices, args.options)
        results[qid] = pred_idx
        outputs[qid] = output
        if pred_idx == answer:
            correct += 1

        acc = correct / len(results) * 100

        if args.debug or qid < 3:
            print("##################################")
            print("# labeled answer:", label)
            print("# predicted answer:", prediction)
            print("# predicted index:", pred_idx)
            print("# predicted output:", output)

        if (qid + 1) % args.save_every == 0 or (qid + 1) == len(test):
            print(f"{len(results)}/{len(test)}, correct: {correct}, acc: {round(acc,2)}%, saving to {result_file}")
            save_results(result_file, acc, correct, qid + 1, args, results, outputs)