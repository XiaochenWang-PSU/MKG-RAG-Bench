import torch
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
from typing import Dict, List, Optional, Tuple
import random
from datasets import load_dataset
from collections import Counter
import pickle

class MedicalVQADataset():
    def __init__(self, 
                 dataset_name: str,
                 split: str,
                 is_close = 1,
                 val_split = 0.1):

        self.samples = []
        self.is_close = is_close

        if dataset_name.lower() == 'slake':
            self._load_slake(split)
        elif dataset_name.lower() == 'vqa_rad':
            self._load_vqa_rad(split, val_split)
        elif dataset_name.lower() == 'pathvqa':
            self._load_pathvqa(split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"Loaded {len(self.samples)} samples from {dataset_name} for {split} split")

    def _process_answer(self, answer: str, is_close = 1):
        answer = str(answer).lower().strip()

        if is_close:
            if answer in ['yes', 'true']:
                return 1
            elif answer in ['no', 'false']:
                return 0
            return None

        if answer in ['yes', 'true', 'no', 'false']:
            return None
        return answer

    def _load_slake(self, split: str):
        base_path = "MEDVQA/Slake/Slake1.0"
        split_files = {'train': 'train.json', 'val': 'validate.json', 'test': 'test.json'}
        json_path = os.path.join(base_path, split_files[split])
        with open(json_path, 'r') as f:
            data = json.load(f)
        for item in data:
            if item.get("q_lang") == "en":
                binary_answer = self._process_answer(item['answer'], is_close=self.is_close)
                if binary_answer is not None:
                    image_path = os.path.join(base_path, 'imgs', item['img_name'])
                    if os.path.exists(image_path):
                        self.samples.append({
                            'image_path': image_path,
                            'question': item['question'],
                            'answer': binary_answer,
                            'answer_text': str(item['answer']).lower().strip(),
                            'dataset': 'slake'
                        })

    def _load_vqa_rad(self, split: str, val_split: float):
        base_path = "MEDVQA/data_RAD"
        json_path = os.path.join(base_path, 'trainset.json' if split != 'test' else 'testset.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
        yes_no_samples = []
        for item in data:
            binary_answer = self._process_answer(item['answer'], is_close=self.is_close)
            if binary_answer is not None:
                image_path = os.path.join(base_path, 'images', item['image_name'])
                if os.path.exists(image_path):
                    yes_no_samples.append({
                        'image_path': image_path,
                        'question': item['question'],
                        'answer': binary_answer,
                        'answer_text': str(item['answer']).lower().strip(),
                        'dataset': 'vqa_rad'
                    })
        if split != 'test':
            random.seed(42)
            split_idx = int(len(yes_no_samples) * (1 - val_split))
            random.shuffle(yes_no_samples)
            self.samples.extend(yes_no_samples[:split_idx] if split == 'train' else yes_no_samples[split_idx:])
        else:
            self.samples.extend(yes_no_samples)

    def _load_pathvqa(self, split: str):
        hf_split = {'train': 'train', 'val': 'validation', 'test': 'test'}[split]
        dataset = load_dataset("flaviagiammarino/path-vqa")[hf_split]
        for item in dataset:
            binary_answer = self._process_answer(item['answer'], is_close=self.is_close)
            if binary_answer is not None:
                image = item['image']
                if image.mode == 'CMYK':
                    image = image.convert('RGB')
                self.samples.append({
                    'image': image,
                    'question': item['question'],
                    'answer': binary_answer,
                    'answer_text': str(item['answer']).lower().strip(),
                    'dataset': 'pathvqa'
                })

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
    
        # Load image
        if 'image_path' in sample:
            image = Image.open(sample['image_path']).convert('RGB')
        else:
            image = sample['image']
        
        return {
            'image': image,
            'question': sample['question'],
            'question_tokens': question_tokens,
            'answer': sample['answer'],
            'answer_text': sample['answer_text'],
            'dataset': sample['dataset'],
        }

    def __len__(self):
        return len(self.samples)

    def get_statistics(self) -> Dict:
        return {
            'total_samples': len(self.samples),
            'answer_distribution': Counter(s['answer'] for s in self.samples),
            'answer_text_distribution': Counter(s['answer_text'] for s in self.samples)
        }

if __name__ == "__main__":
    print("Testing data loading...")
    
    # Test each dataset individually
    datasets = ['slake', 'vqa_rad', 'pathvqa']
    splits = ['test']
    
    for dataset_name in datasets:
        print(f"\nTesting {dataset_name} dataset:")
        
        for split in splits:
            print(f"\n{split} split:")
            dataset = MedicalVQADataset(
                dataset_name=dataset_name,
                split=split,
            )
            
            # Print statistics
            stats = dataset.get_statistics()
            print(f"Total samples: {stats['total_samples']}")
            print("\nBinary answer distribution:")
            print(stats['answer_distribution'])
            
            