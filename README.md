# MKG-RAG-Bench
A repo for sharing codes concerning multimodal retrieval-augmented generation with knowledge graph.

## ScienceQA Folder
Reference: [https://github.com/lupantech/ScienceQA](https://github.com/lupantech/ScienceQA)

To use GPT, prepare your API key:
```
export OPENAI_API_KEY="your_api_key_here"
```

To run GPT model it will store results in folder `results`
```
python3 run_gpt.py
```

Or 
```
python3 run_multimodal_gpt.py
```

Please remember to change `test_number` in `args` to -1 in `run_gpt.py` and `run_multimodal_gpt.py` when running full experiment.

Note:
- `run_gpt.py` use image's caption as input.
- `run_multimodal_gpt.py` use image as input.

## MedicalVQA Folder
Reference: [https://github.com/XiaochenWang-PSU/MedMKG](https://github.com/XiaochenWang-PSU/MedMKG)


To use GPT, prepare your API key:
```
export OPENAI_API_KEY="your_api_key_here"
```

To run GPT model
```
python3 run_gpt.py
```

## MKG_Analogy Folder
Reference: [https://github.com/zjunlp/MKG_Analogy](https://github.com/zjunlp/MKG_Analogy)

To use GPT, prepare your API key:
```
export OPENAI_API_KEY="your_api_key_here"
```

To run GPT model
```
python3 run_gpt.py
```

## FactVQA Folder
Reference: [https://github.com/wangpengnorman/FVQA](https://github.com/wangpengnorman/FVQA)

To use GPT, prepare your API key:
```
export OPENAI_API_KEY="your_api_key_here"
```

To run GPT model
```
python3 run_gpt.py
```

## Useful Resources

### Papers:

#### General Graph RAG
- Knowledge Graph-Guided Retrieval Augmented Generation, NACCL 2025 [[Paper](https://arxiv.org/pdf/2502.06864)][[Code](https://github.com/nju-websoft/KG2RAG/tree/main)]: Given a query, KG2RAG first extracts ⟨h, r, t⟩ triples from the corpus using an LLM prompt (similar to Xiaochen's paper). It then retrieves the top-k semantic seeds via cosine similarity in an embedding space, followed by graph-guided expansion using m-hop BFS from those seeds. Finally, it applies a reranker and graph filtering (e.g., per-component maximum spanning trees) to produce a robust subgraph and context for the LLM.
- From Local to Global: A GraphRAG Approach to Query-Focused Summarization, Microsoft Research [[Paper](https://arxiv.org/pdf/2404.16130)][[Code](https://github.com/microsoft/graphrag)]

#### Healthcare Application
- RULE: Reliable Multimodal RAG for Factuality in Medical Vision Language Models [[Paper](https://arxiv.org/pdf/2407.05131)][[Code](https://github.com/richard-peng-xia/RULE)]: RULE introduces a two-part framework to improve factual accuracy in Medical Large Vision Language Models (Med-LVLMs): (1) a statistical calibration method that adaptively selects the optimal number of retrieved contexts to control factuality risk, and (2) knowledge-balanced preference tuning that fine-tunes models on curated samples where retrieval caused errors, reducing over-reliance on external references (DPO).
- Fact-Aware Multimodal Retrieval Augmentation for Accurate Medical Radiology Report Generation [[Paper](https://arxiv.org/pdf/2407.15268)][[Code](https://github.com/cxcscmu/FactMM-RAG)]: [RadGraph](https://arxiv.org/abs/2106.14463) is used to annotate reports and mine factually consistent pairs, which are then employed to train a [MARVEL](https://arxiv.org/abs/2310.14037)-based multimodal retriever with contrastive learning to align images and text. At inference, given a new chest X-ray, the retriever selects the most factually relevant report, and both the image and retrieved report are passed into LLaVA for retrieval-augmented generation, improving factual correctness in the final radiology report.
