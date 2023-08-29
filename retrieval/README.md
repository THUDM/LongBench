## Introduction
This folder is to conduct retrieval-based context compression on LongBench using 3 retrievers.
- BM25
- [Contriever](https://github.com/facebookresearch/contriever)
- OpenAI Embedding ([text-embedding-ada-002](https://openai.com/blog/new-and-improved-embedding-model))

First, download the LongBench dataset from HuggingFace and save them in `../LongBench/`, resulting in the folder structure:
```
LongBench/
      LongBench/
          data/
              Put raw LongBench data here.
              2wikimqa.jsonl
              ...
      retrieval/
          BM25/
          contriever/
              contriever/: github
                  mcontriever/: huggingface
          embedding/
          README.md: This file.
```
## Usage

Install the requirements with pip: `pip install -r requirements.txt`

### Retrieval

We take contriever method as an example.
1. Clone contriever from https://github.com/facebookresearch/contriever
2. Replace the files in contriever directory with `contriever/passage_retrieval.py` and `contriever/generate_passage_embeddings.py`
3. Get mcontriever model from https://huggingface.co/facebook/mcontriever
4. run `mContriever.sh`
5. Each line within the JSONL file is expanded by adding a new item "retrieved", which represents the retrieval outcomes of the original context. These results are sorted according to the retriever's criteria.

### Evaluation

We take ChatGLM2-6B-32k as an example. First run [pred.py](pred.py):
```bash
python pred.py --model chatglm2-6b-32k --data C200 --top_k 7 
```
Then evaluate via [eval.py](eval.py):
```bash
python eval.py --model chatglm2-6b-32k --data C200_7
```

Then the evaluation files are in `result_chatglm2-6b-32k`.