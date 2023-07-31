![](misc/logo.gif)
<p align="center">
    🤗 <a href="https://huggingface.co/datasets/THUDM/LongBench" target="_blank">HF Repo</a> • 📃 Paper coming soon!
</p>

阅读[中文版本](README_ZH.md).

# LongBench: A Multilingual, Multitask Benchmark for Long Context Understanding

**LongBench** is the first benchmark for multilingual, multitask, and comprehensive assessment of **long context understanding** capabilities of large language models. LongBench includes different languages (Chinese and English) to provide a more comprehensive evaluation of the large models' multilingual capabilities on long contexts. In addition, LongBench consists of 20 different tasks, covering key long context application scenarios such as single-document QA, multi-document QA, summarization, few-shot learning, code completion, and synthetic tasks.

We are fully aware of the potentially high costs involved in the model evaluation process, especially in the context of long context scenarios (such as manual annotation costs or API call costs). Therefore, we adopt a fully automated evaluation method, aimed at measuring and evaluating the model's ability to understand long contexts at the lowest cost.

LongBench includes 13 English tasks, 5 Chinese tasks, and 2 code tasks, with the average length of most tasks ranging from 5k to 15k. For the main task categories, LongBench includes six types of tasks, namely multi-document QA, single-document QA, summaries, Few-shot learning, synthetic tasks, and code completion. For detailed statistics and construction methods of LongBench tasks, please refer [here](task.md).

| Task Type | \#English Task | \#Chinese Task | \#Code Task |
| :-------: | :--------------------: | :--------------------: | :------------------: |
| Multi-document QA | 3 | 1 | - |
| Single-document QA | 3 | 1 | - |
| Summarization | 2 | 1 | - |
| Few-shot learning | 3 | 1 | - |
| Synthetic Tasks | 2 | 1 | - |
| Code Completion | - | - | 2 |

## Table of Contents
- [Leaderboard](#leaderboard)
- [How to evaluate on LongBench](#how-to-evaluate-on-LongBench)
- [Evaluation Result on Each Dataset](#evaluation-result-on-each-dataset)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)

## Leaderboard
Here is the average scores (%) on the main task categories in both Chinese and English languages under the Zero-shot scenario. Please refer to this [link](task.md) for the evaluation metrics used for each task.

#### English
|                   | Avg  | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot Learning | Code Completion | Synthetic Tasks |
| ----------------- | :--: | :-----------: | :----------: | :-----------: | :---------------: | :-------------: | :-------------: |
| GPT-3.5-Turbo-16k | 45.5 |     39.8      |     38.7     |     26.5      |       76.0        |      54.5       |      37.8       |
| Llama2-7B-chat-4k | 29.0 |     24.8      |     21.4     |     23.9      |       50.5        |      47.3       |       5.9       |
| LongChat-7B-16k   | 33.7 |     29.3      |     16.1     |     25.8      |       59.9        |      57.0       |      14.2       |
| XGen-7B-8k        | 28.7 |     24.5      |     20.4     |     24.8      |       58.7        |      38.0       |       5.6       |
| InternLM-7B-8k    | 24.7 |     17.1      |     20.8     |     13.3      |       52.7        |      39.7       |       4.7       |
| ChatGLM2-6B       | 26.0 |     23.1      |     15.0     |     22.9      |       46.1        |      46.1       |       2.7       |
| ChatGLM2-6B-32k   | 42.7 |     32.8      |     34.0     |     28.6      |       68.1        |      52.7       |      39.8       |

#### Chinese

|                   | Avg  | Single-Doc QA | Multi-Doc QA | Summarization | Few-shot Learning | Code Completion | Synthetic Tasks |
| ----------------- | :--: | :-----------: | :----------: | :-----------: | :---------------: | :-------------: | :-------------: |
| GPT-3.5-Turbo-16k | 44.5 |     61.2      |     28.7     |     16.0      |       29.2        |      54.5       |      77.5       |
| Llama2-7B-chat-4k | 13.5 |     11.6      |     1.9      |      0.2      |       19.8        |      47.3       |       0.5       |
| LongChat-7B-16k   | 23.7 |     26.6      |     19.1     |     14.0      |       20.8        |      57.0       |       4.8       |
| XGen-7B-8k        | 14.5 |     14.2      |     9.1      |      1.5      |       20.0        |      38.0       |       4.2       |
| InternLM-7B-8k    | 18.6 |     33.3      |     8.9      |     13.0      |       15.5        |      39.7       |       0.9       |
| ChatGLM2-6B       | 22.5 |     33.0      |     15.2     |     14.6      |       20.5        |      46.1       |       5.5       |
| ChatGLM2-6B-32k   | 41.3 |     52.0      |     34.3     |     16.3      |       29.9        |      52.7       |      62.5       |

#### Radar Chart on Long Context Capability 

![](misc/radar.png)

#### Variation of Abilities under Different Context Lengths
To more specifically analyze the models' relative performance under different context lengths, the following chart shows the average relative scores on all tasks over different context length intervals.
![](misc/curve.png)

> Note: Assume that the model scores x on the data within a specific length range of a task, and y on all data of that task, then the model's **relative score** for that length range is (x/y-1). To better compare the trends of different models, we shift all the lines to 0 on 0-4k.

## How to evaluate on LongBench

#### Load Data
You can download and load the **LongBench** data through the Hugging Face datasets ([🤗 HF Repo](https://huggingface.co/datasets/THUDM/LongBench)):
```python
from datasets import load_dataset

datasets = ["hotpotqa", "2wikimqa", "musique", "dureader", "narrativeqa", "qasper", "multifieldqa_en", \
    "multifieldqa_zh", "gov_report", "qmsum", "vcsum", "trec", "nq", "triviaqa", "lsht", "passage_count", \
    "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

for dataset in datasets:
    data = load_dataset('THUDM/LongBench', dataset, split='test')
```
Alternatively, you can download the folder from [this link](https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip) to load the data.

#### Data Format

All data in **LongBench** are standardized to the following format:

```json
{
    "input": "The input/command for the task, usually short, such as questions in QA, queries in Few-shot tasks, etc",
    "context": "The long context required for the task, such as documents, cross-file code, few-shot examples in Few-shot tasks",
    "answers": "A List of all true answers",
    "length": "Total length of the first three items (counted in characters for Chinese and words for English)",
    "dataset": "The name of the dataset to which this piece of data belongs",
    "language": "The language of this piece of data",
    "all_classes": "All categories in classification tasks, null for non-classification tasks",
    "_id": "Random id for each piece of data"
}
```

#### Evaluation
We provide an evaluation code using ChatGLM2-6B as an example. First, run the [pred.py](pred.py) under the repository:
```bash
CUDA_VISIBLE_DEVICES=0 python pred.py
```
You can get the model outputs on all datasets in the `pred/` folder. After that, run the evaluation code in [eval.py](eval.py):
```bash
python eval.py
```
You can get the evaluation results on all datasets in `result.json`. Please note that in `config/`, we provide the input format suitable for each dataset and the maximum output length. Feel free to modify them to better suit the model you want to evaluate. After modification, when evaluating with [pred.py](pred.py), the data will be automatically organized according to the new format to get the corresponding model output.

## Evaluation Result on Each Dataset

The following tables show the Zero-shot evaluation results (%) on all datasets, where Chinese datasets are denoted by "zh" (please refer to this [link](task.md) for the evaluation metrics used for each task).

#### Single-Document QA
|                   | NarrativeQA | Qasper | MultiFieldQA-en | MultiFieldQA-zh |
| ----------------- | :---------: | :----: | :-------------: | :-------------: |
| GPT-3.5-Turbo-16k |    23.6     |  43.3  |      52.3       |      61.2       |
| Llama2-7B-chat-4k |    19.1     |  19.6  |      35.8       |      11.6       |
| LongChat-7B-16k   |    21.6     |  21.6  |      44.6       |      26.6       |
| XGen-7B-8k        |    17.9     |  18.3  |      37.2       |      14.2       |
| InternLM-7B-8k    |    12.4     |  16.8  |      22.3       |      33.3       |
| ChatGLM2-6B       |    11.2     |  23.7  |      34.2       |      33.0       |
| ChatGLM2-6B-32k   |    20.4     |  32.2  |      45.7       |      52.0       |

#### Multi-Document QA

|                   | HotpotQA | 2WikiMQA | Musique | DuReader (zh) |
| ----------------- | :------: | :------: | :-----: | :-----------: |
| GPT-3.5-Turbo-16k |   51.6   |   37.7   |  26.9   |     28.7      |
| Llama2-7B-chat-4k |   24.3   |   31.4   |   8.6   |      1.9      |
| LongChat-7B-16k   |   22.4   |   16.8   |   9.1   |     19.1      |
| XGen-7B-8k        |   28.3   |   21.5   |  11.5   |      9.1      |
| InternLM-7B-8k    |   27.9   |   24.0   |  10.3   |      8.9      |
| ChatGLM2-6B       |   20.2   |   19.6   |   5.3   |     15.2      |
| ChatGLM2-6B-32k   |   44.9   |   34.9   |  22.2   |     34.3      |

#### Summarization

|                   | GovReport | QMSum | VCSUM (zh) |
| :---------------- | :-------: | :---: | :--------: |
| GPT-3.5-Turbo-16k |   29.5    | 23.4  |    16.0    |
| Llama2-7B-chat-4k |   27.3    | 20.6  |    0.2     |
| LongChat-7B-16k   |   28.4    | 23.2  |    14.0    |
| XGen-7B-8k        |   27.8    | 21.7  |    1.5     |
| InternLM-7B-8k    |    9.8    | 16.8  |    13.0    |
| ChatGLM2-6B       |   23.7    | 22.2  |    14.6    |
| ChatGLM2-6B-32k   |   33.3    | 23.9  |    16.3    |

#### Few-shot Learning

|                   | TREC |  NQ  | TriviaQA | LSHT (zh) |
| ----------------- | :--: | :--: | :------: | :-------: |
| GPT-3.5-Turbo-16k | 68.0 | 73.0 |   87.1   |   29.2    |
| Llama2-7B-chat-4k | 60.5 | 31.4 |   59.7   |   19.8    |
| LongChat-7B-16k   | 61.5 | 44.8 |   73.5   |   20.8    |
| XGen-7B-8k        | 66.0 | 43.2 |   67.0   |   20.0    |
| InternLM-7B-8k    | 49.0 | 47.6 |   61.6   |   15.5    |
| ChatGLM2-6B       | 44.0 | 34.5 |   59.8   |   20.5    |
| ChatGLM2-6B-32k   | 62.0 | 64.9 |   77.6   |   29.9    |

#### Code Completion

|                   | LCC  | RepoBench-P |
| ----------------- | :--: | :---------: |
| GPT-3.5-Turbo-16k | 54.7 |    54.3     |
| Llama2-7B-chat-4k | 52.3 |    42.4     |
| LongChat-7B-16k   | 59.2 |    54.7     |
| XGen-7B-8k        | 38.8 |    37.3     |
| InternLM-7B-8k    | 45.5 |    34.0     |
| ChatGLM2-6B       | 48.4 |    43.7     |
| ChatGLM2-6B-32k   | 55.4 |    50.0     |

#### Synthetic Tasks

|                   | PassageRetrieval-en | Passage Count | PassageRetrieval-zh |
| ----------------- | :-----------------: | :-----------: | :-----------------: |
| GPT-3.5-Turbo-16k |        71.0         |      4.5      |        77.5         |
| Llama2-7B-chat-4k |         9.2         |      2.5      |         0.5         |
| LongChat-7B-16k   |        24.0         |      4.5      |         4.8         |
| XGen-7B-8k        |         9.0         |      2.2      |         4.2         |
| InternLM-7B-8k    |         6.5         |      2.9      |         0.9         |
| ChatGLM2-6B       |         3.2         |      2.1      |         5.5         |
| ChatGLM2-6B-32k   |        77.5         |      2.0      |        62.5         |

## Acknowledgement

- Some of the tasks of **LongBench** are based on the datasets proposed by previous researchers, including [HotpotQA](https://hotpotqa.github.io/), [2WikiMultihopQA](https://aclanthology.org/2020.coling-main.580/), [Musique](https://arxiv.org/abs/2108.00573), [DuReader](https://github.com/baidu/DuReader), [NarrativeQA](https://arxiv.org/pdf/1712.07040.pdf), [Qasper](https://arxiv.org/pdf/2105.03011.pdf), [GovReport](https://arxiv.org/pdf/2104.02112.pdf), [QMSum](https://arxiv.org/pdf/2104.05938.pdf), [VCSUM](https://arxiv.org/abs/2305.05280), [TriviaQA](https://nlp.cs.washington.edu/triviaqa/), [NQ](https://ai.google.com/research/NaturalQuestions/), [TREC](https://aclanthology.org/C02-1150.pdf), [LSHT](http://tcci.ccf.org.cn/conference/2014/dldoc/evatask6.pdf), [LCC](https://arxiv.org/abs/2306.14893) and [RepoBench-P](https://arxiv.org/abs/2306.03091).

## Citation
This is a joint work by **THU-KEG** and **Zhipu AI**. We are currently working on the paper, and the citation information will be updated when it's ready. Please stay tuned~

When citing our work, please cite all of the original dataset papers. The relevant citation information is listed [here](refs/ref.bib).
