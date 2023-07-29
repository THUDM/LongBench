# Task statistics

| Task          | Task Type | Eval metric |     Avg len                            |Language | \#Sample |
| --------- | -------------| ------------- |--------- | ------------- |--------- |
| HotpotQA   | Multi-doc QA | F1                        |9149                           |EN                           |200                           |
| 2WikiMultihopQA| Multi-doc QA | F1                        |4885                           |EN                           |200                           |
| Musique| Multi-doc QA | F1                        |7798                           |EN                           |200                           |
| DuReader| Multi-doc QA | Rouge-L                 |15768                           |ZH                           |200                           |
| MultiFieldQA-en| Single-doc QA | F1                        |4559                           |EN                           |150                           |
| MultiFieldQA-zh| Single-doc QA | F1                        |6771                           |ZH                           |200                           |
| NarrativeQA| Single-doc QA | F1                        |18405                           |EN                           |200                           |
| Qasper| Single-doc QA | F1                        |3619                           |EN                           |200                           |
| GovReport| Summarization | Rouge-L                 |8169                           |EN                           |200                           |
| QMSum| Summarization | Rouge-L                 |10546                           |EN                           |200                           |
| VCSUM| Summarization | Rouge-L                 |15147                           |ZH                           |200                           |
| TriviaQA| Few shot  | F1                        |8015                           |EN                           |200                           |
| NQ| Few shot | F1                        |8210                           |EN                           |200                           |
| TREC| Few shot | Accuracy                |5176                           |EN                           |200                           |
| LSHT| Few shot | Accuracy                |22333                           |ZH                           |200                           |
| PassageRetrieval-en| Synthetic | Accuracy                |9288                           |EN                           |200                           |
| PassageCount| Synthetic | Accuracy                |11141                           |EN                           |200  |
| PassageRetrieval-zh | Synthetic | Accuracy                |6745                           |ZH                           |200                           |
| LCC| Code | Edit Sim              |1235                           |Python/C#/Java                           |500                           |
| RepoBench-P| Code | Edit Sim                |5622                           |Python/Java                           |500                           |

> Note: In order to avoid discrepancies caused by different tokenizers, we use the word count (using Python's split function) to calculate the average length of English datasets and code datasets, and use the character count to calculate the average length of Chinese datasets.

# Task description

| Task              | Task Description                                            |
| ----------------- | ------------------------------------------------------------ |
| HotpotQA          | Answer related questions based on multiple given documents   |
| 2WikiMultihopQA   | Answer related questions based on multiple given documents   |
| Musique           | Answer related questions based on multiple given documents   |
| DuReader          | Answer related Chinese questions based on multiple retrieved documents |
| MultiFieldQA-en   | Answer English questions based on a single document, which comes from a relatively diverse field |
| MultiFieldQA-zh   | Answer Chinese questions based on a single document, which comes from a relatively diverse field |
| NarrativeQA       | Ask questions based on stories or scripts, including understanding of important elements such as characters, plots, themes, etc. |
| Qasper            | Ask questions based on a single paper, questions proposed by NLP readers, and answered by NLP practitioners |
| GovReport         | A summarization task that requires summarizing government work reports |
| QMSum             | A summarization task that requires summarizing meeting records based on user queries |
| VCSUM             | A summarization task that requires summarizing Chinese meeting records |
| TriviaQA          | Single document question answering task, providing several Few Shot examples |
| NQ                | Single document question answering task, providing several Few Shot examples |
| TREC              | A classification task that requires categorizing questions, includes 50 categories in total |
| LSHT              | A Chinese classification task that requires categorizing news, includes 24 categories in total |
| PassageRetrieval-en | Given 30 English Wikipedia paragraphs, determine which paragraph the given summary belongs to |
| PassageCount | Determine the number of non-repeating paragraphs in a given number of paragraphs |
| PassageRetrieval-zh | Given several Chinese paragraphs from the C4 data set, determine which paragraph the given abstract belongs to |
| LCC               | Given a longer piece of code, predict the next line of code |
| RepoBench-P       | Given code in multiple files within a GitHub repository (including inter-file dependencies), predict the next line of code |


# Task construction

> Note: For all tasks constructed from existing datasets, we use data from the validation or test set of the existing dataset (except for VCSUM).

- The tasks of [HotpotQA](https://hotpotqa.github.io/), [2WikiMultihopQA](https://aclanthology.org/2020.coling-main.580/), [Musique](https://arxiv.org/abs/2108.00573), and [DuReader](https://github.com/baidu/DuReader) are built based on the original datasets and processed to make them suitable for long text evaluation. Specifically, for questions in the validation set, we select the evidence passage that contains the answer and several distracting articles. These articles together with the original question constitute the input of the related tasks.
- The tasks of MultiFiedQA-zh and MultiFieldQA-en consist of long-text data from about 10 sources, including Latex papers, judicial documents, government work reports, and PDF documents indexed by Google. For each long text, we invite several PhD and master students to annotate, i.e., to ask questions based on the long text and give the correct answers. To better automate evaluation, we require the annotators to ask questions with definitive answers as much as possible.
- The tasks of [NarrativeQA](https://arxiv.org/pdf/1712.07040.pdf), [Qasper](https://arxiv.org/pdf/2105.03011.pdf), [GovReport](https://arxiv.org/pdf/2104.02112.pdf), and [QMSum](https://arxiv.org/pdf/2104.05938.pdf) directly use the data provided by the original papers. In the specific construction, we use the template provided by [ZeroSCROLLS](https://www.zero.scrolls-benchmark.com/) to convert the corresponding data into pure text input.
- The [VCSUM](https://arxiv.org/abs/2305.05280) task is built based on the original dataset, and we have designed a corresponding template to convert the corresponding data into pure text input.
- The tasks of [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) and [NQ](https://ai.google.com/research/NaturalQuestions/) are constructed in the manner of [CoLT5](https://arxiv.org/abs/2303.09752), which provides several examples of question and answering based on documents, and requires the language model to answer related questions based on new documents.
- The tasks of [TREC](https://aclanthology.org/C02-1150.pdf) and [LSHT](http://tcci.ccf.org.cn/conference/2014/dldoc/evatask6.pdf) are built based on the original datasets. For each question in the validation set, we sample several data from the training set to form few-shot examples. These examples together with the questions in the validation set constitute the input for this task.
- The PassageRetrieval-en task is constructed based on English Wikipedia. For each piece of data, we randomly sample 30 paragraphs from English Wikipedia and select one for summarization (using GPT3.5 Turbo). The task requires the model to specify which original paragraph the summary corresponds to.
- The PassageCount task is constructed based on the English wiki. For each piece of data, we randomly sample several passages from English Wikipedia, repeat each paragraph at random several times, and finally shuffle the paragraphs at random. This task requires the model to determine how many paragraphs are not repeated in a given number of paragraphs.
- The PasskeyRetrieval-zh task is constructed based on [C4](https://arxiv.org/abs/1910.10683). For each piece of data, we randomly sample several Chinese paragraphs from C4 and select one of them for summary (using GPT3.5 Turbo). This task requires the model to give the original paragraph to which the summary should correspond.
- For the [LCC](https://arxiv.org/abs/2306.14893) task, we sample from the original code completion dataset. In the [RepoBench-P](https://arxiv.org/abs/2306.03091) task, we select the most challenging XF-F (Cross-File-First) setting from the original dataset and refer to the Oracle-Filled scenario in the original text. For each original piece of data, we randomly extract multiple cross-file code snippets, including the effective cross-file code snippet (gold snippet), and concatenate them as input, requiring the model to use the effective cross-file code to complete the current file's code.