# LongBench任务统计

| 任务              |      任务类型  |      评价指标  |     平均长度                                       |语言 | Sample数量|
| :--------- | :-----------:| :-----------: |:---------: | :-------------: |:---------: |
| HotpotQA   | 多文档QA       | F1                        |9,151                           |英文                           |200                           |
| 2WikiMultihopQA| 多文档QA | F1                        |4,887                           |英文                           |200                           |
| MuSiQue| 多文档QA   | F1                        |11,214                           |英文                           |200                           |
| DuReader| 多文档QA  | Rouge-L                 |15,768                           |中文                           |200                           |
| MultiFieldQA-en| 单文档QA | F1                        |4,559                           |英文                           |150                           |
| MultiFieldQA-zh| 单文档QA | F1                        |6,701                           |中文                           |200                           |
| NarrativeQA| 单文档QA | F1                        |18,409                           |英文                           |200                           |
| Qasper| 单文档QA    | F1                        |3,619                           |英文                           |200                           |
| GovReport| 摘要 | Rouge-L                 |8,734                           |英文                           |200                           |
| QMSum| 摘要     | Rouge-L                 |10,614                           |英文                           |200                           |
| MultiNews| 摘要     | Rouge-L                 |2,113                           |英文                           |200                           |
| VCSUM| 摘要     | Rouge-L                 |15,380                           |中文                           |200                           |
| TriviaQA| Few shot  | F1                        |8,209                           |英文                           |200                           |
| SAMSum| Few shot | Rouge-L                        |6,258                           |英文                           |200                           |
| TREC| Few shot | Accuracy                |5,177                           |英文                           |200                           |
| LSHT| Few shot | Accuracy                |22,337                           |中文                           |200                           |
| PassageRetrieval-en| 合成任务 | Accuracy                |9,289                           |英文                           |200                           |
| PassageCount| 合成任务 | Accuracy                |11,141                           |英文                           |200                           |
| PassageRetrieval-zh | 合成任务 | Accuracy                |6,745                           |中文                           |200                           |
| LCC| 代码 | Edit Sim              |1,235                           |Python/C#/Java                           |500                           |
| RepoBench-P| 代码 | Edit Sim                |4,206                           |Python/Java                           |500                           |

> 注：为了避免不同Tokenizer统计的差距，我们使用单词数（Python的split函数）来统计英文数据集和代码数据集的平均长度，使用汉字数来统计中文数据集的平均长度。

# 任务说明

| 任务              | 任务说明                                                     |
| :----------------- | :----------------------------------------------------------- |
| HotpotQA          | 基于多篇给定的文档，回答相关问题                             |
| 2WikiMultihopQA   | 基于多篇给定的文档，回答相关问题                             |
| MuSiQue           | 基于多篇给定的文档，回答相关问题                             |
| DuReader          | 基于多篇给定的检索文档，回答相关的中文问题                   |
| MultiFieldQA-en   | 基于单篇文档，回答英文问题，文档所属的领域相对多元           |
| MultiFieldQA-zh   | 基于单篇文档，回答中文问题，文档所属的领域相对多元           |
| NarrativeQA       | 基于故事或剧本提问，包括对人物、情节、主题等重要元素的理解   |
| Qasper            | 基于单篇论文的提出，问题由NLP的读者提出，并由NLP从业者回答   |
| GovReport         | 摘要任务，要求对政府的工作报告进行总结摘要                   |
| QMSum             | 摘要任务，要求基于用户的查询对会议记录进行摘要               |
| MultiNews             | 多文档摘要任务，要求基于多篇新闻进行摘要               |
| VCSUM             | 摘要任务，要求对中文会议记录进行总结摘要                     |
| TriviaQA          | 单文档问答任务，提供若干的Few Shot样例                       |
| SAMSum            | 对话摘要任务，提供若干的Few Shot样例                       |
| TREC              | 分类任务，要求对问题进行分类，一共包含50个类别               |
| LSHT              | 中文分类任务，要求对新闻进行分类，一共包含24个类别           |
| PassageRetrieval-en | 给定30个英文维基的段落，判断给定的摘要属于哪个段落           |
| PassageCount | 判断给定的若干的段落中不重复的段落一共有几个           |
| PassageRetrieval-zh | 给定若干个出自C4数据集的中文段落，判断给定的摘要属于哪个段落 |
| LCC               | 给定一段较长代码，要求预测出下一行代码                       |
| RepoBench-P       | 给定一个github仓库内多个文件中的代码（包含文件间依赖），要求预测出下一行代码 |

# 数据构造方式

> 注：对于所有基于已有数据集构造的任务，我们均选用原有数据集的验证集或测试集的数据（VCSUM任务除外）

- [HotpotQA](https://hotpotqa.github.io/), [2WikiMultihopQA](https://aclanthology.org/2020.coling-main.580/), [MuSiQue](https://arxiv.org/abs/2108.00573)和[DuReader](https://github.com/baidu/DuReader)任务基于原始的数据集构建，并进行相关处理使其适用于长文本评测。具体地，对于验证集中的问题，我们会选取包含答案的evidence passage和若干干扰的文章，这些文章和原始的问题共同组成了相关任务的输入。
- MultiFiedQA-zh和MultiFieldQA-en任务由约10种来源的长文本数据组成，包含Latex论文、裁判文书、政府工作报告和谷歌索引的PDF文档等。对于每篇长文本，我们邀请了若干博士生和硕士生来进行标注，即基于长文本提问，并给出正确的答案。为了更好地进行自动化评测，我们要求标注员尽可能提出有确定性答案的问题。
- [NarrativeQA](https://arxiv.org/pdf/1712.07040.pdf), [Qasper](https://arxiv.org/pdf/2105.03011.pdf), [GovReport](https://arxiv.org/pdf/2104.02112.pdf)，[QMSum](https://arxiv.org/pdf/2104.05938.pdf)和[MultiNews](https://aclanthology.org/P19-1102.pdf)任务直接使用原论文提供的数据。在具体的构建中，我们使用[ZeroSCROLLS](https://www.zero.scrolls-benchmark.com/)提供的模板来将对应的数据转换为纯文本的输入。
- [VCSUM](https://arxiv.org/abs/2305.05280)任务基于原始的数据集构建，我们针对该数据设计了相应的模板将对应的数据转换为纯文本的输入。
- [TriviaQA](https://nlp.cs.washington.edu/triviaqa/)任务参考[CoLT5](https://arxiv.org/abs/2303.09752)的方式进行构建，即会提供若干基于文档进行问答的样例，并要求语言模型基于新的文档回答相关问题。
- [SAMSum](https://aclanthology.org/D19-5409.pdf)，[TREC](https://aclanthology.org/C02-1150.pdf)和[LSHT](http://tcci.ccf.org.cn/conference/2014/dldoc/evatask6.pdf)任务基于原始的数据集构建。对于验证集中的每个问题，我们采样训练集中的若干数据组成Few-shot样例。这些样例会和验证集中的问题共同组成该任务的输入。
- PassageRetrieval-en任务基于英文维基进行构造。对于每条数据，我们随机采样30段英文维基的段落，并选取其中一段进行摘要（使用GPT-3.5-Turbo）。该任务要求模型给出摘要应该对应哪个的原始段落。
- PassageCount任务基于英文维基进行构造。对于每条数据，我们随机采样若干英文维基的段落，并将其中的每个段落随机重复若干次，最后将段落随机打乱。该任务要求模型判断给定的若干的段落中不重复的段落一共有几个。
- PassageRetrieval-zh任务基于[C4](https://arxiv.org/abs/1910.10683)进行构造。对于每条数据，我们随机采样若干段来自于C4的中文段落，并选取其中一段进行摘要（使用GPT-3.5-Turbo）。该任务要求模型给出摘要对应的那个原始段落名称。
- [LCC](https://arxiv.org/abs/2306.14893)任务我们基于原始的代码补全数据集采样构建。[RepoBench-P](https://arxiv.org/abs/2306.03091)任务中我们选取了原数据集最具挑战性的XF-F（Cross-File-First）设定，并且参考原文中的Oracle-Filled场景，对于每一条原始数据我们随机抽取包括有效跨文件代码片段（gold snippet）在内的多个跨文件代码片段，将其拼接后作为输入，要求模型从其中利用有效的跨文件代码以补全当前文件中的代码。

# LongBench-E数据统计
| 任务              |      任务类型  |      0-4k数据量  |     4-8k数据量                                       |8k+数据量|
| :--------- | :-----------:| :-----------: |:---------: | :-------------: |
| HotpotQA   | 多文档QA       | 100                        |100                           |100   |
| 2WikiMultihopQA| 多文档QA | 100                        |100                           |100     |
| MultiFieldQA-en| 单文档QA | 67                        |70                           |13      |
| Qasper| 单文档QA    | 100                        |100                           |24      |
| GovReport| 摘要 | 100                 |100                           |100        |
| MultiNews| 摘要     | 100                 |100                           |94            |
| TriviaQA| Few shot  | 100                        |100                           |100 |
| SAMSum| Few shot | 100                        |100                           |100   |
| TREC| Few shot | 100                |100                           |100     |
| PassageRetrieval-en| 合成任务 | 100                |100                           |100     |
| PassageCount| 合成任务 | 100                |100                           |100   |
| LCC| 代码 | 100              |100                           |100  |
| RepoBench-P| 代码 | 100               |100                          |100  |
