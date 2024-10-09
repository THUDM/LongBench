# A single-file evaluation pipeline for huggingface models
# Adapted from LongBench at https://github.com/THUDM/LongBench

__author__ = "Yongchang Hao"
__email__ = "yongchanghao.w@gmail.com"
__license__ = "MIT"

import gc
import logging
import random
import re
import string
from collections import Counter

import datasets
import numpy as np
import torch
from tqdm.auto import tqdm

TASK_TO_MAXLEN = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}

TASK_TO_PROMPT = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "passage_retrieval_zh": '以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是"段落1"，"段落2"等格式\n\n答案是：',
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}


TASKS_E = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "gov_report",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]


TASK_TO_METRIC_NAME = {
    "narrativeqa": "F1",
    "qasper": "F1",
    "multifieldqa_en": "F1",
    "multifieldqa_zh": "F1 (Zh)",
    "hotpotqa": "F1",
    "2wikimqa": "F1",
    "musique": "F1",
    "dureader": "Rouge (Zh)",
    "gov_report": "Rouge",
    "qmsum": "Rouge",
    "multi_news": "Rouge",
    "vcsum": "Rouge (Zh)",
    "trec": "Accuracy",
    "triviaqa": "F1",
    "samsum": "Rouge",
    "lsht": "Accuracy",
    "passage_retrieval_en": "Accuracy",
    "passage_count": "Accuracy",
    "passage_retrieval_zh": "Accuracy (Zh)",
    "lcc": "Similarity",
    "repobench-p": "Similarity",
}


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def normalize_zh_answer(s):
    """Lower text and remove punctuation, extra whitespace."""

    def white_space_fix(text):
        return "".join(text.split())

    def remove_punc(text):
        cn_punctuation = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
        all_punctuation = set(string.punctuation + cn_punctuation)
        return "".join(ch for ch in text if ch not in all_punctuation)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_score(prediction, ground_truth, **kwargs):
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def retrieval_zh_score(prediction, ground_truth, **kwargs):
    pattern = r"段落(\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = 0
    for number in numbers:
        if str(number) == str(ground_truth_id):
            right_num += 1
    final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
    return float(final_score)


def code_sim_score(prediction, ground_truth, **kwargs):
    from fuzzywuzzy import fuzz  # type: ignore

    all_lines = prediction.lstrip("\n").split("\n")
    prediction = ""
    for line in all_lines:
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            prediction = line
            break
    return fuzz.ratio(prediction, ground_truth) / 100


def classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        score = 1.0 / len(em_match_list)
    else:
        score = 0.0
    return score


def rouge_score(prediction, ground_truth, **kwargs):
    from rouge import Rouge  # type: ignore

    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception as e:
        logger.error(f"Error calculating rouge score: {e}")
        return 0.0
    return scores["rouge-l"]["f"]


def rouge_zh_score(prediction, ground_truth, **kwargs):
    import jieba

    prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
    ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
    score = rouge_score(prediction, ground_truth)
    return score


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)


def qa_f1_zh_score(prediction, ground_truth, **kwargs):
    import jieba  # type: ignore

    prediction_tokens = list(jieba.cut(prediction, cut_all=False))
    ground_truth_tokens = list(jieba.cut(ground_truth, cut_all=False))
    prediction_tokens = [normalize_zh_answer(token) for token in prediction_tokens]
    ground_truth_tokens = [normalize_zh_answer(token) for token in ground_truth_tokens]
    prediction_tokens = [token for token in prediction_tokens if len(token) > 0]
    ground_truth_tokens = [token for token in ground_truth_tokens if len(token) > 0]
    return f1_score(prediction_tokens, ground_truth_tokens)


TASK_TO_METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


class Task:
    def __init__(self, name, task_type, language):
        self.name = name
        self.task_type = task_type
        self.language = language
        self.has_e = name in TASKS_E

        self.prompt_formatter = TASK_TO_PROMPT[name]
        self.metric_name = TASK_TO_METRIC_NAME[name]
        self.metric_fn = TASK_TO_METRIC[name]
        self.maximun_generation_length = TASK_TO_MAXLEN[name]


TASKS = [
    Task("narrativeqa", "single_doc_qa", "en"),
    Task("qasper", "single_doc_qa", "en"),
    Task("multifieldqa_en", "single_doc_qa", "en"),
    Task("multifieldqa_zh", "single_doc_qa", "zh"),
    Task("hotpotqa", "multi_doc_qa", "en"),
    Task("2wikimqa", "multi_doc_qa", "en"),
    Task("musique", "multi_doc_qa", "en"),
    Task("dureader", "multi_doc_qa", "zh"),
    Task("gov_report", "summarization", "en"),
    Task("qmsum", "summarization", "en"),
    Task("multi_news", "summarization", "en"),
    Task("vcsum", "summarization", "zh"),
    Task("trec", "few_shot", "en"),
    Task("triviaqa", "few_shot", "en"),
    Task("samsum", "few_shot", "en"),
    Task("lsht", "few_shot", "zh"),
    Task("passage_count", "synthetic", "en"),
    Task("passage_retrieval_en", "synthetic", "en"),
    Task("passage_retrieval_zh", "synthetic", "zh"),
    Task("lcc", "code", "code"),
    Task("repobench-p", "code", "code"),
]


logger = logging.getLogger(__name__)


def check_denpendencies(task, model_name):
    if "longchat" in model_name or "vicuna" in model_name:
        try:
            from fastchat.model import (  # type: ignore # noqa: F401
                get_conversation_template,
            )
        except ImportError:
            raise ImportError(f"Please install fastchat to use {model_name}.")
    metric_fn = task.metric_fn
    if metric_fn in [code_sim_score]:
        try:
            from fuzzywuzzy import fuzz  # type: ignore # noqa: F401
        except ImportError:
            raise ImportError(f"Please install fuzzywuzzy to use {metric_fn.__name__}.")

    if metric_fn in [rouge_zh_score, qa_f1_zh_score]:
        try:
            import jieba  # type: ignore # noqa: F401
        except ImportError:
            raise ImportError(f"Please install jieba to use {metric_fn.__name__}.")

    if metric_fn in [rouge_zh_score, rouge_score]:
        try:
            from rouge import Rouge  # type: ignore # noqa: F401
        except ImportError:
            raise ImportError(f"Please install rouge to use {metric_fn.__name__}.")


# This is the customized building prompt for chat models
def maybe_build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template  # type: ignore

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama-2" in model_name and "chat" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_task(name):
    for task in TASKS:
        if task.name == name:
            return task
    raise ValueError(f"Task {name} not found.")


def get_pred(
    task: Task,
    model,
    tokenizer,
    version_e=False,
    seed: int = 42,
    disable_tqdm=False,
    num_samples=None,
    callback_fn=None,
):
    seed_everything(seed)
    model_name = model.config._name_or_path

    if version_e:
        data = datasets.load_dataset("THUDM/LongBench", f"{task.name}_e", split="test")
    else:
        data = datasets.load_dataset("THUDM/LongBench", task.name, split="test")

    if num_samples is not None:
        data = data.select(list(range(num_samples)))
        logger.info(f"Using the first {num_samples} samples.")

    # set max length according to THUDM/LongBench/config/model2maxlen.json
    max_length = model.config.max_position_embeddings // 1000 * 1000 - 500
    max_gen = task.maximun_generation_length
    prompt_formatter = task.prompt_formatter

    logger.info(f"Max length is set to {max_length}")
    logger.info(f"Max generation length is set to {max_gen}")
    logger.info(f"Prompt formatter is the following:\n{prompt_formatter}")

    device = model.device

    if getattr(model.generation_config, "pad_token_id", None) is None:
        setattr(model.generation_config, "pad_token_id", tokenizer.eos_token_id)
        logger.warning("Setting pad_token_id to eos_token_id to avoid warnings.")

    if getattr(model.generation_config, "top_p", 1.0) < 1.0:
        setattr(model.generation_config, "top_p", 1.0)
        logger.warning("Setting top_p to 1.0 to avoid warnings.")

    results = []
    for json_obj in tqdm(
        data,
        dynamic_ncols=True,
        desc=f"Evaluating on {task.name}",
        disable=disable_tqdm,
    ):
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        prompt = prompt_formatter.format(**json_obj)

        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        else:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]

        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(
                tokenized_prompt[-half:], skip_special_tokens=True
            )

        if task.task_type not in [
            "few_shot",
            "code",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = maybe_build_chat(tokenizer, prompt, model_name)

        if isinstance(prompt, str):
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        else:
            input = prompt.to(device)

        context_length = input.input_ids.shape[-1]
        if (
            task.name == "samsum"
        ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                use_cache=True,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                use_cache=True,
            )[0]
        if callback_fn is not None:
            callback_fn(model)
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)

        results.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )

    return results


def scorer_e(task: Task, predictions, answers, lengths, all_classes):
    # metric_name = TASK_TO_METRIC_NAME[task]
    # metric_fn = TASK_TO_METRIC[task]

    metric_fn = task.metric_fn
    metric_name = task.metric_name

    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if task.task_type == "few_shot":
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                metric_fn(prediction, ground_truth, all_classes=all_classes),
            )
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)

    for key in scores.keys():
        scores[key] = {
            "task": task.name,
            "metric": metric_name,
            "metric_fn": metric_fn.__name__,
            "task_type": task.task_type,
            "mean": np.mean(scores[key]).item() * 100,
            "std": np.std(scores[key]).item() * 100,
        }


def scorer(task: Task, predictions, answers, all_classes):
    # metric_name = TASK_TO_METRIC_NAME[task]
    # metric_fn = TASK_TO_METRIC[task]
    metric_name = task.metric_name
    metric_fn = task.metric_fn

    scores = []
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if task.task_type == "few_shot":
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(score, metric_fn(prediction, ground_truth, all_classes=all_classes))
        scores.append(score)
    return {
        "task": task.name,
        "metric": metric_name,
        "metric_fn": metric_fn.__name__,
        "task_type": task.task_type,
        "mean": np.mean(scores).item() * 100,
        "std": np.std(scores).item() * 100,
    }


@torch.inference_mode()
def evalute(
    task_name,
    model,
    tokenizer,
    version_e=False,
    seed: int = 42,
    disable_tqdm=False,
    num_samples=None,
    callback_fn=None,
):
    task = get_task(task_name)
    check_denpendencies(task, model.config._name_or_path)
    results = get_pred(
        task=task,
        model=model,
        tokenizer=tokenizer,
        version_e=version_e,
        seed=seed,
        disable_tqdm=disable_tqdm,
        num_samples=num_samples,
        callback_fn=callback_fn,
    )
    if version_e:
        score = scorer_e(
            task,
            [r["pred"] for r in results],
            [r["answers"] for r in results],
            [r["length"] for r in results],
            results[0]["all_classes"],
        )
    else:
        score = scorer(
            task,
            [r["pred"] for r in results],
            [r["answers"] for r in results],
            results[0]["all_classes"],
        )
    score["details"] = results
    return score


def available_tasks(task_name=None, category=None, language=None, version_e=False):
    tasks = TASKS
    if category is not None:
        if not isinstance(category, list):
            category = [category]
        tasks = [t for t in tasks if t.task_type in category]
    if language is not None:
        if not isinstance(language, list):
            language = [language]
        tasks = [t for t in tasks if t.language in language]
    if task_name is not None:
        if not isinstance(task_name, list):
            task_name = [task_name]
        tasks = [t for t in tasks if t.name in task_name]

    if version_e:
        tasks = [t for t in tasks if t.has_e]

    return [t.name for t in tasks]


def available_categories():
    return list(set([t.task_type for t in TASKS]))


def available_languages():
    return list(set([t.language for t in TASKS]))


__all__ = [
    "evalute",
    "available_tasks",
    "available_categories",
    "available_languages",
]
