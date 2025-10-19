# -*- coding: utf-8 -*-
# rag/ragas_eval.py
from rag.g_pipe import rag_function
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

def eval_pairs(pairs):  # pairs: list[dict] with "question","ground_truth"
    ds = Dataset.from_list(pairs)
    answers = []
    for row in ds:
        _, a = rag_function(row["question"], k=3)
        answers.append(a)
    ds = ds.add_column("answer", answers)
    return evaluate(ds, metrics=[faithfulness, answer_relevancy])
