import pandas as pd

from minsearch import Index
from tqdm.auto import tqdm

from ingest import load_faq_data

df_ground_truth = pd.read_csv("./data/ground_truth-new.csv")
ground_truth = df_ground_truth.to_dict(orient="records")

faq_data = load_faq_data()
llm_zc_data = []

for doc in faq_data:
    if doc["course"] == "llm-zoomcamp":
        llm_zc_data.append(doc)

llmzc_text_index = Index(
    text_fields=["question", "answer", "section"],
    keyword_fields=["course"]
)
llmzc_text_index.fit(llm_zc_data)

def text_search(query):
    boost_dict = {"question": 3, "section":  0.5}

    return llmzc_text_index.search(
        query=query,
        boost_dict=boost_dict,
        num_results=5
    )

def compute_relevance(question_info, search_fn):
    
    question_info = ground_truth[0]
    question = question_info["question"]
    results = search_fn(question)

    relevance = []
    for result in results:
        relevance.append(int(question_info["document"]==result["id"]))

    return relevance

def compute_relevance_for_all(truth_records, search_fn):
    all_the_relevance = []

    for question_info in truth_records:
        relevance = compute_relevance(question_info=question_info, search_fn=search_fn)
        all_the_relevance.append(relevance)

    return all_the_relevance

def hit_rate(relevance):
    count = 0

    for row in relevance:
        if 1 in row:
            count += 1

    return count/len(relevance)


def mrr(relevance):
    total_score = 0.0

    for row in relevance:
        for idx, value in enumerate(row, start=1):
            if value:
                score = 1/(idx)
                total_score += score
                break

    return total_score/len(relevance)

def evaluate(ground_truth, search_fn):

    relevance = compute_relevance_for_all(ground_truth, search_fn)

    return {
        "hit_rate": hit_rate(relevance),
        "mrr": mrr(relevance)
    }

print(evaluate(ground_truth, text_search))
