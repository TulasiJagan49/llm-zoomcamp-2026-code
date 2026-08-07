import pandas as pd

from ingest import build_minsearch_text_index, filter_faq_data_by_course

df_ground_truth = pd.read_csv("./data/ground_truth-new.csv")
ground_truth = df_ground_truth.to_dict(orient="records")

llm_zc_data = filter_faq_data_by_course(course="llm-zoomcamp")
llmzc_text_index = build_minsearch_text_index(
    data=llm_zc_data,
    text_fields=["question", "answer", "section"],
    keyword_fields=["course"],
)


def text_search(query):
    boost_dict = {"question": 3, "section": 0.5}

    return llmzc_text_index.search(query=query, boost_dict=boost_dict, num_results=5)


def compute_relevance(question_info, search_fn):

    question = question_info["question"]
    results = search_fn(question)

    relevance = []
    for result in results:
        relevance.append(int(question_info["document"] == result["id"]))

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

    return count / len(relevance)


def mrr(relevance):
    total_score = 0.0

    for row in relevance:
        for idx, value in enumerate(row, start=1):
            if value:
                score = 1 / (idx)
                total_score += score
                break

    return total_score / len(relevance)


def evaluate(ground_truth, search_fn):

    relevance = compute_relevance_for_all(ground_truth, search_fn)

    return {"hit_rate": hit_rate(relevance), "mrr": mrr(relevance)}


# print(evaluate(ground_truth, text_search))


# def search_boost(query, boost_question):

#     boost_dict = {"question": boost_question, "section": 0.5}

#     return llmzc_text_index.search(query=query, boost_dict=boost_dict, num_results=5)


# for boost in [0.5, 1.0, 2.0, 3.0, 5.0]:
#     result = evaluate(
#         ground_truth=ground_truth,
#         search_fn=lambda query, boost=boost: search_boost(query, boost),
#     )
#     print(f"boost={boost}: {result}")


def search_boosts(query, question_boost, answer_boost, section_boost):
    boost_dict = {
        "question": question_boost,
        "section": section_boost,
        "answer": answer_boost,
    }

    return llmzc_text_index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
    )

results = []

for question_boost in [1.0, 2.0, 5.0]:
    for answer_boost in [1.0, 2.0, 4.0, 10.0]:
        for section_boost in [0.1, 0.2, 0.5]:
            print(
                f"Evaluating question_boost={question_boost},"
                f" answer_boost={answer_boost},"
                f" section_boost={section_boost}..."
            )
            result = evaluate(
                ground_truth,
                lambda query, question_boost=question_boost, answer_boost=answer_boost, section_boost=section_boost: search_boosts(
                    query,
                    question_boost,
                    answer_boost,
                    section_boost
                )
            )

            results.append({
                "question": question_boost,
                "answer": answer_boost,
                "section": section_boost,
                "hit_rate": result["hit_rate"],
                "mrr": result["mrr"],
            })

df_results = pd.DataFrame(results)
print(df_results.sort_values("mrr", ascending=False).head(10))
