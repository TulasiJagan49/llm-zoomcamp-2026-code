"""
Generate answers for the generated questions using RAG
"""

import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI

from evaluation_utils import RAGWithUsage, map_progress
from ingest import filter_faq_data_by_course, build_minsearch_text_index

load_dotenv()
openai_client = OpenAI()

llm_zc_faq_data = filter_faq_data_by_course(course="llm-zoomcamp")
llm_zc_index = build_minsearch_text_index(
    data=llm_zc_faq_data,
    text_fields=["question", "answer", "section"],
    keyword_fields=["course"],
)

generated_questions_info = pd.read_csv("./data/ground_truth-new.csv").to_dict(
    orient="records"
)

llm_zc_faq_data_dict = {doc["id"]: doc for doc in llm_zc_faq_data}


generation_assistant = RAGWithUsage(
    llm_client=openai_client,
    index=llm_zc_index,
    course="llm-zoomcamp",
    model="gpt-5.4-mini"
)

def generate_rag_answer(question_info):
    question = question_info["question"]
    document_id = question_info["document"]
    original_document = llm_zc_faq_data_dict[document_id]

    generated_answer = generation_assistant.rag(query=question)

    return {
        "document": document_id,
        "question": question,
        "original_answer": original_document["answer"],
        "llm_answer": generated_answer
    }

with ThreadPoolExecutor(max_workers=4) as pool:
    results = map_progress(pool, generated_questions_info, generate_rag_answer)

generated_answers = [result for result in results]

print(generation_assistant.total_cost())

df_answers = pd.DataFrame(generated_answers)
df_answers.to_csv("./data/rag_answers.csv", index=False)

