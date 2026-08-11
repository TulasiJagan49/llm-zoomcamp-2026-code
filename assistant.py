"""
Creating a new rag base assistant to practice online evaluation method
"""
import sys

from dotenv import load_dotenv
from openai import OpenAI

from ingest import load_faq_data, build_minsearch_text_index
from metrics import RAGWithMetrics


def create_assistant():

    load_dotenv()

    llm_client = OpenAI()

    faq_index = build_minsearch_text_index(
        data=load_faq_data(),
        text_fields=["question", "answer", "section"],
        keyword_fields=["course"],
    )

    return RAGWithMetrics(
        llm_client=llm_client,
        model="gpt-5.4-mini",
        index=faq_index
    )


if __name__ == "__main__":
    assistant = create_assistant()

    query = "How do I join the course?"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    answer = assistant.rag(query)
    print(answer)