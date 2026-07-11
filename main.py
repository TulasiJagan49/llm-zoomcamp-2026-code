from openai import OpenAI

from sqlitesearch import TextSearchIndex
from rag_helper import RAGBase


index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
    db_path="faq.db"
)

ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def main():

    print("Hello from llm-zoomcamp-assistant!")
    
    llm_zc_assistant = RAGBase(
        index=index,
        llm_client=ollama_client
    )

    while True:
        user_query = input("Plese enter your question: ").strip()
        print(llm_zc_assistant.rag(user_query))


if __name__ == "__main__":
    main()
