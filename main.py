from openai import OpenAI


from rag_helper import RAGVector
from sentence_transformers import SentenceTransformer
from sqlitesearch import VectorSearchIndex


model = SentenceTransformer("all-MiniLM-L6-v2")
vector_index = VectorSearchIndex(
    mode="ivf",
    keyword_fields=["course"],
    db_path="faq_vectors.db"
)

ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def main():

    print("Hello from llm-zoomcamp-assistant!")
    
    llm_zc_assistant = RAGVector(
        embedder=model,
        index=vector_index,
        llm_client=ollama_client
    )

    while True:
        user_query = input("Plese enter your question: ").strip()
        print(llm_zc_assistant.rag(user_query))


if __name__ == "__main__":
    main()
