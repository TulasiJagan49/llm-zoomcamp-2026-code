import psycopg
from openai import OpenAI


from rag_helper import RAGPgVector
from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")
psql_conn = psycopg.connect(
    "postgresql://user:pswd@localhost:5432/faq"
)

ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def main():

    print("Hello from llm-zoomcamp-assistant!")
    
    llm_zc_assistant = RAGPgVector(
        embedder=model,
        conn=psql_conn,
        llm_client=ollama_client
    )

    while True:
        user_query = input("Plese enter your question: ").strip()
        print(llm_zc_assistant.rag(user_query))


if __name__ == "__main__":
    main()
