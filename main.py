from openai import OpenAI

from prompt import INSTRUCTIONS, build_prompt
from retrieval import search

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def llm(instructions, prompt, model):
    message_history = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": prompt}
    ]
    response = client.responses.create(model=model, input=message_history)
    return response.output_text

def rag(query, model):
    search_results = search(question=query, course="llm-zoomcamp")
    prompt = build_prompt(question=query, search_results=search_results)
    llm_response = llm(instructions=INSTRUCTIONS, prompt=prompt, model=model)
    return llm_response

def main():
    print("Hello from llm-zoomcamp-2026-code!")

    while True:
        user_query = input("Plese enter your question: ").strip()
        print(rag(user_query, model="llama3.1"))

if __name__ == "__main__":
    main()
