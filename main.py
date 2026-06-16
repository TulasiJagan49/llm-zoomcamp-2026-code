from openai import OpenAI

def main():
    print("Hello from llm-zoomcamp-2026-code!")

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    response = client.chat.completions.create(
        model="llama3.1",
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant"},
            {"role": "user", "content": "Write me a python function to reverse a string"},
        ]
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
