from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


def llm(prompt):
    response = client.responses.create(model="llama3.1", input=prompt)
    return response.output_text


def main():
    print("Hello from llm-zoomcamp-2026-code!")

    question = (
        "Do you know about the ongoing football world cup? How many nations are in it??"
    )

    # Run this to see the answer provided by the llm
    # before providing context about ongoing world cup
    # answer = llm(question)
    # print(answer)

    # Now, we add context about the ongoing 2026 worldcup
    context = """
    Get the latest information from this website: 
    https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026,
    and give me the answers to my queries regarding the tournament.
    """

    prompt = f"""
    Context: {context}
    Question: {question}

    If you are not able to access the url, let me know.
    Please do not make any assumptions about the information
    not known to you.
    """
    print(llm(prompt))

if __name__ == "__main__":
    main()
