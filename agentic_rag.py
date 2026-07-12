import json
from openai import OpenAI

from sqlitesearch import TextSearchIndex


index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
    db_path="faq.db"
)


def search(query, num_results=5):
    boost_dict = {"question": 3.0, "section": 0.5}
    filter_dict = {"course": "llm-zoomcamp"}

    return index.search(
        query,
        num_results=num_results,
        boost_dict=boost_dict,
        filter_dict=filter_dict
    )

search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ Database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

instructions = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function. 
Use as many keywords from the user question as possible when making first requests.

Make multiple searches. First perform search, analyze the results 
and then perform more searches. 

The question has to be about the course or its logistics, offtopic questions 
shouldn't be answered. If the search returns nothing, it's likely an off-topic question.
If you can't answer the question using FAQ, don't do it yourself. Only use the 
facts from the FAQ database.

At the end, ask if there are other areas that the user wants to explore.
""".strip()


def make_call(call):
    args = json.loads(call.arguments)

    if call.name == "search":
        result = search(**args)

    result_json = json.dumps(result, indent=2)

    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": result_json,
    }


def main():

    print("Hello from llm-zoomcamp-agentic-assistant!")
    
    
    llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    messages = list()
    messages.append({"role": "developer", "content": instructions})
    query = input("Please enter your query: ")
    messages.append({"role": "user", "content": query})

    iter_count = 1
    total_tokens_used = 0

    while True:

        has_function_calls = False
        
        print(f"Iteration #{iter_count}...")
        response = llm_client.responses.create(
            model="llama3.1",
            input = messages,
            tools=[search_tool]
        )

        total_tokens_used += (response.usage.input_tokens + response.usage.output_tokens)

        messages.extend(response.output)
        for item in response.output:

            if item.type == "function_call":
                print(f"Call Information:", item.name, item.arguments)
                call_output = make_call(item)
                messages.append(call_output)
                has_function_calls = True
            
            elif item.type == "message":
                print(f"Final Answer...", item.content[0].text)
        

        if not has_function_calls:
            print(total_tokens_used)
            break

        iter_count += 1

if __name__ == "__main__":
    main()