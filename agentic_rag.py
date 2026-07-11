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


def main():

    print("Hello from llm-zoomcamp-agentic-assistant!")
    
    
    llm_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    messages = list()
    query = input("Please enter your query: ")
    messages.append({"role": "user", "content": query})
    
    response = llm_client.responses.create(
        model="llama3.1",
        input = messages,
        tools=[search_tool]
    )

    print(response.output)
    print(response.usage)

    import json

    call = response.output[0]
    args=json.loads(call.arguments)

    results = search(**args)
    result_json = json.dumps(results, indent=2)

    print(result_json)

    messages.extend(response.output)

    messages.append({
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": result_json
    })

    response = llm_client.responses.create(
        model="llama3.1",
        input=messages,
        tools=[search_tool]
    )

    print(response.output_text)
    print(response.usage)

if __name__ == "__main__":
    main()