from re import S

from retrieval import search


'''
    Prompt is divided into two parts:
    1. INSTRUCTIONS - which remain the same for every request sent to LLM
    2. USER QUERY - which changes with every request
'''

INSTRUCTIONS=   """
                    Your task is to answer questions from the course participants
                    based on the provided context.

                    Use the context to find relevant information and provide accurate
                    answers. If the answer is not found in the context,
                    respond with "I don't know."
                """

USER_QUERY_TEMPLATE = """
Question:
{question}

Context:
{context}
"""

def build_context(search_results):
    lines = []

    for doc in search_results:
        lines.append(doc["section"])
        lines.append("Q: " + doc["question"])
        lines.append("A: " + doc["answer"])
        lines.append("")

    return "\n".join(lines).strip()

def prompt(question, search_results):
    context = build_context(search_results=search_results)
    return USER_QUERY_TEMPLATE.format(question=question, context=context)

question = "I just discovered the course. Can I still join now?"
search_results = search(question=question, course="llm-zoomcamp")
print(prompt(question=question, search_results=search_results))
    