import json
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat.runners import OpenAIResponsesRunner
    
from ingest import filter_faq_data_by_course, build_minsearch_text_index
from evaluation_utils import map_progress


load_dotenv()
llm_client = OpenAI()

llm_zc_faq_data = filter_faq_data_by_course(course="llm-zoomcamp")
llm_zc_faq_data_dict = {doc["id"]: doc for doc in llm_zc_faq_data}
llm_zc_index = build_minsearch_text_index(
    data=llm_zc_faq_data,
    text_fields=["question", "answer", "section"],
    keyword_fields=["course"],
)

generated_questions_info = pd.read_csv("./data/ground_truth-new.csv").to_dict(
    orient="records"
)

def search(query: str) -> list[dict]:
    """
    Search the FAQ database for entries matching the given query.
    """
    return llm_zc_index.search(
        query,
        num_results=5,
        boost_dict={"question": 1.0, "answer": 2.0, "section": 0.1},
        filter_dict={"course": "llm-zoomcamp"}
    )

agent_tools = Tools()
agent_tools.add_tool(search)


instructions = """
You're a course teaching assistant. Answer student questions based on
the results obtained from search tool.
""".strip()

runner = OpenAIResponsesRunner(
    tools=agent_tools,
    developer_prompt=instructions,
    llm_client=OpenAIClient(model="gpt-5.4-mini")
)


def extract_tool_calls(messages):
    tool_calls = []

    for message in messages:
        if isinstance(message, dict):
            continue

        if message.type == "function_call":
            tool_calls.append({
                "name": message.name,
                "arguments": message.arguments,
            })

    return tool_calls

def generate_agent_answer(rec):
    doc_id = rec["document"]
    original_doc = llm_zc_faq_data_dict[doc_id]

    result = runner.loop(prompt=rec["question"])

    tool_calls = extract_tool_calls(result.all_messages)

    answer_record = {
        "question": rec["question"],
        "answer_agent": result.last_message,
        "answer_orig": original_doc["answer"],
        "tool_calls": json.dumps(tool_calls),
        "cost": result.cost.total_cost,
        "document": doc_id,
    }

    return answer_record

with ThreadPoolExecutor(max_workers=6) as pool:
    agent_answers = map_progress(pool, generated_questions_info[:50], generate_agent_answer)

df_agent = pd.DataFrame(agent_answers)
print(df_agent["cost"].sum())
df_agent.to_csv("./data/agent-answers.csv", index=False)

