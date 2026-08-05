import json
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

from evaluation_utils import llm_structured_retry, map_progress, calc_total_price
from ingest import load_faq_data


load_dotenv()

faq_documents = load_faq_data()

llm_zc_faqs = []

for doc in faq_documents:
    if doc["course"] == "llm-zoomcamp":
        llm_zc_faqs.append(doc)

print(len(llm_zc_faqs))

class Question(BaseModel):
    questions: list[str]

data_gen_instructions = """
You emulate a student who's taking our course.
Formulate 5 questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record.

The output should resemble how people ask questions
on the internet. Not too formal, not too short, not too long.
""".strip()

llm_client = OpenAI()

def generate_ground_truth(doc):

    user_prompt = json.dumps(doc)

    output, usage = llm_structured_retry(
        client=llm_client,
        instructions=data_gen_instructions,
        user_prompt=user_prompt,
        output_type=Question
    )

    results = []

    for question in output.questions:
        results.append({"question": question, "document": doc["id"]})

    return results, usage

with ThreadPoolExecutor(max_workers=6) as pool:
    results = map_progress(pool, llm_zc_faqs, generate_ground_truth)

ground_truth = []
usages = []

for records, usage in results:
    ground_truth.extend(records)
    usages.append(usage)

print(len(ground_truth))
print(calc_total_price(usages))

df_ground_truth = pd.DataFrame(ground_truth)
df_ground_truth.to_csv("./data/ground_truth-new.csv", index=False)