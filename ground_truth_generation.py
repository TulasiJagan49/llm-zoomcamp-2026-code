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
You are generating search queries for evaluating a search engine.

The FAQ record answers the user's question.

Generate FIVE queries from FIVE different users.

1. Beginner who doesn't know the terminology.
2. Experienced engineer.
3. Someone frustrated because something failed.
4. Someone searching very quickly (2-5 words).
5. Someone writing a long natural-language question.

Rules:

- Never copy sentences from the FAQ.
- Avoid unique nouns whenever possible.
- Use synonyms.
- Leave out details that a real user wouldn't know.
- Sometimes describe the problem instead of asking directly.
- Include one query with a typo or abbreviation.
- Make each query look like something copied from real search logs.

Return only the five questions.
"""

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