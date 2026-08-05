import json
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

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

# Let us test the generation with one document
faq = llm_zc_faqs[0]
print(faq["id"], faq["question"], faq["answer"])
user_prompt = json.dumps(faq)

messages = [
    {"role": "developer", "content": data_gen_instructions},
    {"role": "user", "content": user_prompt}
]

response = llm_client.responses.parse(
    model = "gpt-5.4-mini",
    input = messages,
    text_format=Question
)

print(response.output_parsed.questions)