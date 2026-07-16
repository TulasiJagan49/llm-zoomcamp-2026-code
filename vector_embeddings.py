import numpy as np

from sentence_transformers import SentenceTransformer
from minsearch import VectorSearch

from ingest import load_faq_data

model = SentenceTransformer("all-MiniLM-L6-v2")

faq_info = load_faq_data()

faq_texts = []

for doc in faq_info:
    text = doc["question"] + " " +doc["answer"]
    faq_texts.append(text)

from tqdm.auto import tqdm

batch_size = 50
faqs_as_vectors = []

for i in range(0, len(faq_texts), batch_size):
    batch = faq_texts[i:i+batch_size]
    faqs_as_vectors.extend(model.encode(batch))

# print(len(faqs_as_vectors), faqs_as_vectors[10].shape)

query = "Can I still join the course after the start date?"
query_vector = model.encode(query)

np_vectors = np.array(faqs_as_vectors)

# print(np_vectors.shape)

# scores = np_vectors.dot(query_vector)

# idx = np.argmax(scores)
# print(idx, scores[idx], faq_info[idx])

# top_five = np.argsort(-scores)[:5]

# for idx in top_five:
#     print(scores[idx])
#     print(faq_info[idx])

vector_index = VectorSearch(keyword_fields=["course"])
vector_index.fit(np_vectors, faq_info)

results = vector_index.search(query_vector=query_vector, filter_dict={"course": "llm-zoomcamp"}, num_results=5)

print(results)

