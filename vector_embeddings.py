from sentence_transformers import SentenceTransformer

from ingest import load_faq_data

model = SentenceTransformer("all-MiniLM-L6-v2")

# q1 = "I just discovered about the course, can I join now?"

# v1 = model.encode(q1)

# q2 = "Can I still enroll to this course, even if it has started?"

# v2 = model.encode(q2)

# q3 = "How do I install Docker locally?"

# v3 = model.encode(q3)

# a  = "You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."
# av = model.encode(a)

# print(v1.dot(v2), v1.dot(v3), av.dot(v1), av.dot(v2), av.dot(v3))

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

print(len(faqs_as_vectors), faqs_as_vectors[10].shape)

