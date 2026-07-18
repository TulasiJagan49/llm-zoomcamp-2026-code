import numpy as np

from sentence_transformers import SentenceTransformer
from sqlitesearch import VectorSearchIndex
from tqdm.auto import tqdm

from ingest import load_faq_data

model = SentenceTransformer("all-MiniLM-L6-v2")

faq_info = load_faq_data()

def process_data():   

    faq_texts = []

    for doc in faq_info:
        text = doc["question"] + " " +doc["answer"]
        faq_texts.append(text)
    
    return faq_texts

def embed_data(faq_texts):

    batch_size = 50
    faqs_as_vectors = []

    for i in range(0, len(faq_texts), batch_size):
        batch = faq_texts[i:i+batch_size]
        faqs_as_vectors.extend(model.encode(batch))
    
    np_vectors = np.array(faqs_as_vectors)

    return np_vectors

def build_index(np_vectors, faq_info):

    vector_index = VectorSearchIndex(
        mode='ivf',
        keyword_fields=["course"],
        db_path="faq_vectors.db"
    )
    vector_index.fit(np_vectors, faq_info)

    return vector_index

def search(vector_index):

    query = "Can I still join the course after the start date?"
    query_vector = model.encode(query)
    results = vector_index.search(query_vector=query_vector, filter_dict={"course": "llm-zoomcamp"}, num_results=5)

    print(results)


if __name__=="__main__":
    # If the data is loaded, otherwise run the boilerplate code
    # to load it.
    vector_index = VectorSearchIndex(
        mode='ivf',
        keyword_fields=["course"],
        db_path="faq_vectors.db"
    )
    search()

