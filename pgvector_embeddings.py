'''
We should be running the code in this file once
to insert data into postgres. We are using pg db
as a container. And this is the command to do that
docker run -it \
    --name pgvector \
    -e POSTGRES_USER=user \
    -e POSTGRES_PASSWORD=pswd \
    -e POSTGRES_DB=faq \
    -v pgvector_data:/var/lib/postgresql/data \
    -p 5432:5432 \
    pgvector/pgvector:pg17

After that, we should use the agent in main.py to 
get our questions answered.
'''
import psycopg
from tqdm.auto import tqdm

from ingest import load_faq_data
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = load_faq_data()
texts = [doc["question"] + " " + doc["answer"] for doc in documents]

batch_size = 50
vectors = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)


conn = psycopg.connect(
    "postgresql://user:pswd@localhost:5432/faq"
)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

conn.execute("""
    DROP TABLE IF EXISTS documents
""")

conn.execute("""
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        course TEXT,
        section TEXT,
        question TEXT,
        answer TEXT,
        embedding vector(384) --  stores our 384-dimensional embeddings from all-MiniLM-L6-v2
    )
""")

def vec_to_str(vector):
    return "[" + ",".join(str(x) for x in vector) + "]"

for doc, vec in tqdm(zip(documents, vectors), total=len(documents)):
    conn.execute(
        """
        INSERT INTO documents (course, section, question, answer, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
        """,
        (doc["course"], doc["section"], doc["question"], doc["answer"],
         vec_to_str(vec))
    )

conn.commit()

query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)
query_str = vec_to_str(query_vector)

results = conn.execute(
    """
    SELECT course, question, answer,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    WHERE course = %s
    ORDER BY embedding <=> %s::vector
    LIMIT 5
    """,
    (query_str, "llm-zoomcamp", query_str)
).fetchall()

for row in results:
    print(f"[{row[0]}] {row[1]} (similarity: {row[3]:.4f})")

# The above process runs brute-force search,
# comparing our query against every row. 
# We can build an index on embeddings to make our search faster,
# at the cost of a small accuracy trade-off.
conn.execute("""
    CREATE INDEX ON documents
    USING hnsw (embedding vector_cosine_ops)
""")