from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import re


document = """
Customers can return products within 30 days after delivery.
The product must be unused and in its original packaging.
Refunds are processed within 5 business days after the returned product is received.

Standard shipping usually takes between 3 and 5 business days.
Express shipping is available and normally takes 1 to 2 business days.
Shipping times may be longer during holidays.

All electronic products include a two-year warranty.
The warranty covers manufacturing defects but does not cover accidental damage
or damage caused by improper use.

Premium members receive free standard shipping on all orders.
They also receive early access to discounts and special promotions.
Premium membership can be cancelled at any time.
"""


# --------------------------------------------------
# 1) SENTENCE SPLITTING
# --------------------------------------------------

def split_into_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# --------------------------------------------------
# 2) SENTENCE-BASED CHUNKING
# --------------------------------------------------

def create_sentence_chunks(
    text,
    sentences_per_chunk=2,
    overlap_sentences=1
):
    sentences = split_into_sentences(text)

    chunks = []

    start = 0

    while start < len(sentences):

        end = start + sentences_per_chunk

        selected_sentences = sentences[start:end]

        chunk = " ".join(selected_sentences)

        chunks.append(chunk)

        start = end - overlap_sentences

    return chunks


# --------------------------------------------------
# 3) CHUNKS
# --------------------------------------------------

chunks = create_sentence_chunks(
    document,
    sentences_per_chunk=2,
    overlap_sentences=1
)


print("\nOLUŞTURULAN CHUNK'LAR:")

for i, chunk in enumerate(chunks):
    print("\n" + "=" * 80)
    print(f"CHUNK {i}")
    print(chunk)


# --------------------------------------------------
# 4) EMBEDDING MODEL
# --------------------------------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 5) EMBEDDINGS
# --------------------------------------------------

chunk_embeddings = model.encode(
    chunks
)


# --------------------------------------------------
# 6) INDEX
# --------------------------------------------------

index = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

index.fit(
    chunk_embeddings
)


# --------------------------------------------------
# 7) QUERY
# --------------------------------------------------

query = "Can I return an opened product after 20 days?"

query_embedding = model.encode(
    [query]
)


# --------------------------------------------------
# 8) RETRIEVAL
# --------------------------------------------------

distances, indices = index.kneighbors(
    query_embedding,
    n_neighbors=3
)


# --------------------------------------------------
# 9) RESULTS
# --------------------------------------------------

print("\nEN İLGİLİ CHUNK'LAR:")

for rank, (distance, idx) in enumerate(
    zip(distances[0], indices[0]),
    start=1
):
    similarity = 1 - distance

    print("\n" + "=" * 80)

    print(f"SONUÇ {rank}")
    print(f"Similarity: {similarity:.4f}")
    print("Chunk:")
    print(chunks[idx])