from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import re


# ==================================================
# 1) BİLGİ KAYNAĞI
# ==================================================

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


# ==================================================
# 2) SENTENCE SPLITTING
# ==================================================

def split_into_sentences(text):

    text = re.sub(
        r"\s+",
        " ",
        text.strip()
    )

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


# ==================================================
# 3) CHUNKING
# ==================================================

def create_chunks(
    text,
    sentences_per_chunk=2,
    overlap_sentences=1
):

    sentences = split_into_sentences(text)

    chunks = []

    start = 0

    while start < len(sentences):

        end = start + sentences_per_chunk

        selected = sentences[start:end]

        chunk = " ".join(selected)

        chunks.append(chunk)

        start = end - overlap_sentences

    return chunks


chunks = create_chunks(document)


# ==================================================
# 4) EMBEDDING MODEL
# ==================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# 5) VECTOR INDEX
# ==================================================

chunk_embeddings = model.encode(
    chunks
)

index = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

index.fit(
    chunk_embeddings
)


# ==================================================
# 6) EN İYİ RETRIEVAL SKORUNU BUL
# ==================================================

def get_best_score(query):

    query_embedding = model.encode(
        [query]
    )

    distances, indices = index.kneighbors(
        query_embedding,
        n_neighbors=1
    )

    best_distance = distances[0][0]

    best_score = 1 - best_distance

    return best_score


# ==================================================
# 7) GELİŞTİRİLMİŞ EVAL SET
# ==================================================

eval_set = [

    # --------------------------------------------------
    # ANSWERABLE - NORMAL
    # --------------------------------------------------

    {
        "question": "How many days do I have to return a product?",
        "answerable": True
    },

    {
        "question": "How long does standard shipping take?",
        "answerable": True
    },

    {
        "question": "How long is the warranty for electronics?",
        "answerable": True
    },

    {
        "question": "Do premium members get free shipping?",
        "answerable": True
    },


    # --------------------------------------------------
    # ANSWERABLE - PARAPHRASE
    # --------------------------------------------------

    {
        "question": "Can I send my order back after three weeks?",
        "answerable": True
    },

    {
        "question": "When should I expect normal delivery to arrive?",
        "answerable": True
    },

    {
        "question": "How many years are electronic devices protected?",
        "answerable": True
    },

    {
        "question": "Is shipping free if I have a premium account?",
        "answerable": True
    },

    {
        "question": "How quickly will I receive my money after returning an item?",
        "answerable": True
    },


    # --------------------------------------------------
    # NOT ANSWERABLE - EASY
    # --------------------------------------------------

    {
        "question": "Can I pay with Bitcoin?",
        "answerable": False
    },

    {
        "question": "Do you have physical stores?",
        "answerable": False
    },

    {
        "question": "Can I pay with PayPal?",
        "answerable": False
    },


    # --------------------------------------------------
    # NOT ANSWERABLE - HARD NEGATIVES
    # --------------------------------------------------

    {
        "question": "Do premium members get a student discount?",
        "answerable": False
    },

    {
        "question": "Can premium members get free express shipping?",
        "answerable": False
    },

    {
        "question": "Does the warranty cover stolen electronics?",
        "answerable": False
    },

    {
        "question": "Can I return a used product after 10 days?",
        "answerable": False
    },

    {
        "question": "Do you offer same-day delivery?",
        "answerable": False
    },

    {
        "question": "Can I change my shipping address after ordering?",
        "answerable": False
    }
]


# ==================================================
# 8) TEST EDECEĞİMİZ THRESHOLD'LAR
# ==================================================

thresholds = [
    0.20,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


# ==================================================
# 9) EVALUATION
# ==================================================

for threshold in thresholds:

    correct = 0

    true_positive = 0
    true_negative = 0

    false_positive = 0
    false_negative = 0

    print("\n" + "=" * 100)

    print(
        f"THRESHOLD = {threshold}"
    )

    print("=" * 100)


    for item in eval_set:

        question = item["question"]

        expected = item["answerable"]

        score = get_best_score(
            question
        )

        predicted = (
            score >= threshold
        )


        # --------------------------------------------------
        # DOĞRU / YANLIŞ
        # --------------------------------------------------

        if predicted == expected:

            correct += 1


        # --------------------------------------------------
        # CONFUSION MATRIX
        # --------------------------------------------------

        if predicted and expected:

            true_positive += 1

        elif not predicted and not expected:

            true_negative += 1

        elif predicted and not expected:

            false_positive += 1

        elif not predicted and expected:

            false_negative += 1


        # --------------------------------------------------
        # HER SORUNUN SONUCUNU GÖSTER
        # --------------------------------------------------

        print(
            f"{score:.4f} | "
            f"Beklenen: {str(expected):5} | "
            f"Tahmin: {str(predicted):5} | "
            f"{question}"
        )


    # ==================================================
    # 10) METRİKLER
    # ==================================================

    accuracy = (
        correct / len(eval_set)
    )


    precision = (
        true_positive /
        (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )


    recall = (
        true_positive /
        (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )


    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0
    )


    # ==================================================
    # 11) SONUÇLAR
    # ==================================================

    print("\nSONUÇ:")

    print(
        f"Accuracy:  {accuracy:.2%}"
    )

    print(
        f"Precision: {precision:.2%}"
    )

    print(
        f"Recall:    {recall:.2%}"
    )

    print(
        f"F1 Score:  {f1:.2%}"
    )

    print()

    print(
        f"True Positive:  {true_positive}"
    )

    print(
        f"True Negative:  {true_negative}"
    )

    print(
        f"False Positive: {false_positive}"
    )

    print(
        f"False Negative: {false_negative}"
    )