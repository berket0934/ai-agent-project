from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn.functional as F
import numpy as np
import re
import json


# ==================================================
# 1) CONFIG
# ==================================================

RETRIEVAL_THRESHOLD = 0.50
EVIDENCE_RELEVANCE_THRESHOLD = 0.35
ENTAILMENT_THRESHOLD = 0.70

TOP_K = 3


# ==================================================
# 2) BİLGİ KAYNAĞI
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
# 3) SENTENCE SPLITTING
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
# 4) CHUNKING
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

        chunks.append(
            " ".join(selected)
        )

        start = end - overlap_sentences

    return chunks


chunks = create_chunks(
    document
)


# ==================================================
# 5) MODELLER
# ==================================================

print("Embedding modeli yükleniyor...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print("Local LLM yükleniyor...")

llm_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    llm_name
)

llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype="auto"
)


print("Verifier modeli yükleniyor...")

verifier = CrossEncoder(
    "cross-encoder/nli-deberta-v3-base"
)


# ==================================================
# 6) VECTOR INDEX
# ==================================================

chunk_embeddings = embedding_model.encode(
    chunks,
    normalize_embeddings=True
)


index = NearestNeighbors(
    metric="cosine",
    algorithm="brute"
)

index.fit(
    chunk_embeddings
)


# ==================================================
# 7) RETRIEVAL
# ==================================================

def retrieve(query):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    distances, indices = index.kneighbors(
        query_embedding,
        n_neighbors=min(
            TOP_K,
            len(chunks)
        )
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        results.append(
            {
                "text": chunks[idx],
                "score": float(
                    1 - distance
                )
            }
        )

    return results


# ==================================================
# 8) EVIDENCE CANDIDATES
# ==================================================

def create_evidence_candidates(
    retrieved_chunks
):

    candidates = []

    seen = set()

    for result in retrieved_chunks:

        chunk_text = result["text"]


        # Full chunk
        if chunk_text not in seen:

            candidates.append(
                {
                    "text": chunk_text,
                    "type": "full_chunk",
                    "retrieval_score":
                        result["score"]
                }
            )

            seen.add(
                chunk_text
            )


        # Sentence level
        for sentence in split_into_sentences(
            chunk_text
        ):

            if sentence not in seen:

                candidates.append(
                    {
                        "text": sentence,
                        "type": "sentence",
                        "retrieval_score":
                            result["score"]
                    }
                )

                seen.add(
                    sentence
                )

    return candidates


# ==================================================
# 9) EVIDENCE SIMILARITY
# ==================================================

def evidence_similarities(
    claim,
    evidence_candidates
):

    claim_embedding = embedding_model.encode(
        [claim],
        normalize_embeddings=True
    )[0]


    evidence_texts = [
        item["text"]
        for item
        in evidence_candidates
    ]


    evidence_embeddings = embedding_model.encode(
        evidence_texts,
        normalize_embeddings=True
    )


    similarities = np.dot(
        evidence_embeddings,
        claim_embedding
    )

    return similarities


# ==================================================
# 10) CLAIM VERIFICATION
# ==================================================

def verify_claims(
    retrieved_chunks,
    answer
):

    claims = split_into_sentences(
        answer
    )


    evidence_candidates = create_evidence_candidates(
        retrieved_chunks
    )


    results = []


    for claim in claims:

        similarities = evidence_similarities(
            claim,
            evidence_candidates
        )


        relevant_evidence = []


        for evidence, similarity in zip(
            evidence_candidates,
            similarities
        ):

            if (
                similarity
                >= EVIDENCE_RELEVANCE_THRESHOLD
            ):

                relevant_evidence.append(
                    {
                        **evidence,
                        "evidence_similarity":
                            float(similarity)
                    }
                )


        # ------------------------------------------
        # Relevant evidence bulunamadı
        # ------------------------------------------

        if len(relevant_evidence) == 0:

            results.append(
                {
                    "claim": claim,
                    "supported": False,
                    "reason":
                        "no_relevant_evidence",
                    "evidence": None,
                    "evidence_similarity": 0.0,
                    "entailment": 0.0
                }
            )

            continue


        # ------------------------------------------
        # NLI
        # ------------------------------------------

        pairs = [
            (
                evidence["text"],
                claim
            )
            for evidence
            in relevant_evidence
        ]


        logits = verifier.predict(
            pairs
        )


        probabilities = F.softmax(
            torch.tensor(
                logits,
                dtype=torch.float32
            ),
            dim=1
        )


        entailment_scores = probabilities[
            :,
            1
        ]


        best_index = torch.argmax(
            entailment_scores
        ).item()


        best_evidence = relevant_evidence[
            best_index
        ]


        best_probs = probabilities[
            best_index
        ]


        entailment = best_probs[
            1
        ].item()


        supported = (
            best_evidence[
                "evidence_similarity"
            ]
            >= EVIDENCE_RELEVANCE_THRESHOLD

            and

            entailment
            >= ENTAILMENT_THRESHOLD
        )


        results.append(
            {
                "claim": claim,

                "supported":
                    supported,

                "evidence":
                    best_evidence["text"],

                "evidence_similarity":
                    best_evidence[
                        "evidence_similarity"
                    ],

                "entailment":
                    entailment,

                "contradiction":
                    best_probs[0].item(),

                "neutral":
                    best_probs[2].item(),

                "reason":
                    (
                        "supported"
                        if supported
                        else
                        "evidence_does_not_entail_claim"
                    )
            }
        )


    return results


# ==================================================
# 11) GENERATION
# ==================================================

def generate_answer(
    question,
    retrieved_chunks
):

    context = "\n\n".join(
        item["text"]
        for item
        in retrieved_chunks
    )


    messages = [

        {
            "role": "system",

            "content": (
                "Answer the user's question using only "
                "the provided context. "

                "Do not make assumptions beyond the context. "

                "Do not infer that something is false simply "
                "because it is not mentioned. "

                "If the requested information is not explicitly "
                "provided, say that it is not specified. "

                "Keep factual wording close to the context. "

                "Do not invent policies, benefits, restrictions, "
                "prices, shipping options, or conditions. "

                "Be precise and concise."
            )
        },

        {
            "role": "user",

            "content": f"""
CONTEXT:

{context}

QUESTION:

{question}
"""
        }
    ]


    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )


    with torch.no_grad():

        outputs = llm.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )


    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]


    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    return answer


# ==================================================
# 12) FULL RAG PIPELINE
# ==================================================

def run_rag(question):

    retrieved = retrieve(
        question
    )


    best_retrieval_score = retrieved[
        0
    ]["score"]


    # ----------------------------------------------
    # Retrieval Gate
    # ----------------------------------------------

    if (
        best_retrieval_score
        < RETRIEVAL_THRESHOLD
    ):

        return {
            "status": "not_supported",

            "answer": (
                "I do not have enough information "
                "in the provided documents."
            ),

            "reason":
                "retrieval_similarity_too_low",

            "retrieval_score":
                best_retrieval_score,

            "claims": []
        }


    # ----------------------------------------------
    # Generation
    # ----------------------------------------------

    answer = generate_answer(
        question,
        retrieved
    )


    # ----------------------------------------------
    # Verification
    # ----------------------------------------------

    claim_results = verify_claims(
        retrieved,
        answer
    )


    all_supported = (

        len(claim_results) > 0

        and

        all(
            claim["supported"]
            for claim
            in claim_results
        )
    )


    if all_supported:

        status = "supported"

    else:

        status = "not_supported"


    return {

        "status":
            status,

        "answer":
            answer,

        "retrieval_score":
            best_retrieval_score,

        "claims":
            claim_results
    }


# ==================================================
# 13) FULL PIPELINE EVAL SET
# ==================================================

eval_set = [

    # ==================================================
    # SUPPORTED / ANSWERABLE
    # ==================================================

    {
        "question":
            "How many days do I have to return a product?",

        "expected_supported":
            True
    },

    {
        "question":
            "How long does standard shipping take?",

        "expected_supported":
            True
    },

    {
        "question":
            "How long does express shipping take?",

        "expected_supported":
            True
    },

    {
        "question":
            "How long is the warranty for electronics?",

        "expected_supported":
            True
    },

    {
        "question":
            "Does the warranty cover accidental damage?",

        "expected_supported":
            True
    },

    {
        "question":
            "Do premium members get free standard shipping?",

        "expected_supported":
            True
    },

    {
        "question":
            "Can I send my order back after three weeks?",

        "expected_supported":
            True
    },

    {
        "question":
            "When will I receive my refund after you get my return?",

        "expected_supported":
            True
    },

    {
        "question":
            "Can premium membership be cancelled anytime?",

        "expected_supported":
            True
    },

    {
        "question":
            "Can shipping take longer during holidays?",

        "expected_supported":
            True
    },

    # IMPORTANT:
    # Cevabın "hayır" olması bu soruyu
    # unsupported yapmaz.
    #
    # Doküman "product must be unused"
    # dediği için cevap verilebilir.

    {
        "question":
            "Can I return a used product after 10 days?",

        "expected_supported":
            True
    },


    # ==================================================
    # NOT SUPPORTED
    # ==================================================

    {
        "question":
            "Can I pay with Bitcoin?",

        "expected_supported":
            False
    },

    {
        "question":
            "Can I pay with PayPal?",

        "expected_supported":
            False
    },

    {
        "question":
            "Do you have physical stores?",

        "expected_supported":
            False
    },

    {
        "question":
            "Do premium members get a student discount?",

        "expected_supported":
            False
    },

    {
        "question":
            "Can premium members get free express shipping?",

        "expected_supported":
            False
    },

    {
        "question":
            "Does the warranty cover stolen electronics?",

        "expected_supported":
            False
    },

    {
        "question":
            "Do you offer same-day delivery?",

        "expected_supported":
            False
    },

    {
        "question":
            "Can I change my delivery address after ordering?",

        "expected_supported":
            False
    },

    {
        "question":
            "Do you ship internationally?",

        "expected_supported":
            False
    },

    {
        "question":
            "Can I extend the electronics warranty to three years?",

        "expected_supported":
            False
    },

    {
        "question":
            "Do refunds go back to the original payment method?",

        "expected_supported":
            False
    }
]


# ==================================================
# 14) RUN EVALUATION
# ==================================================

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0


all_results = []


print(
    "\n\nFULL RAG EVALUATION BAŞLIYOR..."
)


for number, item in enumerate(
    eval_set,
    start=1
):

    question = item[
        "question"
    ]


    expected = item[
        "expected_supported"
    ]


    result = run_rag(
        question
    )


    predicted = (
        result["status"]
        == "supported"
    )


    # ----------------------------------------------
    # Confusion Matrix
    # ----------------------------------------------

    if predicted and expected:

        true_positive += 1

        classification = "TP"


    elif (
        not predicted
        and
        not expected
    ):

        true_negative += 1

        classification = "TN"


    elif (
        predicted
        and
        not expected
    ):

        false_positive += 1

        classification = "FP"


    else:

        false_negative += 1

        classification = "FN"


    print(
        "\n" + "=" * 100
    )

    print(
        f"TEST {number}"
    )

    print(
        f"QUESTION: {question}"
    )

    print(
        f"EXPECTED: "
        f"{'SUPPORTED' if expected else 'NOT SUPPORTED'}"
    )

    print(
        f"PREDICTED: "
        f"{result['status'].upper()}"
    )

    print(
        f"CLASS: {classification}"
    )

    print(
        f"RETRIEVAL SCORE: "
        f"{result['retrieval_score']:.4f}"
    )

    print(
        f"ANSWER: {result['answer']}"
    )


    # ----------------------------------------------
    # En zayıf claim
    # ----------------------------------------------

    if result["claims"]:

        minimum_entailment = min(

            claim.get(
                "entailment",
                0
            )

            for claim
            in result["claims"]
        )


        print(
            f"MIN ENTAILMENT: "
            f"{minimum_entailment:.4f}"
        )


    # ----------------------------------------------
    # JSON için kaydet
    # ----------------------------------------------

    all_results.append(
        {
            "question":
                question,

            "expected_supported":
                expected,

            "predicted_supported":
                predicted,

            "classification":
                classification,

            "result":
                result
        }
    )


# ==================================================
# 15) METRICS
# ==================================================

total = len(
    eval_set
)


accuracy = (

    true_positive
    +
    true_negative

) / total


precision = (

    true_positive
    /
    (
        true_positive
        +
        false_positive
    )

    if (
        true_positive
        +
        false_positive
    ) > 0

    else 0
)


recall = (

    true_positive
    /
    (
        true_positive
        +
        false_negative
    )

    if (
        true_positive
        +
        false_negative
    ) > 0

    else 0
)


f1 = (

    2
    *
    precision
    *
    recall
    /
    (
        precision
        +
        recall
    )

    if (
        precision
        +
        recall
    ) > 0

    else 0
)


# ==================================================
# 16) FINAL REPORT
# ==================================================

print(
    "\n\n"
    + "=" * 100
)

print(
    "FULL PIPELINE EVAL SONUCU"
)

print(
    "=" * 100
)


print(
    f"\nTotal: {total}"
)

print(
    f"True Positive:  "
    f"{true_positive}"
)

print(
    f"True Negative:  "
    f"{true_negative}"
)

print(
    f"False Positive: "
    f"{false_positive}"
)

print(
    f"False Negative: "
    f"{false_negative}"
)


print(
    "\nMETRICS"
)

print(
    f"Accuracy:  "
    f"{accuracy:.2%}"
)

print(
    f"Precision: "
    f"{precision:.2%}"
)

print(
    f"Recall:    "
    f"{recall:.2%}"
)

print(
    f"F1 Score:  "
    f"{f1:.2%}"
)


# ==================================================
# 17) SAVE JSON REPORT
# ==================================================

report = {

    "config": {

        "retrieval_threshold":
            RETRIEVAL_THRESHOLD,

        "evidence_relevance_threshold":
            EVIDENCE_RELEVANCE_THRESHOLD,

        "entailment_threshold":
            ENTAILMENT_THRESHOLD,

        "top_k":
            TOP_K
    },

    "metrics": {

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "true_positive":
            true_positive,

        "true_negative":
            true_negative,

        "false_positive":
            false_positive,

        "false_negative":
            false_negative
    },

    "tests":
        all_results
}


with open(
    "full_rag_eval_results.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4,
        ensure_ascii=False
    )


print(
    "\nRapor kaydedildi:"
)

print(
    "full_rag_eval_results.json"
)