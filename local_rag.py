from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import torch.nn.functional as F
import numpy as np
import re
import json


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


chunks = create_chunks(
    document,
    sentences_per_chunk=2,
    overlap_sentences=1
)


# ==================================================
# 4) EMBEDDING MODEL
# ==================================================

print("Embedding modeli yükleniyor...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==================================================
# 5) DOCUMENT EMBEDDINGS
# ==================================================

chunk_embeddings = embedding_model.encode(
    chunks,
    normalize_embeddings=True
)


# ==================================================
# 6) VECTOR INDEX
# ==================================================

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

def retrieve(query, top_k=3):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )

    distances, indices = index.kneighbors(
        query_embedding,
        n_neighbors=min(
            top_k,
            len(chunks)
        )
    )

    results = []

    for distance, idx in zip(
        distances[0],
        indices[0]
    ):

        similarity = 1 - distance

        results.append(
            {
                "text": chunks[idx],
                "score": float(similarity)
            }
        )

    return results


# ==================================================
# 8) LOCAL LLM
# ==================================================

print("Local LLM yükleniyor...")

llm_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    llm_name
)

llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype="auto"
)


# ==================================================
# 9) VERIFIER MODEL
# ==================================================

print("Verifier modeli yükleniyor...")

verifier = CrossEncoder(
    "cross-encoder/nli-deberta-v3-base"
)


# ==================================================
# 10) EVIDENCE CANDIDATES
# ==================================================

def create_evidence_candidates(
    retrieved_chunks
):

    evidence_candidates = []

    seen = set()


    for result in retrieved_chunks:

        chunk_text = result["text"]


        # ------------------------------------------
        # FULL CHUNK
        # ------------------------------------------

        if chunk_text not in seen:

            evidence_candidates.append(
                {
                    "text": chunk_text,
                    "type": "full_chunk",
                    "retrieval_score": result["score"]
                }
            )

            seen.add(
                chunk_text
            )


        # ------------------------------------------
        # SENTENCES
        # ------------------------------------------

        sentences = split_into_sentences(
            chunk_text
        )

        for sentence in sentences:

            if sentence not in seen:

                evidence_candidates.append(
                    {
                        "text": sentence,
                        "type": "sentence",
                        "retrieval_score": result["score"]
                    }
                )

                seen.add(
                    sentence
                )


    return evidence_candidates


# ==================================================
# 11) CLAIM → EVIDENCE SIMILARITY
# ==================================================

def calculate_evidence_similarities(
    claim,
    evidence_candidates
):

    claim_embedding = embedding_model.encode(
        [claim],
        normalize_embeddings=True
    )[0]


    evidence_texts = [
        evidence["text"]
        for evidence
        in evidence_candidates
    ]


    evidence_embeddings = embedding_model.encode(
        evidence_texts,
        normalize_embeddings=True
    )


    # Normalize ettiğimiz için dot product
    # doğrudan cosine similarity olur.

    similarities = np.dot(
        evidence_embeddings,
        claim_embedding
    )


    return similarities


# ==================================================
# 12) CLAIM VERIFICATION
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


    verification_results = []


    # Evidence önce claim ile yeterince ilgili olmalı.
    EVIDENCE_RELEVANCE_THRESHOLD = 0.35


    for claim in claims:

        # ------------------------------------------
        # A) SEMANTIC RELEVANCE
        # ------------------------------------------

        similarities = calculate_evidence_similarities(
            claim,
            evidence_candidates
        )


        relevant_evidence = []


        for evidence, similarity in zip(
            evidence_candidates,
            similarities
        ):

            if similarity >= EVIDENCE_RELEVANCE_THRESHOLD:

                relevant_evidence.append(
                    {
                        **evidence,

                        "evidence_similarity":
                            float(similarity)
                    }
                )


        # ------------------------------------------
        # Hiç alakalı evidence yoksa direkt reject
        # ------------------------------------------

        if len(relevant_evidence) == 0:

            verification_results.append(
                {
                    "claim": claim,

                    "supported": False,

                    "reason":
                        "no_relevant_evidence",

                    "evidence": None,

                    "evidence_type": None,

                    "evidence_similarity": 0.0,

                    "entailment": 0.0,

                    "contradiction": 0.0,

                    "neutral": 1.0
                }
            )

            continue


        # ------------------------------------------
        # B) NLI
        # ------------------------------------------

        pairs = [

            (
                evidence["text"],
                claim
            )

            for evidence
            in relevant_evidence
        ]


        scores = verifier.predict(
            pairs
        )


        scores_tensor = torch.tensor(
            scores,
            dtype=torch.float32
        )


        probabilities = F.softmax(
            scores_tensor,
            dim=1
        )


        # ------------------------------------------
        # C) EN YÜKSEK ENTAILMENT
        # ------------------------------------------

        entailment_scores = probabilities[
            :,
            1
        ]


        best_index = torch.argmax(
            entailment_scores
        ).item()


        best_probabilities = probabilities[
            best_index
        ]


        best_evidence = relevant_evidence[
            best_index
        ]


        verification_results.append(
            {
                "claim": claim,

                "evidence":
                    best_evidence["text"],

                "evidence_type":
                    best_evidence["type"],

                "retrieval_score":
                    best_evidence[
                        "retrieval_score"
                    ],

                "evidence_similarity":
                    best_evidence[
                        "evidence_similarity"
                    ],

                "contradiction":
                    best_probabilities[0].item(),

                "entailment":
                    best_probabilities[1].item(),

                "neutral":
                    best_probabilities[2].item(),

                "supported": None,

                "reason": None
            }
        )


    return verification_results


# ==================================================
# 13) USER QUESTION
# ==================================================

question = input(
    "\nSorunu yaz:\n> "
)


# ==================================================
# 14) RETRIEVAL
# ==================================================

retrieved_chunks = retrieve(
    question,
    top_k=3
)


print(
    "\nRETRIEVE EDİLEN CHUNK'LAR:"
)


for i, result in enumerate(
    retrieved_chunks,
    start=1
):

    print(
        f"\n{i}. skor = "
        f"{result['score']:.4f}"
    )

    print(
        result["text"]
    )


# ==================================================
# 15) RETRIEVAL GATE
# ==================================================

SIMILARITY_THRESHOLD = 0.50


best_score = retrieved_chunks[0][
    "score"
]


if best_score < SIMILARITY_THRESHOLD:

    final_output = {

        "status": "not_supported",

        "answer": (
            "I do not have enough information "
            "in the provided documents to answer this question."
        ),

        "reason":
            "retrieval_similarity_too_low",

        "retrieval_score":
            round(
                best_score,
                4
            ),

        "evidence": [],

        "confidence": 0.0
    }


    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL STRUCTURED OUTPUT:"
    )

    print(
        json.dumps(
            final_output,
            indent=4,
            ensure_ascii=False
        )
    )

    raise SystemExit


# ==================================================
# 16) GENERATION CONTEXT
# ==================================================

context = "\n\n".join(
    result["text"]
    for result
    in retrieved_chunks
)


# ==================================================
# 17) PROMPT
# ==================================================

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


# ==================================================
# 18) CHAT TEMPLATE
# ==================================================

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)


inputs = tokenizer(
    prompt,
    return_tensors="pt"
)


# ==================================================
# 19) GENERATION
# ==================================================

with torch.no_grad():

    outputs = llm.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False
    )


generated_tokens = outputs[0][
    inputs["input_ids"].shape[1]:
]


candidate_answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
).strip()


# ==================================================
# 20) LLM ADAY CEVABI
# ==================================================

print(
    "\n" + "=" * 80
)

print(
    "LLM ADAY CEVABI:"
)

print(
    candidate_answer
)


# ==================================================
# 21) CLAIM VERIFICATION
# ==================================================

verification_results = verify_claims(
    retrieved_chunks,
    candidate_answer
)


# ==================================================
# 22) CLAIM GATE
# ==================================================

ENTAILMENT_THRESHOLD = 0.70

EVIDENCE_RELEVANCE_THRESHOLD = 0.35


for result in verification_results:

    evidence_similarity = result.get(
        "evidence_similarity",
        0.0
    )

    entailment = result.get(
        "entailment",
        0.0
    )


    result["supported"] = (

        evidence_similarity
        >= EVIDENCE_RELEVANCE_THRESHOLD

        and

        entailment
        >= ENTAILMENT_THRESHOLD
    )


    if result["supported"]:

        result["reason"] = "supported"

    elif evidence_similarity < EVIDENCE_RELEVANCE_THRESHOLD:

        result["reason"] = (
            "evidence_not_relevant_enough"
        )

    else:

        result["reason"] = (
            "evidence_does_not_entail_claim"
        )


# ==================================================
# 23) DEBUG OUTPUT
# ==================================================

print(
    "\nCLAIM VERIFICATION:"
)


for i, result in enumerate(
    verification_results,
    start=1
):

    print(
        "\n" + "-" * 80
    )

    print(
        f"CLAIM {i}:"
    )

    print(
        result["claim"]
    )


    print(
        "\nBEST EVIDENCE:"
    )

    print(
        result["evidence"]
    )


    print(
        f"\nEVIDENCE TYPE: "
        f"{result['evidence_type']}"
    )


    print(
        "\nRELEVANCE:"
    )

    print(
        f"Evidence similarity: "
        f"{result['evidence_similarity']:.4f}"
    )


    print(
        "\nNLI:"
    )

    print(
        f"Entailment:    "
        f"{result['entailment']:.4f}"
    )

    print(
        f"Contradiction: "
        f"{result['contradiction']:.4f}"
    )

    print(
        f"Neutral:       "
        f"{result['neutral']:.4f}"
    )


    print(
        "\nCLAIM STATUS:"
    )

    print(
        "SUPPORTED"
        if result["supported"]
        else "NOT SUPPORTED"
    )


# ==================================================
# 24) FINAL VERIFICATION
# ==================================================

all_claims_supported = (

    len(verification_results) > 0

    and all(

        result["supported"]

        for result
        in verification_results
    )
)


# ==================================================
# 25) STRUCTURED OUTPUT
# ==================================================

if all_claims_supported:

    evidence_list = []


    for result in verification_results:

        evidence_list.append(
            {
                "claim":
                    result["claim"],

                "evidence":
                    result["evidence"],

                "evidence_similarity":
                    round(
                        result[
                            "evidence_similarity"
                        ],
                        4
                    ),

                "entailment":
                    round(
                        result["entailment"],
                        4
                    )
            }
        )


    # En zayıf claim sistemin genel güvenini belirlesin

    confidence = min(

        result["entailment"]

        for result
        in verification_results
    )


    final_output = {

        "status": "supported",

        "answer":
            candidate_answer,

        "evidence":
            evidence_list,

        "confidence":
            round(
                confidence,
                4
            )
    }


else:

    failed_claims = []


    for result in verification_results:

        if not result["supported"]:

            failed_claims.append(
                {
                    "claim":
                        result["claim"],

                    "best_evidence":
                        result["evidence"],

                    "evidence_similarity":
                        round(
                            result[
                                "evidence_similarity"
                            ],
                            4
                        ),

                    "entailment":
                        round(
                            result["entailment"],
                            4
                        ),

                    "reason":
                        result["reason"]
                }
            )


    final_output = {

        "status": "not_supported",

        "answer": (
            "The provided documents do not contain "
            "enough verified information to answer "
            "this question reliably."
        ),

        "failed_claims":
            failed_claims,

        "confidence":
            0.0
    }


# ==================================================
# 26) FINAL OUTPUT
# ==================================================

print(
    "\n" + "=" * 80
)

print(
    "FINAL STRUCTURED OUTPUT:"
)

print(
    json.dumps(
        final_output,
        indent=4,
        ensure_ascii=False
    )
)