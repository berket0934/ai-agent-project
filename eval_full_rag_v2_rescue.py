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

        # Sentence-level
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
#
# BU KISIM V2 MANTIĞI:
# Relevant evidence içinden en yüksek
# ENTAILMENT skorunu seçiyoruz.
# ==================================================

def verify_claims(
    retrieved_chunks,
    claims
):

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
        # Relevant evidence yok
        # ------------------------------------------

        if len(relevant_evidence) == 0:

            results.append(
                {
                    "claim": claim,

                    "supported": False,

                    "reason":
                        "no_relevant_evidence",

                    "evidence": None,

                    "evidence_similarity":
                        0.0,

                    "entailment":
                        0.0,

                    "contradiction":
                        0.0,

                    "neutral":
                        1.0
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


        # ------------------------------------------
        # V2:
        # EN YÜKSEK ENTAILMENT
        # ------------------------------------------

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
                "claim":
                    claim,

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
# 12) NORMAL V2 ANSWER VERIFICATION
# ==================================================

def verify_generated_answer(
    retrieved_chunks,
    answer
):

    claims = split_into_sentences(
        answer
    )


    if len(claims) == 0:

        return {
            "supported":
                False,

            "claims":
                []
        }


    claim_results = verify_claims(
        retrieved_chunks,
        claims
    )


    all_supported = all(

        claim["supported"]

        for claim
        in claim_results
    )


    return {
        "supported":
            all_supported,

        "claims":
            claim_results
    }


# ==================================================
# 13) RESCUE HELPERS
#
# BURADAN SONRASI YENİ.
#
# V2'NİN ÇALIŞAN PIPELINE'I DEĞİŞMİYOR.
# Rescue SADECE V2 BAŞARISIZ OLURSA devreye giriyor.
# ==================================================


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60
}


def extract_duration_days(text):

    text = text.lower()


    word_pattern = "|".join(
        NUMBER_WORDS.keys()
    )


    pattern = (

        rf"(\d+|{word_pattern})"

        r"\s*"

        r"(day|days|week|weeks)"
    )


    match = re.search(
        pattern,
        text
    )


    if not match:

        return None


    number_text = match.group(1)

    unit = match.group(2)


    if number_text.isdigit():

        number = int(
            number_text
        )

    else:

        number = NUMBER_WORDS.get(
            number_text
        )


    if number is None:

        return None


    if unit in (
        "week",
        "weeks"
    ):

        days = number * 7

    else:

        days = number


    return {
        "original":
            match.group(0),

        "days":
            days
    }


# ==================================================
# 14) GET RETURN WINDOW FROM DOCUMENT
# ==================================================

def get_return_window_days():

    match = re.search(
        r"return products within\s+(\d+)\s+days",
        document,
        flags=re.IGNORECASE
    )


    if not match:

        return None


    return int(
        match.group(1)
    )


RETURN_WINDOW_DAYS = (
    get_return_window_days()
)


# ==================================================
# 15) IS RETURN QUESTION?
# ==================================================

def is_return_question(
    question
):

    q = question.lower()


    if "return" in q:

        return True


    if (
        "send" in q
        and
        "back" in q
    ):

        return True


    return False


# ==================================================
# 16) DETERMINISTIC RESCUE
#
# ÇOK DAR KAPSAMLI:
#
# 1) used product
# 2) week/day duration
#
# Diğer hiçbir soruya dokunmuyor.
# ==================================================

def deterministic_return_rescue(
    question,
    retrieved_chunks
):

    # ------------------------------------------
    # Return sorusu değilse rescue yok
    # ------------------------------------------

    if not is_return_question(
        question
    ):

        return None


    q = question.lower()


    # ==================================================
    # RESCUE CASE 1:
    # USED PRODUCT
    # ==================================================

    if re.search(
        r"\bused\b",
        q
    ):

        policy_claim = (
            "The product must be unused "
            "and in its original packaging."
        )


        verification = verify_claims(
            retrieved_chunks,
            [policy_claim]
        )


        if (
            len(verification) > 0
            and
            verification[0][
                "supported"
            ]
        ):

            return {
                "status":
                    "supported",

                "answer":
                    (
                        "No. The return policy requires "
                        "the product to be unused and "
                        "in its original packaging."
                    ),

                "reason":
                    "used_product_policy_rescue",

                "calculation":
                    None,

                "claims":
                    verification
            }


        return None


    # ==================================================
    # RESCUE CASE 2:
    # DURATION
    # ==================================================

    duration = extract_duration_days(
        question
    )


    if duration is None:

        return None


    if RETURN_WINDOW_DAYS is None:

        return None


    # Önce gerçek 30-day policy'i verifier ile doğrula.

    policy_claim = (

        f"Customers can return products within "
        f"{RETURN_WINDOW_DAYS} days after delivery."
    )


    verification = verify_claims(
        retrieved_chunks,
        [policy_claim]
    )


    if not (

        len(verification) > 0

        and

        verification[0][
            "supported"
        ]

    ):

        return None


    requested_days = duration[
        "days"
    ]


    inside_window = (

        requested_days
        <=
        RETURN_WINDOW_DAYS
    )


    # ------------------------------------------
    # Within window
    # ------------------------------------------

    if inside_window:

        answer = (

            f"{duration['original'].capitalize()} "
            f"is {requested_days} days, which is within "
            f"the {RETURN_WINDOW_DAYS}-day return window. "
            f"The other return conditions must also be satisfied."
        )


    # ------------------------------------------
    # Outside window
    # ------------------------------------------

    else:

        answer = (

            f"No. {duration['original'].capitalize()} "
            f"is {requested_days} days, which exceeds "
            f"the {RETURN_WINDOW_DAYS}-day return window."
        )


    return {
        "status":
            "supported",

        "answer":
            answer,

        "reason":
            "duration_policy_rescue",

        "calculation":
            {
                "input":
                    duration[
                        "original"
                    ],

                "normalized_days":
                    requested_days,

                "return_window_days":
                    RETURN_WINDOW_DAYS,

                "inside_window":
                    inside_window
            },

        "claims":
            verification
    }


# ==================================================
# 17) FULL V2 + RESCUE PIPELINE
# ==================================================

def run_rag(
    question
):

    # ==================================================
    # A) RETRIEVAL
    # ==================================================

    retrieved = retrieve(
        question
    )


    best_retrieval_score = retrieved[
        0
    ]["score"]


    # ==================================================
    # B) ORIGINAL V2 RETRIEVAL GATE
    #
    # Rescue burada ÇALIŞMIYOR.
    # V2 kararını aynen koruyoruz.
    # ==================================================

    if (
        best_retrieval_score
        < RETRIEVAL_THRESHOLD
    ):

        return {
            "status":
                "not_supported",

            "answer":
                (
                    "I do not have enough information "
                    "in the provided documents."
                ),

            "reason":
                "retrieval_similarity_too_low",

            "pipeline_stage":
                "retrieval_reject",

            "rescue_used":
                False,

            "retrieval_score":
                best_retrieval_score,

            "claims":
                []
        }


    # ==================================================
    # C) ORIGINAL V2 GENERATION
    # ==================================================

    answer = generate_answer(
        question,
        retrieved
    )


    # ==================================================
    # D) ORIGINAL V2 VERIFICATION
    # ==================================================

    verification = verify_generated_answer(
        retrieved,
        answer
    )


    # ==================================================
    # E) V2 SUCCESS
    #
    # Çalışan V2 cevabına dokunmuyoruz.
    # ==================================================

    if verification[
        "supported"
    ]:

        return {
            "status":
                "supported",

            "answer":
                answer,

            "reason":
                "normal_v2_supported",

            "pipeline_stage":
                "normal_v2_supported",

            "rescue_used":
                False,

            "retrieval_score":
                best_retrieval_score,

            "claims":
                verification[
                    "claims"
                ]
        }


    # ==================================================
    # F) RESCUE
    #
    # YENİ TEK NOKTA BURASI.
    #
    # V2 normalde NOT_SUPPORTED diyecekti.
    # Şimdi sadece iki spesifik return durumunu
    # kurtarmayı deniyoruz.
    # ==================================================

    rescue = deterministic_return_rescue(
        question,
        retrieved
    )


    if rescue is not None:

        return {
            "status":
                "supported",

            "answer":
                rescue[
                    "answer"
                ],

            "reason":
                rescue[
                    "reason"
                ],

            "pipeline_stage":
                "deterministic_rescue",

            "rescue_used":
                True,

            "retrieval_score":
                best_retrieval_score,

            "calculation":
                rescue[
                    "calculation"
                ],

            "original_v2_answer":
                answer,

            "claims":
                rescue[
                    "claims"
                ]
        }


    # ==================================================
    # G) ORIGINAL V2 REJECTION
    # ==================================================

    return {
        "status":
            "not_supported",

        "answer":
            (
                "The provided documents do not contain "
                "enough verified information to answer "
                "this question reliably."
            ),

        "reason":
            "normal_v2_verification_failed",

        "pipeline_stage":
            "normal_v2_reject",

        "rescue_used":
            False,

        "retrieval_score":
            best_retrieval_score,

        "original_v2_answer":
            answer,

        "claims":
            verification[
                "claims"
            ]
    }


# ==================================================
# 18) EVAL SET
# ==================================================

eval_set = [

    # ==================================================
    # SUPPORTED
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
# 19) EVALUATION
# ==================================================

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

rescue_success_count = 0

all_results = []


print(
    "\n"
    + "=" * 100
)

print(
    "V2 BASELINE + DETERMINISTIC RESCUE EVALUATION"
)

print(
    "=" * 100
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
        ==
        "supported"
    )


    # ------------------------------------------
    # Confusion Matrix
    # ------------------------------------------

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


    # ------------------------------------------
    # Rescue counter
    # ------------------------------------------

    if result[
        "rescue_used"
    ]:

        rescue_success_count += 1


    # ------------------------------------------
    # Error Type
    # ------------------------------------------

    error_type = None


    if classification == "FP":

        error_type = (
            "safety_failure_false_positive"
        )


    elif classification == "FN":

        if (
            result[
                "pipeline_stage"
            ]
            ==
            "retrieval_reject"
        ):

            error_type = (
                "retrieval_false_negative"
            )

        else:

            error_type = (
                "generation_or_verification_false_negative"
            )


    # ==================================================
    # PRINT TEST
    # ==================================================

    print(
        "\n"
        + "-" * 100
    )

    print(
        f"TEST {number}"
    )

    print(
        f"QUESTION: {question}"
    )

    print(
        f"EXPECTED: "
        f"{'SUPPORTED' if expected else 'NOT_SUPPORTED'}"
    )

    print(
        f"PREDICTED: "
        f"{result['status'].upper()}"
    )

    print(
        f"CLASS: {classification}"
    )

    print(
        f"PIPELINE STAGE: "
        f"{result['pipeline_stage']}"
    )

    print(
        f"RESCUE USED: "
        f"{result['rescue_used']}"
    )

    print(
        f"RETRIEVAL SCORE: "
        f"{result['retrieval_score']:.4f}"
    )

    print(
        f"ANSWER: "
        f"{result['answer']}"
    )


    # ------------------------------------------
    # Show original V2 answer when rescued
    # ------------------------------------------

    if result.get(
        "original_v2_answer"
    ):

        print(
            f"ORIGINAL V2 ANSWER: "
            f"{result['original_v2_answer']}"
        )


    # ------------------------------------------
    # Calculation
    # ------------------------------------------

    if result.get(
        "calculation"
    ):

        print(
            "CALCULATION:"
        )

        print(
            json.dumps(
                result[
                    "calculation"
                ],
                indent=4
            )
        )


    # ------------------------------------------
    # Claim debug
    # ------------------------------------------

    for claim in result.get(
        "claims",
        []
    ):

        print(
            "\nCLAIM:"
        )

        print(
            claim.get(
                "claim"
            )
        )

        print(
            f"SUPPORTED: "
            f"{claim.get('supported')}"
        )

        print(
            f"EVIDENCE SIM: "
            f"{claim.get('evidence_similarity', 0):.4f}"
        )

        print(
            f"ENTAILMENT: "
            f"{claim.get('entailment', 0):.4f}"
        )

        print(
            f"EVIDENCE: "
            f"{claim.get('evidence')}"
        )


    if error_type:

        print(
            f"ERROR TYPE: "
            f"{error_type}"
        )


    # ------------------------------------------
    # Store
    # ------------------------------------------

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

            "error_type":
                error_type,

            "result":
                result
        }
    )


# ==================================================
# 20) METRICS
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
# 21) FINAL REPORT
# ==================================================

print(
    "\n\n"
    + "=" * 100
)

print(
    "V2 + RESCUE EVAL SONUCU"
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


print(
    "\nRESCUE"
)

print(
    f"Deterministic rescue ile kurtarılan: "
    f"{rescue_success_count}"
)


# ==================================================
# 22) SAVE REPORT
# ==================================================

report = {

    "version":
        "v2_baseline_plus_deterministic_rescue",

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
            false_negative,

        "rescue_success_count":
            rescue_success_count
    },

    "tests":
        all_results
}


with open(
    "full_rag_eval_v2_rescue_results.json",
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
    "full_rag_eval_v2_rescue_results.json"
)