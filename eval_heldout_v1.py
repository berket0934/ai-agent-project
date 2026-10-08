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


chunks = create_chunks(document)


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

        # Sentence-level evidence
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
        for item in evidence_candidates
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


        # ------------------------------------------
        # EVIDENCE RELEVANCE GATE
        # ------------------------------------------

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
                    "claim":
                        claim,

                    "supported":
                        False,

                    "reason":
                        "no_relevant_evidence",

                    "evidence":
                        None,

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
        # NLI VERIFICATION
        # ------------------------------------------

        pairs = [
            (
                evidence["text"],
                claim
            )
            for evidence in relevant_evidence
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
        # V2 mantığı:
        # En yüksek ENTAILMENT veren evidence
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

                "reason":
                    (
                        "supported"
                        if supported
                        else
                        "evidence_does_not_entail_claim"
                    ),

                "evidence":
                    best_evidence["text"],

                "evidence_type":
                    best_evidence["type"],

                "evidence_similarity":
                    best_evidence[
                        "evidence_similarity"
                    ],

                "entailment":
                    entailment,

                "contradiction":
                    best_probs[0].item(),

                "neutral":
                    best_probs[2].item()
            }
        )


    return results


# ==================================================
# 11) JSON PARSER
# ==================================================

def parse_structured_output(raw_text):

    text = raw_text.strip()


    # Markdown code fence varsa temizle

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )


    start = text.find("{")
    end = text.rfind("}")


    if (
        start == -1
        or
        end == -1
    ):

        return None


    json_text = text[
        start:end + 1
    ]


    try:

        data = json.loads(
            json_text
        )

    except json.JSONDecodeError:

        return None


    answer = data.get(
        "answer"
    )

    claims = data.get(
        "claims"
    )


    if not isinstance(
        answer,
        str
    ):

        return None


    if not isinstance(
        claims,
        list
    ):

        return None


    clean_claims = []


    for claim in claims:

        if (
            isinstance(claim, str)
            and
            claim.strip()
        ):

            clean_claims.append(
                claim.strip()
            )


    return {
        "answer":
            answer.strip(),

        "claims":
            clean_claims
    }


# ==================================================
# 12) SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = """
You are a grounded RAG answer generator.

You must answer using ONLY the supplied context.

Return ONLY valid JSON.
Do not use markdown.

The JSON schema must be exactly:

{
    "answer": "final user-facing answer",
    "claims": [
        "complete factual claim",
        "complete factual claim"
    ]
}

Rules:

1. Every claim must be a complete standalone factual statement.

2. Claims must contain the actual factual meaning of the answer.

3. Do not place conversational words such as "Yes" or "No"
   inside the claims.

4. For yes/no questions, convert the answer into a factual claim.

Example:

Question:
Does the warranty cover accidental damage?

If the context says it does NOT cover accidental damage:

{
    "answer": "No, the warranty does not cover accidental damage.",
    "claims": [
        "The warranty does not cover accidental damage."
    ]
}

5. For short factual answers, the claim must still be a complete sentence.

Example:

Question:
How long is the warranty for electronics?

{
    "answer": "The warranty is two years.",
    "claims": [
        "Electronic products include a two-year warranty."
    ]
}

6. Do not infer that something is false merely because it is not mentioned.

7. If the context does NOT contain enough information to answer,
   use:

{
    "answer": "The provided context does not specify this information.",
    "claims": []
}

8. Never invent policies, conditions, prices, benefits,
   payment methods, shipping options, or restrictions.

9. Output JSON only.
"""


# ==================================================
# 13) RAW LLM CALL
# ==================================================

def call_llm(messages):

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
            max_new_tokens=180,
            do_sample=False
        )


    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]


    raw_output = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    return raw_output


# ==================================================
# 14) INITIAL STRUCTURED GENERATION
# ==================================================

def generate_structured_answer(
    question,
    retrieved_chunks
):

    context = "\n\n".join(
        item["text"]
        for item in retrieved_chunks
    )


    messages = [

        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT
        },

        {
            "role":
                "user",

            "content": f"""
CONTEXT:

{context}

QUESTION:

{question}

Return only the JSON object.
"""
        }
    ]


    raw_output = call_llm(
        messages
    )


    parsed = parse_structured_output(
        raw_output
    )


    return {
        "raw":
            raw_output,

        "parsed":
            parsed
    }


# ==================================================
# 15) REPAIR GENERATION
# ==================================================

def repair_structured_answer(
    question,
    retrieved_chunks,
    previous_output,
    verification_results=None
):

    context = "\n\n".join(
        item["text"]
        for item in retrieved_chunks
    )


    feedback_lines = []


    if verification_results:

        for result in verification_results:

            feedback_lines.append(
                (
                    f"Claim: {result['claim']}\n"
                    f"Supported: {result['supported']}\n"
                    f"Best evidence: {result['evidence']}\n"
                    f"Entailment: {result['entailment']:.4f}\n"
                    f"Reason: {result['reason']}"
                )
            )


    verifier_feedback = "\n\n".join(
        feedback_lines
    )


    messages = [

        {
            "role":
                "system",

            "content":
                (
                    SYSTEM_PROMPT
                    +
                    """

You are now repairing a previous answer.

The previous answer could not be safely verified.

Correct it using ONLY the supplied context.

If the context truly does not answer the question,
return an empty claims list.

Do not repeat an unsupported claim.
"""
                )
        },

        {
            "role":
                "user",

            "content": f"""
CONTEXT:

{context}

QUESTION:

{question}

PREVIOUS OUTPUT:

{previous_output}

VERIFIER FEEDBACK:

{verifier_feedback}

Produce a corrected JSON answer.
"""
        }
    ]


    raw_output = call_llm(
        messages
    )


    parsed = parse_structured_output(
        raw_output
    )


    return {
        "raw":
            raw_output,

        "parsed":
            parsed
    }


# ==================================================
# 16) CHECK STRUCTURED ANSWER
# ==================================================

def check_structured_answer(
    retrieved_chunks,
    parsed_output
):

    if parsed_output is None:

        return {
            "supported":
                False,

            "reason":
                "invalid_structured_output",

            "claims":
                []
        }


    claims = parsed_output[
        "claims"
    ]


    if len(claims) == 0:

        return {
            "supported":
                False,

            "reason":
                "no_claims_generated",

            "claims":
                []
        }


    verification_results = verify_claims(
        retrieved_chunks,
        claims
    )


    all_supported = (

        len(verification_results) > 0

        and

        all(
            claim["supported"]
            for claim in verification_results
        )
    )


    return {
        "supported":
            all_supported,

        "reason":
            (
                "all_claims_supported"
                if all_supported
                else
                "claim_verification_failed"
            ),

        "claims":
            verification_results
    }


# ==================================================
# 17) DETERMINISTIC RESCUE HELPERS
#
# BURADAN SONRASI V2'YE EKLENEN KISIM.
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


# ==================================================
# 18) EXTRACT DURATION
# ==================================================

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


    number_text = match.group(
        1
    )

    unit = match.group(
        2
    )


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

        "number":
            number,

        "unit":
            unit,

        "days":
            days
    }


# ==================================================
# 19) GET RETURN WINDOW
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
# 20) RETURN QUESTION DETECTION
# ==================================================

def is_return_question(question):

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
# 21) DETERMINISTIC RETURN RESCUE
# ==================================================

def deterministic_return_rescue(
    question,
    retrieved_chunks
):

    # Rescue yalnızca return-policy sorularına bakıyor.

    if not is_return_question(
        question
    ):

        return None


    q = question.lower()


    # ==================================================
    # RESCUE CASE 1
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


        if not (
            len(verification) > 0

            and

            verification[0][
                "supported"
            ]
        ):

            return None


        return {

            "answer":
                (
                    "No. The return policy requires "
                    "the product to be unused and "
                    "in its original packaging."
                ),

            "reason":
                "used_product_rescue",

            "calculation":
                None,

            "claims":
                verification
        }


    # ==================================================
    # RESCUE CASE 2
    # DAY / WEEK NORMALIZATION
    # ==================================================

    duration = extract_duration_days(
        question
    )


    if duration is None:

        return None


    if RETURN_WINDOW_DAYS is None:

        return None


    # Önce 30-day policy gerçekten retrieved evidence
    # içinde var mı onu doğruluyoruz.

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
    # Within policy window
    # ------------------------------------------

    if inside_window:

        answer = (

            f"{duration['original'].capitalize()} "
            f"is {requested_days} days, which is "
            f"within the {RETURN_WINDOW_DAYS}-day "
            f"return window. Other return conditions "
            f"must also be satisfied."
        )


    # ------------------------------------------
    # Outside policy window
    # ------------------------------------------

    else:

        answer = (

            f"No. {duration['original'].capitalize()} "
            f"is {requested_days} days, which exceeds "
            f"the {RETURN_WINDOW_DAYS}-day return window."
        )


    return {

        "answer":
            answer,

        "reason":
            "duration_rescue",

        "calculation": {

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
# 22) FULL RAG V2 + RESCUE
# ==================================================

def run_rag_v2(question):

    # ==================================================
    # RETRIEVAL
    # ==================================================

    retrieved = retrieve(
        question
    )


    best_retrieval_score = retrieved[
        0
    ]["score"]


    # ==================================================
    # RETRIEVAL GATE
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

            "pipeline_stage":
                "retrieval_reject",

            "repair_used":
                False,

            "rescue_used":
                False,

            "retrieval_score":
                best_retrieval_score,

            "initial_generation":
                None,

            "repair_generation":
                None,

            "claims":
                []
        }


    # ==================================================
    # ORIGINAL V2 INITIAL GENERATION
    # ==================================================

    initial = generate_structured_answer(
        question,
        retrieved
    )


    initial_parsed = initial[
        "parsed"
    ]


    initial_check = check_structured_answer(
        retrieved,
        initial_parsed
    )


    # ==================================================
    # INITIAL PASS BAŞARILI
    # ==================================================

    if initial_check[
        "supported"
    ]:

        return {

            "status":
                "supported",

            "answer":
                initial_parsed[
                    "answer"
                ],

            "pipeline_stage":
                "initial_supported",

            "repair_used":
                False,

            "rescue_used":
                False,

            "retrieval_score":
                best_retrieval_score,

            "initial_generation":
                initial,

            "repair_generation":
                None,

            "claims":
                initial_check[
                    "claims"
                ]
        }


    # ==================================================
    # ORIGINAL V2 REPAIR PASS
    # ==================================================

    repair = repair_structured_answer(

        question=
            question,

        retrieved_chunks=
            retrieved,

        previous_output=
            initial["raw"],

        verification_results=
            initial_check["claims"]
    )


    repair_parsed = repair[
        "parsed"
    ]


    repair_check = check_structured_answer(
        retrieved,
        repair_parsed
    )


    # ==================================================
    # REPAIR BAŞARILI
    # ==================================================

    if repair_check[
        "supported"
    ]:

        return {

            "status":
                "supported",

            "answer":
                repair_parsed[
                    "answer"
                ],

            "pipeline_stage":
                "repair_supported",

            "repair_used":
                True,

            "rescue_used":
                False,

            "retrieval_score":
                best_retrieval_score,

            "initial_generation":
                initial,

            "repair_generation":
                repair,

            "claims":
                repair_check[
                    "claims"
                ]
        }


    # ==================================================
    # YENİ:
    # DETERMINISTIC RESCUE
    #
    # SADECE initial + repair başarısız olduktan sonra.
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

            "pipeline_stage":
                "deterministic_rescue",

            "repair_used":
                True,

            "rescue_used":
                True,

            "retrieval_score":
                best_retrieval_score,

            "initial_generation":
                initial,

            "repair_generation":
                repair,

            "rescue_reason":
                rescue[
                    "reason"
                ],

            "calculation":
                rescue[
                    "calculation"
                ],

            "claims":
                rescue[
                    "claims"
                ]
        }


    # ==================================================
    # ORIGINAL V2 FINAL REJECT
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

        "pipeline_stage":
            "repair_failed",

        "repair_used":
            True,

        "rescue_used":
            False,

        "retrieval_score":
            best_retrieval_score,

        "initial_generation":
            initial,

        "repair_generation":
            repair,

        "initial_failure_reason":
            initial_check[
                "reason"
            ],

        "repair_failure_reason":
            repair_check[
                "reason"
            ],

        "claims":
            repair_check[
                "claims"
            ]
    }


# ==================================================
# 23) EVAL SET
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
# LOAD HELD-OUT EVAL SET
# ==================================================

with open("heldout_eval_v1.json", "r", encoding="utf-8") as file:
    eval_set = json.load(file)


# ==================================================
# 24) EVALUATION
# ==================================================

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

repair_success_count = 0
rescue_success_count = 0

all_results = []


print(
    "\n"
    + "=" * 100
)

print(
    "FULL RAG V2 + RESCUE EVALUATION"
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


    result = run_rag_v2(
        question
    )


    predicted = (

        result["status"]
        ==
        "supported"
    )


    # ------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------

    if (
        predicted
        and
        expected
    ):

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
    # COUNTERS
    # ------------------------------------------

    if (
        result[
            "pipeline_stage"
        ]
        ==
        "repair_supported"
    ):

        repair_success_count += 1


    if result.get(
        "rescue_used",
        False
    ):

        rescue_success_count += 1


    # ------------------------------------------
    # ERROR TYPE
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
    # TEST OUTPUT
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
        f"REPAIR USED: "
        f"{result['repair_used']}"
    )

    print(
        f"RESCUE USED: "
        f"{result.get('rescue_used', False)}"
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
    # Rescue calculation
    # ------------------------------------------

    if result.get(
        "calculation"
    ):

        print(
            "\nCALCULATION:"
        )

        print(
            json.dumps(
                result[
                    "calculation"
                ],
                indent=4,
                ensure_ascii=False
            )
        )


    # ------------------------------------------
    # Claim debug
    # ------------------------------------------

    if result[
        "claims"
    ]:

        for claim_result in result[
            "claims"
        ]:

            print(
                "\nCLAIM:"
            )

            print(
                claim_result[
                    "claim"
                ]
            )

            print(
                f"SUPPORTED: "
                f"{claim_result['supported']}"
            )

            print(
                f"EVIDENCE SIM: "
                f"{claim_result['evidence_similarity']:.4f}"
            )

            print(
                f"ENTAILMENT: "
                f"{claim_result['entailment']:.4f}"
            )

            print(
                f"CONTRADICTION: "
                f"{claim_result['contradiction']:.4f}"
            )

            print(
                f"NEUTRAL: "
                f"{claim_result['neutral']:.4f}"
            )

            print(
                f"EVIDENCE: "
                f"{claim_result['evidence']}"
            )


    if error_type:

        print(
            f"\nERROR TYPE: "
            f"{error_type}"
        )


    # ------------------------------------------
    # STORE RESULT
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
# 25) METRICS
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
# 26) FINAL REPORT
# ==================================================

print(
    "\n\n"
    + "=" * 100
)

print(
    "FULL RAG V2 + RESCUE EVAL SONUCU"
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
    "\nPIPELINE STATS"
)

print(
    f"Repair ile kurtarılan: "
    f"{repair_success_count}"
)

print(
    f"Deterministic rescue ile kurtarılan: "
    f"{rescue_success_count}"
)


# ==================================================
# 27) SAVE REPORT
# ==================================================

report = {

    "version":
        "rag_v2_structured_claims_repair_plus_deterministic_rescue",

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

        "repair_success_count":
            repair_success_count,

        "rescue_success_count":
            rescue_success_count
    },

    "tests":
        all_results
}


with open(
    "heldout_eval_v1_results.json",
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
    "heldout_eval_v1_results.json"
)
