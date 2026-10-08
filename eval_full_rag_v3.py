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

# NO cevabını contradiction üzerinden kabul ederken
# daha sıkı davranıyoruz.
CONTRADICTION_THRESHOLD = 0.85

TOP_K = 3


# ==================================================
# 2) DOCUMENT
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

    sentences = split_into_sentences(
        text
    )

    chunks = []

    start = 0

    while start < len(sentences):

        end = (
            start
            +
            sentences_per_chunk
        )

        selected = sentences[
            start:end
        ]

        chunks.append(
            " ".join(selected)
        )

        start = (
            end
            -
            overlap_sentences
        )

    return chunks


chunks = create_chunks(
    document
)


# ==================================================
# 5) MODELS
# ==================================================

print(
    "Embedding modeli yükleniyor..."
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print(
    "Local LLM yükleniyor..."
)

llm_name = (
    "Qwen/Qwen2.5-0.5B-Instruct"
)

tokenizer = AutoTokenizer.from_pretrained(
    llm_name
)

llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype="auto"
)


print(
    "Verifier modeli yükleniyor..."
)

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

        similarity = (
            1
            -
            distance
        )

        results.append(
            {
                "text":
                    chunks[idx],

                "score":
                    float(similarity)
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

        chunk_text = result[
            "text"
        ]


        # ------------------------------------------
        # FULL CHUNK
        # ------------------------------------------

        if chunk_text not in seen:

            candidates.append(
                {
                    "text":
                        chunk_text,

                    "type":
                        "full_chunk",

                    "retrieval_score":
                        result["score"]
                }
            )

            seen.add(
                chunk_text
            )


        # ------------------------------------------
        # INDIVIDUAL SENTENCES
        # ------------------------------------------

        sentences = split_into_sentences(
            chunk_text
        )

        for sentence in sentences:

            if sentence not in seen:

                candidates.append(
                    {
                        "text":
                            sentence,

                        "type":
                            "sentence",

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

def calculate_evidence_similarities(
    statement,
    evidence_candidates
):

    statement_embedding = (
        embedding_model.encode(
            [statement],
            normalize_embeddings=True
        )[0]
    )


    evidence_texts = [

        evidence["text"]

        for evidence
        in evidence_candidates
    ]


    evidence_embeddings = (
        embedding_model.encode(
            evidence_texts,
            normalize_embeddings=True
        )
    )


    similarities = np.dot(
        evidence_embeddings,
        statement_embedding
    )


    return similarities


# ==================================================
# 10) GENERIC NLI CHECK
# ==================================================

def run_nli_against_evidence(
    statement,
    retrieved_chunks
):

    evidence_candidates = (
        create_evidence_candidates(
            retrieved_chunks
        )
    )


    similarities = (
        calculate_evidence_similarities(
            statement,
            evidence_candidates
        )
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


    if len(relevant_evidence) == 0:

        return {
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


    pairs = [

        (
            evidence["text"],
            statement
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


    # Burada yalnızca entailment'e göre
    # best evidence seçmek istemiyoruz.
    #
    # Hem entailment hem contradiction
    # ileride lazım olacak.
    #
    # Bu yüzden en güçlü non-neutral
    # evidence'ı seçiyoruz.

    non_neutral_scores = (

        probabilities[:, 0]
        +
        probabilities[:, 1]
    )


    best_index = torch.argmax(
        non_neutral_scores
    ).item()


    best_probs = probabilities[
        best_index
    ]


    best_evidence = relevant_evidence[
        best_index
    ]


    return {

        "evidence":
            best_evidence["text"],

        "evidence_type":
            best_evidence["type"],

        "evidence_similarity":
            best_evidence[
                "evidence_similarity"
            ],

        "entailment":
            best_probs[1].item(),

        "contradiction":
            best_probs[0].item(),

        "neutral":
            best_probs[2].item()
    }


# ==================================================
# 11) NORMAL CLAIM VERIFICATION
# ==================================================

def verify_claims(
    retrieved_chunks,
    claims
):

    results = []


    for claim in claims:

        nli_result = (
            run_nli_against_evidence(
                claim,
                retrieved_chunks
            )
        )


        supported = (

            nli_result[
                "evidence_similarity"
            ]
            >= EVIDENCE_RELEVANCE_THRESHOLD

            and

            nli_result[
                "entailment"
            ]
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
                        "claim_not_entailed"
                    ),

                **nli_result
            }
        )


    return results


# ==================================================
# 12) EXPLICIT NEGATION CHECK
# ==================================================

def contains_explicit_negative_language(
    text
):

    text = text.lower()


    negative_patterns = [

        r"\bnot\b",

        r"\bno\b",

        r"\bnever\b",

        r"\bcannot\b",

        r"\bcan't\b",

        r"\bdoes not\b",

        r"\bdo not\b",

        r"\bmust not\b",

        r"\bwithout\b"
    ]


    for pattern in negative_patterns:

        if re.search(
            pattern,
            text
        ):

            return True


    return False


# ==================================================
# 13) YES / NO POLARITY VERIFICATION
# ==================================================

def verify_polarity(
    retrieved_chunks,
    proposition,
    polarity
):

    if not proposition:

        return {
            "supported": False,
            "reason":
                "missing_proposition"
        }


    nli_result = (
        run_nli_against_evidence(
            proposition,
            retrieved_chunks
        )
    )


    # ------------------------------------------
    # YES
    #
    # proposition doğru olmalı
    # ------------------------------------------

    if polarity == "yes":

        supported = (

            nli_result[
                "evidence_similarity"
            ]
            >= EVIDENCE_RELEVANCE_THRESHOLD

            and

            nli_result[
                "entailment"
            ]
            >= ENTAILMENT_THRESHOLD
        )


        reason = (

            "yes_supported_by_entailment"

            if supported

            else

            "yes_not_entailed"
        )


    # ------------------------------------------
    # NO
    #
    # positive proposition evidence ile
    # çelişmeli.
    #
    # Ayrıca false-negative değil,
    # false-positive riskini azaltmak için
    # explicit negative evidence arıyoruz.
    # ------------------------------------------

    elif polarity == "no":

        evidence_text = (
            nli_result["evidence"]
            or
            ""
        )


        explicit_negative = (
            contains_explicit_negative_language(
                evidence_text
            )
        )


        supported = (

            nli_result[
                "evidence_similarity"
            ]
            >= EVIDENCE_RELEVANCE_THRESHOLD

            and

            nli_result[
                "contradiction"
            ]
            >= CONTRADICTION_THRESHOLD

            and

            explicit_negative
        )


        reason = (

            "no_supported_by_explicit_contradiction"

            if supported

            else

            "no_not_safely_verified"
        )


    else:

        supported = False

        reason = (
            "unknown_or_not_applicable"
        )


    return {

        "supported":
            supported,

        "reason":
            reason,

        "polarity":
            polarity,

        "proposition":
            proposition,

        **nli_result
    }


# ==================================================
# 14) NUMBER WORDS
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
    "thirty": 30
}


# ==================================================
# 15) EXTRACT DURATION
# ==================================================

def extract_duration_days(
    text
):

    text = text.lower()


    number_pattern = (
        r"(\d+|"
        +
        "|".join(
            NUMBER_WORDS.keys()
        )
        +
        r")"
    )


    pattern = (

        number_pattern

        +

        r"\s*"

        +

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


    if unit in [
        "week",
        "weeks"
    ]:

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
# 16) GET RETURN WINDOW FROM DOCUMENT
# ==================================================

def extract_return_window_days(
    document_text
):

    match = re.search(
        r"return products within\s+(\d+)\s+days",
        document_text,
        flags=re.IGNORECASE
    )


    if not match:

        return None


    return int(
        match.group(1)
    )


RETURN_WINDOW_DAYS = (
    extract_return_window_days(
        document
    )
)


# ==================================================
# 17) RETURN INTENT
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
# 18) DETERMINISTIC RETURN POLICY REASONING
# ==================================================

def deterministic_return_reasoning(
    question,
    retrieved_chunks
):

    if not is_return_question(
        question
    ):

        return None


    q = question.lower()


    duration = extract_duration_days(
        question
    )


    used_product = bool(
        re.search(
            r"\bused\b",
            q
        )
    )


    # ------------------------------------------
    # CASE A:
    # USED PRODUCT
    # ------------------------------------------

    if used_product:

        claims = [

            (
                "The product must be unused "
                "and in its original packaging."
            )
        ]


        claim_results = verify_claims(
            retrieved_chunks,
            claims
        )


        all_supported = (

            len(claim_results) > 0

            and

            all(
                result["supported"]

                for result
                in claim_results
            )
        )


        if all_supported:

            return {

                "status":
                    "supported",

                "answer":
                    (
                        "No. The return policy requires "
                        "the product to be unused and "
                        "in its original packaging."
                    ),

                "reasoning_type":
                    "deterministic_return_condition",

                "calculation":
                    None,

                "claims":
                    claim_results
            }


        return {

            "status":
                "not_supported",

            "reasoning_type":
                "deterministic_return_condition_failed_verification",

            "claims":
                claim_results
        }


    # ------------------------------------------
    # CASE B:
    # TIME WINDOW
    # ------------------------------------------

    if (
        duration is not None
        and
        RETURN_WINDOW_DAYS is not None
    ):

        requested_days = duration[
            "days"
        ]


        inside_window = (

            requested_days
            <=
            RETURN_WINDOW_DAYS
        )


        policy_claim = (

            f"Customers can return products within "
            f"{RETURN_WINDOW_DAYS} days after delivery."
        )


        claim_results = verify_claims(
            retrieved_chunks,
            [
                policy_claim
            ]
        )


        policy_verified = (

            len(claim_results) > 0

            and

            claim_results[0][
                "supported"
            ]
        )


        if not policy_verified:

            return {

                "status":
                    "not_supported",

                "reasoning_type":
                    "duration_policy_not_verified",

                "claims":
                    claim_results
            }


        # ------------------------------------------
        # TIME IS INSIDE WINDOW
        # ------------------------------------------

        if inside_window:

            answer = (

                f"{duration['original'].capitalize()} "
                f"is {requested_days} days, which is "
                f"within the {RETURN_WINDOW_DAYS}-day "
                f"return window. The product must also "
                f"meet the return conditions."
            )


        # ------------------------------------------
        # TIME EXCEEDS WINDOW
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

            "reasoning_type":
                "deterministic_duration",

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
                claim_results
        }


    return None


# ==================================================
# 19) STRUCTURED OUTPUT PARSER
# ==================================================

def parse_structured_output(
    raw_text
):

    text = raw_text.strip()


    # Markdown fences temizle
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


    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )


    if (
        start == -1
        or
        end == -1
    ):

        return None


    try:

        data = json.loads(
            text[
                start:end + 1
            ]
        )

    except json.JSONDecodeError:

        return None


    answer = data.get(
        "answer"
    )

    claims = data.get(
        "claims"
    )

    is_yes_no = data.get(
        "is_yes_no",
        False
    )

    polarity = data.get(
        "polarity",
        "not_applicable"
    )

    proposition = data.get(
        "proposition"
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


    allowed_polarities = [

        "yes",
        "no",
        "unknown",
        "not_applicable"
    ]


    if polarity not in allowed_polarities:

        polarity = (
            "not_applicable"
        )


    if not isinstance(
        proposition,
        str
    ):

        proposition = None


    return {

        "answer":
            answer.strip(),

        "claims":
            clean_claims,

        "is_yes_no":
            bool(is_yes_no),

        "polarity":
            polarity,

        "proposition":
            (
                proposition.strip()

                if proposition

                else None
            )
    }


# ==================================================
# 20) SYSTEM PROMPT
# ==================================================

SYSTEM_PROMPT = """
You are a grounded RAG answer generator.

Use ONLY the provided context.

Return ONLY valid JSON.
Do not use markdown.

Schema:

{
    "answer": "final user-facing answer",
    "claims": [
        "standalone factual claim"
    ],
    "is_yes_no": true,
    "polarity": "yes",
    "proposition": "positive factual proposition"
}

Rules:

1. claims must be complete standalone factual statements.

2. Do not put conversational Yes or No inside claims.

3. If the user asks a yes/no question:
   - is_yes_no must be true.
   - polarity must be "yes", "no", or "unknown".
   - proposition must express the positive version of the question.

Example:

Question:
Does the warranty cover accidental damage?

Positive proposition:
"The warranty covers accidental damage."

If context says it does NOT cover it:

{
    "answer": "No, the warranty does not cover accidental damage.",
    "claims": [
        "The warranty does not cover accidental damage."
    ],
    "is_yes_no": true,
    "polarity": "no",
    "proposition": "The warranty covers accidental damage."
}

4. For non yes/no questions:

{
    "is_yes_no": false,
    "polarity": "not_applicable",
    "proposition": null
}

5. If the context does not contain enough information:

{
    "answer": "The provided context does not specify this information.",
    "claims": [],
    "is_yes_no": true,
    "polarity": "unknown",
    "proposition": "positive factual proposition"
}

6. Never infer that something is false just because it is not mentioned.

7. Never invent policies, prices, benefits, payment methods,
shipping options, restrictions, or conditions.

8. Output JSON only.
"""


# ==================================================
# 21) LLM CALL
# ==================================================

def call_llm(
    messages
):

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
            max_new_tokens=220,
            do_sample=False
        )


    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]


    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


# ==================================================
# 22) INITIAL GENERATION
# ==================================================

def generate_structured_answer(
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

Return only JSON.
"""
        }
    ]


    raw = call_llm(
        messages
    )


    return {

        "raw":
            raw,

        "parsed":
            parse_structured_output(
                raw
            )
    }


# ==================================================
# 23) REPAIR GENERATION
# ==================================================

def repair_structured_answer(
    question,
    retrieved_chunks,
    previous_output,
    verification_feedback
):

    context = "\n\n".join(

        item["text"]

        for item
        in retrieved_chunks
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

The previous answer failed verification.

Correct it using only the supplied evidence.

Pay special attention to negation.

If the context says "does not cover",
do not answer that it covers.

If the context is insufficient,
use polarity "unknown" and an empty claims list.
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

VERIFICATION FEEDBACK:

{verification_feedback}

Return a corrected JSON object only.
"""
        }
    ]


    raw = call_llm(
        messages
    )


    return {

        "raw":
            raw,

        "parsed":
            parse_structured_output(
                raw
            )
    }


# ==================================================
# 24) CHECK STRUCTURED ANSWER
# ==================================================

def check_structured_answer(
    retrieved_chunks,
    parsed
):

    if parsed is None:

        return {

            "supported":
                False,

            "reason":
                "invalid_json",

            "claims":
                [],

            "polarity_result":
                None
        }


    claims = parsed[
        "claims"
    ]


    # ------------------------------------------
    # UNKNOWN / EMPTY
    # ------------------------------------------

    if (
        parsed[
            "polarity"
        ]
        == "unknown"
    ):

        return {

            "supported":
                False,

            "reason":
                "model_says_unknown",

            "claims":
                [],

            "polarity_result":
                None
        }


    if len(claims) == 0:

        return {

            "supported":
                False,

            "reason":
                "no_claims",

            "claims":
                [],

            "polarity_result":
                None
        }


    # ------------------------------------------
    # NORMAL CLAIM VERIFICATION
    # ------------------------------------------

    claim_results = verify_claims(
        retrieved_chunks,
        claims
    )


    claims_supported = all(

        result["supported"]

        for result
        in claim_results
    )


    # ------------------------------------------
    # YES / NO AUXILIARY VERIFICATION
    # ------------------------------------------

    polarity_result = None


    if parsed[
        "is_yes_no"
    ]:

        polarity_result = verify_polarity(

            retrieved_chunks,

            parsed[
                "proposition"
            ],

            parsed[
                "polarity"
            ]
        )


    # ------------------------------------------
    # FINAL CHECK
    #
    # Direct factual claims remain mandatory.
    #
    # Polarity is an extra consistency check.
    # ------------------------------------------

    if parsed[
        "is_yes_no"
    ]:

        polarity_ok = (

            polarity_result
            is not None

            and

            polarity_result[
                "supported"
            ]
        )

    else:

        polarity_ok = True


    supported = (

        claims_supported

        and

        polarity_ok
    )


    return {

        "supported":
            supported,

        "reason":
            (
                "supported"
                if supported
                else
                "verification_failed"
            ),

        "claims":
            claim_results,

        "polarity_result":
            polarity_result
    }


# ==================================================
# 25) FEEDBACK BUILDER
# ==================================================

def build_verification_feedback(
    check_result
):

    lines = []


    for result in check_result.get(
        "claims",
        []
    ):

        lines.append(
            (
                f"Claim: {result['claim']}\n"
                f"Supported: {result['supported']}\n"
                f"Evidence: {result['evidence']}\n"
                f"Entailment: {result['entailment']:.4f}\n"
                f"Contradiction: {result['contradiction']:.4f}\n"
                f"Neutral: {result['neutral']:.4f}"
            )
        )


    polarity_result = check_result.get(
        "polarity_result"
    )


    if polarity_result:

        lines.append(
            (
                "YES/NO POLARITY CHECK:\n"
                f"Proposition: "
                f"{polarity_result.get('proposition')}\n"
                f"Polarity: "
                f"{polarity_result.get('polarity')}\n"
                f"Supported: "
                f"{polarity_result.get('supported')}\n"
                f"Reason: "
                f"{polarity_result.get('reason')}\n"
                f"Evidence: "
                f"{polarity_result.get('evidence')}\n"
                f"Entailment: "
                f"{polarity_result.get('entailment', 0):.4f}\n"
                f"Contradiction: "
                f"{polarity_result.get('contradiction', 0):.4f}"
            )
        )


    return "\n\n".join(
        lines
    )


# ==================================================
# 26) FULL RAG V3
# ==================================================

def run_rag_v3(
    question
):

    # ------------------------------------------
    # RETRIEVAL
    # ------------------------------------------

    retrieved = retrieve(
        question
    )


    best_score = retrieved[
        0
    ]["score"]


    # ------------------------------------------
    # RETRIEVAL GATE
    # ------------------------------------------

    if (
        best_score
        < RETRIEVAL_THRESHOLD
    ):

        return {

            "status":
                "not_supported",

            "answer":
                (
                    "The provided documents do not "
                    "contain enough information."
                ),

            "pipeline_stage":
                "retrieval_reject",

            "reasoning_type":
                "retrieval",

            "repair_used":
                False,

            "retrieval_score":
                best_score,

            "claims":
                []
        }


    # ==================================================
    # DETERMINISTIC REASONING FIRST
    # ==================================================

    deterministic = (
        deterministic_return_reasoning(
            question,
            retrieved
        )
    )


    if (
        deterministic is not None
        and
        deterministic[
            "status"
        ]
        == "supported"
    ):

        return {

            "status":
                "supported",

            "answer":
                deterministic[
                    "answer"
                ],

            "pipeline_stage":
                "deterministic_supported",

            "reasoning_type":
                deterministic[
                    "reasoning_type"
                ],

            "calculation":
                deterministic.get(
                    "calculation"
                ),

            "repair_used":
                False,

            "retrieval_score":
                best_score,

            "claims":
                deterministic[
                    "claims"
                ]
        }


    # ==================================================
    # LLM INITIAL GENERATION
    # ==================================================

    initial = generate_structured_answer(
        question,
        retrieved
    )


    initial_check = (
        check_structured_answer(
            retrieved,
            initial["parsed"]
        )
    )


    if initial_check[
        "supported"
    ]:

        return {

            "status":
                "supported",

            "answer":
                initial[
                    "parsed"
                ]["answer"],

            "pipeline_stage":
                "initial_supported",

            "reasoning_type":
                "llm_verified",

            "repair_used":
                False,

            "retrieval_score":
                best_score,

            "claims":
                initial_check[
                    "claims"
                ],

            "polarity_result":
                initial_check[
                    "polarity_result"
                ]
        }


    # ==================================================
    # REPAIR
    # ==================================================

    feedback = (
        build_verification_feedback(
            initial_check
        )
    )


    repair = repair_structured_answer(

        question,

        retrieved,

        initial["raw"],

        feedback
    )


    repair_check = (
        check_structured_answer(
            retrieved,
            repair["parsed"]
        )
    )


    if repair_check[
        "supported"
    ]:

        return {

            "status":
                "supported",

            "answer":
                repair[
                    "parsed"
                ]["answer"],

            "pipeline_stage":
                "repair_supported",

            "reasoning_type":
                "llm_repaired_verified",

            "repair_used":
                True,

            "retrieval_score":
                best_score,

            "claims":
                repair_check[
                    "claims"
                ],

            "polarity_result":
                repair_check[
                    "polarity_result"
                ]
        }


    # ==================================================
    # FINAL REJECT
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

        "reasoning_type":
            "verification_failure",

        "repair_used":
            True,

        "retrieval_score":
            best_score,

        "initial_raw":
            initial["raw"],

        "repair_raw":
            repair["raw"],

        "claims":
            repair_check[
                "claims"
            ],

        "polarity_result":
            repair_check[
                "polarity_result"
            ]
    }


# ==================================================
# 27) EVAL SET
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
# 28) EVALUATION
# ==================================================

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

repair_success_count = 0

deterministic_success_count = 0

all_results = []


print(
    "\n"
    +
    "=" * 100
)

print(
    "FULL RAG V3 EVALUATION"
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


    result = run_rag_v3(
        question
    )


    predicted = (

        result["status"]
        ==
        "supported"
    )


    # ------------------------------------------
    # CLASSIFICATION
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


    if (
        result[
            "pipeline_stage"
        ]
        ==
        "deterministic_supported"
    ):

        deterministic_success_count += 1


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
                "reasoning_or_verification_false_negative"
            )


    # ------------------------------------------
    # PRINT
    # ------------------------------------------

    print(
        "\n"
        +
        "-" * 100
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
        f"REASONING TYPE: "
        f"{result.get('reasoning_type')}"
    )


    print(
        f"REPAIR USED: "
        f"{result['repair_used']}"
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
    # DETERMINISTIC CALCULATION
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
    # CLAIM DEBUG
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
            f"CONTRADICTION: "
            f"{claim.get('contradiction', 0):.4f}"
        )

        print(
            f"EVIDENCE: "
            f"{claim.get('evidence')}"
        )


    # ------------------------------------------
    # POLARITY DEBUG
    # ------------------------------------------

    polarity_result = result.get(
        "polarity_result"
    )


    if polarity_result:

        print(
            "\nPOLARITY CHECK:"
        )

        print(
            f"POLARITY: "
            f"{polarity_result.get('polarity')}"
        )

        print(
            f"PROPOSITION: "
            f"{polarity_result.get('proposition')}"
        )

        print(
            f"SUPPORTED: "
            f"{polarity_result.get('supported')}"
        )

        print(
            f"ENTAILMENT: "
            f"{polarity_result.get('entailment', 0):.4f}"
        )

        print(
            f"CONTRADICTION: "
            f"{polarity_result.get('contradiction', 0):.4f}"
        )

        print(
            f"EVIDENCE: "
            f"{polarity_result.get('evidence')}"
        )


    if error_type:

        print(
            f"\nERROR TYPE: "
            f"{error_type}"
        )


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
# 29) METRICS
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
# 30) FINAL REPORT
# ==================================================

print(
    "\n\n"
    +
    "=" * 100
)

print(
    "FULL RAG V3 EVAL SONUCU"
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
    f"Deterministic reasoning ile çözülen: "
    f"{deterministic_success_count}"
)


# ==================================================
# 31) SAVE REPORT
# ==================================================

report = {

    "version":
        "rag_v3_polarity_and_deterministic_reasoning",

    "config": {

        "retrieval_threshold":
            RETRIEVAL_THRESHOLD,

        "evidence_relevance_threshold":
            EVIDENCE_RELEVANCE_THRESHOLD,

        "entailment_threshold":
            ENTAILMENT_THRESHOLD,

        "contradiction_threshold":
            CONTRADICTION_THRESHOLD,

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

    "pipeline_stats": {

        "repair_success_count":
            repair_success_count,

        "deterministic_success_count":
            deterministic_success_count
    },

    "tests":
        all_results
}


with open(
    "full_rag_eval_v3_results.json",
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
    "full_rag_eval_v3_results.json"
)