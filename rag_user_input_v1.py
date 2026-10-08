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

def read_user_document():

    print("\n" + "=" * 70)
    print("KNOWLEDGE TEXT")
    print("=" * 70)

    print(
        "\nPaste your knowledge text below."
        "\nWhen you are finished, type END on a new line."
        "\n"
    )

    lines = []

    while True:

        line = input()

        if line.strip() == "END":
            break

        lines.append(line)

    document_text = "\n".join(lines).strip()

    if not document_text:

        raise ValueError(
            "Knowledge text cannot be empty."
        )

    return document_text




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


chunks = []

# ==================================================
# 5) MODELLER
# ==================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print("Loading local language model...")

llm_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    llm_name
)

llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype="auto"
)


print("Loading verification model...")

verifier = CrossEncoder(
    "cross-encoder/nli-deberta-v3-base"
)


# ==================================================
# 6) VECTOR INDEX
# ==================================================

chunk_embeddings = None
index = None


def build_knowledge_base(document_text):

    global document
    global chunks
    global chunk_embeddings
    global index
    global RETURN_WINDOW_DAYS

    if not document_text.strip():
        raise ValueError(
            "Knowledge text cannot be empty."
        )
    document = document_text
    chunks = create_chunks(document_text)

    print(
        "\nKnowledge text processed successfully."
    )

    print(
        f"Created {len(chunks)} chunks."
    )

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
    RETURN_WINDOW_DAYS = get_return_window_days()
    print(
        "Vector index ready."
    )
def build_knowledge_base_from_chunks(document_chunks):

    global document
    global chunks
    global chunk_embeddings
    global index
    global RETURN_WINDOW_DAYS

    if not document_chunks:
        raise ValueError(
            "Document chunks cannot be empty."
        )

    chunk_texts = [
        chunk["text"]
        for chunk in document_chunks
        if chunk.get("text", "").strip()
    ]

    if not chunk_texts:
        raise ValueError(
            "Document chunks do not contain any text."
        )

    chunks = [
        chunk
        for chunk in document_chunks
        if chunk.get("text", "").strip()
    ]

    document = "\n".join(
        chunk["text"]
        for chunk in chunks
    )

    print(
        "\nDocument chunks processed successfully."
    )

    print(
        f"Created {len(chunks)} metadata chunks."
    )

    chunk_embeddings = embedding_model.encode(
        [
            chunk["text"]
            for chunk in chunks
        ],
        normalize_embeddings=True
    )

    index = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    index.fit(
        chunk_embeddings
    )

    RETURN_WINDOW_DAYS = get_return_window_days()

    print(
        "Vector index ready."
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

          chunk_item = chunks[idx]

          if isinstance(chunk_item, dict):

            result = {
                "text": chunk_item["text"],
                "score": float(
                    1 - distance
                ),
                "source": chunk_item.get(
                    "source"
                ),
                "page": chunk_item.get(
                    "page"
                )
            }

          else:

            result = {
                "text": chunk_item,
                "score": float(
                    1 - distance
                )
            }

          results.append(
            result
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
            result["score"],
        "source":
            result.get("source"),
        "page":
            result.get("page")
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
            result["score"],
        "source":
            result.get("source"),
        "page":
            result.get("page")
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
                    "source":
                        None,

                    "page":
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
                "source":
                    best_evidence.get("source"),

                "page":
                     best_evidence.get("page"),    

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


RETURN_WINDOW_DAYS = None


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
# 23) INTERACTIVE USER QUESTION LOOP
# ==================================================

def print_rag_result(result):

    print("\n" + "=" * 70)
    print("RAG RESULT")
    print("=" * 70)

    print(
        f"\nSTATUS: "
        f"{result['status'].upper()}"
    )

    print(
        f"\nANSWER:\n"
        f"{result['answer']}"
    )


    # ------------------------------------------
    # Pipeline info
    # ------------------------------------------

    if result.get("pipeline_stage"):

        print(
            f"\nPIPELINE STAGE: "
            f"{result['pipeline_stage']}"
        )


    if result.get("retrieval_score") is not None:

        print(
            f"RETRIEVAL SCORE: "
            f"{result['retrieval_score']:.4f}"
        )


    if result.get("repair_used"):

        print(
            "REPAIR USED: True"
        )


    if result.get("rescue_used"):

        print(
            "DETERMINISTIC RESCUE USED: True"
        )


    # ------------------------------------------
    # Evidence / Claim details
    # ------------------------------------------

    claims = result.get(
        "claims",
        []
    )


    if claims:

        print(
            "\nVERIFIED CLAIMS / EVIDENCE"
        )

        print(
            "-" * 70
        )


        for number, claim in enumerate(
            claims,
            start=1
        ):

            print(
                f"\nCLAIM {number}:"
            )

            print(
                claim.get(
                    "claim",
                    "N/A"
                )
            )


            print(
                f"\nSUPPORTED: "
                f"{claim.get('supported', False)}"
            )


            evidence = claim.get(
                "evidence"
            )


            if evidence:

                print(
                    f"\nEVIDENCE:\n"
                    f"{evidence}"
                )


            evidence_similarity = claim.get(
                "evidence_similarity"
            )


            if evidence_similarity is not None:

                print(
                    f"\nEVIDENCE SIMILARITY: "
                    f"{evidence_similarity:.4f}"
                )


            entailment = claim.get(
                "entailment"
            )


            if entailment is not None:

                print(
                    f"ENTAILMENT: "
                    f"{entailment:.4f}"
                )


    print(
        "\n" + "=" * 70
    )


# ==================================================
# 24) INTERACTIVE CHAT
# ==================================================

def interactive_chat():
    document_text = read_user_document()
    build_knowledge_base(document_text)

    print("\n" + "=" * 70)
    print("LOCAL VERIFIED RAG")
    print("=" * 70)

    print(
        "\nYour knowledge text has been indexed successfully."
    )

    print(
        "\nYou can now ask questions about the text."
    )

    print(
        "Type 'exit' when you want to stop."
    )


    while True:

        question = input(
            "\nAsk a question: "
        ).strip()


        # ------------------------------------------
        # Exit
        # ------------------------------------------

        if question.lower() in {
            "exit",
            "quit"
        }:

            print(
                "\nSession ended."
            )

            break


        # ------------------------------------------
        # Empty question
        # ------------------------------------------

        if not question:

            print(
                "Please enter a question."
            )

            continue


        # ------------------------------------------
        # Run RAG
        # ------------------------------------------

        try:

            result = run_rag_v2(
                question
            )


            print_rag_result(
                result
            )


        except Exception as error:

            print(
                f"\nAn error occurred: "
                f"{error}"
            )


# ==================================================
# 25) START PROGRAM
# ==================================================

if __name__ == "__main__":

    interactive_chat()
