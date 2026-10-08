# AI Agent / RAG Project

Local, fully open-source RAG experimentation project focused on grounded question answering, claim verification, and local inference.

## Current Baseline

**Main working implementation:**

`eval_full_rag_v2.py`

**Version tag:**

`rag-baseline-v1.0`

This baseline includes:

- Semantic retrieval with Sentence Transformers
- Cosine similarity retrieval
- Evidence relevance filtering
- NLI-based claim verification
- Structured JSON claim generation
- Repair pass
- Deterministic rescue for selected return-policy reasoning cases

## Development Evaluation

22-question development set:

- Accuracy: 100%
- Precision: 100%
- Recall: 100%
- F1 Score: 100%

Results file:

`full_rag_eval_v2_rescue_results.json`

> Note: These results are from the development/tuning set and should not be interpreted as general test performance.

## Held-Out Evaluation v1

The baseline was also evaluated on a separate 30-question held-out set that was committed before evaluation.

Results:

- Accuracy: 96.67%
- Precision: 93.75%
- Recall: 100.00%
- F1 Score: 96.77%
- True Positive: 15
- True Negative: 14
- False Positive: 1
- False Negative: 0

Held-out set:

`heldout_eval_v1.json`

Results file:

`heldout_eval_v1_results.json`

The single false positive was an answer-relevance failure: the generated claim was supported by the document, but it did not actually answer the user's question.

---

# User Input RAG v1

Development branch:

`rag-user-input-v1`

This version extends the original verified RAG baseline by allowing users to provide their own knowledge text dynamically.

Instead of relying on a hard-coded document, the system builds a new knowledge base from user-provided text at runtime.

## Pipeline

```text
User Knowledge Text
        ↓
Sentence Splitting
        ↓
Overlapping Chunks
        ↓
SentenceTransformer Embeddings
        ↓
Cosine Nearest-Neighbor Index
        ↓
User Question
        ↓
Semantic Retrieval
        ↓
Qwen Local Generation
        ↓
Claim-Level NLI Verification
        ↓
Repair Pass
        ↓
Optional Deterministic Rescue
        ↓
Grounded Answer + Evidence
```

## Main Files

- `rag_user_input_v1.py` — dynamic verified RAG pipeline
- `app.py` — Streamlit web interface
- `eval_user_input_v1.py` — automated evaluation runner
- `user_input_eval_v1.json` — multi-domain evaluation dataset
- `requirements.txt` — project dependencies

## Web Interface

The Streamlit application allows the user to:

- Paste custom knowledge text
- Build a knowledge base dynamically
- Ask questions about that knowledge
- See whether the answer is supported
- Inspect the pipeline stage
- View retrieval scores
- Inspect verified claims and supporting evidence
- View evidence similarity scores
- View entailment scores
- See whether the repair stage was used
- See whether deterministic rescue was used

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on macOS/Linux:

```bash
source .venv/bin/activate
```

Install project dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Web Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will normally be available locally at:

```text
http://localhost:8501
```

All inference runs locally using open-source models.

No paid API is required.

## How the User Input Version Works

The user first provides a knowledge text.

The application then:

1. Splits the text into sentences
2. Creates overlapping chunks
3. Generates semantic embeddings
4. Builds a cosine-similarity nearest-neighbor index
5. Accepts user questions
6. Retrieves the most relevant chunks
7. Generates a structured answer using a local language model
8. Verifies generated claims against the retrieved evidence
9. Attempts a repair pass when necessary
10. Rejects answers that cannot be sufficiently supported

The knowledge text is not used to retrain the underlying models.

Instead, it acts as an external knowledge source that can be replaced dynamically.

## Models

### Embedding Model

`all-MiniLM-L6-v2`

Used for:

- Knowledge chunk embeddings
- Question embeddings
- Semantic similarity
- Retrieval
- Evidence relevance

### Local Language Model

`Qwen/Qwen2.5-0.5B-Instruct`

Used for:

- Structured answer generation
- Claim generation
- Repair generation

### Verification Model

`cross-encoder/nli-deberta-v3-base`

Used for:

- Claim-level entailment verification
- Determining whether evidence actually supports generated claims

## User Input Evaluation

The dynamic RAG version is currently evaluated across three different knowledge domains:

- University library policy
- Hotel policy
- Software subscription policy

The current regression evaluation contains:

```text
Total Questions: 14

True Positive:  9
True Negative:  5
False Positive: 0
False Negative: 0
```

Metrics:

- Accuracy: 100.00%
- Precision: 100.00%
- Recall: 100.00%
- F1 Score: 100.00%

> Important: This is a small development/regression evaluation set. The 100% result should not be interpreted as general real-world performance.

## Example Behaviors

### Supported Question

Knowledge:

```text
Students can borrow up to 5 books at the same time.
```

Question:

```text
How many books can students borrow at the same time?
```

Expected behavior:

```text
SUPPORTED
```

The answer is generated and verified against evidence from the supplied knowledge text.

### Unsupported Question

Knowledge:

```text
Students can borrow up to 5 books at the same time.
```

Question:

```text
Can students reserve private study rooms?
```

Expected behavior:

```text
NOT_SUPPORTED
```

The system should reject the question because the knowledge text does not contain sufficient information.

### Hard Negative

Knowledge:

```text
Students can borrow up to 5 books at the same time.
```

Question:

```text
Can students borrow up to 5 laptops at the same time?
```

Although the question is semantically similar to the source text, the information about laptops is not present.

Expected behavior:

```text
NOT_SUPPORTED
```

This demonstrates why semantic retrieval alone is not sufficient and why claim verification is included in the pipeline.

## Evaluation Philosophy

The project separates two different questions:

```text
Is the question semantically related to the document?

and

Does the document actually support the generated answer?
```

Semantic similarity is handled by the embedding and retrieval stages.

Answer support is handled by the verification stage.

This distinction helps reduce cases where a highly similar document chunk is incorrectly treated as sufficient evidence.

## Current Architecture

```text
                    User Knowledge Text
                            │
                            ▼
                    Sentence Splitting
                            │
                            ▼
                       Chunking
                            │
                            ▼
                 SentenceTransformer
                       Embeddings
                            │
                            ▼
                NearestNeighbors Index
                            │
                            │
User Question ──────────────┘
        │
        ▼
Semantic Retrieval
        │
        ▼
Retrieval Threshold
        │
        ├── Low relevance
        │       ↓
        │   NOT_SUPPORTED
        │
        ▼
Local Qwen Generation
        │
        ▼
Structured Answer + Claims
        │
        ▼
Evidence Relevance Check
        │
        ▼
DeBERTa NLI Verification
        │
        ├── Supported
        │       ↓
        │     Answer
        │
        └── Unsupported
                ↓
           Repair Pass
                │
                ▼
        Verification Again
                │
                ├── Supported
                │       ↓
                │     Answer
                │
                └── Unsupported
                        ↓
             Optional Deterministic Rescue
                        │
                        ▼
              Answer or NOT_SUPPORTED
```

## Current Goal

The goal of `rag-user-input-v1` is to turn the original verified RAG research prototype into a reusable local application where arbitrary user-provided knowledge can be indexed and queried without retraining the underlying language model.

The project currently focuses on:

- Grounded question answering
- Local inference
- Open-source models
- Dynamic knowledge input
- Claim-level verification
- Hallucination reduction
- Automated regression evaluation
- Transparent evidence inspection

## Known Limitation

The original held-out evaluation revealed an answer-relevance failure.

In that case, a generated claim was correctly supported by the source document, but the claim did not actually answer the user's question.

This means that:

```text
Evidence supports claim
```

does not always guarantee:

```text
Claim answers question
```

Question-answer relevance is therefore a known area for future experimentation.

## Future Work

Possible next steps include:

- Dedicated question-answer relevance verification
- Larger multi-domain evaluation sets
- More adversarial hard-negative examples
- Improved web interface
- Persistent knowledge bases
- File-based document ingestion
- Multiple-document support
- More general deterministic reasoning tools
- Better observability and evaluation reports