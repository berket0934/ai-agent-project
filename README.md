# AI Agent / RAG Project

Local, fully open-source RAG experimentation project.

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
