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
