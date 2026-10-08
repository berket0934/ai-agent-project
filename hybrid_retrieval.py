from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


class HybridRetriever:

    def __init__(
        self,
        embedding_model,
        semantic_candidates=10,
        lexical_candidates=10
    ):

        self.embedding_model = embedding_model

        self.semantic_candidates = semantic_candidates
        self.lexical_candidates = lexical_candidates

        self.chunks = []

        self.semantic_index = None
        self.semantic_embeddings = None

        self.lexical_vectorizer = None
        self.lexical_matrix = None

        self.reranker = CrossEncoder(
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        )


    def build(
        self,
        chunks
    ):

        if not chunks:

            raise ValueError(
                "Chunks cannot be empty."
            )


        self.chunks = [
            chunk
            for chunk in chunks
            if chunk.get(
                "text",
                ""
            ).strip()
        ]


        if not self.chunks:

            raise ValueError(
                "Chunks do not contain text."
            )


        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]


        # ==========================================
        # SEMANTIC INDEX
        # ==========================================

        self.semantic_embeddings = (
            self.embedding_model.encode(
                texts,
                normalize_embeddings=True
            )
        )


        self.semantic_index = (
            NearestNeighbors(
                metric="cosine",
                algorithm="brute"
            )
        )


        self.semantic_index.fit(
            self.semantic_embeddings
        )


        # ==========================================
        # LEXICAL INDEX
        # ==========================================

        self.lexical_vectorizer = (
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2)
            )
        )


        self.lexical_matrix = (
            self.lexical_vectorizer.fit_transform(
                texts
            )
        )


    def retrieve(
        self,
        query,
        top_k=5
    ):

        if self.semantic_index is None:

            raise ValueError(
                "Hybrid retriever has not been built."
            )


        # ==========================================
        # SEMANTIC SEARCH
        # ==========================================

        query_embedding = (
            self.embedding_model.encode(
                [query],
                normalize_embeddings=True
            )
        )


        semantic_count = min(
            self.semantic_candidates,
            len(self.chunks)
        )


        distances, indices = (
            self.semantic_index.kneighbors(
                query_embedding,
                n_neighbors=semantic_count
            )
        )


        semantic_candidates = {}

        for rank, (
            distance,
            idx
        ) in enumerate(
            zip(
                distances[0],
                indices[0]
            ),
            start=1
        ):

            idx = int(idx)

            semantic_candidates[
                idx
            ] = {
                "rank": rank,
                "score": float(
                    1 - distance
                )
            }


        # ==========================================
        # LEXICAL SEARCH
        # ==========================================

        query_vector = (
            self.lexical_vectorizer.transform(
                [query]
            )
        )


        lexical_scores_array = (
            cosine_similarity(
                query_vector,
                self.lexical_matrix
            )[0]
        )


        lexical_count = min(
            self.lexical_candidates,
            len(self.chunks)
        )


        lexical_indices = (
            lexical_scores_array
            .argsort()[::-1][
                :lexical_count
            ]
        )


        lexical_candidates = {}

        for rank, idx in enumerate(
            lexical_indices,
            start=1
        ):

            idx = int(idx)

            lexical_candidates[
                idx
            ] = {
                "rank": rank,
                "score": float(
                    lexical_scores_array[
                        idx
                    ]
                )
            }


        # ==========================================
        # CANDIDATE UNION
        # ==========================================

        candidate_indices = list(
            set(
                semantic_candidates.keys()
            )
            |
            set(
                lexical_candidates.keys()
            )
        )


        pairs = [
            [
                query,
                self.chunks[
                    idx
                ]["text"]
            ]
            for idx in candidate_indices
        ]


        # ==========================================
        # MULTILINGUAL RERANKING
        # ==========================================

        reranker_scores = (
            self.reranker.predict(
                pairs
            )
        )


        ranked_candidates = sorted(
            zip(
                candidate_indices,
                reranker_scores
            ),
            key=lambda item: float(
                item[1]
            ),
            reverse=True
        )


        # ==========================================
        # FINAL RESULTS
        # ==========================================

        results = []


        for idx, reranker_score in (
            ranked_candidates[:top_k]
        ):

            chunk = self.chunks[
                idx
            ]


            semantic_info = (
                semantic_candidates.get(
                    idx
                )
            )


            lexical_info = (
                lexical_candidates.get(
                    idx
                )
            )


            results.append(
                {
                    "text":
                        chunk["text"],

                    "source":
                        chunk.get(
                            "source"
                        ),

                    "page":
                        chunk.get(
                            "page"
                        ),

                    "reranker_score":
                        float(
                            reranker_score
                        ),

                    "semantic_score":
                        (
                            semantic_info[
                                "score"
                            ]
                            if semantic_info
                            else None
                        ),

                    "semantic_rank":
                        (
                            semantic_info[
                                "rank"
                            ]
                            if semantic_info
                            else None
                        ),

                    "lexical_score":
                        (
                            lexical_info[
                                "score"
                            ]
                            if lexical_info
                            else None
                        ),

                    "lexical_rank":
                        (
                            lexical_info[
                                "rank"
                            ]
                            if lexical_info
                            else None
                        )
                }
            )


        return results