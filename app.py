import streamlit as st
import rag_user_input_v1 as rag

st.set_page_config(
    page_title="Local Verified RAG",
    page_icon="🧠",
    layout="centered"
)


st.title("Local Verified RAG")

st.write(
    "Paste your own knowledge text and ask grounded questions locally."
)

knowledge_text = st.text_area(
    "Knowledge Text",
    height=250,
    placeholder="Paste your knowledge text here..."
)

question = st.text_input(
    "Question",
    placeholder="Ask a question about the knowledge text..."
)

if st.button("Build Knowledge Base"):

    if not knowledge_text.strip():

        st.warning(
            "Please enter some knowledge text first."
        )

    else:

        with st.spinner(
            "Building knowledge base..."
        ):

            rag.build_knowledge_base(
                knowledge_text
            )

        st.session_state["knowledge_built"] = True

        st.success(
            "Knowledge base is ready."
        )
if st.button("Ask"):

    if not st.session_state.get(
        "knowledge_built",
        False
    ):

        st.warning(
            "Please build the knowledge base first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner(
            "Checking the knowledge base..."
        ):

            result = rag.run_rag_v2(
                question
            )

        st.subheader("Answer")

        if result["status"] == "supported":

            st.success(
                result["answer"]
            )

        else:

            st.warning(
                result["answer"]
            )

        st.write(
            f"**Status:** {result['status'].upper()}"
        )

        st.write(
            f"**Pipeline Stage:** {result['pipeline_stage']}"
        )

        st.write(
            f"**Retrieval Score:** "
            f"{result['retrieval_score']:.4f}"
        )
        if result.get("repair_used"):
            st.write("**Repair Used:** Yes")

        if result.get("rescue_used"):
            st.write("**Deterministic Rescue Used:** Yes")
        claims = result.get(
            "claims",
            []
        )

        if claims:

            st.subheader(
                "Verified Evidence"
            )

            for number, claim in enumerate(
                claims,
                start=1
            ):

                with st.expander(
                    f"Claim {number}"
                ):

                    st.write(
                        f"**Claim:** "
                        f"{claim.get('claim', 'N/A')}"
                    )

                    st.write(
                        f"**Supported:** "
                        f"{claim.get('supported', False)}"
                    )

                    evidence = claim.get(
                        "evidence"
                    )

                    if evidence:

                        st.write(
                            f"**Evidence:** {evidence}"
                        )

                    evidence_similarity = claim.get(
                        "evidence_similarity"
                    )

                    if evidence_similarity is not None:

                        st.write(
                            f"**Evidence Similarity:** "
                            f"{evidence_similarity:.4f}"
                        )

                    entailment = claim.get(
                        "entailment"
                    )

                    if entailment is not None:

                        st.write(
                            f"**Entailment:** "
                            f"{entailment:.4f}"
                        )