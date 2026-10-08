import streamlit as st
import rag_user_input_v1 as rag

from document_loader import (
    load_pdf_pages,
    create_page_chunks
)

st.set_page_config(
    page_title="Local Verified RAG",
    page_icon="🧠",
    layout="centered"
)


st.title("Local Verified RAG")

st.write(
    "Paste your own knowledge text and ask grounded questions locally."
)
source_type = st.radio(
    "Knowledge Source",
    [
        "Paste Text",
        "Upload PDF"
    ],
    horizontal=True
)

knowledge_text = ""
uploaded_pdf = None


if source_type == "Paste Text":

    knowledge_text = st.text_area(
        "Knowledge Text",
        height=250,
        placeholder="Paste your knowledge text here..."
    )


elif source_type == "Upload PDF":

    uploaded_pdf = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

question = st.text_input(
    "Question",
    placeholder="Ask a question about the knowledge text..."
)

if st.button("Build Knowledge Base"):

    # ==========================================
    # TEXT INPUT
    # ==========================================

    if source_type == "Paste Text":

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
            st.session_state["source_type"] = "text"

            st.success(
                "Knowledge base is ready."
            )


    # ==========================================
    # PDF INPUT
    # ==========================================

    elif source_type == "Upload PDF":

        if uploaded_pdf is None:

            st.warning(
                "Please upload a PDF first."
            )

        else:

            with st.spinner(
                "Reading and indexing PDF..."
            ):

                pages = load_pdf_pages(
                    uploaded_pdf.getvalue(),
                    uploaded_pdf.name
                )

                document_chunks = create_page_chunks(
                    pages
                )

                rag.build_knowledge_base_from_chunks(
                    document_chunks
                )

            st.session_state["knowledge_built"] = True
            st.session_state["source_type"] = "pdf"
            st.session_state["document_name"] = uploaded_pdf.name

            st.success(
                f"Knowledge base ready: "
                f"{len(pages)} pages, "
                f"{len(document_chunks)} chunks."
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

                        source = claim.get(
                         "source"
                        )

                        page = claim.get(
                            "page"
                        )

                        if source:

                            st.write(
                                f"**Source:** {source}"
                            )

                        if page is not None:

                            st.write(
                                f"**Page:** {page}"
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