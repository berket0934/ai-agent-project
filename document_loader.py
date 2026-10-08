import pymupdf
import re

def load_pdf_pages(pdf_bytes, source_name):
    """
    Extract text from a PDF page by page.

    Returns:
    [
        {
            "text": "...",
            "source": "document.pdf",
            "page": 1
        },
        ...
    ]
    """

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_index in range(
        len(document)
    ):

        page = document[
            page_index
        ]

        text = page.get_text(
            "text"
        ).strip()

        # Skip empty pages
        if not text:
            continue

        pages.append(
            {
                "text": text,
                "source": source_name,
                "page": page_index + 1
            }
        )

    document.close()

    return pages
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


def create_page_chunks(
    pages,
    sentences_per_chunk=2,
    overlap_sentences=1
):

    chunks = []

    for page in pages:

        sentences = split_into_sentences(
            page["text"]
        )

        start = 0

        while start < len(sentences):

            end = (
                start
                + sentences_per_chunk
            )

            selected = sentences[
                start:end
            ]

            chunk_text = " ".join(
                selected
            )

            if chunk_text.strip():

                chunks.append(
                    {
                        "text": chunk_text,
                        "source": page["source"],
                        "page": page["page"]
                    }
                )

            start = (
                end
                - overlap_sentences
            )

    return chunks