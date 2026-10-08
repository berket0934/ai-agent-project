from sentence_transformers import CrossEncoder
import numpy as np


# ==================================================
# 1) NLI VERIFIER MODEL
# ==================================================

model = CrossEncoder(
    "cross-encoder/nli-deberta-v3-base"
)


# Modelin sınıf sırası
label_mapping = [
    "contradiction",
    "entailment",
    "neutral"
]


# ==================================================
# 2) TEST ÖRNEKLERİ
# ==================================================

tests = [

    # ----------------------------------------------
    # ENTAILMENT
    # ----------------------------------------------
    {
        "premise": (
            "All electronic products include "
            "a two-year warranty."
        ),

        "hypothesis": (
            "Electronic products have "
            "a two-year warranty."
        )
    },


    # ----------------------------------------------
    # CONTRADICTION
    # ----------------------------------------------
    {
        "premise": (
            "Customers can return products within "
            "30 days after delivery. "
            "The product must be unused."
        ),

        "hypothesis": (
            "A used product can be returned "
            "within 10 days."
        )
    },


    # ----------------------------------------------
    # NEUTRAL
    # ----------------------------------------------
    {
        "premise": (
            "Premium members receive free "
            "standard shipping. "
            "Express shipping normally takes "
            "1 to 2 business days."
        ),

        "hypothesis": (
            "Premium members receive "
            "free express shipping."
        )
    },


    # ----------------------------------------------
    # NEUTRAL - BITCOIN
    # ----------------------------------------------
    {
        "premise": (
            "Premium membership can be "
            "cancelled at any time."
        ),

        "hypothesis": (
            "Customers can pay with Bitcoin."
        )
    }
]


# ==================================================
# 3) VERIFICATION
# ==================================================

for i, test in enumerate(
    tests,
    start=1
):

    pair = [
        (
            test["premise"],
            test["hypothesis"]
        )
    ]

    scores = model.predict(
        pair
    )[0]


    # En yüksek skorlu sınıf
    best_index = np.argmax(scores)

    label = label_mapping[
        best_index
    ]


    print("\n" + "=" * 80)

    print(
        f"TEST {i}"
    )

    print(
        "\nPREMISE:"
    )

    print(
        test["premise"]
    )

    print(
        "\nHYPOTHESIS:"
    )

    print(
        test["hypothesis"]
    )

    print(
        "\nSONUÇ:"
    )

    print(
        label.upper()
    )

    print(
        "\nSKORLAR:"
    )

    print(
        f"Contradiction: {scores[0]:.4f}"
    )

    print(
        f"Entailment:    {scores[1]:.4f}"
    )

    print(
        f"Neutral:       {scores[2]:.4f}"
    )