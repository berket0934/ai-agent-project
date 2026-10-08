import json
import rag_user_input_v1 as rag


with open("user_input_eval_v1.json", "r", encoding="utf-8") as f:
    eval_scenarios = json.load(f)


print(f"Loaded {len(eval_scenarios)} evaluation scenario(s).")

total_questions = 0
passed_questions = 0
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for scenario in eval_scenarios:

    print("\n" + "=" * 70)
    print(f"SCENARIO: {scenario['name']}")
    print("=" * 70)

    rag.build_knowledge_base(
        scenario["knowledge_text"]
    )

    for item in scenario["questions"]:

        question = item["question"]
        expected_supported = item["expected_supported"]

        result = rag.run_rag_v2(
            question
        )

        actual_supported = (
            result["status"] == "supported"
        )

        passed = (
            actual_supported == expected_supported
        )
        if expected_supported and actual_supported:
             true_positive += 1

        elif not expected_supported and not actual_supported:
             true_negative += 1

        elif not expected_supported and actual_supported:
             false_positive += 1

        else:
              false_negative += 1
        total_questions += 1

        if passed:
             passed_questions += 1

        print(f"\nQuestion: {question}")
        print(f"Expected supported: {expected_supported}")
        print(f"Actual supported:   {actual_supported}")
        print(f"Pipeline stage:     {result['pipeline_stage']}")
        print(f"Result:             {'PASS' if passed else 'FAIL'}")


        print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

accuracy = (
    passed_questions / total_questions
) * 100

print(f"Total questions: {total_questions}")
print(f"Passed:          {passed_questions}")
print(f"Failed:          {total_questions - passed_questions}")
print(f"Accuracy:        {accuracy:.2f}%")

precision = (
    true_positive / (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)

recall = (
    true_positive / (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)

print()
print(f"TP:              {true_positive}")
print(f"TN:              {true_negative}")
print(f"FP:              {false_positive}")
print(f"FN:              {false_negative}")
print()
print(f"Precision:       {precision * 100:.2f}%")
print(f"Recall:          {recall * 100:.2f}%")
print(f"F1 Score:        {f1 * 100:.2f}%")