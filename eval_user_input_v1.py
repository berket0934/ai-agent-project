import json
import rag_user_input_v1 as rag


with open("user_input_eval_v1.json", "r", encoding="utf-8") as f:
    eval_scenarios = json.load(f)


print(f"Loaded {len(eval_scenarios)} evaluation scenario(s).")

total_questions = 0
passed_questions = 0

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