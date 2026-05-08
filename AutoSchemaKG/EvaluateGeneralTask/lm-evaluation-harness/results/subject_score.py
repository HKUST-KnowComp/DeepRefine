import json

subjects = {
    "formal_logic": "Logic",
    "high_school_european_history": "History",
    "high_school_us_history": "History",
    "high_school_world_history": "History",
    "international_law": "Law",
    "jurisprudence": "Law",
    "logical_fallacies": "Logic",
    "moral_disputes": "Philosophy_and_Ethics",
    "moral_scenarios": "Philosophy_and_Ethics",
    "philosophy": "Philosophy_and_Ethics",
    "prehistory": "History",
    "professional_law": "Law",
    "world_religions": "Religion",
    "business_ethics": "Philosophy_and_Ethics",
    "clinical_knowledge": "Medicine_and_Health",
    "college_medicine": "Medicine_and_Health",
    "global_facts": "Global_Facts",
    "human_aging": "Medicine_and_Health",
    "management": "Business_and_Management",
    "marketing": "Business_and_Management",
    "medical_genetics": "Medicine_and_Health",
    "miscellaneous": "Global_Facts",
    "nutrition": "Medicine_and_Health",
    "professional_accounting": "Business_and_Management",
    "professional_medicine": "Medicine_and_Health",
    "virology": "Medicine_and_Health",
    "econometrics": "Economics",
    "high_school_geography": "Social_Sciences",
    "high_school_government_and_politics": "Social_Sciences",
    "high_school_macroeconomics": "Economics",
    "high_school_microeconomics": "Economics",
    "high_school_psychology": "Social_Sciences",
    "human_sexuality": "Social_Sciences",
    "professional_psychology": "Social_Sciences",
    "public_relations": "Business_and_Management",
    "security_studies": "Social_Sciences",
    "sociology": "Social_Sciences",
    "us_foreign_policy": "Social_Sciences",
    "abstract_algebra": "Mathematics",
    "anatomy": "Medicine_and_Health",
    "astronomy": "Natural_Sciences",
    "college_biology": "Natural_Sciences",
    "college_chemistry": "Natural_Sciences",
    "college_computer_science": "Computer_Science_and_Engineering",
    "college_mathematics": "Mathematics",
    "college_physics": "Natural_Sciences",
    "computer_security": "Computer_Science_and_Engineering",
    "conceptual_physics": "Natural_Sciences",
    "electrical_engineering": "Computer_Science_and_Engineering",
    "elementary_mathematics": "Mathematics",
    "high_school_biology": "Natural_Sciences",
    "high_school_chemistry": "Natural_Sciences",
    "high_school_computer_science": "Computer_Science_and_Engineering",
    "high_school_mathematics": "Mathematics",
    "high_school_physics": "Natural_Sciences",
    "high_school_statistics": "Mathematics",
    "machine_learning": "Computer_Science_and_Engineering",
}


# File paths for MMLU
base_file = "FILEPATH_TO_MMLU_RESULTS_BASELINE.json"

random_file = "FILEPATH_TO_MMLU_RESULTS_RANDOM.json"
bm25_file = "FILEPATH_TO_MMLU_RESULTS_BM25.json"
vector_file = "FILEPATH_TO_MMLU_RESULTS_VECTOR.json"
kg_file = "FILEPATH_TO_MMLU_RESULTS_KG.json"


for file in [base_file, random_file, bm25_file, vector_file, kg_file]:

    with open(file, 'r') as f:
        result = json.load(f)["results"]

    score_list = []
    for name, score in result.items():
        if "exact_match,get_response" not in score:
            continue
        score_list.append(score["exact_match,get_response"])
    # mean
    mean_score = sum(score_list) / len(score_list) * 100
    print(len(score_list))
    print(mean_score)

    score_dic = {
        "Logic": [],
        "History": [],
        "Law": [],
        "Philosophy_and_Ethics": [],
        "Religion": [],
        "Medicine_and_Health": [],
        "Global_Facts": [],
        "Business_and_Management": [],
        "Economics": [],
        "Mathematics": [],
        "Social_Sciences": [],
        "Natural_Sciences": [],
        "Computer_Science_and_Engineering": []
    }

    for _, score in result.items():
        if "exact_match,get_response" not in score:
            continue
        name = score["alias"].replace("  - ", "")
        score_dic[subjects[name]].append(score["exact_match,get_response"])

    for name, scores in score_dic.items():
        if len(scores) == 0:
            continue
        mean_score = sum(scores) / len(scores)
        print(name, f"{mean_score * 100:.2f}")

    print("============")
