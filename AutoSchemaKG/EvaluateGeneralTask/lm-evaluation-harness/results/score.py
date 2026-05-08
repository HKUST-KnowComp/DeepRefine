import json
with open("FILE_PATH.json", 'r') as f:
    result = json.load(f)["results"]

score_list = []
stderr_list = []
for name, score in result.items():
    if "exact_match,get_response" not in score:
        continue
    score_list.append(score["exact_match,get_response"])
    stderr_list.append(score["exact_match_stderr,get_response"])
# mean
mean_score = sum(score_list) / len(score_list) * 100
print(mean_score)

score_list = []
for name, score in result.items():
    
    if name == "mmlu_generative":
        continue
    if name == "humanities":
        score_list = []
        continue
    if name in ["other", "social sciences", "stem"]:
        # mean
        mean_score = sum(score_list) / len(score_list)
        print(mean_score*100)
        score_list = []
        continue
    score_list.append(score["exact_match,get_response"])
# mean
mean_score = sum(score_list) / len(score_list)
print(mean_score*100)

