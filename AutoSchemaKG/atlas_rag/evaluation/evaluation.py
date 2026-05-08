import re
from collections import Counter
from typing import Tuple
import argparse
import json
from tqdm import tqdm

class QAJudger:
    def __init__(self):
        pass
    def normalize_for_retrieval(self, text: str) -> str:
        """Normalize text for retrieval comparison."""
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Normalize whitespace again
        return text
    def split_answer(self, generated_text):
        if "Answer:" in generated_text:
            generated_text = generated_text.split("Answer:")[-1]
        elif "answer:" in generated_text:
            generated_text = generated_text.split("answer:")[-1]
        # if answer is none
        if not generated_text:
            return "none"
        return generated_text

    def normalize_answer(self, answer: str) -> str:
        """Direct copy of the normalization from QAExactMatch/QAF1Score"""
        # Lowercase and normalize whitespace
        answer = answer.lower()
        # Replace hyphens with spaces
        answer = answer.replace('-', ' ')
        # Remove all other punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        # Standardize whitespace
        return ' '.join(answer.split())

    def judge(self, generated_text: str, reference_text: str) -> Tuple[int, float]:
        """Direct port of the original scoring logic"""
        # Extract answer from generated text
        pred_answer = self.split_answer(generated_text)
        
        # Normalize both answers
        pred_norm = self.normalize_answer(pred_answer)
        ref_norm = self.normalize_answer(reference_text)

        # Exact match calculation
        em = 1 if pred_norm == ref_norm else 0

        # F1 calculation (direct port from QAF1Score)
        pred_tokens = pred_norm.split()
        ref_tokens = ref_norm.split()
        
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return em, 0.0

        precision = num_same / len(pred_tokens) if pred_tokens else 0.0
        recall = num_same / len(ref_tokens) if ref_tokens else 0.0

        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return em, f1

    def recall_at_k(self, retrieved_text: list, reference_text: list, k: int) -> float:
        """Calculates recall at k based on the top k retrieved texts."""
        successful_retrievals = 0

        # Limit the retrieved texts to the top k entries
        limited_retrieved_text = retrieved_text[:k]
        normalized_reference_text = [self.normalize_for_retrieval(ref) for ref in reference_text]
        normalized_retrieved_text = [self.normalize_for_retrieval(ret) for ret in limited_retrieved_text]
        for ref_text in normalized_reference_text:
            for ret_text in normalized_retrieved_text:
                if ret_text in ref_text:
                    successful_retrievals += 1
                    break

        recall = successful_retrievals / len(reference_text) if reference_text else 0
        return recall

    # recall for 1 answer
    def recall(self, retrieved_text: list, reference_text: list) -> dict:
        """Calculates recall values at different k levels."""
        recall_values = {
            'recall@2': self.recall_at_k(retrieved_text, reference_text, 2),
            'recall@5': self.recall_at_k(retrieved_text, reference_text, 5),
        }
        return recall_values['recall@2'], recall_values['recall@5']
    

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--file_path", type=str, required=True, help="Path to the JSON file containing results.")
    args = argument_parser.parse_args()

    # Initialize the QAJudger
    llm_judge = QAJudger()

    # Load results from the JSON file
    result_list = []
    with open(args.file_path, 'r') as file:
        for line in file:
            if line.strip():  # Make sure the line is not empty
                try:
                    result = json.loads(line.strip())
                    result_list.append(result)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")

    # Debugging output to inspect the loaded data structure
    # print("Loaded data structure:", result_list)

    # Evaluate each entry in result_list
    for result in tqdm(result_list):
        if isinstance(result, dict):  # Ensure each result is a dictionary
            question = result["question"]
            answer = result["answer"]

            # Evaluate generated answers with Hippo and Hippo2
            hippo_generated_answer = result["hippo_generated_answer"]
            hippo2_generated_answer = result["hippo2_generated_answer"]

            # Split and judge the answers
            hippo_short_answer = llm_judge.split_answer(hippo_generated_answer)
            hippo_em, hippo_f1 = llm_judge.judge(hippo_short_answer, answer)

            hippo2_short_answer = llm_judge.split_answer(hippo2_generated_answer)
            hippo2_em, hippo2_f1 = llm_judge.judge(hippo2_short_answer, answer)

            # Store the scores back in the result dictionary
            result["hippo_em"] = hippo_em
            result["hippo_f1"] = hippo_f1
            result["hippo2_em"] = hippo2_em
            result["hippo2_f1"] = hippo2_f1

            result['recall@2'], result['recall@5'] = llm_judge.recall(result['hippo2_id'], result['gold_file_ids'])
            result['recall@2_hippo'], result['recall@5_hippo'] = llm_judge.recall(result['hippo_id'], result['gold_file_ids'])
            
    # Calculate averages
    average_em_with_hippo = sum(result["hippo_em"] for result in result_list) / len(result_list)
    average_em_with_hippo2 = sum(result["hippo2_em"] for result in result_list) / len(result_list)

    average_f1_with_hippo = sum(result["hippo_f1"] for result in result_list) / len(result_list)
    average_f1_with_hippo2 = sum(result["hippo2_f1"] for result in result_list) / len(result_list)
    
    average_recall2_with_hippo = sum(result['recall@2'] for result in result_list) / len(result_list)
    average_recall5_with_hippo = sum(result['recall@5'] for result in result_list) / len(result_list)
    average_recall2 = sum(result['recall@2_hippo'] for result in result_list) / len(result_list)
    average_recall5 = sum(result['recall@5_hippo'] for result in result_list) / len(result_list)
    # Output the averages
    print(f"Average EM with Hippo: {average_em_with_hippo:.4f}")
    print(f"Average EM with Hippo2: {average_em_with_hippo2:.4f}")
    print(f"Average F1 with Hippo: {average_f1_with_hippo:.4f}")
    print(f"Average F1 with Hippo2: {average_f1_with_hippo2:.4f}")
    
    print(f"Average Recall@2: {average_recall2:.4f}")
    print(f"Average Recall@5: {average_recall5:.4f}")
    print(f"Average Recall@2 with Hippo: {average_recall2_with_hippo:.4f}")
    print(f"Average Recall@5 with Hippo: {average_recall5_with_hippo:.4f}")