import os
import pandas as pd
from collections import defaultdict

input_folder = "./../data/results"
output_folder = "./../data/grouped_results"
os.makedirs(output_folder, exist_ok=True)

files_by_model = defaultdict(
    lambda: {"easy": [], "medium": [], "hard": []})

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        hops = int(filename[4])
        model_name = "gemini" if "gemini" in filename else "deepseek"
        full_path = os.path.join(input_folder, filename)

        files_by_model[model_name]["easy" if hops == 1 else (
            "medium" if hops == 2 else "hard")].append(full_path)

os.makedirs(output_folder, exist_ok=True)


def change_word_order(pred, true):
    print(pred, true)
    pred = sorted([word.lower().strip() for word in pred.split(',')])
    true = sorted([word.lower().strip() for word in true.split(',')])
    print(pred, true)
    result_pred = []
    result_true = []
    for word in pred:
        if word in true:
            result_pred.append(word)
            result_true.append(word)
    print(result_pred, result_true)
    for word in pred:
        if word not in result_pred:
            result_pred.append(word)
    for word in true:
        if word not in result_true:
            result_true.append(word)
    print(result_pred, result_true)
    print()
    return ','.join(result_pred), ','.join(result_true)


for model, parts in files_by_model.items():
    for difficulty in ["easy", "medium", "hard"]:
        dfs = [pd.read_csv(f) for f in parts[difficulty]]
        df_combined = pd.concat(dfs, ignore_index=True)
        print(df_combined.columns)
        df_combined[['llm_response', 'answer']] = df_combined.apply(
            lambda row: pd.Series(change_word_order(
                row['llm_response'], row['answer'])),
            axis=1
        )
        df_combined.to_csv(os.path.join(
            output_folder, f"{difficulty}_{model}.csv"), index=False)
