import os
import pandas as pd
from collections import defaultdict

input_folder = "./../data/results"
output_folder = "./../data/grouped_results"
os.makedirs(output_folder, exist_ok=True)

files_by_model = defaultdict(lambda: {"easy": None, "medium": [], "hard": None})

for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):
        parts = filename.split("_", 1)
        level = int(parts[0][1])
        model_name = parts[1].replace(".csv", "")
        full_path = os.path.join(input_folder, filename)

        if level == 1:
            files_by_model[model_name]["easy"] = full_path
        elif 2 <= level <= 4:
            files_by_model[model_name]["medium"].append(full_path)
        elif level == 5:
            files_by_model[model_name]["hard"] = full_path

os.makedirs(output_folder, exist_ok=True)

for model, parts in files_by_model.items():
    if parts["easy"]:
        df = pd.read_csv(parts["easy"])
        df.to_csv(os.path.join(output_folder, f"easy_{model}.csv"), index=False)

    if parts["medium"]:
        dfs = [pd.read_csv(f) for f in parts["medium"]]
        df_combined = pd.concat(dfs, ignore_index=True)
        df_combined.to_csv(os.path.join(output_folder, f"medium_{model}.csv"), index=False)

    if parts["hard"]:
        df = pd.read_csv(parts["hard"])
        df.to_csv(os.path.join(output_folder, f"hard_{model}.csv"), index=False)

