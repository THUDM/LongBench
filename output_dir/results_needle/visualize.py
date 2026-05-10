
import argparse
import json
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get("TMPDIR", "/tmp")) / "qwen35_compression_matplotlib_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

RESULTS_DIR = BASE_DIR / "results"
IMG_DIR = BASE_DIR / "img"
PRETRAINED_LEN = 33000
RESULT_FILE_RE = re.compile(r"^(?P<model>.+)_len_(?P<context>\d+)_depth_(?P<depth>\d+)_results\.json$")


def parse_result_filename(path):
    match = RESULT_FILE_RE.match(path.name)
    if not match:
        return None, None
    return int(match.group("context")), int(match.group("depth")) / 100


def score_response(json_data):
    model_response = (json_data.get("model_response") or "").lower()
    expected_answer = "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
    if model_response:
        return len(set(model_response.split()).intersection(set(expected_answer))) / len(set(expected_answer))

    score = json_data.get("score")
    if score is None:
        return 0.0
    return min(float(score) / 10, 1.0)


def load_results(model_dir):
    json_files = sorted(model_dir.glob("*_len_*_depth_*_results.json"))

    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, "r") as f:
            json_data = json.load(f)

        filename_context_length, filename_document_depth = parse_result_filename(file)
        document_depth = json_data.get("depth_percent", filename_document_depth)
        context_length = json_data.get("context_length", filename_context_length)
        if document_depth is None or context_length is None:
            print(f"Skipping unparseable result file: {file}")
            continue

        # Appending to the list
        data.append(
            {
                "Document Depth": float(document_depth),
                "Context Length": int(context_length),
                "Score": score_response(json_data),
            }
        )

    return pd.DataFrame(data)


def plot_model_results(model_dir):
    model_name = model_dir.name
    print("model_name = %s" % model_name)

    df = load_results(model_dir)
    if df.empty:
        print(f"No result files found in {model_dir}, skipping.")
        return


    locations = list(df["Context Length"].unique())
    locations.sort()
    pretrained_len = len(locations)
    for li, l in enumerate(locations):
        if l > PRETRAINED_LEN:
            pretrained_len = li
            break

    print(df.head())
    print("Overall score %.3f" % df["Score"].mean())

    pivot_table = pd.pivot_table(df, values='Score', index=['Document Depth', 'Context Length'], aggfunc='mean').reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(index="Document Depth", columns="Context Length", values="Score") # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    # Create the heatmap with better aesthetics
    plt.figure(figsize=(38, 8))  # Can adjust these dimensions as needed
    sns.heatmap(
        pivot_table,
        vmin=0, vmax=1,
        cmap=cmap,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,  # Adjust the thickness of the grid lines here
        linecolor='grey',  # Set the color of the grid lines
        linestyle='--'
    )


    # More aesthetics
    plt.title(f'Pressure Testing {model_name} \nFact Retrieval Across Context Lengths ("Needle In A HayStack")', fontsize=18)  # Adds a title
    plt.xlabel('Token Limit', fontsize=18)  # X-axis label
    plt.ylabel('Depth Percent', fontsize=18)  # Y-axis label
    plt.xticks(rotation=45, fontsize=18)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=18)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Add a vertical line at the desired column index
    plt.axvline(x=pretrained_len + 0.8, color='white', linestyle='--', linewidth=4)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    save_path = IMG_DIR / f"{model_name}.png"
    print("saving at %s" % save_path)
    plt.savefig(save_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing per-model result folders.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional model result folder names to visualize. Defaults to all folders under results_dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir
    model_dirs = [results_dir / model for model in args.models] if args.models else sorted(p for p in results_dir.iterdir() if p.is_dir())

    for model_dir in model_dirs:
        plot_model_results(model_dir)


if __name__ == "__main__":
    main()
