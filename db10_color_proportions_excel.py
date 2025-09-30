import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from db8_color_predictions import get_trained_models

def analyze_color_proportions(
    dataset_path,
    data_folder,
    output_excel_path=None,
    output_plot_folder=None,
    output_plots=False,
    verbose=False
):
    models, label_encoder = get_trained_models(dataset_path)
    output_dataframe = []

    color_columns = ["Red", "Orange", "Yellow", "Green", "Blue", "Purple", "Pink", "Brown", "Grey", "Black", "White"]

    for file in os.listdir(data_folder):
        file_name = '_'.join(file.split('_')[1:])
        if verbose:
            print(f"\nProcessing: {file_name}")
        file_path = os.path.join(data_folder, file)
        rgbs = np.loadtxt(file_path, dtype=int)

        color_percentages = []

        for name, model in models.items():
            pred = model.predict(rgbs)
            pred_labels = label_encoder.inverse_transform(pred)
            counts = pd.Series(pred_labels).value_counts()
            percentages = (counts / counts.sum()) * 100
            color_percentages.append(percentages)

        df_percentages = pd.DataFrame(color_percentages)
        average_percentages = df_percentages.mean(axis=0).sort_values(ascending=False)
        std_dev = df_percentages.std(axis=0).reindex(average_percentages.index)

        result_dict = defaultdict(float)
        result_dict["File"] = file_name 

        for color, percentage in average_percentages.items():
            if verbose:
                print(f"{color}: {percentage:.2f} ± {std_dev[color]:.2f}")
            result_dict[color] = round(percentage, 2)

        for color in color_columns:
            if color not in result_dict:
                result_dict[color] = 0.00

        output_dataframe.append(result_dict)

        if output_plots and output_plot_folder:
            os.makedirs(output_plot_folder, exist_ok=True)
            colors = [label.lower() for label in average_percentages.index]
            x_pos = np.arange(len(average_percentages))

            plt.figure(figsize=(10, 6))
            plt.bar(x_pos, average_percentages, yerr=std_dev, capsize=5, color=colors, edgecolor='black', width=1)
            plt.xlabel("Predicted colors")
            plt.ylabel("Average percentage")
            plt.title(file_name.replace(".txt", ""))
            plt.xticks(x_pos, average_percentages.index, rotation=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_plot_folder, f"{file_name}_color_distribution.png"))
            plt.close()

    results = pd.DataFrame(output_dataframe)
    for color in color_columns:
        if color not in results:
            results[color] = 0.00
    results = results.round(2)
    results = results[["File"] + color_columns]

    if output_excel_path:
        results.to_excel(output_excel_path, index=False)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze color proportions using trained models.")
    parser.add_argument("--dataset_path", type=str, required=True, help="train dataset csv")
    parser.add_argument("--data_folder", type=str, required=True, help="directory w/ txt files w/ rgb values")
    parser.add_argument("--output_excel", type=str, help="output excel database")
    parser.add_argument("--output_plot_dir", type=str, help="enable plot saving in this directory")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    save_plots = args.output_plot_dir is not None

    analyze_color_proportions(
        dataset_path=args.dataset_path,
        data_folder=args.data_folder,
        output_excel_path=args.output_excel,
        output_plot_folder=args.output_plot_dir,
        output_plots=save_plots,
        verbose=args.verbose
    )
