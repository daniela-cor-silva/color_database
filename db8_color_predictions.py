import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import argparse

def train_and_evaluate_models(dataset_path):
    data = pd.read_csv(dataset_path)
    rgb_values = data[['red', 'green', 'blue']].values
    labels = data['label'].values

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    encoded_labels = label_encoder.transform(labels)

    rgbs_train, rgbs_val, labels_train, labels_val = train_test_split(
        rgb_values, encoded_labels, test_size=0.2, random_state=0
    )

    models = {
        "GradientBoosting": GradientBoostingClassifier(
            loss='log_loss', learning_rate=0.1, n_estimators=50,
            min_samples_split=2, min_samples_leaf=1, max_depth=5,
            random_state=0, verbose=0, ccp_alpha=0
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, criterion='gini', max_depth=None,
            min_samples_split=2, min_samples_leaf=1, max_features='sqrt',
            random_state=0, bootstrap=True
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5, weights='uniform', metric='manhattan'
        ),
        "QDA": QuadraticDiscriminantAnalysis(),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(50,), activation='relu', solver='adam',
            learning_rate='adaptive', max_iter=200, random_state=0,
            early_stopping=True, validation_fraction=0.1
        )
    }

    for name, model in models.items():
        print(f"\nTraining and evaluating model: {name}")
        model.fit(rgbs_train, labels_train)

        train_pred = model.predict(rgbs_train)
        val_pred = model.predict(rgbs_val)

        train_accuracy = accuracy_score(labels_train, train_pred) * 100
        val_accuracy = accuracy_score(labels_val, val_pred) * 100
        train_precision = precision_score(labels_train, train_pred, average='weighted') * 100
        val_precision = precision_score(labels_val, val_pred, average='weighted') * 100
        train_recall = recall_score(labels_train, train_pred, average='weighted') * 100
        val_recall = recall_score(labels_val, val_pred, average='weighted') * 100
        train_f1 = f1_score(labels_train, train_pred, average='weighted') * 100
        val_f1 = f1_score(labels_val, val_pred, average='weighted') * 100

        print(f'Accuracy (train/val): {train_accuracy:.2f}% / {val_accuracy:.2f}%')
        print(f'Precision (train/val): {train_precision:.2f}% / {val_precision:.2f}%')
        print(f'Recall (train/val): {train_recall:.2f}% / {val_recall:.2f}%')
        print(f'F1 Score (train/val): {train_f1:.2f}% / {val_f1:.2f}%')
        print(f'Overfitting gap: {train_accuracy - val_accuracy:.2f}%')
        print('********************************************************')
        print(f"\nClassification Report for {name}:\n")
        print(classification_report(labels_val, val_pred, target_names=label_encoder.classes_))
        print('********************************************************')

        cf = confusion_matrix(labels_val, val_pred)
        norm_cf = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(
            norm_cf,
            annot=True,
            annot_kws={"size": 16},
            fmt=".2f",
            cmap="flare",
            linewidth=.5,
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16, rotation=0)
        plt.xlabel('Predicted labels', fontsize=24)
        plt.ylabel('True labels', fontsize=24)
        plt.title(f'{name}', fontsize=30)
        plt.savefig(f"normalized_confusion_matrix_{name}.png")
        plt.close()

    return models, label_encoder

def predict_new_data(models, label_encoder, data_path):
    rgbs = np.loadtxt(data_path, dtype=int)

    color_percentages = []

    for name, model in models.items():
        pred = model.predict(rgbs)
        pred_labels = label_encoder.inverse_transform(pred)
        counts = pd.Series(pred_labels).value_counts()
        percentages = (counts / counts.sum()) * 100
        color_percentages.append(percentages)

    df_percentages = pd.DataFrame(color_percentages).fillna(0)
    average_percentages = df_percentages.mean(axis=0).sort_values(ascending=False)
    std_dev = df_percentages.std(axis=0).reindex(average_percentages.index)

    print("\nAverage predicted colors ± standard deviation:")
    for color, percentage in average_percentages.items():
        print(f"{color}: {percentage:.2f} ± {std_dev[color]:.2f}")

    colors = [label.lower() for label in average_percentages.index]
    x_pos = np.arange(len(average_percentages))

    plt.figure(figsize=(12, 8))
    plt.bar(x_pos, average_percentages, yerr=std_dev, capsize=5, color=colors, edgecolor='black', width=1)
    plt.xlabel("Predicted colors", fontsize=24)
    plt.ylabel("Percentage", fontsize=24)
    plt.title("Predicted color proportions", fontsize=30)
    plt.xticks(x_pos, average_percentages.index, rotation=0, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig("average_color_distribution.png")
    plt.close()

    for name, model in models.items():
        pred = model.predict(rgbs)
        pred_labels = label_encoder.inverse_transform(pred)
        counts = pd.Series(pred_labels).value_counts()
        percentages = (counts / counts.sum()) * 100
        plt.figure(figsize=(12, 8))
        colors_model = [label.lower() for label in percentages.index]
        x_pos_model = np.arange(len(percentages))
        plt.bar(x_pos_model, percentages, capsize=5, color=colors_model, edgecolor='black', width=1)
        plt.xlabel("Predicted colors", fontsize=24)
        plt.ylabel("Percentage", fontsize=24)
        plt.title(f"{name}", fontsize=30)
        plt.xticks(x_pos_model, percentages.index, rotation=0, fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(f"{name}_color_distribution.png")
        plt.close()

def get_trained_models(dataset_path):
    models, label_encoder = train_and_evaluate_models(dataset_path)
    return models, label_encoder

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train color classifiers and analyze single illustration.")
    parser.add_argument("--train_dataset", type=str, required=True, help="csv file")
    parser.add_argument("--image_rgb_values", type=str, required=False, help="txt file, optional")
    args = parser.parse_args()

    models, label_encoder = train_and_evaluate_models(args.train_dataset)

    if args.image_rgb_values: # if txt file provided to analyze single image
        predict_new_data(models, label_encoder, args.image_rgb_values)
