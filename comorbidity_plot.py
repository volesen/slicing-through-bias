import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import sklearn.mixture
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from load_chexpert import load_chexpert
from load_nih import load_nih

# Use Helvetica for plots
plt.rcParams["font.family"] = "Helvetica"

sns.set_style("whitegrid")

NIH_DISEASE_COLUMNS = [
    "Effusion",
    "Emphysema",
    "Nodule",
    "Atelectasis",
    "Infiltration",
    "Mass",
    "Pleural_Thickening",
    "Pneumothorax",
    "Consolidation",
    "Fibrosis",
    "Cardiomegaly",
    "Pneumonia",
    "Edema",
    "Hernia",
    "No Finding",
]

CHEXPERT_DISEASE_COLUMNS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

DRAIN_COLUMNS_PNEUMOTHORAX = [
    "Drain_0.0",
    "Drain_1.0",
]

DRAIN_COLUMNS_ATELECTASIS = [
    "Drain_0.0",
    "Drain_1.0",
    "Drain_nan",
]

COLUMN_TO_LABEL = {
    "Drain_0.0": "No Drain",
    "Drain_1.0": "Drain",
    "Drain_nan": "No Drain Label",
    "Pleural_Thickening": "Pleural Thickening",
    "Sex_Male": "Male",
}


def load_dataset(path, dataset):
    load = load_nih if dataset == "NIH" else load_chexpert

    full_df, full_embeddings = load(path, split="test")
    _, nn_128_embeddings = load(path, split="test", embedding_size=128)

    # Dummyify the column
    full_df = pd.get_dummies(full_df, columns=["Drain"], dummy_na=True)

    return full_df, full_embeddings, nn_128_embeddings


## Clustering algorithms



class GMM(BaseEstimator, TransformerMixin):
    cluster_array: np.ndarray = None
    _n_components: int = None
    _gmm: sklearn.mixture.GaussianMixture = None

    def __init__(self, cluster_array=None, ic="bic"):
        self.cluster_array = cluster_array
        self.ic = ic

    def fit(self, X, y=None):
        ics = []

        for n_components in tqdm(self.cluster_array):
            gmm = sklearn.mixture.GaussianMixture(n_components=n_components)
            gmm.fit(X)
            ics.append(gmm.bic(X) if self.ic == "bic" else gmm.aic(X))

        self._n_components = self.cluster_array[np.argmin(ics)]
        self._gmm = sklearn.mixture.GaussianMixture(n_components=self._n_components)
        self._gmm.fit(X)

        return self

    def transform(self, X, y=None):
        return self._gmm.predict(X)


## Clustering evaluation


def summarize_clusters(label_df, cluster, include_columns=[], threshold=0.5):
    df = label_df.groupby(cluster).agg(
        Size=("target_0", "count"),
        Disease_Prevalence=("target_0", "mean"),
        Confidence=("class_0", "mean"),
        Sex_Female=("Sex_Female", "mean"),
        Age=("Age", "mean"),
        **{col: (col, "mean") for col in include_columns},
    )

    # Size of cluster
    df["Size"] = label_df.groupby(cluster).size()

    # TP, FP, TN, FN
    df["TP"] = label_df.groupby(cluster).apply(
        lambda x: ((x["target_0"] == 1) & (x["class_0"] > threshold)).sum()
    )
    df["FP"] = label_df.groupby(cluster).apply(
        lambda x: ((x["target_0"] == 0) & (x["class_0"] > threshold)).sum()
    )
    df["FN"] = label_df.groupby(cluster).apply(
        lambda x: ((x["target_0"] == 1) & (x["class_0"] <= threshold)).sum()
    )
    df["TN"] = label_df.groupby(cluster).apply(
        lambda x: ((x["target_0"] == 0) & (x["class_0"] <= threshold)).sum()
    )

    # TPR, FPR
    df["TPR"] = df["TP"] / (df["TP"] + df["FN"])
    df["FPR"] = df["FP"] / (df["FP"] + df["TN"])

    # Accuracy, Precision, Recall
    df["Accuracy"] = (df["TP"] + df["TN"]) / (df["TP"] + df["TN"] + df["FP"] + df["FN"])
    df["Precision"] = df["TP"] / (df["TP"] + df["FP"])
    df["Recall"] = df["TP"] / (df["TP"] + df["FN"])

    # AUROC and Brier Score
    try:
        df["AUROC"] = label_df.groupby(cluster).apply(
            lambda x: sklearn.metrics.roc_auc_score(x["target_0"], x["class_0"])
        )
    except:
        df["AUROC"] = np.nan

    df["Brier Score"] = label_df.groupby(cluster).apply(
        lambda x: sklearn.metrics.brier_score_loss(x["target_0"], x["class_0"])
    )

    df["Bootstrapped Brier Score"] = label_df.groupby(cluster).apply(
        lambda x: bootstrapped_brier_score(x["target_0"], x["class_0"])
    )

    df["Lower Brier Score"] = df["Bootstrapped Brier Score"].apply(lambda x: x[0])
    df["Upper Brier Score"] = df["Bootstrapped Brier Score"].apply(lambda x: x[1])

    return df


## Evaluate the clustering algorithms


def bootstrapped_brier_score(y_true, y_prob):
    assert y_true.shape == y_prob.shape

    N = 1000

    brier_scores = np.zeros((N,))
    for i in range(N):
        indexes = np.random.choice(len(y_true), size=len(y_true), replace=True)

        y_true_samples = y_true.iloc[indexes]
        y_prob_samples = y_prob.iloc[indexes]

        brier_scores[i] = sklearn.metrics.brier_score_loss(
            y_true_samples, y_prob_samples
        )

    return np.quantile(brier_scores, 0.025), np.quantile(brier_scores, 0.975)


def plot_confidence_distribution(label_df, cluster_labels, density=False):
    cluster_ids = np.unique(cluster_labels)

    fig, ax = plt.subplots(
        nrows=len(cluster_ids),
        ncols=1,
        figsize=(5, 5 * len(cluster_ids)),
        dpi=300,
        sharex=True,
    )

    # Color the bars by the disease
    colors = matplotlib.colormaps["tab20"].colors

    bins = np.linspace(0, 1, 20)

    for i, cluster_id in enumerate(cluster_ids):
        # Plot the distribution of confidence (`class_0`) as a histogram
        ax[i].hist(
            label_df[cluster_labels == cluster_id]["class_0"],
            bins=bins,
            color=colors[cluster_id],
            density=density,
        )

        ax[cluster_id].set_title(f"Cluster {cluster_id}")

        ax[cluster_id].set_xlabel("Confidence")
        ax[cluster_id].set_ylabel("Density" if density else "Count")

        ax[cluster_id].set_xlim(0, 1)

    fig.tight_layout()

    return fig, ax


def plot_comorbidity_distribution(label_df, cluster_df, comorbidities, wrt_mean=False):
    # Select first and last cluster
    cluster_ids = cluster_df.index.values
    cluster_ids = cluster_ids[[0, -1]]

    fig, ax = plt.subplots(
        nrows=1,
        ncols=len(cluster_ids),
        figsize=(5 * len(cluster_ids), 5),
        sharex=True,
        sharey=True,
        dpi=300,
    )

    for i, cluster_id in enumerate(cluster_ids):
        row = cluster_df.loc[cluster_id]

        ax[i].set_title(
            f"Brier score: [{row['Lower Brier Score']:.2f}, {row['Upper Brier Score']:.2f}]. Size: {row['Size']}"
        )
        ax[i].set_xlabel("Proportion of cases")
        ax[i].set_ylabel("Comorbidity")
        ax[i].set_xlim(0, 1)

        y = row[comorbidities].values

        # Horizontal bar plot
        sns.barplot(
            x=y,
            y=comorbidities,
            ax=ax[i],
            orient="h",
            edgecolor="#4c4c4c",
        )

        # Add the proportion of cases
        for j, comorbidity in enumerate(comorbidities):
            ax[i].text(
                y[j] + 0.01,
                j,
                f"{row[comorbidity]:.2f}",
                verticalalignment="center",
            )

    fig.tight_layout()

    # We assume the last 3 comordbidites are `Drain_nan`, `Drain_0.0`, `Drain_1.0`.
    # Rename the ticks to `No Drain Label`, `No Drain`, `Drain`
    comorbidities = [
        COLUMN_TO_LABEL[comorbidity] if comorbidity in COLUMN_TO_LABEL else comorbidity
        for comorbidity in comorbidities
    ]

    ax[0].set_yticklabels(comorbidities)

    return fig, ax


def main(path, dataset, disease, case, drain, output):
    df, _, nn_128_embeddings = load_dataset(
        path, dataset
    )

    if case == "positive":
        case_df = df[df["target_0"] == 1]
        nn_128_embeddings = nn_128_embeddings[df["target_0"] == 1]

    elif case == "negative":
        case_df = df[df["target_0"] == 0]
        nn_128_embeddings = nn_128_embeddings[df["target_0"] == 0]

    dim_reduced_embeddings = {
        "nn_128": (case_df, nn_128_embeddings),
    }

    cluster_dfs = {}
    cluster_labels = {}

    DISEASE_COLUMNS = (
        NIH_DISEASE_COLUMNS if dataset == "NIH" else CHEXPERT_DISEASE_COLUMNS
    )

    COMORBIDITIES = [
        col for col in DISEASE_COLUMNS if col not in [disease, "No Finding"]
    ]

    comorbidity_df = df[COMORBIDITIES].sum() / len(df)
    comorbidity_df = comorbidity_df.sort_values(ascending=False)

    if drain:
        COMORBIDITIES = list(comorbidity_df.index) + (
            DRAIN_COLUMNS_PNEUMOTHORAX
            if disease == "Pneumothorax"
            else DRAIN_COLUMNS_ATELECTASIS
        )
    else:
        COMORBIDITIES = list(comorbidity_df.index)

    # Run clustering algorithms
    clustering_algorithms = {
        "GMM": GMM(cluster_array=[4, 6, 8, 10, 12, 14, 16], ic="bic"),
    }

    for name, (df, embeddings) in dim_reduced_embeddings.items():
        for algorithm_name, algorithm in clustering_algorithms.items():

            # Fit on the embeddings, that do not include drain
            algorithm.fit(embeddings)

            if drain and disease == "Pneumothorax":
                selected_df = df[df["Drain_nan"] == 0]
                selected_embeddings = embeddings[df["Drain_nan"] == 0]
            else:
                selected_df = df
                selected_embeddings = embeddings

            cluster_id = algorithm.transform(selected_embeddings)
            cluster_labels[(name, algorithm_name)] = cluster_id

            cluster_df = summarize_clusters(
                selected_df,
                cluster_id,
                include_columns=(
                    DISEASE_COLUMNS
                    + (
                        DRAIN_COLUMNS_PNEUMOTHORAX
                        if disease == "Pneumothorax"
                        else DRAIN_COLUMNS_ATELECTASIS
                    )
                    if drain
                    else DISEASE_COLUMNS
                ),
            )

            cluster_df["Algorithm"] = algorithm_name
            cluster_df["Embedding"] = name
            cluster_dfs[(name, algorithm_name)] = cluster_df

    total_cluster_df = (
        pd.concat(cluster_dfs.values())
        .reset_index()
        .drop(
            columns=[
                "TPR",
                "FPR",
                "Accuracy",
                "Precision",
                "Recall",
                "AUROC",
            ]
        )
        .sort_values(["Embedding", "Algorithm"], ascending=False)
    )

    # Export results
    os.makedirs(f"{output}/{dataset}/{disease}", exist_ok=True)
    os.makedirs(f"{output}/{dataset}/{disease}/confidence", exist_ok=True)
    os.makedirs(f"{output}/{dataset}/{disease}/comorbidities", exist_ok=True)

    # Export cluster labels
    total_cluster_df.to_excel(
        f"{output}/{dataset}/{disease}/{case}_cluster_results.xlsx", index=False
    )

    # Export plots
    for (name, algorithm_name), cluster_df in cluster_dfs.items():
        # Skip
        if len(cluster_df) <= 1:
            continue

        upper_df = cluster_df.sort_values(["Upper Brier Score"], ascending=True)
        best_cluster_df = upper_df.iloc[[0]]

        lower_df = cluster_df.sort_values(["Lower Brier Score"], ascending=False)
        worst_cluster_df = lower_df.iloc[[0]]

        cluster_df = pd.concat([worst_cluster_df, best_cluster_df])

        fig, _ = plot_comorbidity_distribution(
            case_df, cluster_df, COMORBIDITIES, wrt_mean=False
        )
        fig.savefig(
            f"{output}/{dataset}/{disease}/comorbidities/{case}_{name}_{algorithm_name}.png"
        )
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--disease", type=str, required=True)

    # Positive vs negative
    parser.add_argument("--case", type=str, choices=["positive", "negative"])

    # Include drain in analysis
    parser.add_argument("--drain", action="store_true")

    # Output directory
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    main(args.path, args.dataset, args.disease, args.case, args.drain, args.output)
