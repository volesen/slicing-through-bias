import os
from typing import Literal

import pandas as pd

Split = Literal["train", "val", "test"]


def load_nih(
    root_dir: str,
    split: Split = "test",
    version: int = 0,
    embedding_size: int | None = None,
):
    # Load labels
    nih_labels_df = pd.read_csv(
        os.path.join(root_dir, f"{split}.version_{version}.csv"),
        dtype={
            "Patient Gender": "category",
            "View Position": "category",
        },
    )

    # Remove unnecessary columns
    nih_labels_df = nih_labels_df.drop(
        columns=[
            "Finding Labels",
            "Unnamed: 0",
            "OriginalImage[Width",
            "Height]",
            "OriginalImagePixelSpacing[x",
            "y]",
        ]
    )

    # Rename columns, to be consistent with chexpert
    nih_labels_df = nih_labels_df.rename(
        columns={
            "index": "Index",
            "Patient ID": "Id",
            "Patient Age": "Age",
            "Patient Gender": "Sex",
            "split": "Split",
        }
    )
    # Change Sex to be consistent with chexpert
    nih_labels_df["Sex"] = nih_labels_df["Sex"].replace({"M": "Male", "F": "Female"})

    # Dummy encode categorical variables
    nih_labels_df = pd.get_dummies(nih_labels_df, columns=["Sex"])

    # Load predictions
    nih_predictions_df = pd.read_csv(
        os.path.join(root_dir, f"predictions.{split}.version_{version}.csv")
    )

    # Load embeddings
    if embedding_size is not None:
        nih_embeddings_df = pd.read_csv(
            os.path.join(root_dir, f"embeddings_{split}_{embedding_size}.csv")
        )
    else:
        nih_embeddings_df = pd.read_csv(
            os.path.join(root_dir, f"embeddings.{split}.version_{version}.csv")
        )

    nih_embeddings = nih_embeddings_df.iloc[:, :-1].values

    chexpert_df = pd.concat([nih_predictions_df, nih_labels_df], axis=1)

    return chexpert_df, nih_embeddings
