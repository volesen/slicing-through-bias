import os
from typing import Literal, Optional

import pandas as pd

Split = Literal["train", "val", "test"]


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


def load_chexpert(
    root_dir: str,
    split="test",
    version: int = 0,
    embedding_size: Optional[int] = None,
):
    # Load labels
    chexpert_labels_df = pd.read_csv(
        os.path.join(root_dir, f"{split}.version_{version}.csv"),
        dtype={
            "race": "category",
            "ethnicity": "category",
            "sex": "category",
            "Frontal/Lateral": "category",
            "AP/PA": "category",
        },
    )

    # Remove duplicate columns
    chexpert_labels_df = chexpert_labels_df.drop(
        columns=["Unnamed: 0", "Unnamed: 0.1", "path_preproc", "sex_label"],
    )

    # Rename columns to be consistent with NIH
    chexpert_labels_df = chexpert_labels_df.rename(
        columns={
            "index": "Index",
            "path_preproc_new": "Image Index",
            "sex": "Sex",
            "age": "Age",
            "ethnicity": "Ethnicity",
            "race": "Race",
            "patient_id": "Id",
        }
    )

    chexpert_labels_df[CHEXPERT_DISEASE_COLUMNS] = (
        chexpert_labels_df[CHEXPERT_DISEASE_COLUMNS].fillna(0) > 0
    )

    # Preprocess labels
    chexpert_labels_df["No Finding"].fillna(0, inplace=True)

    chexpert_labels_df["Support Devices"] = chexpert_labels_df[
        "Support Devices"
    ].fillna(0)
    chexpert_labels_df["Support Devices"] = chexpert_labels_df["Support Devices"] == 1

    # Dummy encode Sex
    chexpert_labels_df = pd.get_dummies(chexpert_labels_df, columns=["Sex"])

    # Load predictions
    chexpert_predictions_df = pd.read_csv(
        os.path.join(root_dir, f"predictions.{split}.version_{version}.csv")
    )

    # Load embeddings
    if embedding_size is not None:
        chexpert_embeddings_df = pd.read_csv(
            os.path.join(root_dir, f"embeddings_{split}_{embedding_size}.csv")
        )
    else:
        chexpert_embeddings_df = pd.read_csv(
            os.path.join(root_dir, f"embeddings.{split}.version_{version}.csv")
        )
    chexpert_embeddings = chexpert_embeddings_df.iloc[:, :-1].values

    chexpert_df = pd.concat([chexpert_predictions_df, chexpert_labels_df], axis=1)

    return chexpert_df, chexpert_embeddings
