import argparse
import os.path

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import Tensor, nn, optim, utils
from torch.utils.data import DataLoader

from load_chexpert import load_chexpert
from load_nih import load_nih

RESNET_LATENT_DIM = 2048


class Model(L.LightningModule):
    def __init__(self, embedding_size):
        super().__init__()

        self.embedding_size = embedding_size

        self.fc1 = nn.Linear(RESNET_LATENT_DIM, embedding_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(embedding_size, 1)

    def forward(self, x):
        # Returns the result of the penultimate layer
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)

        return x

    def remove_last_layer(self):
        self.fc2 = nn.Identity(self.embedding_size)

    def _step(self, split, batch, batch_idx):
        X, y = batch

        y_hat = self(X)

        # Convert y_hat to tensor of shape (batch_size,) even if batch_size == 1
        y_hat = y_hat.squeeze(1)

        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)

        self.log(f"{split}_loss", loss)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-6)
        return optimizer


def load_data(
    path: str,
    dataset: str,
    batch_size: bool = 64,
    shuffle_train: bool = True,
):
    load = load_nih if dataset == "NIH" else load_chexpert

    train_df, train_embeddings = load(path, split="train")
    val_df, val_embeddings = load(path, split="val")
    test_df, test_embeddings = load(path, split="test")

    train_dataloader = DataLoader(
        utils.data.TensorDataset(
            Tensor(train_embeddings),
            Tensor(train_df["target_0"]),
        ),
        batch_size=batch_size,
        shuffle=shuffle_train,
    )

    val_dataloader = DataLoader(
        utils.data.TensorDataset(
            Tensor(val_embeddings),
            Tensor(val_df["target_0"]),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    test_dataloader = DataLoader(
        utils.data.TensorDataset(
            Tensor(test_embeddings),
            Tensor(test_df["target_0"]),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_dataloader, val_dataloader, test_dataloader


def predict(model, dataloader):
    y_true = []
    y_pred = []

    for batch in dataloader:
        X, y = batch

        y_true.append(y.detach().numpy())

        with torch.no_grad():
            y_hat = model(X).detach().numpy()

        y_pred.append(y_hat)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return pd.DataFrame({"target_0": y_true.flatten(), "class_0": y_pred.flatten()})



def extract_embeddings(model, dataloader):
    y_true = []
    embeddings = []

    for batch in dataloader:
        X, y = batch

        y_true.append(y.detach().numpy())

        with torch.no_grad():
            embedding = model(X).detach().numpy()

        embeddings.append(embedding)

    y_true = np.concatenate(y_true)
    embeddings = np.concatenate(embeddings)

    df = pd.DataFrame(embeddings)
    df["target_0"] = y_true

    return df


def main(path: str, dataset: str, dimension: str):
    print(f"Running with path '{path}', dataset '{dataset}' and dimension {dimension}")

    # Load data
    train_dataloader, val_dataloader, test_dataloader = load_data(path, dataset)

    # Make instance of model
    model = Model(embedding_size=dimension)

    # Train model
    trainer = L.Trainer(
        callbacks=[L.pytorch.callbacks.EarlyStopping(monitor="val_loss")],
        logger=L.pytorch.loggers.WandbLogger(
            project="embed",
            config={
                "path": path,
                "dataset": dataset,
                "emedding_size": model.embedding_size,
            },
        ),
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Save predictions to access performance
    dataloaders = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }

    accs = {}
    aurocs = {}

    for split, dataloader in dataloaders.items():
        pred_df = predict(model, dataloader)

        accs[split] = accuracy_score(pred_df["target_0"], pred_df["class_0"] > 0.5)
        aurocs[split] = roc_auc_score(pred_df["target_0"], pred_df["class_0"])

        # Save predictions
        export_path = os.path.join(path, f"pred_embeddings_{split}_{dimension}.csv")
        pred_df.to_csv(export_path, index=False)

    # Log performance to STDOUT
    print("Accuracy ", accs)
    print("AUROC ", aurocs)

    # Compare to the orifginal model
    load = load_nih if dataset == "NIH" else load_chexpert
    test_df, test_embeddings = load(path, split="test")

    print(
        "Accuracy model", accuracy_score(test_df["target_0"], test_df["class_0"] > 0.5)
    )
    print("AUROC model", roc_auc_score(test_df["target_0"], test_df["class_0"]))

    # Extract embeddings for train, val and test set
    model.remove_last_layer()
    for split, dataloader in dataloaders.items():
        embedding_df = extract_embeddings(model, dataloader)

        # Save embeddings
        export_path = os.path.join(path, f"embeddings_{split}_{dimension}.csv")
        embedding_df.to_csv(export_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Embed",
        description="Generates dimensionality reduced latent space embeddings, by adding a layer",
    )

    parser.add_argument("-p", "--path", type=str)
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("-n", "--dimension", type=int)

    args = parser.parse_args()

    main(path=args.path, dataset=args.dataset, dimension=args.dimension)
