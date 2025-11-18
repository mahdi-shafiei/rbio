import hashlib
import json
import os
import pickle
import random

import click
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler


def compute_embeddings_hash(emb_dict: dict) -> str:
    """Compute MD5 hash of embeddings dictionary for validation."""
    emb_str = json.dumps({k: v.tolist() for k, v in sorted(emb_dict.items())})
    return hashlib.md5(emb_str.encode()).hexdigest()


class MLPClassifier(nn.Module):
    """Simple MLP classifier for gene pair classification."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GeneDataset(Dataset):
    """Dataset for gene perturbation and monitoring pairs."""
    
    def __init__(self, df: pd.DataFrame, name_to_embedding: dict):
        self.df = df.reset_index(drop=True)
        self.name_to_embedding = name_to_embedding
        self.pos_indices = self.df[self.df["label"] == 1].index.tolist()
        self.neg_indices = self.df[self.df["label"] == 0].index.tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        gene_pert = torch.tensor(
            self.name_to_embedding[row["gene_perturbed"].lower()], dtype=torch.float32
        )
        gene_mon = torch.tensor(
            self.name_to_embedding[row["gene_monitored"].lower()], dtype=torch.float32
        )
        label = torch.tensor(row["label"], dtype=torch.float32)
        return gene_pert, gene_mon, label


class BalancedBatchSampler(Sampler):
    """Sampler that ensures balanced positive/negative samples in each batch."""
    
    def __init__(self, pos_indices, neg_indices, batch_size):
        super().__init__()
        assert batch_size % 2 == 0, "Batch size must be even for balanced sampling"
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices
        self.batch_size = batch_size
        self.half_batch = batch_size // 2

    def __iter__(self):
        pos_pool = random.sample(self.pos_indices, len(self.pos_indices))
        neg_pool = random.sample(self.neg_indices, len(self.neg_indices))
        min_len = min(len(pos_pool), len(neg_pool))

        for i in range(0, min_len, self.half_batch):
            pos_batch = pos_pool[i : i + self.half_batch]
            neg_batch = neg_pool[i : i + self.half_batch]
            if len(pos_batch) == self.half_batch and len(neg_batch) == self.half_batch:
                batch = pos_batch + neg_batch
                random.shuffle(batch)
                yield batch

    def __len__(self):
        return min(len(self.pos_indices), len(self.neg_indices)) // self.half_batch


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    train_df: pd.DataFrame,
    emb_dict: dict,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    """Train the MLP classifier on gene pair data."""
    train_dataset = GeneDataset(train_df, emb_dict)
    sampler = BalancedBatchSampler(
        train_dataset.pos_indices, train_dataset.neg_indices, batch_size
    )
    train_loader = DataLoader(train_dataset, batch_sampler=sampler)

    input_dim = len(next(iter(emb_dict.values())))
    model = MLPClassifier(input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for gene_pert, gene_mon, label in train_loader:
            gene_pert = gene_pert.to(device)
            gene_mon = gene_mon.to(device)
            label = label.to(device).unsqueeze(1)

            inputs = torch.cat([gene_pert, gene_mon], dim=1)
            logits = model(inputs)
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * gene_pert.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    return model


def embedding_training(
    training_set_paths,
    emb_dict: dict,
    batch_size: int,
    num_epochs: int,
    checkpoint_dir: os.PathLike,
):
    """Train model on gene pair data using pre-computed embeddings."""
    dfs = [pd.read_csv(path) for path in training_set_paths]
    df_training = pd.concat(dfs, ignore_index=True)

    genes = pd.unique(df_training[["gene_perturbed", "gene_monitored"]].values.ravel())
    all_genes = sorted(set(genes))

    # Verify all genes are in emb_dict
    missing_genes = [gene for gene in all_genes if gene.lower() not in emb_dict]
    if missing_genes:
        raise ValueError(f"Missing embeddings for genes: {missing_genes}")

    model = train_model(
        df_training, emb_dict, batch_size=batch_size, num_epochs=num_epochs
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "mlp_model.pt")
    embeddings_hash_path = os.path.join(checkpoint_dir, "embeddings_hash.txt")

    # Save model and embeddings hash
    torch.save(model.state_dict(), checkpoint_path)
    embeddings_hash = compute_embeddings_hash(emb_dict)
    with open(embeddings_hash_path, "w") as f:
        f.write(embeddings_hash)

    print(f"Model checkpoint saved to {checkpoint_path}")
    print(f"Embeddings hash saved to {embeddings_hash_path}")


def test_model(
    model: nn.Module,
    test_df: pd.DataFrame,
    emb_dict: dict,
    output_csv: os.PathLike,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """Test the trained model and save predictions to CSV."""
    test_dataset = GeneDataset(test_df, emb_dict)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model.eval()
    results = []
    idx = 0

    with torch.no_grad():
        for gene_pert, gene_mon, label in test_loader:
            gene_pert = gene_pert.to(device)
            gene_mon = gene_mon.to(device)
            label = label.cpu().numpy().flatten()

            inputs = torch.cat([gene_pert, gene_mon], dim=1)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            for gt, pred in zip(label, preds):
                results.append(
                    {
                        "prompt": "",
                        "completion": "",
                        "answer": pred,
                        "binary_answer": int(pred),
                        "ground_truth": int(gt),
                        "gene_perturbed": test_df.iloc[idx]["gene_perturbed"],
                        "gene_monitored": test_df.iloc[idx]["gene_monitored"],
                    }
                )
                idx += 1

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"\nPrediction CSV saved to: {output_csv}")


@click.group()
def cli():
    """Gene pair classification training and testing."""
    pass


@cli.command()
@click.option(
    "--train-dataset-path",
    required=True,
    multiple=True,
    help="Training dataset CSV file(s)",
)
@click.option("--batch-size", default=32, help="Batch size")
@click.option("--num-epochs", default=10, help="Number of training epochs")
@click.option("--embedding-file", required=True, help="Path to embedding .pkl file")
@click.option(
    "--checkpoint-dir", required=True, help="Directory to save model checkpoint"
)
def train(
    train_dataset_path,
    batch_size,
    num_epochs,
    embedding_file,
    checkpoint_dir,
):
    """Train the MLP classifier on gene pair data."""
    set_seed(42)
    with open(embedding_file, "rb") as f:
        emb_dict = pickle.load(f)
    embedding_training(
        train_dataset_path, emb_dict, batch_size, num_epochs, checkpoint_dir
    )


@cli.command()
@click.option(
    "--test-dataset-path",
    help="Test dataset CSV file path",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--mlp-model-path",
    help="Path to the trained MLP model checkpoint",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--embedding-file",
    help="Path to the gene embedding dictionary pickle file",
    required=True,
    type=click.Path(exists=True),
)
@click.option(
    "--output-csv-path",
    help="Output CSV file path for predictions",
    required=True,
)
@click.option(
    "--batch-size",
    help="Batch size for testing",
    default=32,
)
def test(
    test_dataset_path: os.PathLike,
    mlp_model_path: os.PathLike,
    embedding_file: os.PathLike,
    output_csv_path: os.PathLike,
    batch_size: int,
):
    """Test the trained model and generate predictions."""
    # Load test dataset
    test_df = pd.read_csv(test_dataset_path)

    # Load embeddings
    with open(embedding_file, "rb") as f:
        emb_dict = pickle.load(f)

    # Verify all genes are in emb_dict
    genes = pd.unique(test_df[["gene_perturbed", "gene_monitored"]].values.ravel())
    missing_genes = [gene for gene in genes if gene.lower() not in emb_dict]
    if missing_genes:
        raise ValueError(f"Missing embeddings for genes: {missing_genes}")

    # Check embeddings hash
    embeddings_hash_path = os.path.join(
        os.path.dirname(mlp_model_path), "embeddings_hash.txt"
    )
    if os.path.exists(embeddings_hash_path):
        with open(embeddings_hash_path, "r") as f:
            expected_hash = f.read().strip()
        current_hash = compute_embeddings_hash(emb_dict)
        if current_hash != expected_hash:
            print(
                "\033[93mWARNING: Embeddings hash does not match! Results will be random.\033[0m"
            )
            print(f"Expected hash: {expected_hash}")
            print(f"Current hash:  {current_hash}")

    # Load model
    input_dim = len(next(iter(emb_dict.values())))
    model = MLPClassifier(input_dim)
    model.load_state_dict(torch.load(mlp_model_path, map_location=torch.device("cpu")))
    model.eval()

    # Run testing
    test_model(
        model,
        test_df,
        emb_dict,
        output_csv=output_csv_path,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    cli()
