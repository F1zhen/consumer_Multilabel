"""Comprehensive training pipeline for multi-label text classification.

The CLI tool provided by this module loads the Amazon clothing reviews dataset
and trains a rich collection of models grouped in three families:

* **Classical machine learning** – 10 TF-IDF based one-vs-rest classifiers.
* **Neural networks** – 5 sequence models implemented with PyTorch.
* **Transformers** – 5 Hugging Face checkpoints fine-tuned for the task.

For every trained model the script exports evaluation artefacts (confusion
matrices, classification reports) and collects a consolidated comparison table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import (
    LogisticRegression,
    PassiveAggressiveClassifier,
    RidgeClassifier,
    SGDClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    multilabel_confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

TOKEN_PATTERN = re.compile(r"\b\w+\b")
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

CLASSIC_MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "logistic_regression": "Logistic Regression (One-vs-Rest)",
    "linear_svc": "Linear SVC (One-vs-Rest)",
    "sgd_classifier": "SGD Classifier (log-loss)",
    "passive_aggressive": "Passive Aggressive Classifier",
    "ridge_classifier": "Ridge Classifier",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "knn": "K-Nearest Neighbours",
    "multinomial_nb": "Multinomial Naive Bayes",
    "complement_nb": "Complement Naive Bayes",
}
CLASSIC_MODEL_KEYS = list(CLASSIC_MODEL_DISPLAY_NAMES.keys())
DEFAULT_CLASSIC_MODELS = CLASSIC_MODEL_KEYS.copy()

NEURAL_MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "bilstm": "Bidirectional LSTM",
    "bigru": "Bidirectional GRU",
    "textcnn": "Text CNN",
    "self_attention": "Self-Attention Pooler",
    "mlp": "Averaged Embedding MLP",
}
NEURAL_MODEL_KEYS = list(NEURAL_MODEL_DISPLAY_NAMES.keys())
DEFAULT_NEURAL_MODELS = NEURAL_MODEL_KEYS.copy()

DEFAULT_TRANSFORMER_MODELS = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "distilroberta-base",
    "roberta-base",
    "albert-base-v2",
]


@dataclass
class ModelResult:
    """Container for storing model evaluation metadata."""

    approach: str
    method: str
    dataset: str
    samples: int
    duration_seconds: float
    metrics: dict
    output_dir: Path

    def to_table_row(self) -> dict:
        """Return a serialisable dictionary for the summary table."""

        return {
            "Approach": self.approach,
            "Method": self.method,
            "Dataset": self.dataset,
            "Amount of data": self.samples,
            "Speed (s)": round(self.duration_seconds, 2),
            "Subset accuracy": round(self.metrics.get("subset_accuracy", 0), 4),
            "F1 micro": round(self.metrics.get("f1_micro", 0), 4),
            "F1 macro": round(self.metrics.get("f1_macro", 0), 4),
            "Hamming loss": round(self.metrics.get("hamming_loss", 0), 4),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/data_amazon.xlsx - Sheet1.csv"),
        help="Path to the CSV dataset file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where artefacts (plots, metrics) will be stored.",
    )
    parser.add_argument(
        "--label-columns",
        nargs="+",
        default=None,
        help=(
            "Optional list of label column names. If not provided the standard "
            "Amazon dataset labels are used."
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optionally limit the number of samples used for training to speed "
            "experimentation."
        ),
    )
    parser.add_argument(
        "--tfidf-max-features",
        type=int,
        default=5000,
        help="Maximum number of features for the TF-IDF vectoriser.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=2,
        help="Number of components for PCA visualisation.",
    )

    parser.add_argument(
        "--classic-models",
        nargs="+",
        default=None,
        help=(
            "Subset of classical models to run. Available keys: "
            + ", ".join(CLASSIC_MODEL_KEYS)
        ),
    )
    parser.add_argument(
        "--classic-max-iter",
        type=int,
        default=1000,
        help="Maximum number of iterations for linear classical models.",
    )
    parser.add_argument(
        "--classic-n-jobs",
        type=int,
        default=-1,
        help="Parallelism for One-vs-Rest wrappers (where supported).",
    )
    parser.add_argument(
        "--classic-forest-estimators",
        type=int,
        default=200,
        help="Number of trees for ensemble-based classical models.",
    )
    parser.add_argument(
        "--classic-knn-neighbors",
        type=int,
        default=5,
        help="Number of neighbours for the KNN classifier.",
    )

    parser.add_argument(
        "--neural-models",
        nargs="+",
        default=None,
        help=("Subset of neural models to run. Available keys: " + ", ".join(NEURAL_MODEL_KEYS)),
    )
    parser.add_argument(
        "--neural-epochs",
        "--lstm-epochs",
        dest="neural_epochs",
        type=int,
        default=5,
        help="Number of training epochs for neural sequence models.",
    )
    parser.add_argument(
        "--neural-batch-size",
        "--lstm-batch-size",
        dest="neural_batch_size",
        type=int,
        default=32,
        help="Batch size for neural model training.",
    )
    parser.add_argument(
        "--neural-embedding-dim",
        "--lstm-embedding-dim",
        dest="neural_embedding_dim",
        type=int,
        default=128,
        help="Embedding dimensionality for neural models.",
    )
    parser.add_argument(
        "--neural-hidden-dim",
        "--lstm-hidden-dim",
        dest="neural_hidden_dim",
        type=int,
        default=128,
        help="Hidden state size for recurrent and feed-forward layers.",
    )
    parser.add_argument(
        "--neural-max-length",
        "--lstm-max-length",
        dest="neural_max_length",
        type=int,
        default=200,
        help="Maximum sequence length for the neural tokeniser.",
    )
    parser.add_argument(
        "--neural-min-frequency",
        "--lstm-min-frequency",
        dest="neural_min_frequency",
        type=int,
        default=2,
        help="Minimum token frequency to be included in the vocabulary.",
    )
    parser.add_argument(
        "--neural-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for neural network optimisation.",
    )
    parser.add_argument(
        "--neural-dropout",
        type=float,
        default=0.3,
        help="Dropout probability applied in neural architectures.",
    )
    parser.add_argument(
        "--cnn-kernel-sizes",
        nargs="+",
        type=int,
        default=[3, 4, 5],
        help="Kernel sizes for the Text CNN model.",
    )
    parser.add_argument(
        "--cnn-num-filters",
        type=int,
        default=128,
        help="Number of filters per kernel size for the Text CNN model.",
    )
    parser.add_argument(
        "--attention-heads",
        type=int,
        default=4,
        help="Number of attention heads for the self-attention model.",
    )
    parser.add_argument(
        "--mlp-hidden-dim",
        type=int,
        default=256,
        help="Hidden layer size for the MLP neural model.",
    )

    parser.add_argument(
        "--transformer-models",
        nargs="+",
        default=None,
        help=(
            "List of Hugging Face model identifiers to fine-tune. Default models: "
            + ", ".join(DEFAULT_TRANSFORMER_MODELS)
        ),
    )
    parser.add_argument(
        "--bert-model",
        dest="legacy_bert_model",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--transformer-epochs",
        "--bert-epochs",
        dest="transformer_epochs",
        type=int,
        default=1,
        help="Number of epochs for transformer fine-tuning.",
    )
    parser.add_argument(
        "--transformer-batch-size",
        "--bert-batch-size",
        dest="transformer_batch_size",
        type=int,
        default=8,
        help="Batch size for transformer fine-tuning and evaluation.",
    )
    parser.add_argument(
        "--transformer-max-length",
        "--bert-max-length",
        dest="transformer_max_length",
        type=int,
        default=256,
        help="Maximum tokenised sequence length for transformer models.",
    )
    parser.add_argument(
        "--transformer-learning-rate",
        "--bert-learning-rate",
        dest="transformer_learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate for transformer fine-tuning.",
    )
    parser.add_argument(
        "--transformer-weight-decay",
        "--bert-weight-decay",
        dest="transformer_weight_decay",
        type=float,
        default=0.01,
        help="Weight decay applied during transformer optimisation.",
    )
    parser.add_argument(
        "--transformer-threshold",
        "--bert-threshold",
        dest="transformer_threshold",
        type=float,
        default=0.5,
        help="Decision threshold applied to transformer probabilities.",
    )

    parser.add_argument(
        "--skip-classic",
        action="store_true",
        help="Skip training classical machine learning baselines.",
    )
    parser.add_argument(
        "--skip-tree",
        dest="skip_classic",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-neural",
        action="store_true",
        help="Skip training neural network baselines.",
    )
    parser.add_argument(
        "--skip-lstm",
        dest="skip_neural",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-transformers",
        action="store_true",
        help="Skip transformer fine-tuning baselines.",
    )
    parser.add_argument(
        "--skip-bert",
        dest="skip_transformers",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    args = parser.parse_args()

    if args.classic_models is None:
        args.classic_models = DEFAULT_CLASSIC_MODELS.copy()
    else:
        unknown = sorted(set(args.classic_models) - set(CLASSIC_MODEL_KEYS))
        if unknown:
            raise ValueError(f"Unknown classical model keys provided: {unknown}")
    if args.neural_models is None:
        args.neural_models = DEFAULT_NEURAL_MODELS.copy()
    else:
        unknown = sorted(set(args.neural_models) - set(NEURAL_MODEL_KEYS))
        if unknown:
            raise ValueError(f"Unknown neural model keys provided: {unknown}")
    if args.transformer_models is None:
        if args.legacy_bert_model is not None:
            args.transformer_models = [args.legacy_bert_model]
        else:
            args.transformer_models = DEFAULT_TRANSFORMER_MODELS.copy()
    else:
        args.transformer_models = list(dict.fromkeys(args.transformer_models))

    args.cnn_kernel_sizes = [int(size) for size in args.cnn_kernel_sizes]
    if any(size <= 0 for size in args.cnn_kernel_sizes):
        raise ValueError("CNN kernel sizes must be positive integers.")

    if "self_attention" in args.neural_models and args.neural_hidden_dim % args.attention_heads != 0:
        raise ValueError("--neural-hidden-dim must be divisible by --attention-heads for the self-attention model.")
    return args



def load_dataset(data_path: Path, label_columns: Sequence[str] | None = None) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """Load the Amazon dataset and return texts and label matrix."""

    df = pd.read_csv(data_path)
    df["Title"] = df["Title"].fillna("")
    df["Review"] = df["Review"].fillna("")
    df["text"] = (df["Title"].str.strip() + " " + df["Review"].str.strip()).str.strip()

    if label_columns is None:
        default_labels = ["Materials", "Construction", "Color", "Finishing", "Durability"]
        label_columns = [col for col in default_labels if col in df.columns]
    else:
        missing = [col for col in label_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Label columns not found in dataset: {missing}")

    labels = df[label_columns].fillna(0).astype(int)
    texts = df["text"].fillna("").to_numpy()
    label_matrix = labels.to_numpy()
    return texts, label_matrix, list(label_columns), df


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute a suite of multi-label classification metrics."""

    return {
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }


def save_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: Sequence[str], output_path: Path
) -> None:
    """Persist the sklearn classification report as a text file."""

    report = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    output_path.write_text(report, encoding="utf-8")


def plot_multilabel_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: Sequence[str], output_path: Path
) -> None:
    """Visualise the multilabel confusion matrix for each target."""

    matrices = multilabel_confusion_matrix(y_true, y_pred)
    n_labels = len(label_names)
    cols = min(3, n_labels)
    rows = math.ceil(n_labels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (matrix, label) in enumerate(zip(matrices, label_names)):
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cbar=False,
            ax=axes[idx],
            cmap="Blues",
            xticklabels=["0", "1"],
            yticklabels=["0", "1"],
        )
        axes[idx].set_title(label)
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def simple_tokenizer(text: str) -> List[str]:
    """Tokenise text into lower-cased word tokens."""

    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def build_vocabulary(texts: Iterable[str], min_frequency: int) -> dict:
    """Create a word-index mapping based on training texts."""

    counter: Dict[str, int] = {}
    for text in texts:
        for token in simple_tokenizer(text):
            counter[token] = counter.get(token, 0) + 1

    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, frequency in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if frequency < min_frequency:
            continue
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(text: str, vocabulary: dict, max_length: int) -> List[int]:
    """Convert a piece of text into a fixed-length sequence of token ids."""

    tokens = [vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in simple_tokenizer(text)]
    if len(tokens) >= max_length:
        return tokens[:max_length]
    padding_needed = max_length - len(tokens)
    return tokens + [vocabulary[PAD_TOKEN]] * padding_needed


class SequenceDataset(Dataset):
    """PyTorch dataset wrapping padded token sequences and labels."""

    def __init__(self, sequences: Sequence[Sequence[int]], labels: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        return self.sequences[idx], self.labels[idx]


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM for multi-label classification."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
        padding_idx: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embedded = self.embedding(inputs)
        outputs, _ = self.lstm(embedded)
        pooled = outputs.mean(dim=1)
        logits = self.fc(self.dropout(pooled))
        return logits


class GRUClassifier(nn.Module):
    """Bidirectional GRU baseline."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
        padding_idx: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embedded = self.embedding(inputs)
        outputs, _ = self.gru(embedded)
        pooled = outputs.mean(dim=1)
        logits = self.fc(self.dropout(pooled))
        return logits


class TextCNNClassifier(nn.Module):
    """Text CNN with multiple kernel sizes."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: Sequence[int],
        num_labels: int,
        padding_idx: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embedding_dim, num_filters, kernel_size, padding=kernel_size // 2) for kernel_size in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embedded = self.embedding(inputs)
        embedded = embedded.transpose(1, 2)
        conv_outputs = [torch.relu(conv(embedded)).max(dim=2).values for conv in self.convs]
        features = torch.cat(conv_outputs, dim=1)
        logits = self.fc(self.dropout(features))
        return logits


class SelfAttentionClassifier(nn.Module):
    """Self-attention pooling architecture."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_labels: int,
        padding_idx: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embedded = self.embedding(inputs)
        projected = self.projection(embedded)
        attn_output, _ = self.attention(projected, projected, projected)
        pooled = attn_output.mean(dim=1)
        logits = self.fc(self.dropout(pooled))
        return logits


class MLPClassifier(nn.Module):
    """MLP on averaged embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
        padding_idx: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        embedded = self.embedding(inputs)
        pooled = embedded.mean(dim=1)
        hidden = self.activation(self.fc1(self.dropout(pooled)))
        logits = self.fc2(self.dropout(hidden))
        return logits


NEURAL_MODEL_BUILDERS: Dict[str, Callable[[int, int, argparse.Namespace, int], nn.Module]] = {
    "bilstm": lambda vocab_size, num_labels, args, padding_idx: LSTMClassifier(
        vocab_size,
        args.neural_embedding_dim,
        args.neural_hidden_dim,
        num_labels,
        padding_idx,
        args.neural_dropout,
    ),
    "bigru": lambda vocab_size, num_labels, args, padding_idx: GRUClassifier(
        vocab_size,
        args.neural_embedding_dim,
        args.neural_hidden_dim,
        num_labels,
        padding_idx,
        args.neural_dropout,
    ),
    "textcnn": lambda vocab_size, num_labels, args, padding_idx: TextCNNClassifier(
        vocab_size,
        args.neural_embedding_dim,
        args.cnn_num_filters,
        args.cnn_kernel_sizes,
        num_labels,
        padding_idx,
        args.neural_dropout,
    ),
    "self_attention": lambda vocab_size, num_labels, args, padding_idx: SelfAttentionClassifier(
        vocab_size,
        args.neural_embedding_dim,
        args.neural_hidden_dim,
        args.attention_heads,
        num_labels,
        padding_idx,
        args.neural_dropout,
    ),
    "mlp": lambda vocab_size, num_labels, args, padding_idx: MLPClassifier(
        vocab_size,
        args.neural_embedding_dim,
        args.mlp_hidden_dim,
        num_labels,
        padding_idx,
        args.neural_dropout,
    ),
}

CLASSIC_MODEL_BUILDERS: Dict[str, Callable[[argparse.Namespace], OneVsRestClassifier]] = {
    "logistic_regression": lambda args: OneVsRestClassifier(
        LogisticRegression(
            max_iter=args.classic_max_iter,
            class_weight="balanced",
            solver="lbfgs",
        ),
        n_jobs=args.classic_n_jobs,
    ),
    "linear_svc": lambda args: OneVsRestClassifier(
        LinearSVC(C=1.0, max_iter=args.classic_max_iter),
        n_jobs=args.classic_n_jobs,
    ),
    "sgd_classifier": lambda args: OneVsRestClassifier(
        SGDClassifier(
            loss="log_loss",
            penalty="l2",
            max_iter=args.classic_max_iter,
            class_weight="balanced",
        ),
        n_jobs=args.classic_n_jobs,
    ),
    "passive_aggressive": lambda args: OneVsRestClassifier(
        PassiveAggressiveClassifier(max_iter=args.classic_max_iter, class_weight="balanced"),
        n_jobs=args.classic_n_jobs,
    ),
    "ridge_classifier": lambda args: OneVsRestClassifier(
        RidgeClassifier(class_weight="balanced", max_iter=args.classic_max_iter),
        n_jobs=args.classic_n_jobs,
    ),
    "random_forest": lambda args: OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=args.classic_forest_estimators,
            random_state=args.random_seed,
            n_jobs=args.classic_n_jobs,
            class_weight="balanced_subsample",
        ),
        n_jobs=args.classic_n_jobs,
    ),
    "extra_trees": lambda args: OneVsRestClassifier(
        ExtraTreesClassifier(
            n_estimators=args.classic_forest_estimators,
            random_state=args.random_seed,
            n_jobs=args.classic_n_jobs,
            class_weight="balanced",
        ),
        n_jobs=args.classic_n_jobs,
    ),
    "knn": lambda args: OneVsRestClassifier(
        KNeighborsClassifier(n_neighbors=args.classic_knn_neighbors, weights="distance"),
        n_jobs=args.classic_n_jobs,
    ),
    "multinomial_nb": lambda args: OneVsRestClassifier(MultinomialNB()),
    "complement_nb": lambda args: OneVsRestClassifier(ComplementNB()),
}


def compute_pca_visualisation(
    vectorizer: TfidfVectorizer,
    texts: Sequence[str],
    label_matrix: np.ndarray,
    n_components: int,
    output_path: Path,
) -> None:
    """Reduce the TF-IDF space for visualisation and persist a scatter plot."""

    transformed = vectorizer.transform(texts)
    reducer = TruncatedSVD(n_components=n_components, random_state=42)
    components = reducer.fit_transform(transformed)

    label_cardinality = label_matrix.sum(axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(components[:, 0], components[:, 1], c=label_cardinality, cmap="viridis", alpha=0.7)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("PCA of TF-IDF features")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Number of active labels")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)



def train_classic_models(
    base_vectorizer: TfidfVectorizer,
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> List[ModelResult]:
    """Train the suite of classical machine learning models."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[ModelResult] = []

    vectorizer = clone(base_vectorizer)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)

    for key in args.classic_models:
        display_name = CLASSIC_MODEL_DISPLAY_NAMES[key]
        classifier = CLASSIC_MODEL_BUILDERS[key](args)
        model_output_dir = output_dir / key
        model_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Classical] Training {display_name}...")
        start_time = time.time()
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)
        duration = time.time() - start_time

        metrics = compute_classification_metrics(y_test, predictions)
        save_classification_report(
            y_test, predictions, label_names, model_output_dir / "classification_report.txt"
        )
        plot_multilabel_confusion_matrix(
            y_test, predictions, label_names, model_output_dir / "confusion_matrix.png"
        )

        results.append(
            ModelResult(
                approach="Classical ML",
                method=f"{display_name} (TF-IDF)",
                dataset=args.data_path.name,
                samples=len(train_texts),
                duration_seconds=duration,
                metrics=metrics,
                output_dir=model_output_dir,
            )
        )

    return results


def train_neural_models(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> List[ModelResult]:
    """Train multiple neural architectures on the tokenised dataset."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[ModelResult] = []

    vocabulary = build_vocabulary(train_texts, args.neural_min_frequency)
    train_sequences = [encode_text(text, vocabulary, args.neural_max_length) for text in train_texts]
    test_sequences = [encode_text(text, vocabulary, args.neural_max_length) for text in test_texts]

    train_dataset = SequenceDataset(train_sequences, y_train)
    test_dataset = SequenceDataset(test_sequences, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.neural_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.neural_batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = len(label_names)
    padding_idx = vocabulary[PAD_TOKEN]

    for key in args.neural_models:
        display_name = NEURAL_MODEL_DISPLAY_NAMES[key]
        model_output_dir = output_dir / key
        model_output_dir.mkdir(parents=True, exist_ok=True)

        model = NEURAL_MODEL_BUILDERS[key](len(vocabulary), num_labels, args, padding_idx).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.neural_learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        print(f"\n[Neural] Training {display_name}...")
        history = []
        start_time = time.time()
        for epoch in range(1, args.neural_epochs + 1):
            model.train()
            running_loss = 0.0
            for batch_inputs, batch_labels in train_loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                logits = model(batch_inputs)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * batch_inputs.size(0)
            epoch_loss = running_loss / len(train_loader.dataset)
            history.append({"epoch": epoch, "loss": epoch_loss})

        model.eval()
        all_probs = []
        with torch.no_grad():
            for batch_inputs, _ in test_loader:
                batch_inputs = batch_inputs.to(device)
                logits = model(batch_inputs)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        probabilities = np.vstack(all_probs)
        predictions = (probabilities >= 0.5).astype(int)
        duration = time.time() - start_time

        metrics = compute_classification_metrics(y_test, predictions)

        (model_output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        save_classification_report(
            y_test, predictions, label_names, model_output_dir / "classification_report.txt"
        )
        plot_multilabel_confusion_matrix(
            y_test, predictions, label_names, model_output_dir / "confusion_matrix.png"
        )

        results.append(
            ModelResult(
                approach="Neural Networks",
                method=display_name,
                dataset=args.data_path.name,
                samples=len(train_texts),
                duration_seconds=duration,
                metrics=metrics,
                output_dir=model_output_dir,
            )
        )

    return results


class TransformersDataset(Dataset):
    """Dataset wrapper for Hugging Face tokeniser outputs."""

    def __init__(self, encodings: dict, labels: np.ndarray) -> None:
        self.encodings = {key: torch.tensor(value) for key, value in encodings.items()}
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:  # type: ignore[override]
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def slugify_model_name(name: str) -> str:
    """Convert a Hugging Face model name into a filesystem-friendly slug."""

    return name.replace("/", "__").replace(" ", "_")


def train_transformer_model(
    model_name: str,
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> ModelResult:
    """Fine-tune a single transformer for the multi-label task."""

    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DISABLED", "true")

    print(f"\n[Transformers] Fine-tuning {model_name}...")
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_encodings = tokenizer(
        list(train_texts),
        truncation=True,
        padding=True,
        max_length=args.transformer_max_length,
    )
    test_encodings = tokenizer(
        list(test_texts),
        truncation=True,
        padding=True,
        max_length=args.transformer_max_length,
    )

    train_dataset = TransformersDataset(train_encodings, y_train)
    test_dataset = TransformersDataset(test_encodings, y_test)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.transformer_epochs,
        per_device_train_batch_size=args.transformer_batch_size,
        per_device_eval_batch_size=args.transformer_batch_size,
        learning_rate=args.transformer_learning_rate,
        weight_decay=args.transformer_weight_decay,
        logging_strategy="epoch",
        save_strategy="no",
        evaluation_strategy="no",
        report_to=[],
    )

    def trainer_compute_metrics(predictions):
        logits = predictions.predictions
        labels = predictions.label_ids
        probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probabilities >= args.transformer_threshold).astype(int)
        metrics = compute_classification_metrics(labels, preds)
        return metrics

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=trainer_compute_metrics,
    )
    trainer.train()

    prediction_output = trainer.predict(test_dataset)
    probabilities = torch.sigmoid(torch.tensor(prediction_output.predictions)).numpy()
    predictions = (probabilities >= args.transformer_threshold).astype(int)

    metrics = compute_classification_metrics(y_test, predictions)
    duration = time.time() - start_time

    save_classification_report(y_test, predictions, label_names, output_dir / "classification_report.txt")
    plot_multilabel_confusion_matrix(y_test, predictions, label_names, output_dir / "confusion_matrix.png")

    return ModelResult(
        approach="Transformers",
        method=f"{model_name} (fine-tuned)",
        dataset=args.data_path.name,
        samples=len(train_texts),
        duration_seconds=duration,
        metrics=metrics,
        output_dir=output_dir,
    )


def train_transformer_models(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> List[ModelResult]:
    """Fine-tune multiple transformer checkpoints."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[ModelResult] = []

    for model_name in args.transformer_models:
        model_output_dir = output_dir / slugify_model_name(model_name)
        result = train_transformer_model(
            model_name,
            train_texts,
            test_texts,
            y_train,
            y_test,
            label_names,
            model_output_dir,
            args,
        )
        results.append(result)

    return results


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    texts, labels, label_names, _ = load_dataset(args.data_path, args.label_columns)
    if args.max_samples is not None:
        texts = texts[: args.max_samples]
        labels = labels[: args.max_samples]

    train_texts, test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_seed,
        shuffle=True,
    )

    base_vectorizer = TfidfVectorizer(max_features=args.tfidf_max_features, ngram_range=(1, 2))
    visual_vectorizer = clone(base_vectorizer).fit(texts)
    compute_pca_visualisation(
        visual_vectorizer,
        texts,
        labels,
        args.pca_components,
        output_dir / "tfidf_pca.png",
    )

    results: List[ModelResult] = []

    if not args.skip_classic:
        classic_results = train_classic_models(
            base_vectorizer,
            train_texts,
            test_texts,
            y_train,
            y_test,
            label_names,
            output_dir / "classical",
            args,
        )
        results.extend(classic_results)

    if not args.skip_neural:
        neural_results = train_neural_models(
            train_texts,
            test_texts,
            y_train,
            y_test,
            label_names,
            output_dir / "neural",
            args,
        )
        results.extend(neural_results)

    if not args.skip_transformers:
        transformer_results = train_transformer_models(
            train_texts,
            test_texts,
            y_train,
            y_test,
            label_names,
            output_dir / "transformers",
            args,
        )
        results.extend(transformer_results)

    if results:
        summary_table = pd.DataFrame([result.to_table_row() for result in results])
        summary_table.to_csv(output_dir / "results_summary.csv", index=False)
        (output_dir / "results_summary.json").write_text(
            json.dumps(summary_table.to_dict(orient="records"), indent=2),
            encoding="utf-8",
        )
        print("\n=== Model comparison ===")
        print(summary_table.to_markdown(index=False))
    else:
        print("No models were trained because all were skipped.")


if __name__ == "__main__":
    main()

