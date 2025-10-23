# Consumer Multi-label Classification

This repository contains tooling for experimenting with multi-label text
classification on the Amazon clothing reviews dataset (``data/raw/data_amazon.xlsx - Sheet1.csv``).

## Pipeline overview

The ``src/multilabel_pipeline.py`` script orchestrates the following steps:

1. **Vectorisation and visualisation** – TF-IDF features are computed for the
   combined ``Title`` and ``Review`` text fields. A PCA plot coloured by label
   cardinality is exported to help understand the dataset structure.
2. **Model training** – three model families can be trained via command-line
   switches and cover a broad portfolio of architectures:
   - Classical ML: 10 one-vs-rest classifiers including logistic regression,
     linear SVM, random forest, extra trees, KNN and several Naive Bayes
     variants.
   - Neural Networks: 5 PyTorch models (BiLSTM, BiGRU, TextCNN,
     self-attention pooling and an averaged-embedding MLP).
   - Transformers: 5 Hugging Face checkpoints are fine-tuned by default
     (DistilBERT, BERT, DistilRoBERTa, RoBERTa and ALBERT) and can be replaced
     with custom identifiers.
3. **Evaluation artefacts** – for every model the script saves a confusion
   matrix heatmap, a classification report and records the metrics into
   ``results_summary.csv`` and ``results_summary.json``.

## Getting started

1. Create and activate a Python environment (Python 3.10 or newer is
   recommended).
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the pipeline (use ``--max-samples`` during prototyping to speed things
   up):

   ```bash
   python src/multilabel_pipeline.py --max-samples 500 --output-dir reports/experiment_01
   ```

4. To skip heavy models during quick iterations you can use the dedicated
   switches:

   ```bash
   python src/multilabel_pipeline.py --skip-transformers --skip-neural
   ```

   You can also request specific subsets, e.g. only logistic regression and
   random forest on the classical side:

   ```bash
   python src/multilabel_pipeline.py --classic-models logistic_regression random_forest \
       --neural-models bilstm textcnn --transformer-models distilbert-base-uncased roberta-base
   ```

Artefacts are stored inside the directory provided via ``--output-dir`` and the
terminal prints a comparison table once the requested models finish training.
