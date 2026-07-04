# Mushroom Classification

Machine-learning analysis of the **mushroom dataset** — predicting whether a mushroom is
edible or poisonous from its physical characteristics. The project moves from classic
supervised classifiers to a PyTorch model and finally to unsupervised structure discovery.

## Notebooks

| Notebook | Topic |
|---|---|
| [`01_classifiers.ipynb`](01_classifiers.ipynb) | Feature visualization and classic classifiers — majority-vote and random baselines, decision trees, and k-nearest neighbors, with hyperparameter selection (best k, best depth) |
| [`02_logistic_regression_pytorch.ipynb`](02_logistic_regression_pytorch.ipynb) | Logistic regression in PyTorch — a linear layer with a sigmoid output trained with binary cross-entropy |
| [`03_pca_kmeans_clustering.ipynb`](03_pca_kmeans_clustering.ipynb) | Unsupervised analysis — PCA for dimensionality reduction and K-means clustering |

## Tech stack

Python · scikit-learn · PyTorch · pandas · NumPy · matplotlib

## Getting started

```bash
pip install -r requirements.txt
jupyter notebook
```

The notebooks download the mushroom dataset automatically, so no manual setup is required.
