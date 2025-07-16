# Animal Disease Dataset - Exploratory and Predictive Analysis

This project analyzes a dataset of animal health records, including symptoms, age, temperature, animal types, and diagnosed diseases. The goal is to explore feature-disease relationships, build predictive models, and uncover hidden syndromic clusters using both statistical and machine learning techniques.


# Outline of Analysis

## Data Exploration
Dataset structure and variable types

Summary statistics and distribution plots (KDE, barplots, histograms)

## Statistical Analysis
Chi-square tests for feature–disease associations (categorical features)

ANOVA for age and temperature across disease classes

KDE comparisons for visual distribution insights

## Predictive Modeling
One-hot encoding for symptoms and categorical features

Logistic Regression and Random Forest models

Model evaluation (Accuracy, Precision, Recall, F1-score)

SHAP values for feature importance and interpretability

### Clustering & Pattern Discovery
KMeans clustering

DBSCAN clustering (density-based)

Hierarchical Clustering

t-SNE and UMAP used for visualization and embedding

Clustering on UMAP-reduced space (KMeans + DBSCAN)

Cluster profiling by age, temperature, animal type, and symptom prevalence

Comparison of clusters with disease labels


# Key Outputs
Cleaned dataset with all predictions, cluster labels, and transformed features

Visualizations: SHAP plots, UMAP/t-SNE projections, KDE, heatmaps

Performance metrics for models

Silhouette scores for clustering quality


# Achievement
Identified symptom patterns predictive of disease

Revealed potential syndromic clusters across animal populations


