# utils.py

from cellarium.ml import CellariumAnnDataDataModule, CellariumModule
from cellarium.ml.data import DistributedAnnDataCollection
from cellarium.ml.utilities.data import AnnDataField, collate_fn, densify, categories_to_codes
from cellarium.ml.callbacks import PredictionWriter
from cellarium.ml.data.dadc_dataset import IterableDistributedAnnDataCollectionDataset
from cellarium.ml.models import ImputationModel

import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
import pandas as pd
import numpy as np
import anndata as ad
import glob
import os
import torch
import shutil
import random

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report

def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

def mask_data(adata, masking_probability, random_seed, excluded_genes=None):
    """
    Randomly mask a fraction of genes per cell in an AnnData object, with the option to exclude specific genes from masking.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object (n_obs x n_vars).
    masking_probability : float
        Fraction of genes to mask per cell (e.g., 0.2 = 20%).
    random_seed : int
        Random seed for reproducibility.
    excluded_genes : list of str, optional
        List of genes to exclude from masking.

    Returns
    -------
    masked_adata : AnnData
        New AnnData object with masked genes.
    """

    rng = np.random.default_rng(random_seed)
    n_cells, n_genes = adata.X.shape

    # --- Identify genes allowed for masking ---
    if excluded_genes is not None:
        excluded_genes = set(excluded_genes)
        excluded_indices = [
            i for i, g in enumerate(adata.var_names) if g in excluded_genes
        ]
    else:
        excluded_indices = []
    
    excluded_indices = np.array(excluded_indices, dtype=int)

    # Genes available for masking
    all_indices = np.arange(n_genes)
    maskable_indices = np.setdiff1d(all_indices, excluded_indices)

    n_maskable = len(maskable_indices)
    n_mask = int(masking_probability * n_maskable)

    print(f"Total genes: {n_genes}")
    print(f"Excluded genes: {len(excluded_indices)}")
    print(f"Maskable genes: {n_maskable}")
    print(f"Masking {n_mask} genes per cell ({masking_probability*100}% of maskable genes).")

    # Create mask matrix
    mask = np.zeros((n_cells, n_genes), dtype=bool)

    # For each cell, randomly select n_mask gene included indices to mask
    masked_indices_per_cell = []
    for i in range(n_cells):
        # Sample only from maskable genes
        mask_indices = rng.choice(maskable_indices, n_mask, replace=False)
        mask[i, mask_indices] = True
        masked_indices_per_cell.append(mask_indices)

    X_masked = adata.X.copy()
    X_masked[mask] = 0

    # Create new AnnData to store masked data and mask matrix
    masked_adata = ad.AnnData(
        X_masked,
        obs=adata.obs.copy(),
        var=adata.var.copy(),
    )

    # Store masked data and mask matrix in layers
    masked_adata.layers["X_masked"] = X_masked
    masked_adata.layers[f"mask_{int(masking_probability*100)}"] = mask.astype(np.int8)
    masked_adata.obsm = adata.obsm.copy()

    # Store which gene indices were masked per cell (as strings for convenience)
    masked_adata.obs[f"masked_gene_indices_{int(masking_probability*100)}"] = [
        ",".join(map(str, idxs)) for idxs in masked_indices_per_cell
    ]

    # Create DataModule for masked data
    dm_mask = CellariumAnnDataDataModule(
        dadc=masked_adata,
        batch_keys={
            "x_ng": AnnDataField(attr="X", convert_fn=densify),
            "var_names_g": AnnDataField(attr="var_names"),
            "obs_names_n": AnnDataField(attr="obs_names"),
            "batch_index_n": AnnDataField(
                attr="obs",
                key="batch",
                convert_fn=categories_to_codes,
            ),
        },
        batch_size=512,
        shuffle=False,
    )

    return masked_adata, dm_mask, masked_indices_per_cell

def save_predictions_latent(predictions, prediction_path, adata, label, prob):
    """
    Save prediction latent representations with proper labeling.
    
    Parameters
    ----------
    predictions : list
        List of prediction batches
    prediction_path : str
        Base path for saving predictions
    adata : AnnData
        AnnData object to store results
    label : str
        Label for the predictions ('whole' or 'masked')
    prob : float
        Masking probability used
    """

    # Create subdirectory for this prediction type
    pred_path = os.path.join(prediction_path, label)
    os.makedirs(pred_path, exist_ok=True)
    
    # Load and combine all batches
    dfs = []
    for file in glob.glob(os.path.join(prediction_path, f'batch_*')):
        dfs.append(pd.read_csv(file, index_col=0, header=None))
    latent_df = pd.concat(dfs, axis=0)
    latent_df = latent_df.loc[adata.obs_names]
    
    # Move batch files in prediction_path to pred_path
    batch_files = glob.glob(os.path.join(prediction_path, f'batch_*'))

    for src in batch_files:
        base = os.path.basename(src)
        dest = os.path.join(pred_path, base)
        try:
            shutil.move(src, dest)
        except Exception:
            if os.path.exists(dest):
                try:
                    os.remove(dest)
                except Exception as e:
                    raise RuntimeError(f"Could not remove existing destination {dest}: {e}") from e
            shutil.move(src, dest)

    return latent_df.values

def train_and_predict_imputation(imputation_module, adata, dm, masking_prob, n2s_weight, 
                                 prediction_base_path='runs/imputation_annealing',
                                 excluded_genes=None, 
                                 epochs=1):
    """
    Train imputation model and perform both whole dataset and masked predictions.
    
    Parameters
    ----------
    imputation_module : CellariumModule
        Imputation model to be trained
    adata : AnnData
        Input dataset
    dm : CellariumAnnDataDataModule
        Data module for training
    masking_prob : float
        Probability for masking genes
    n2s_weight : float
        Noise2Self weight for the model
    prediction_base_path : str
        Base path for storing predictions
    excluded_genes : list or None
        List of genes to exclude from masking
    Returns
    -------
    dict
        Dictionary containing whole and masked predictions, 
        whole and masked latent representations, 
        and masked gene indices.
    """

    # Setup paths and names
    name_label = f"mask{int(masking_prob*100)}_n2s{int(n2s_weight*100)}"
    prediction_path = f"{prediction_base_path}/predictions_{name_label}"
    
    # Define imputation module
    print(f"\nDefining imputation module with masking_probability={masking_prob} and noise2self_weight={n2s_weight}")
    # Train model
    print(f"Training model...")
    logger = CSVLogger("logs", name=f"imputation_{name_label}")
    trainer = pl.Trainer(
        accelerator="cpu",
        precision="32-true",
        devices=1,
        max_epochs=epochs,
        default_root_dir=f"{prediction_base_path}_{name_label}/",
        logger=logger,
        callbacks=[PredictionWriter(output_dir=prediction_path)])
    trainer.fit(imputation_module, dm)
    
    # Get metrics from the trained model
    imputation_module.eval()
    batch = next(iter(dm.train_dataloader()))
    batch = {k: v for k, v in batch.items() if k in {"x_ng", "batch_index_n","var_names_g"}}
    print(batch.keys())
    with torch.no_grad():
        metric_outputs = imputation_module.model(**batch)
    print(f"Loss: {metric_outputs['loss'].item():.4f}")
    print(f"Reconstruction Loss: {metric_outputs['reconstruction_loss'].mean().item():.4f}")
    print(f"Noise2Self Rec Loss: {metric_outputs['noise2self_rec_loss_n'].mean().item():.4f}")
    print(f"KL Divergence Z: {metric_outputs['kl_divergence_z'].mean().item():.4f}")
    print(f"KL Annealing Weight: {metric_outputs['kl_annealing_weight']:.4f}")

    # Predict on the whole dataset
    print("Predicting on whole dataset...")
    whole_predictions = trainer.predict(imputation_module, dm)
    whole_collated = collate_fn(whole_predictions)
    latent_whole = save_predictions_latent(whole_collated, prediction_path, adata, 'whole', masking_prob)

    # Create and predict on masked dataset
    print("Predicting on masked dataset...")
    masked_adata, dm_mask, masked_indices = mask_data(dm.dadc, masking_probability=masking_prob, 
                                                      random_seed=42, excluded_genes=excluded_genes)
    masked_predictions = trainer.predict(imputation_module, dm_mask)
    masked_collated = collate_fn(masked_predictions)
    latent_masked = save_predictions_latent(masked_collated, prediction_path, adata, 'masked', masking_prob)    
    
    return {
        f"{name_label}_whole": whole_collated,
        f"{name_label}_masked": masked_collated,
        f"{name_label}_latent_whole": latent_whole,
        f"{name_label}_latent_masked":latent_masked,
        f"{name_label}_masked_gene_indices": masked_indices,
        f"{name_label}_output_metrics": metric_outputs
    }

def train_classifiers(X_train, y_train):
    """Train multiple classifier models on the provided training data."""

    # Multinomial Logistic Regression -- this is considered to be a strong baseline model for cell type classification 
    print("Training Logistic Regression model...") 
    lr_model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
    lr_model.fit(X_train, y_train)

    # Random Forest Classifier
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Support Vector Machine Classifier
    print("Training SVM model...")
    svm_model = SVC(gamma='auto', random_state=42)
    svm_model.fit(X_train, y_train)

    return lr_model, rf_model, svm_model

def evaluate_classifiers(models, X_test, y_test, prob, noise2self_ratio):
    """Evaluate the trained classifiers on the test data and print classification reports."""
    
    model_names = ["Logistic Regression", "Random Forest", "SVM"]
    model_names_shortcut = ["LR", "RF", "SVM"]
    metrics_df = []
    for model, name in zip(models, model_names):
        print(f"\nEvaluating {name} model for masking probability {prob}...")
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        # --- Extract per-class metrics ---
        for cell_type, stats in report.items():
            if cell_type in ["accuracy", "macro avg", "weighted avg"]:
                continue

            metrics_df.append({
                "classifier": model_names_shortcut[model_names.index(name)],
                "masking_prob": prob,
                "noise2self_ratio": noise2self_ratio,
                "cell_type": cell_type,
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1_score": stats["f1-score"],
            })

    return pd.DataFrame(metrics_df)