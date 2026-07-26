# SOCAM Inference And Hop-Scoring Plan

## Current State

- `SOCAM_inference` is based on the latest `origin/main`.
- The lambda-trials SOCAM device fix is included in `cellarium/ml/models/socam.py`.
- Current `main` already has the modern SOCAM implementation, including memory and numerical-stability changes that should not be replaced by the older `SOCAM` branch implementation.
- The old `SOCAM` branch contains useful inference pieces:
  - `cellarium/ml/hop_scoring/hop_score_calculation.py`
  - `cellarium/ml/hop_scoring/utils.py`
  - a specialized `cellarium/ml/callbacks/prediction_writer.py`
  - `cellarium/ml/models/custom_logistic_regression.py` as the reference for `predict()` output shape and probability propagation behavior.

## Goal

Add a SOCAM prediction callback that runs `socam predict`, writes per-batch hop-based metrics for hop levels 0 through 4, and supports output to a GCS bucket.

## Implementation Plan

1. Add a new hop-scoring package under `cellarium/ml/hop_scoring`.
   - Port the scoring logic from `origin/SOCAM`.
   - Keep the public API small, for example `calculate_hop_metrics_for_batch(...)`.
   - Avoid committing large metadata artifacts from the old branch.
   - Load ontology resources from a user-provided local or GCS path.

2. Add a dedicated callback rather than replacing the generic `PredictionWriter`.
   - Keep `cellarium.ml.callbacks.PredictionWriter` as the generic writer used by other models.
   - Add `cellarium.ml.callbacks.SOCAMHopScoringPredictionWriter`.
   - Export it from `cellarium/ml/callbacks/__init__.py`.
   - Required callback args should include `output_dir`, `co_resource_path`, `class_names_path` or an equivalent source, `num_hops=4`, `key="cell_type_probs_nc"`, and `gzip` or CSV options.

3. Make the callback match current SOCAM predict output.
   - Current `SOCAM.predict()` returns `{"y_logits_nc": logits_nc, "cell_type_probs_nc": probs_nc}`.
   - The callback should read `prediction["cell_type_probs_nc"]`.
   - Ground-truth labels should come from a prediction batch key, likely `cl_names_n`.
   - Cell IDs should come from `obs_names_n`.
   - Class columns should align to `pl_module.model.active_cl_names`, not old `y_categories` or `valid_mask` assumptions from `CustomLogisticRegression`.

4. Support GCS output through `pandas.to_csv("gs://...")`.
   - `gcsfs` is expected to be present in the runtime environment.
   - Use deterministic shard names that include `batch_idx`, `global_rank`, and `world_size`.

5. Avoid expensive per-batch resource reloads.
   - Load the Cell Ontology resource and class-name metadata once in callback `__init__` or `setup`.
   - Validate that all `active_cl_names` and ground-truth labels exist in the resource before long prediction runs when possible.

6. Add a SOCAM predict config.
   - Use `examples/cli_workflow/socam_predict_hop_scoring_config.yaml` as the starting template.
   - Use `cellarium-ml socam predict --config ...`.
   - Set `return_predictions: false`.
   - Include callback config for `SOCAMHopScoringPredictionWriter`.
   - Include `batch_keys` for `x_ng`, `var_names_g`, `obs_names_n`, and `cl_names_n`.
   - Set `ckpt_path` to the lambda=100 checkpoint.

7. Add focused tests.
   - Unit-test hop scoring on a tiny synthetic ontology resource.
   - Unit-test callback batch writing with a synthetic `prediction` dict and batch containing `obs_names_n` and `cl_names_n`.
   - Add a SOCAM `predict()` compatibility test confirming `cell_type_probs_nc` columns match `active_cl_names`.

8. Benchmarking outputs.
   - Write one CSV per prediction shard initially.
   - Add an optional aggregation utility later if a single summary CSV is needed.
   - Include per-cell columns for `query_cell_id`, `ground_truth_cl_name`, and hop-level metrics from `hop_0_*` through `hop_4_*`.

## Main Differences From The Old SOCAM Branch

- Do not port `CustomLogisticRegression` wholesale; it is useful as a reference, but current `SOCAM` already owns inference.
- Do not replace the generic `PredictionWriter`; add a SOCAM-specific callback.
- Do not use old `actual_categories`, `valid_mask`, or `y_categories_path` assumptions. Current SOCAM prediction uses `active_cl_names`.
- Do not merge the old `SOCAM` branch directly because it removes or rewrites many files that are newer on `main`.
