# Verification record

The following checks were executed locally before the comparison runs:

- Unoptimized `Baseline` forward output matches the corrected upstream `GraphModel` exactly for identical state dictionaries and four real microstructures.
- `FastBaseline` forward outputs match `Baseline` within `rtol=1e-5, atol=1e-6`; all parameter gradients match within `rtol=1e-4, atol=1e-5`.
- The pooled candidate preserves its prediction after a shared permutation of node features, adjacency rows/columns and node masks, within float32 tolerances.
- Official validation/test indices match the original scikit-learn `KFold(10, shuffle=True, random_state=123)` after the original graph shuffle.
- Inputs are finite; there are 492 structures and 2287 retained observations.
- A two-epoch smoke training run completed and wrote metrics/predictions/checkpoints. Its scores are excluded from the formal comparison.
- The single-structure `predict.py` entry point completed using a saved GNN checkpoint and produced finite magnetostriction/microstrain predictions.
- Three GNN configurations finished 300 epochs. The grouped baseline was finalized at 138 epochs after validation stagnation; its best checkpoint was at epoch 15.
- Six descriptor candidates (three per split) were fit, with model selection on validation RMSE and final test evaluation of the selected candidate only.
- Final learning-curve and prediction plots were rendered and visually inspected.

Re-run model checks using `python check_port.py`. Syntax checks passed for all added Python entry points. A fresh Conda environment installation and GPU execution have not been verified.
