# Polycrystal GNN reproduction and improvement experiments

Windows / Anaconda reproduction of **Graph neural networks for an accurate and interpretable prediction of the properties of polycrystalline materials**, npj Computational Materials 7, 103 (2021), DOI [10.1038/s41524-021-00574-w](https://doi.org/10.1038/s41524-021-00574-w).

## 中文说明

本项目保留作者源码，并增加可在 Windows 的 Anaconda 环境直接运行的 CPU 实验脚本、数据下载脚本、原模型一致性检查及分组评测。**作者已更正原始代码中的消息传递错误**，因此本项目以更正后的官方模型为基线，不将修复该错误算作方法提升。

- 官方仓库：https://github.com/mehmetfdemirel/PolycrystalGraph
- 作者更正：https://www.nature.com/articles/s41524-022-00804-9
- 上游版本：`853719eb0ddaa7251881471a2b8524c060f830a3`（下载时 master 的 API 查询结果；使用源码 ZIP，不含上游 Git 历史）
- 上游 README 保存在 `README.upstream.md`；原始 Python 文件及 `hyper/` 均保留。
- 数据来自作者 `download_data.sh` 指定地址，492 个微结构，2410 条原始观测；按官方规则排除 123 条负值观测后，保留 2287 条有效观测。

## Run with Anaconda

The actual local run uses Python 3.13.5, PyTorch 2.14.0+cpu and NumPy 2.5.3 in an existing Anaconda environment. `experiment.py` needs only PyTorch and NumPy; original scripts additionally need SciPy and scikit-learn. The original `env.yml` describes the historical Python 3.7.4 / PyTorch 1.8.1 environment; this is a port, not a bit-for-bit recreation of that environment.

```powershell
conda activate YOUR_ENV
python download_data.py
python check_port.py
python experiment.py --split official --model baseline --epochs 1500
python experiment.py --split official --model pooled --epochs 1500
python experiment.py --split grouped --model baseline --epochs 1500
python experiment.py --split grouped --model pooled --epochs 1500
```

Or run all four configurations:

```powershell
./run_experiments.ps1 -Python 'C:\path\to\anaconda3\envs\YOUR_ENV\python.exe'
```

For the first exploratory comparison, use `-Epochs 300`. The wrapper enables `--fast`: the baseline's second linear projection is moved before adjacency aggregation using `A(XW)+b = (AX)W+b`. The bias remains after aggregation. This preserves the mathematical model and parameter count while reducing the expensive adjacency multiplication from 80 channels to 3. `check_port.py` checks forward outputs and parameter gradients against the unoptimized port within float32 tolerances. Direct commands without `--fast` retain the original operation order. Floating-point associativity changes can change the training trajectory.

Completed configurations are skipped at the same epoch budget; a different budget requires archiving the existing result directory and passing `--overwrite` (or `-Overwrite` to the wrapper). A `.running` marker prevents duplicate concurrent runs. If a process is forcibly terminated, remove its stale marker only after confirming that training has stopped. Avoid launching concurrent runs before the shared data cache has been created once.

## Models and evaluation

**Baseline:** exact corrected upstream architecture: normalized adjacency with self loops, 5→80→3 sigmoid graph layers, flatten over 300 padded nodes, concatenate applied field, 901→1024→128→1 ReLU MLP. Adam learning rate 0.0005, batch size up to 32, 1500 epochs, seed 124; hyperparameter set 20. Forward output is checked against the actual upstream class with identical weights and inputs.

**Pooled candidate:** encode the three Euler angles with sine/cosine (8 input channels including size and degree), then masked 8→64→64 ReLU graph layers with a residual connection; concatenate mean/max pooling, field and grain count divided by 300; 130→128→64→1 readout, Adam learning rate 0.001. Padding and the dataset's dummy node 0 are excluded from pooling. Count retains global size information that mean pooling would lose; trigonometric features avoid the artificial 0/2π angle discontinuity. This candidate changes several choices together; it is an exploratory model comparison, not an attribution of improvement to pooling alone. Node permutation invariance is verified numerically.

**Official-compatible split:** reproduces the original graph shuffle (123), observation-level shuffled ten-fold split (123), validation fold 8 and test fold 9. As in upstream, target normalization uses all observations. The same microstructure can occur in both training and test data. This answers the original observation-level task, not generalization to unseen microstructures.

**Grouped split:** partitions microstructure IDs, holds out complete structures, fits target normalization on training data only, and reports train/test structure overlap (must be zero). This is a stricter task; its numbers should not be compared directly with the original paper's numbers.

For both models, the checkpoint is chosen by validation RMSE only; the test set is evaluated after training. This differs from the upstream final-epoch reporting. Experiments use float32 and a reshuffled batch order, so exact historical training trajectories are not expected. Relative absolute error is `sum(abs(pred-true))/sum(abs(true))`, matching `macro_avg_err`; it is not mean sample-wise percentage error. RMSE/MAE use the raw magnetostriction units.

## Results and reproducibility

See `REPORT.md` after the experiments finish. Each run saves `metrics.json`, `learning.csv`, `predictions.csv`, exact split indices, and best/last state dictionaries under `results/`. Large data, cache and checkpoints are excluded from Git. Download data and rerun commands to regenerate them.

`environment-port.yml` pins the locally observed environment versions; creation of a fresh environment from that file has not been tested. The successful run uses the existing Anaconda environment. `plot_results.py` optionally generates charts with Matplotlib; `summarize.py` builds the report without extra dependencies.

These initial experiments use one seed and one split. Any improvement is preliminary until confirmed with multiple seeds and grouped cross-validation. No hyperparameter choice is made using test scores. Integrated gradients and phase-field data generation are not rerun by `experiment.py`; the original interpretation scripts are retained for follow-up.

The first comparison used a 300-epoch cap. The grouped baseline was stopped early after more than 100 epochs without validation improvement; its actual epoch count is recorded in metrics and the report. This exploratory stopping decision was introduced after observing stagnation. `finalize_stopped_run.py` verifies the patience condition and evaluates the saved best checkpoint after training stops. It is not an automatically applied stopping rule in the main trainer. Comparisons with different actual epoch counts are explicitly flagged. Official baseline used four CPU threads; grouped baseline used two; pooled models used four.

Predict an existing benchmark structure at a supplied field after training:

```powershell
python predict.py --run results/official_pooled_seed124 --structure 1 --field 200
```

This entry point supports the two GNN architectures, requires the local data/cache and checkpoint, and uses training normalization saved in metrics. It does not claim extrapolation accuracy outside the dataset's field range or for new structures. It outputs dimensionless magnetostriction and microstrain.

## Additional descriptor baseline

`descriptor_baseline.py` runs in the existing Anaconda base environment with NumPy, scikit-learn and joblib. It compares Extra Trees (300 trees), Random Forest (300 trees) and Huber Gradient Boosting (500 stages), selects one by validation RMSE, then evaluates that model on test data. Both splits reuse the exact saved GNN split indices. Features include sine/cosine orientation moments and quantiles, volume-weighted moments, grain count, size/degree statistics, field and a fourth-order direction-cosine statistic. The direction-cosine transform assumes Bunge ZXZ angles; this convention has not been independently checked against the original DREAM.3D export. These are exploratory descriptors, not a replacement for the phase-field physics model.

This method has no message-passing layers and is reported as an alternative baseline, not a GNN architectural improvement. It does not use target normalization or test labels for model selection. Run it after generating the baseline split files:

```powershell
python descriptor_baseline.py
python summarize.py
python plot_results.py
```

The descriptor and plotting scripts can use the Anaconda base environment; the successful GNN runs use the existing PyTorch environment described above. They are optional dependencies outside `environment-port.yml`.

## Provenance and license

Downloaded source ZIP SHA256: `ac1f7f66e4f736627314776898867672df91321144578ca328c115d6ac06dc04`.

Author data archive SHA256: `e54cbb9f818b7ca651071d08e58e66ac1f500cf60b3da912234f6a7fabaf3937`.

Upstream code is MIT licensed; its `LICENSE` and attribution are retained. Dataset redistribution rights are not inferred from the code license; data are downloaded separately from the author-provided URL.
