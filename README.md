# GAST: A Graph-Augmented Spectral–Spatial Transformer with Adaptive Gated Fusion for Small-Sample Hyperspectral Image Classification

Hyperspectral image (HSI) classification demands models that can exploit both the hundreds of narrow spectral bands and the fine-grained spatial structure captured by modern sensors, yet remain robust to the severe scarcity of *labelled* pixels. 

We introduce **GAST** — a **Graph-Augmented Spectral–Spatial Transformer** that combines three complementary components:
1. **Spectral Transformer Encoder:** A lightweight Transformer encoder derives context-enriched spectral features for every pixel within a patch.
2. **Spatial Graph Attention:** A stack of Graph Attention Network v2 (GATv2) layers propagates information across an 8-neighbourhood graph to capture local spatial relationships.
3. **Adaptive Fusion:** A channel-wise gating mechanism adaptively fuses the spectral and spatial representations.

A two-stage Optuna **optimisation** procedure tunes hyper-parameters for each scene, whilst a scene-adaptive loss choice (focal versus weighted cross-entropy) mitigates class imbalance.

---

## Features

- **Patch-based HSI Classification:** Efficiently handles high-dimensional HSI cubes with patch extraction and normalization.
- **Spectral-Spatial Modeling:** Integrates spectral context (Transformer) and spatial context (GATv2) for robust pixel classification.
- **Adaptive Fusion:** Learns to combine spectral and spatial features via a gating mechanism.
- **Class Imbalance Handling:** Automatically selects between Focal Loss and weighted Cross-Entropy based on class distribution.
- **Automated Hyperparameter Tuning:** Two-stage (coarse + fine) Optuna search for each dataset.
- **Comprehensive Evaluation:** Includes scripts for multi-run experiments, parameter sweeps, t-SNE visualization, and detailed metric summarization.

---

## Project Structure

```
GAST2/
│
├── main.py                  # Main entry point for training/testing GAST
├── multirun.py              # Run multiple seeds/datasets for robust evaluation
├── summarize_metrics.py     # Summarize OA, AA, Kappa across runs
├── run_all_tsne.py          # Batch t-SNE visualization for all datasets
├── scripts/
│   ├── run_parameter_sweep.sh # Shell script for parameter sweeps
│   └── run_gast_ablation.sh # Shell script for ablation study
├── src/
│   ├── data/                # Data loading, class distribution, dataset info
│   ├── models/              # GAST model architecture
│   ├── training/            # Training, testing, Optuna tuning, reporting
│   └── utils/               # Visualization, utilities
├── models/                  # Saved checkpoints, splits, results
├── reports/                 # Results, figures, Optuna studies, summaries
└── requirements.txt         # Python dependencies
```

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# For GATv2: pip install torch-geometric
```

### 2. Prepare Data

- Place your HSI datasets in the `src/Dataset/` directory, following the structure in `src/data/data_loader.py`.
- Supported datasets: Botswana, Houston13, Indian_Pines, Kennedy_Space_Center, Pavia_Centre, Pavia_University, Salinas, SalinasA.

### 3. Train and Test

**Single run:**
```bash
python main.py --mode train --dataset Indian_Pines --output_dir models/checkpoints/Indian_Pines
python main.py --mode test --dataset Indian_Pines --checkpoint models/checkpoints/Indian_Pines/gast_best_Indian_Pines.pth
```

**Multi-run (all datasets, multiple seeds):**
```bash
python multirun.py
```

### 4. Hyperparameter Tuning

**Two-stage Optuna search:**
```bash
python src/training/gast_hyperparameter_tuning.py
```
- Results and best CLIs are saved in `reports/results/` and `models/final/gast/`.

### 5. Ablation Study

**Run GAST ablation experiments (full, no_spectral, no_spatial, concat) for all datasets:**
```bash
cd scripts
chmod +x run_gast_ablation.sh
./run_gast_ablation.sh
```
This script will automatically train and test all ablation variants for each dataset using the optimal hyperparameters, saving results in `models/gast_ablation/`.

### 6. Summarize Results

**Aggregate metrics across runs:**
```bash
python summarize_metrics.py
```
- Outputs CSV, TXT, and LaTeX tables in `reports/results/`.

### 7. Visualization

A rich set of visualization utilities is provided for dataset exploration and result analysis:

- **Class Distribution:**  
  Plot class histograms and tables for any dataset using functions in `src/utils/visualization.py` and label mappings from `src/data/dataset_info.py`.

- **Ground Truth & RGB:**  
  Visualize ground truth maps, RGB composites, and spectral signatures for any pixel.

- **Prediction vs. Ground Truth:**  
  Use `plot_gt_vs_pred_side_by_side` in `src/utils/visualization.py` to compare model predictions with ground truth, with consistent color mapping and legends.

- **3D Cube Visualization:**  
  Render 3D hyperspectral cubes for qualitative inspection.

- **Example usage:**  
  See the CLI examples and function calls at the bottom of `src/data/dataset_info.py` for quick dataset summaries and visualizations.

- **Customizable:**  
  All visualization functions accept `save_path` arguments for automated figure saving.

- **t-SNE for feature visualization:**  
  ```bash
  python run_all_tsne.py
  ```
  Plots are saved in `reports/figures/`.

For more, see the docstrings in `src/utils/visualization.py` and `src/data/dataset_info.py`.

---

## Key Scripts

- **main.py**: Main training/testing logic, supports all model and data options.
- **multirun.py**: Automates repeated training/testing across datasets and seeds.
- **summarize_metrics.py**: Aggregates and formats results for publication (CSV, TXT, LaTeX).
- **run_all_tsne.py**: Batch t-SNE visualization for all datasets.
- **src/training/gast_hyperparameter_tuning.py**: Two-stage (coarse + fine) Optuna-based hyperparameter search and CLI export.
- **src/training/generate_optuna_reports.py**: Generates visualizations and summary reports from Optuna study databases or pickles.
- **scripts/run_parameter_sweep.sh**: Shell script for systematic parameter sweeps.
- **scripts/run_gast_ablation.sh**: Shell script for running ablation studies (full, no_spectral, no_spatial, concat) across all datasets.
- **src/utils/visualization.py**: Rich visualization utilities for HSI data, predictions, and metrics.
- **src/data/dataset_info.py**: Dataset label mappings, summary utilities, and CLI examples for quick data exploration.

For more details, see the docstrings and CLI examples in each script.

---

## Citation

If you use GAST in your research, please cite:

```
@article{gast_paper,
  title={GAST: A Graph-Augmented Spectral–Spatial Transformer with Adaptive Gated Fusion for Small-Sample Hyperspectral Image Classification},
  author={Author et al.},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- Built on PyTorch, PyTorch Geometric, Optuna, and scikit-learn.
- Inspired by recent advances in spectral-spatial deep learning for remote sensing.

---

For questions or contributions, please open an issue or pull request!

---
