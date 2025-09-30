# MDN Model: Training and Evaluation

This directory contains scripts to train and evaluate a Mixture Density Network (MDN) for modeling collision data, as well as a Direct Simulation Monte Carlo (DSMC) method with a surrogate neural network for the collision term.

## Prerequisites

- Required packages: See `environment.yml` for dependencies (TensorFlow, NumPy, Matplotlib, SciPy, etc.).
- Dataset: `collision_dataset.txt` should be present in the `MDN/data/` directory (or specify a custom path with `--data`).

---

## 1. Training the MDN Model

Use `train.py` to train the MDN model on the dataset.

### Example Usage

```bash
python train.py --epochs 1000 --nr_gaussians 20 --nr_neurons 8 --activation_function relu --show-loss --save-model --name mymodel --data collision_dataset.txt
```

### Arguments

- `--epochs`: Number of training epochs (default: 1000)
- `--patience`: Early stopping patience (default: 10)
- `--nr_gaussians`: Number of Gaussians in the MDN (default: 20)
- `--nr_neurons`: Number of neurons in hidden layers (default: 8)
- `--activation_function`: Activation function for hidden layers (default: relu)
- `--show-loss`: Show loss plot after training
- `--save-model`: Save the trained model weights
- `--test-size`: Proportion of test data (default: 0.3)
- `--include-b`: Include impact parameter in input features
- `--verbose`: Verbosity level during training (default: 1)
- `--save-best-only`: Save only the best model weights (based on validation loss)
- `--name`: Name for the model and config files (default: mdn)
- `--data`: Path to the dataset file (default: collision_dataset.txt)

Model weights and config are saved to `MDN/models/<name>.h5` and `MDN/models/<name>_config.json`.

---

## 2. Evaluating the MDN Model

Use `eval.py` to evaluate the trained model and generate plots/statistics.

### Example Usage

```bash
python eval.py --eps-scatter --E-scatter --plot-density --marginals --correlation --procrustes --pdf --evaluate --name mymodel --data collision_dataset.txt --save-figures
```

### Arguments

- `--eps-scatter`: Show epsilon scatter plots
- `--E-scatter`: Show energy scatter plots
- `--correlation`: Print correlation coefficients
- `--procrustes`: Print Procrustes disparity
- `--plot-density`: Plot density estimation
- `--marginals`: Plot marginal distributions and perform statistical tests
- `--evaluate`: Evaluate model on test set (prints NLL)
- `--pdf`: Plot PDF estimation
- `--name`: Name of the model to load (default: mdn)
- `--data`: Path to the dataset file (default: collision_dataset.txt)
- `--save-figures`: Save figures to `MDN/figures/` instead of displaying them interactively

You can combine multiple flags to generate different analyses and plots.

---

## 3. Output

- Plots and statistics are displayed interactively or saved to the `figures/` directory if `--save-figures` is used.
- Model evaluation metrics (e.g., NLL) are printed to the console.

---

## 4. DSMC-MDN Simulation

Direct Simulation Monte Carlo (DSMC) method with a surrogate neural network for the collision term.

### Usage

Run the DSMC simulation using Python:

```powershell
python dsmc.py
```

### Arguments

You can customize the DSMC simulation by providing command-line arguments:

- `--mdn_model <path>`: Path to the trained MDN model.
- `--n_particles <int>`: Number of particles (default: 5000).
- `--n_steps <int>`: Number of steps (default: 1000).

Simulation results and figures will be saved in the `logs/` directory.

---

## Notes

- Ensure the model weights and config (`<name>.h5` and `<name>_config.json`) exist in `MDN/models/` before running `eval.py`.
