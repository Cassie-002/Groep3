# MDN Model: Training and Evaluation

This directory contains scripts to train and evaluate a Mixture Density Network (MDN) for modeling collision data. The main scripts are `train.py` (for training the model) and `eval.py` (for evaluating and visualizing results).

## Prerequisites
- Required packages: see `environment.yml` for dependencies (TensorFlow, NumPy, Matplotlib, SciPy, etc.)
- Dataset: `collision_dataset.txt` should be present in the `MDN/data/` directory.

## Training the Model

Use `train.py` to train the MDN model on the dataset.

### Example usage
```bash
python train.py --epochs 1000 --nr_gaussians 20 --nr_neurons 8 --activation_function relu --show-loss --save-model
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

Model weights are saved to `MDN/models/mdn_weights.h5`.

## Evaluating the Model

Use `eval.py` to evaluate the trained model and generate plots/statistics.

### Example usage
```bash
python eval.py --eps-scatter --E-scatter --plot-density --marginals --correlation --procrustes --pdf --evaluate
```

### Arguments
- `--eps-scatter`: Show epsilon scatter plots
- `--E-scatter`: Show energy scatter plots
- `--include-b`: Include impact parameter in input features
- `--correlation`: Print correlation coefficients
- `--procrustes`: Print Procrustes disparity
- `--plot-density`: Plot density estimation
- `--marginals`: Plot marginal distributions and perform statistical tests
- `--evaluate`: Evaluate model on test set (prints NLL)
- `--pdf`: Plot PDF estimation

You can combine multiple flags to generate different analyses and plots.

## Output
- Plots and statistics are displayed interactively.
- Model evaluation metrics (e.g., NLL) are printed to the console.

## Notes
- Ensure the model weights (`mdn_weights.h5`) exist before running `eval.py`.