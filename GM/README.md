# Gaussian Mixture (GM) Experiment

This folder contains the implementation of SVGD convergence analysis for Gaussian Mixture models.

## Overview

This experiment tests SVGD convergence on a Gaussian Mixture distribution, which is a common benchmark for variational inference methods. The Gaussian Mixture model provides a non-trivial target distribution that tests SVGD's ability to capture multi-modal posteriors.

## Files

- `model.py`: Implementation of Gaussian Mixture model
- `SVGD.py`: SVGD algorithm implementation
- `running_experiments_decay.py`: Main experiment script
- `plot.ipynb`: Jupyter notebook for visualizing results
- `README.md`: This file

## Model Details

The Gaussian Mixture model includes:

- **Distribution**: Mixture of multiple Gaussian components
- **Parameters**: Means and covariance matrices for each component
- **Challenge**: Multi-modal posterior distribution
- **Evaluation**: Convergence to the true mixture distribution

## Usage

1. **Run experiments**:
   ```bash
   cd GM
   python running_experiments_decay.py
   ```

2. **Visualize results**:
   ```bash
   jupyter notebook plot.ipynb
   ```

## Experiment Parameters

- **Number of particles**: [5, 10, 100, 1000]
- **Iterations**: 100,000
- **Learning rate**: 1e-2
- **Kernel**: RBF kernel
- **Step-size decay**: Various decay factors tested

## Expected Results

The experiment will generate:
- KSD convergence plots
- KL divergence analysis
- Eigenvalue analysis of kernel matrix
- Convergence rate comparison across particle counts

## Dependencies

- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- tqdm

## Cache Management

Cache files (`__pycache__/`, `.pyc`, etc.) are automatically excluded via `.gitignore`.

## Notes

- The experiment focuses on multi-modal convergence
- Results are saved in the `results/` directory
- The model tests SVGD's ability to capture complex posterior structures
- Various step-size decay strategies are compared 