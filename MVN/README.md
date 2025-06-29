# Multivariate Normal (MVN) Experiment

This folder contains the implementation of SVGD convergence analysis for Multivariate Normal (Gaussian) models.

## Overview

This experiment tests SVGD convergence on a Multivariate Normal distribution, which serves as a fundamental benchmark for variational inference methods. The MVN model provides a well-understood target distribution that allows for precise analysis of SVGD's convergence properties.

## Files

- `model.py`: Implementation of Multivariate Normal model
- `SVGD.py`: SVGD algorithm implementation
- `running_experiments_decay.py`: Main experiment script
- `plot.ipynb`: Jupyter notebook for visualizing results
- `README.md`: This file

## Model Details

The Multivariate Normal model includes:

- **Distribution**: Multivariate Gaussian with known mean and covariance
- **Parameters**: Mean vector μ and precision matrix A
- **Challenge**: Convergence to the true posterior distribution
- **Evaluation**: KL divergence and KSD convergence analysis

## Usage

1. **Run experiments**:
   ```bash
   cd MVN
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
- **True parameters**: μ = [1, 1], A = [[1, 0], [0, 1]]

## Expected Results

The experiment will generate:
- KSD convergence plots
- KL divergence analysis
- Eigenvalue analysis of kernel matrix
- Convergence rate comparison across particle counts
- Step-size sensitivity analysis

## Dependencies

- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- tqdm

## Notes

- The experiment provides baseline convergence analysis
- Results are saved in the `results/` directory
- The model allows for exact computation of KL divergence
- Various step-size decay strategies are compared
- The 2D setting enables easy visualization of particle evolution 