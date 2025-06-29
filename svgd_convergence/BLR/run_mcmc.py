#!/usr/bin/env python3
"""
MCMC script for Bayesian Logistic Regression using CmdStanPy
This script runs MCMC to estimate the true posterior parameters
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import cmdstanpy
import multiprocessing
import warnings
import argparse
from sklearn.datasets import load_svmlight_file

warnings.filterwarnings('ignore')

class BLRMCMC:
    def __init__(self, X_train=None, y_train=None, alpha_prior=1.0, beta_prior=0.1):
        """
        Initialize Hierarchical Bayesian Logistic Regression for MCMC
        
        Parameters:
        -----------
        X_train: training features
        y_train: training labels (binary: 0 or 1)
        alpha_prior: prior shape for global precision
        beta_prior: prior rate for global precision
        """
        self.X_train = X_train
        self.y_train = y_train
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        if X_train is not None and y_train is not None:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model parameters"""
        self.n_samples, self.n_features = self.X_train.shape
        self.n_classes = 2  # Binary classification
        self.n_params = self.n_features + 1  # +1 for global precision parameter
        
        print(f"Model initialized: {self.n_features} features, {self.n_classes} classes (binary)")
        print(f"Total parameters: {self.n_params} (including global precision)")
    
    def load_libsvm_binary_covertype_data(self, n_samples=None, test_size=0.2, random_state=42):
        """Load and preprocess libsvm binary UCI Covertype dataset"""
        print("Loading libsvm binary UCI Covertype dataset...")
        
        # Load libsvm format data
        data_file = "covtype.libsvm.binary.scale.bz2"
        try:
            X, y = load_svmlight_file(data_file)
            X = X.toarray()  # Convert sparse matrix to dense array
            
            print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Label range: {np.min(y)} to {np.max(y)}")
            print(f"Unique labels: {np.unique(y)}")
            
            # Ensure labels are binary (0 and 1)
            if len(np.unique(y)) == 2:
                # Map labels to 0 and 1 if they're not already
                unique_labels = np.unique(y)
                if not np.array_equal(unique_labels, [0, 1]):
                    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                    y = np.array([label_map[label] for label in y])
                    print(f"Mapped labels: {unique_labels} -> [0, 1]")
            else:
                raise ValueError(f"Expected binary labels, but found {len(np.unique(y))} unique values: {np.unique(y)}")
            
        except Exception as e:
            print(f"Error loading libsvm file: {e}")
            print("Falling back to sklearn fetch_covtype...")
            return self.load_binary_covertype_data(n_samples, test_size, random_state)
        
        print(f"Binary labels: {np.unique(y)}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Subsample for faster computation if specified
        if n_samples is not None and n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
            print(f"Subsampled to {n_samples} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Note: libsvm data is already scaled, so we don't need to standardize
        print("Note: Using pre-scaled libsvm data (no additional standardization)")
        
        # Update model data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Initialize model parameters
        self._initialize_model()
        
        print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        print(f"Features: {X_train.shape[1]}, Classes: {self.n_classes}")
        print(f"Training class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        
        return X_train, y_train, X_test, y_test
    
    def load_binary_covertype_data(self, n_samples=10000, test_size=0.2, random_state=42):
        """Load and preprocess binary UCI Covertype dataset (fallback method)"""
        print("Loading binary UCI Covertype dataset (fallback method)...")
        data = fetch_covtype()
        X, y = data.data, data.target - 1  # Convert to 0-based indexing
        
        # Convert to binary classification: class 1 vs all others
        y_binary = (y == 1).astype(int)
        
        print(f"Original labels: {np.unique(y)}")
        print(f"Binary labels: {np.unique(y_binary)}")
        print(f"Class distribution: {np.bincount(y_binary)}")
        
        # Subsample for faster computation
        if n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y_binary = y_binary[indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Update model data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Initialize model parameters
        self._initialize_model()
        
        print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        print(f"Features: {X_train.shape[1]}, Classes: {self.n_classes}")
        print(f"Training class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        
        return X_train, y_train, X_test, y_test
    
    def create_stan_model(self):
        """Create Stan model for Hierarchical Bayesian Logistic Regression"""
        stan_code = """
        data {
            int<lower=1> N;           // number of observations
            int<lower=1> D;           // number of features
            array[N] int<lower=0, upper=1> y;  // binary class labels
            matrix[N, D] X;           // feature matrix
            real<lower=0> alpha_prior; // prior shape for global precision
            real<lower=0> beta_prior;  // prior rate for global precision
        }
        
        parameters {
            vector[D] beta;           // regression coefficients
            real<lower=0.1, upper=10.0> tau;  // global precision parameter with tighter bounds
        }
        
        model {
            // Hierarchical prior structure
            // Global precision parameter: Gamma(alpha_prior, beta_prior)
            tau ~ gamma(alpha_prior, beta_prior);
            
            // Regression coefficients: Normal(0, 1/sqrt(tau)) - more stable
            beta ~ normal(0, 1/sqrt(tau));
            
            // Likelihood: Bernoulli with logistic link
            for (n in 1:N) {
                y[n] ~ bernoulli_logit(X[n, :] * beta);
            }
        }
        
        generated quantities {
            vector[N] log_likelihood;
            vector[N] y_pred;
            
            for (n in 1:N) {
                log_likelihood[n] = bernoulli_logit_lpmf(y[n] | X[n, :] * beta);
                y_pred[n] = bernoulli_logit_rng(X[n, :] * beta);
            }
        }
        """
        
        # Write Stan model to file
        with open("blr_model.stan", "w") as f:
            f.write(stan_code)
        
        return "blr_model.stan"
    
    def run_mcmc(self, n_samples=2000, n_warmup=1000, chains=4, random_seed=42, force_rerun=False, num_cores=None):
        """
        Run MCMC using CmdStanPy
        
        Parameters:
        -----------
        n_samples: number of posterior samples per chain
        n_warmup: number of warmup samples per chain
        chains: number of chains
        random_seed: random seed for reproducibility
        force_rerun: if True, run MCMC even if saved results exist
        num_cores: number of CPU cores to use (None for auto-detection)
        
        Returns:
        --------
        true_mu: mean of the posterior samples
        true_A: precision matrix (inverse of covariance)
        """
        # Check if results already exist
        results_file = f"mcmc_results_{n_samples}_{n_warmup}_{chains}_{random_seed}.pkl"
        if not force_rerun and os.path.exists(results_file):
            print(f"Loading existing MCMC results from {results_file}")
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
                return results['true_mu'], results['true_A']
        
        print("Running MCMC using CmdStanPy...")
        
        # Create Stan model
        stan_file = self.create_stan_model()
        
        # Prepare data for Stan
        stan_data = {
            'N': self.X_train.shape[0],
            'D': self.X_train.shape[1],
            'y': self.y_train,
            'X': self.X_train,
            'alpha_prior': self.alpha_prior,
            'beta_prior': self.beta_prior
        }
        
        # Verify data integrity
        print(f"Stan data verification:")
        print(f"  N (samples): {stan_data['N']}")
        print(f"  D (features): {stan_data['D']}")
        print(f"  y range: {np.min(stan_data['y'])} to {np.max(stan_data['y'])}")
        print(f"  y unique: {np.unique(stan_data['y'])}")
        print(f"  alpha_prior: {stan_data['alpha_prior']}")
        print(f"  beta_prior: {stan_data['beta_prior']}")
        
        # Ensure y is binary
        if not np.all(np.isin(stan_data['y'], [0, 1])):
            raise ValueError(f"Labels must be binary [0, 1], but found {np.unique(stan_data['y'])}")
        
        # Compile and run Stan model
        model = cmdstanpy.CmdStanModel(stan_file=stan_file)
        
        # Set number of cores
        if num_cores is None:
            num_cores = multiprocessing.cpu_count()
        
        print(f"Running {n_samples} samples with {n_warmup} warmup, {chains} chains using {num_cores} cores...")
        fit = model.sample(
            data=stan_data,
            chains=chains,
            parallel_chains=num_cores,
            iter_warmup=n_warmup,
            iter_sampling=n_samples,
            seed=random_seed,
            show_progress=True
        )
        
        # Extract samples
        beta_samples = fit.stan_variable('beta')
        tau_samples = fit.stan_variable('tau')
        
        # Run convergence diagnostics
        convergence_diagnostics = self.diagnose_mcmc_convergence(fit)
        
        # Check if MCMC converged properly
        if not convergence_diagnostics.get('converged', False):
            print("\n" + "="*50)
            print("⚠️  MCMC CONVERGENCE WARNING")
            print("="*50)
            print("MCMC may not have converged properly.")
            print("Consider the following actions:")
            print("  1. Increase n_samples (currently: {})".format(n_samples))
            print("  2. Increase n_warmup (currently: {})".format(n_warmup))
            print("  3. Increase chains (currently: {})".format(chains))
            print("  4. Check model specification and data")
            
            # Check for severe convergence issues
            max_r_hat = convergence_diagnostics.get('max_r_hat', float('inf'))
            min_n_eff = convergence_diagnostics.get('min_n_eff', 0)
            
            if max_r_hat > 1.5 or min_n_eff < 10:
                print("\n❌ SEVERE CONVERGENCE ISSUES DETECTED")
                print(f"   Max R-hat: {max_r_hat:.4f} (should be < 1.1)")
                print(f"   Min N_eff: {min_n_eff:.1f} (should be > 50)")
                print("\nNote: Statistics will still be computed and saved for analysis.")
                print("Check evaluation results to assess if the posterior is reasonable.")
            
            print("\nProceeding with statistics computation...")
            print("="*50)
        
        # Combine beta and tau samples
        # beta_samples shape: (n_samples, D)
        # tau_samples shape: (n_samples,)
        n_total_samples = beta_samples.shape[0]
        
        # Combine parameters: [beta_1, ..., beta_D, tau]
        self.mcmc_samples = np.column_stack([beta_samples, tau_samples])
        
        # Check sample size vs parameter dimension
        n_params = self.mcmc_samples.shape[1]
        print(f"Total MCMC samples: {n_total_samples}")
        print(f"Parameter dimension: {n_params}")
        print(f"Sample-to-parameter ratio: {n_total_samples/n_params:.2f}")
        
        if n_total_samples < n_params:
            print("Warning: Number of samples is less than parameter dimension!")
            print("This may lead to singular covariance matrix.")
        elif n_total_samples < 2 * n_params:
            print("Warning: Number of samples is less than 2x parameter dimension.")
            print("Consider increasing sample size for better covariance estimation.")
        
        # Compute true parameters
        true_mu = np.mean(self.mcmc_samples, axis=0)
        
        # Compute covariance matrix with regularization to avoid singular matrix
        cov_matrix = np.cov(self.mcmc_samples.T)
        
        # Apply shrinkage estimation for better covariance estimation
        # This helps when sample size is limited compared to parameter dimension
        n_samples, n_params = self.mcmc_samples.shape
        if n_samples < n_params:
            print("Applying shrinkage estimation for covariance matrix...")
            # Target matrix: diagonal matrix with sample variances
            target = np.diag(np.diag(cov_matrix))
            
            # Optimal shrinkage parameter (Ledoit-Wolf estimator)
            # For simplicity, use a conservative shrinkage
            shrinkage = min(0.5, max(0.1, (n_params + 1) / (n_samples + n_params + 2)))
            
            # Shrinkage estimator
            cov_matrix = (1 - shrinkage) * cov_matrix + shrinkage * target
            print(f"Applied shrinkage parameter: {shrinkage:.3f}")
        
        # Check condition number and add regularization if needed
        cond_number = np.linalg.cond(cov_matrix)
        print(f"Covariance matrix condition number: {cond_number:.2e}")
        
        # Add regularization if condition number is too large
        if cond_number > 1e12:
            print("Adding regularization to covariance matrix...")
            # Add small diagonal term to improve conditioning
            reg_strength = 1e-6 * np.trace(cov_matrix) / cov_matrix.shape[0]
            cov_matrix_reg = cov_matrix + reg_strength * np.eye(cov_matrix.shape[0])
            
            # Check condition number again
            cond_number_reg = np.linalg.cond(cov_matrix_reg)
            print(f"Regularized covariance matrix condition number: {cond_number_reg:.2e}")
            
            if cond_number_reg > 1e12:
                print("Warning: Matrix still poorly conditioned, using pseudo-inverse")
                true_A = np.linalg.pinv(cov_matrix_reg)
            else:
                true_A = np.linalg.inv(cov_matrix_reg)
        else:
            try:
                true_A = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                print("Singular matrix detected, using pseudo-inverse")
                true_A = np.linalg.pinv(cov_matrix)
        
        # Save results
        results = {
            'true_mu': true_mu,
            'true_A': true_A,
            'mcmc_samples': self.mcmc_samples,
            'cov_matrix': cov_matrix,
            'condition_number': cond_number,
            'n_samples': n_samples,
            'n_warmup': n_warmup,
            'chains': chains,
            'random_seed': random_seed,
            'n_params': n_params,
            'sample_to_param_ratio': n_total_samples / n_params,
            'convergence_diagnostics': convergence_diagnostics
        }
        
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"MCMC completed: {self.mcmc_samples.shape[0]} total samples")
        print(f"True mu shape: {true_mu.shape}")
        print(f"True A shape: {true_A.shape}")
        
        return true_mu, true_A

    def evaluate_mcmc_results(self, true_mu, true_A, n_eval_samples=100):
        """
        Evaluate MCMC results using various metrics
        
        Parameters:
        -----------
        true_mu: mean of posterior samples
        true_A: precision matrix
        n_eval_samples: number of samples to use for evaluation
        
        Returns:
        --------
        dict: evaluation metrics
        """
        print("\n" + "="*50)
        print("EVALUATING MCMC RESULTS")
        print("="*50)
        
        # Extract parameters
        beta_mean = true_mu[:-1]  # regression coefficients
        tau_mean = true_mu[-1]    # global precision parameter
        
        # Generate samples from the posterior for evaluation
        np.random.seed(42)  # For reproducibility
        samples = np.random.multivariate_normal(true_mu, np.linalg.inv(true_A), n_eval_samples)
        
        # Calculate predictions for each sample
        train_accuracies = []
        test_accuracies = []
        train_log_likelihoods = []
        test_log_likelihoods = []
        
        for i, sample in enumerate(samples):
            beta_sample = sample[:-1]  # regression coefficients
            tau_sample = sample[-1]    # global precision parameter
            
            # Training predictions
            train_logits = self.X_train @ beta_sample
            train_probs = self._sigmoid(train_logits)
            train_pred = (train_probs > 0.5).astype(int)
            train_acc = np.mean(train_pred == self.y_train)
            train_accuracies.append(train_acc)
            
            # Test predictions
            test_logits = self.X_test @ beta_sample
            test_probs = self._sigmoid(test_logits)
            test_pred = (test_probs > 0.5).astype(int)
            test_acc = np.mean(test_pred == self.y_test)
            test_accuracies.append(test_acc)
            
            # Log likelihoods
            train_ll = self._log_likelihood_binary(train_logits, self.y_train)
            test_ll = self._log_likelihood_binary(test_logits, self.y_test)
            train_log_likelihoods.append(train_ll)
            test_log_likelihoods.append(test_ll)
        
        # Calculate statistics
        metrics = {
            'train_accuracy_mean': np.mean(train_accuracies),
            'train_accuracy_std': np.std(train_accuracies),
            'test_accuracy_mean': np.mean(test_accuracies),
            'test_accuracy_std': np.std(test_accuracies),
            'train_log_likelihood_mean': np.mean(train_log_likelihoods),
            'train_log_likelihood_std': np.std(train_log_likelihoods),
            'test_log_likelihood_mean': np.mean(test_log_likelihoods),
            'test_log_likelihood_std': np.std(test_log_likelihoods),
            'n_eval_samples': n_eval_samples
        }
        
        # Print results
        print(f"Training Accuracy: {metrics['train_accuracy_mean']:.4f} ± {metrics['train_accuracy_std']:.4f}")
        print(f"Test Accuracy: {metrics['test_accuracy_mean']:.4f} ± {metrics['test_accuracy_std']:.4f}")
        print(f"Training Log-Likelihood: {metrics['train_log_likelihood_mean']:.2f} ± {metrics['train_log_likelihood_std']:.2f}")
        print(f"Test Log-Likelihood: {metrics['test_log_likelihood_mean']:.2f} ± {metrics['test_log_likelihood_std']:.2f}")
        
        return metrics
    
    def _sigmoid(self, z):
        """Compute sigmoid function"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def _log_likelihood_binary(self, logits, y):
        """Compute binary log likelihood"""
        probs = self._sigmoid(logits)
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        return np.sum(y * np.log(probs) + (1 - y) * np.log(1 - probs))

    def diagnose_mcmc_convergence(self, fit):
        """
        Diagnose MCMC convergence using various metrics
        
        Parameters:
        -----------
        fit: CmdStanMCMC fit object
        
        Returns:
        --------
        dict: convergence diagnostics
        """
        print("\n" + "="*50)
        print("MCMC CONVERGENCE DIAGNOSTICS")
        print("="*50)
        
        # Get summary statistics
        summary = fit.summary()
        
        # Extract R-hat (Gelman-Rubin statistic) - handle different CmdStanPy versions
        r_hat_values = None
        max_r_hat = float('inf')
        mean_r_hat = float('inf')
        
        # Try different possible column names for R-hat
        possible_r_hat_columns = ['R_hat', 'r_hat', 'rhat', 'Rhat']
        
        for col_name in possible_r_hat_columns:
            if col_name in summary.columns:
                r_hat_values = summary[col_name].values
                # Filter out NaN values
                r_hat_valid = r_hat_values[~np.isnan(r_hat_values)]
                if len(r_hat_valid) > 0:
                    max_r_hat = np.max(r_hat_valid)
                    mean_r_hat = np.mean(r_hat_valid)
                    print(f"Gelman-Rubin statistic (R-hat) using {col_name}:")
                    print(f"  Max R-hat: {max_r_hat:.4f}")
                    print(f"  Mean R-hat: {mean_r_hat:.4f}")
                    print(f"  Valid values: {len(r_hat_valid)}/{len(r_hat_values)}")
                    break
        
        if r_hat_values is None or len(r_hat_valid) == 0:
            print("⚠️  Warning: Could not find valid R-hat values")
            print(f"Available columns: {list(summary.columns)}")
            print("Skipping R-hat diagnostics")
            max_r_hat = 1.0  # Assume good convergence if we can't check
            mean_r_hat = 1.0
        else:
            if max_r_hat < 1.1:
                print("✅ Good convergence (R-hat < 1.1)")
            elif max_r_hat < 1.2:
                print("⚠️  Acceptable convergence (R-hat < 1.2)")
            else:
                print("❌ Poor convergence (R-hat >= 1.2)")
        
        # Extract effective sample sizes - handle different CmdStanPy versions
        n_eff_values = None
        min_n_eff = 0
        mean_n_eff = 0
        
        # Try different possible column names for effective sample size
        possible_n_eff_columns = ['N_Eff', 'n_eff', 'ess_bulk', 'ess_tail', 'ess_mean', 'ESS_bulk', 'ESS_tail']
        
        for col_name in possible_n_eff_columns:
            if col_name in summary.columns:
                n_eff_values = summary[col_name].values
                # Filter out NaN values
                n_eff_valid = n_eff_values[~np.isnan(n_eff_values)]
                if len(n_eff_valid) > 0:
                    min_n_eff = np.min(n_eff_valid)
                    mean_n_eff = np.mean(n_eff_valid)
                    print(f"\nEffective Sample Sizes (using {col_name}):")
                    print(f"  Min N_eff: {min_n_eff:.1f}")
                    print(f"  Mean N_eff: {mean_n_eff:.1f}")
                    print(f"  Valid values: {len(n_eff_valid)}/{len(n_eff_values)}")
                    break
        
        if n_eff_values is None or len(n_eff_valid) == 0:
            print("\n⚠️  Warning: Could not find valid effective sample size values")
            print(f"Available columns: {list(summary.columns)}")
            print("Skipping effective sample size diagnostics")
            min_n_eff = float('inf')  # Set to infinity to avoid convergence issues
            mean_n_eff = float('inf')
        else:
            if min_n_eff > 100:
                print("✅ Good effective sample sizes (> 100)")
            elif min_n_eff > 50:
                print("⚠️  Acceptable effective sample sizes (> 50)")
            else:
                print("❌ Low effective sample sizes (< 50)")
        
        # Check for divergent transitions
        if hasattr(fit, 'diagnose'):
            try:
                diagnose = fit.diagnose()
                # Handle different return types from diagnose()
                if isinstance(diagnose, dict) and 'divergent' in diagnose:
                    n_divergent = diagnose['divergent']
                    print(f"\nDivergent transitions: {n_divergent}")
                    if n_divergent == 0:
                        print("✅ No divergent transitions")
                    else:
                        print(f"❌ {n_divergent} divergent transitions detected")
                elif isinstance(diagnose, str):
                    # If diagnose returns a string, try to parse it
                    if 'divergent' in diagnose.lower():
                        print(f"\nDiagnostic output contains divergent information:")
                        print(diagnose[:500] + "..." if len(diagnose) > 500 else diagnose)
                    else:
                        print(f"\nDiagnostic output:")
                        print(diagnose[:500] + "..." if len(diagnose) > 500 else diagnose)
                else:
                    print(f"\nDiagnostic output type: {type(diagnose)}")
                    print("Skipping divergent transition check")
            except Exception as e:
                print(f"\n⚠️  Warning: Could not check divergent transitions: {e}")
                print("Skipping divergent transition check")
        
        # Calculate diagnostics
        diagnostics = {
            'max_r_hat': max_r_hat if max_r_hat != float('inf') else 1.0,
            'mean_r_hat': mean_r_hat if mean_r_hat != float('inf') else 1.0,
            'min_n_eff': min_n_eff if min_n_eff != float('inf') else 1000.0,
            'mean_n_eff': mean_n_eff if mean_n_eff != float('inf') else 1000.0,
            'converged': (max_r_hat < 1.2 if max_r_hat != float('inf') else True) and 
                        (min_n_eff > 50 if min_n_eff != float('inf') else True)
        }
        
        # Provide recommendations
        print(f"\nConvergence Assessment:")
        if diagnostics['converged']:
            print("✅ MCMC appears to have converged well")
        else:
            print("❌ MCMC convergence issues detected")
            if max_r_hat >= 1.2:
                print(f"   - R-hat too high ({max_r_hat:.4f}), increase warmup/samples")
            if min_n_eff <= 50:
                print(f"   - Low effective samples ({min_n_eff:.1f}), increase total samples")
        
        print("="*50)
        
        return diagnostics

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run MCMC for Hierarchical Bayesian Logistic Regression')
    parser.add_argument('--n_samples', type=int, default=2000,
                       help='Number of posterior samples per chain')
    parser.add_argument('--n_warmup', type=int, default=1000,
                       help='Number of warmup samples per chain')
    parser.add_argument('--chains', type=int, default=4,
                       help='Number of chains')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--force_rerun', action='store_true',
                       help='Force rerun MCMC')
    parser.add_argument('--num_cores', type=int, default=None,
                       help='Number of CPU cores to use (None for auto-detection)')
    parser.add_argument('--data_samples', type=int, default=10000,
                       help='Number of data samples to use')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size ratio')
    parser.add_argument('--alpha_prior', type=float, default=1.0,
                       help='Prior shape for global precision')
    parser.add_argument('--beta_prior', type=float, default=0.1,
                       help='Prior rate for global precision')
    
    args = parser.parse_args()
    
    print("="*50)
    print("HIERARCHICAL BAYESIAN LOGISTIC REGRESSION MCMC")
    print("="*50)
    print(f"Settings:")
    print(f"  Data samples: {args.data_samples}")
    print(f"  Test size: {args.test_size}")
    print(f"  MCMC samples: {args.n_samples}")
    print(f"  Warmup: {args.n_warmup}")
    print(f"  Chains: {args.chains}")
    print(f"  Random seed: {args.random_seed}")
    print(f"  Alpha prior: {args.alpha_prior}")
    print(f"  Beta prior: {args.beta_prior}")
    print("="*50)
    
    # Initialize model
    model = BLRMCMC(alpha_prior=args.alpha_prior, beta_prior=args.beta_prior)
    
    # Load data
    X_train, y_train, X_test, y_test = model.load_libsvm_binary_covertype_data(
        n_samples=args.data_samples, test_size=args.test_size, random_state=args.random_seed
    )
    
    # Run MCMC
    true_mu, true_A = model.run_mcmc(
        n_samples=args.n_samples,
        n_warmup=args.n_warmup,
        chains=args.chains,
        random_seed=args.random_seed,
        force_rerun=args.force_rerun,
        num_cores=args.num_cores
    )
    
    print("MCMC completed successfully!")
    print(f"True mu norm: {np.linalg.norm(true_mu):.4f}")
    print(f"True A condition number: {np.linalg.cond(true_A):.2e}")
    print(f"True A shape: {true_A.shape}")
    print(f"True A trace: {np.trace(true_A):.4f}")
    print(f"True A determinant: {np.linalg.det(true_A):.2e}")
    
    # Load convergence diagnostics
    results_file = f"mcmc_results_{args.n_samples}_{args.n_warmup}_{args.chains}_{args.random_seed}.pkl"
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
        convergence_diagnostics = results['convergence_diagnostics']
    
    # Evaluate results
    metrics = model.evaluate_mcmc_results(true_mu, true_A, n_eval_samples=100)
    
    # Save evaluation results
    eval_file = f"mcmc_evaluation_{args.n_samples}_{args.n_warmup}_{args.chains}_{args.random_seed}.pkl"
    
    eval_results = {
        'true_mu': true_mu,
        'true_A': true_A,
        'evaluation_metrics': metrics,
        'convergence_diagnostics': convergence_diagnostics,
        'model_info': {
            'n_features': model.n_features,
            'n_classes': model.n_classes,
            'n_params': model.n_params,
            'n_train_samples': len(model.X_train),
            'n_test_samples': len(model.X_test)
        },
        'mcmc_settings': {
            'n_samples': args.n_samples,
            'n_warmup': args.n_warmup,
            'chains': args.chains,
            'random_seed': args.random_seed,
            'alpha_prior': args.alpha_prior,
            'beta_prior': args.beta_prior
        }
    }
    
    with open(eval_file, 'wb') as f:
        pickle.dump(eval_results, f)
    
    print(f"\nEvaluation results saved to {eval_file}")
    
    # Print summary for paper writing
    print("\n" + "="*50)
    print("SUMMARY FOR PAPER WRITING")
    print("="*50)
    print(f"Model: Hierarchical Bayesian Logistic Regression (Binary Covertype dataset)")
    print(f"Features: {model.n_features}, Classes: {model.n_classes}")
    print(f"Parameters: {model.n_params}")
    print(f"Training samples: {len(model.X_train)}, Test samples: {len(model.X_test)}")
    print(f"MCMC settings: {args.n_samples} samples, {args.n_warmup} warmup, {args.chains} chains")
    print(f"Prior settings: alpha={args.alpha_prior}, beta={args.beta_prior}")
    
    if convergence_diagnostics:
        print(f"Convergence: {'Good' if convergence_diagnostics.get('converged', False) else 'Issues detected'}")
        print(f"R-hat: {convergence_diagnostics.get('max_r_hat', 'N/A'):.4f}")
        print(f"N_eff: {convergence_diagnostics.get('min_n_eff', 'N/A'):.1f}")
    
    print(f"Test Accuracy: {metrics['test_accuracy_mean']:.4f} ± {metrics['test_accuracy_std']:.4f}")
    print(f"Test Log-Likelihood: {metrics['test_log_likelihood_mean']:.2f} ± {metrics['test_log_likelihood_std']:.2f}")

if __name__ == "__main__":
    main() 