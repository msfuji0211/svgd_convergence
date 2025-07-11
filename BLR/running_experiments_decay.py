#!/usr/bin/env python3
"""
SVGD experiments for Bayesian Logistic Regression with decay
This script runs SVGD experiments using pre-computed MCMC results
"""

import os
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.metrics.pairwise import linear_kernel as linear
from tqdm.auto import tqdm
from SVGD import SVGD
from model import BLR
from run_mcmc import BLRMCMC
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')

def runexp_BLR_decay(model, true_mu, true_A, mcmc_samples, x0, init_sig, n_iter, decay_factor=0.1, beta=1, mode='rbf', step_size=1e-2, cons=1., seed=10, adagrad=False, lr_decay=True):
    np.random.seed(seed)
    
    # Use MCMC-estimated true parameters
    theta, mse_list, kl_list, ksd_list, fisher_list, eig_list, kl_kde_list, kl_mmd_list, likelihood_list = SVGD().update(
        x0, model.dlnprob, n_iter=n_iter, stepsize=step_size, cons=cons, 
        decay_factor=decay_factor, beta=beta, mode=mode, adagrad=adagrad, 
        lr_decay=lr_decay, verbose=True, true_mu=true_mu, true_A=true_A, mcmc_samples=mcmc_samples
    )
    
    # Get final values for backward compatibility
    kl_kde = kl_kde_list[-1] if not np.isnan(kl_kde_list[-1]) else None
    kl_mmd = kl_mmd_list[-1] if not np.isnan(kl_mmd_list[-1]) else None
    
    # Calculate NLL (Negative Log Likelihood) from likelihood_list
    nll_list = -likelihood_list  # Convert log likelihood to negative log likelihood
    
    return theta, ksd_list, eig_list, kl_kde_list, kl_mmd_list, nll_list, kl_kde, kl_mmd

def load_mcmc_results(n_samples=2000, n_warmup=1000, chains=4, random_seed=42, alpha_prior=1.0, beta_prior=0.1):
    """Load MCMC results or run MCMC if not available"""
    results_file = f"mcmc_results_{n_samples}_{n_warmup}_{chains}_{random_seed}.pkl"
    
    if os.path.exists(results_file):
        print(f"Loading existing MCMC results from {results_file}")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
            return results['true_mu'], results['true_A'], results.get('mcmc_samples', None)
    else:
        print("MCMC results not found. Running MCMC...")
        mcmc_model = BLRMCMC(alpha_prior=alpha_prior, beta_prior=beta_prior)
        X_train, y_train, X_test, y_test = mcmc_model.load_libsvm_binary_covertype_data(
            n_samples=None, test_size=0.2, random_state=42
        )
        true_mu, true_A = mcmc_model.run_mcmc(
            n_samples=n_samples,
            n_warmup=n_warmup,
            chains=chains,
            random_seed=random_seed,
            force_rerun=False
        )
        # Load the results again to get mcmc_samples
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
            return results['true_mu'], results['true_A'], results.get('mcmc_samples', None)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run SVGD decay experiments for Hierarchical BLR')
    parser.add_argument('--n_iterations', type=int, default=10000,
                       help='Number of SVGD iterations')
    parser.add_argument('--force_rerun', action='store_true',
                       help='Force rerun MCMC')
    parser.add_argument('--alpha_prior', type=float, default=1.0,
                       help='Prior shape for global precision')
    parser.add_argument('--beta_prior', type=float, default=0.01,
                       help='Prior rate for global precision')
    args = parser.parse_args()
    
    # path for results
    path = os.getcwd()
    if not os.path.exists(path+'/results/'):
        os.mkdir('results')
    results_path = path + '/results/'
    
    print("Initializing Hierarchical BLR model...")
    
    # Initialize model
    model = BLR(alpha_prior=args.alpha_prior, beta_prior=args.beta_prior)
    
    # Load data
    X_train, y_train, X_test, y_test = model.load_libsvm_binary_covertype_data(
        n_samples=None, test_size=0.2, random_state=42
    )
    
    # Load MCMC-estimated true parameters and samples
    true_mu, true_A, mcmc_samples = load_mcmc_results(
        n_samples=2000, n_warmup=1000, chains=4, random_seed=42,
        alpha_prior=args.alpha_prior, beta_prior=args.beta_prior
    )
    
    print(f"Loaded true parameters:")
    print(f"  true_mu shape: {true_mu.shape}")
    print(f"  true_A shape: {true_A.shape}")
    print(f"  true_mu norm: {np.linalg.norm(true_mu):.4f}")
    print(f"  true_A condition number: {np.linalg.cond(true_A):.2e}")
    if mcmc_samples is not None:
        print(f"  MCMC samples shape: {mcmc_samples.shape}")
    else:
        print("  Warning: MCMC samples not available for KDE/k-NN KL calculation")
    
    # Initialize parameters for particles
    n_params = model.n_params
    init_mu = np.zeros(n_params)  # Initialize weights to zero
    init_sig = np.eye(n_params) * 0.1  # Small variance for initialization
    
    n_iter = args.n_iterations
    stepsize = 1e-2
    mode = 'rbf'  # or linear
    beta = [0., 0.5, 0.67, 1.]
    decay_factor = 1.0

    print(f"Running decay experiments with {n_iter} iterations...")
    
    # Run experiments with different numbers of particles
    #n_particles_list = [5, 10, 20, 50]
    n_particles_list = [20]

    for n_particles in n_particles_list:
        print(f"\nRunning experiment with {n_particles} particles...")
        
        # Initialize particles
        x0 = np.random.multivariate_normal(mean=init_mu, cov=init_sig, size=n_particles)
        # Set reasonable initial values for log_tau (log precision parameter)
        x0[:, -1] = np.log(0.1) + 0.1 * np.random.randn(n_particles)  # log_tau ~ log(0.1) + noise
        
        print(f"  Particle initialization:")
        print(f"    x0 shape: {x0.shape}")
        print(f"    init_mu shape: {init_mu.shape}")
        print(f"    init_sig shape: {init_sig.shape}")
        print(f"    n_params: {n_params}")
        
        # Run SVGD for different decay parameters
        for decay_beta in beta:
            print(f"  Running with decay beta = {decay_beta}")
            
            # Run SVGD
            theta, ksd_list, eig_list, kl_kde_list, kl_mmd_list, nll_list, kl_kde, kl_mmd = runexp_BLR_decay(
                model, true_mu, true_A, mcmc_samples, x0, init_sig, n_iter=n_iter, step_size=stepsize, 
                decay_factor=decay_factor, beta=decay_beta, mode=mode
            )
            
            # Save results
            results = {
                'theta': theta,
                'ksd_list': ksd_list,
                'eig_list': eig_list,
                'kl_kde_list': kl_kde_list,  # Add iteration-by-iteration KDE-KL
                'kl_mmd_list': kl_mmd_list,  # Add iteration-by-iteration MMD
                'nll_list': nll_list,  # Add iteration-by-iteration NLL
                'kl_kde': kl_kde,
                'kl_mmd': kl_mmd,
                'n_particles': n_particles,
                'n_iterations': n_iter,
                'decay_beta': decay_beta,
                'stepsize': stepsize,
                'mode': mode,
                'true_mu': true_mu,
                'true_A': true_A,
                'mcmc_samples': mcmc_samples
            }
            
            # Save to file
            filename = f"svgd_results_n{n_particles}_beta{decay_beta}_iter{n_iter}.pkl"
            with open(results_path + filename, 'wb') as f:
                pickle.dump(results, f)
            
            print(f"    Results saved to {filename}")
            
            # Print final metrics
            if nll_list is not None and len(nll_list) > 0:
                final_nll = nll_list[-1, 0] if nll_list.ndim == 2 else nll_list[-1]
                print(f"    Final NLL: {final_nll:.6f}")
            
            if kl_kde is not None:
                kl_kde_val = kl_kde.item() if hasattr(kl_kde, 'item') else kl_kde
                print(f"    Final KL divergence (KDE): {kl_kde_val:.6f}")
            
            if kl_mmd is not None:
                kl_mmd_val = kl_mmd.item() if hasattr(kl_mmd, 'item') else kl_mmd
                print(f"    Final MMD divergence: {kl_mmd_val:.6f}")
            
            if ksd_list is not None and len(ksd_list) > 0:
                final_ksd = ksd_list[-1] if ksd_list[-1] is not None else float('inf')
                if isinstance(final_ksd, np.ndarray):
                    final_ksd = final_ksd.item()
                print(f"    Final KSD: {final_ksd:.6f}")
    
    print("\nAll experiments completed!")
    print(f"Results saved in: {results_path}")

if __name__ == "__main__":
    main() 