#!/usr/bin/env python3
"""
Test script for SVGD likelihood tracking functionality
"""

import numpy as np
import matplotlib.pyplot as plt
from model import BLR
from SVGD import SVGD

def test_svgd_likelihood_tracking():
    """Test SVGD with likelihood tracking"""
    
    print("=== BLR SVGD Likelihood Tracking Test ===")
    
    # Initialize BLR model
    blr = BLR(alpha_prior=1.0, beta_prior=0.01)
    
    # Load data (small subset for quick testing)
    print("\nLoading data...")
    X_train, y_train, X_test, y_test = blr.load_libsvm_binary_covertype_data(
        n_samples=1000, test_size=0.2, random_state=42
    )
    
    # Initialize SVGD
    svgd = SVGD()
    
    # Set parameters
    n_particles = 50
    n_iter = 5000
    stepsize = 0.01
    
    # Initialize particles
    n_params = blr.n_params
    theta0 = np.random.randn(n_particles, n_params)
    
    # Set reasonable initial values for log_tau (log precision parameter)
    # log_tau can be negative, but we want reasonable starting values
    theta0[:, -1] = np.log(0.1) + 0.1 * np.random.randn(n_particles)  # log_tau ~ log(0.1) + noise
    
    print(f"\nStarting SVGD with {n_particles} particles for {n_iter} iterations")
    print(f"Parameters: {n_params} (including global precision)")
    print(f"Training samples: {X_train.shape[0]}")
    
    # Test a single likelihood calculation for debugging
    test_theta = theta0[0]
    test_log_likelihood = blr.log_likelihood(test_theta)
    test_log_likelihood_per_sample = test_log_likelihood / X_train.shape[0]
    test_log_tau = test_theta[-1]
    test_tau = np.exp(test_log_tau)
    print(f"Debug - Single particle log likelihood: {test_log_likelihood:.2f}")
    print(f"Debug - Per-sample log likelihood: {test_log_likelihood_per_sample:.6f}")
    print(f"Debug - log_tau: {test_log_tau:.6f}, tau: {test_tau:.6f}")
    
    # Run SVGD with likelihood tracking
    theta_final, mse_list, kl_list, ksd_list, fisher_list, eig_list, kl_kde_list, kl_mmd_list, likelihood_list = svgd.update(
        x0=theta0,
        lnprob=blr.dlnprob,  # Use gradient function
        n_iter=n_iter,
        stepsize=stepsize,
        adagrad=False,  # Use vanilla SGD instead of AdaGrad
        lr_decay=True,
        verbose=True,
        debug=True
    )
    
    print(f"\nSVGD completed!")
    print(f"Final average log likelihood: {likelihood_list[-1, 0]:.6f}")
    print(f"Likelihood improvement: {likelihood_list[-1, 0] - likelihood_list[0, 0]:.6f}")
    
    # Save likelihood history
    svgd.save_likelihood_history(likelihood_list, 'blr_likelihood_history.npy')
    
    # Plot likelihood convergence
    svgd.plot_likelihood_convergence(likelihood_list, 'blr_likelihood_convergence.png')
    
    # Additional analysis
    print(f"\nLikelihood Statistics:")
    print(f"Initial: {likelihood_list[0, 0]:.6f}")
    print(f"Final: {likelihood_list[-1, 0]:.6f}")
    print(f"Min: {np.min(likelihood_list):.6f}")
    print(f"Max: {np.max(likelihood_list):.6f}")
    print(f"Mean: {np.mean(likelihood_list):.6f}")
    print(f"Std: {np.std(likelihood_list):.6f}")
    
    # Compare with MCMC reference
    print(f"\nComparison with MCMC reference (-0.52):")
    print(f"Difference from MCMC: {likelihood_list[-1, 0] - (-0.52):.6f}")
    
    # Analyze final particle likelihoods
    print(f"\nFinal Particle Analysis:")
    final_likelihoods = []
    for i in range(theta_final.shape[0]):
        theta_i = theta_final[i]
        if hasattr(blr, 'log_likelihood'):
            log_prob_i = blr.log_likelihood(theta_i)
            if hasattr(blr, 'n_samples'):
                log_prob_i = log_prob_i / blr.n_samples
            elif hasattr(blr, 'X_train'):
                # Fallback: use X_train shape
                log_prob_i = log_prob_i / blr.X_train.shape[0]
            final_likelihoods.append(log_prob_i)
    
    if final_likelihoods:
        final_likelihoods = np.array(final_likelihoods)
        print(f"Final particle likelihoods:")
        print(f"  Mean: {np.mean(final_likelihoods):.6f}")
        print(f"  Std: {np.std(final_likelihoods):.6f}")
        print(f"  Min: {np.min(final_likelihoods):.6f}")
        print(f"  Max: {np.max(final_likelihoods):.6f}")
        print(f"  Range: {np.max(final_likelihoods) - np.min(final_likelihoods):.6f}")
        
        # Show distribution
        print(f"  Percentiles: 25%={np.percentile(final_likelihoods, 25):.6f}, 50%={np.percentile(final_likelihoods, 50):.6f}, 75%={np.percentile(final_likelihoods, 75):.6f}")
    
    # Test prediction accuracy
    if hasattr(blr, 'accuracy'):
        final_theta = np.mean(theta_final, axis=0)
        train_acc = blr.accuracy(final_theta, X_train, y_train)
        test_acc = blr.accuracy(final_theta, X_test, y_test)
        print(f"\nPrediction Accuracy:")
        print(f"Training: {train_acc:.4f}")
        print(f"Test: {test_acc:.4f}")
    
    return theta_final, likelihood_list

if __name__ == "__main__":
    theta_final, likelihood_list = test_svgd_likelihood_tracking() 