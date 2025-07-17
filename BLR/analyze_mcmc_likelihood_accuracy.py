#!/usr/bin/env python3
"""
Analyze MCMC results for likelihood and prediction accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
from model import BLR
from load_mcmc_results import load_mcmc_results_from_pickle, extract_mcmc_components

def calculate_mcmc_likelihoods(mcmc_samples, blr_model):
    """
    Calculate likelihood for each MCMC sample
    
    Parameters:
    -----------
    mcmc_samples : np.ndarray
        MCMC samples array (n_samples, n_params)
    blr_model : BLR
        BLR model instance with data loaded
    
    Returns:
    --------
    np.ndarray : Likelihood values for each sample
    """
    
    if mcmc_samples is None or len(mcmc_samples) == 0:
        print("No MCMC samples provided")
        return None
    
    print(f"Calculating likelihoods for {len(mcmc_samples)} MCMC samples...")
    
    likelihoods = []
    for i, theta in enumerate(mcmc_samples):
        # Calculate log likelihood for this sample
        log_likelihood = blr_model.log_likelihood(theta)
        
        # Convert to per-sample log likelihood
        per_sample_log_likelihood = log_likelihood / blr_model.X_train.shape[0]
        
        likelihoods.append(per_sample_log_likelihood)
        
        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(mcmc_samples)} samples")
    
    likelihoods = np.array(likelihoods)
    print(f"Completed! Calculated likelihoods for {len(likelihoods)} samples")
    
    return likelihoods

def calculate_mcmc_predictions(mcmc_samples, blr_model, X_data, y_data):
    """
    Calculate predictions and accuracy for each MCMC sample
    
    Parameters:
    -----------
    mcmc_samples : np.ndarray
        MCMC samples array (n_samples, n_params)
    blr_model : BLR
        BLR model instance with data loaded
    X_data : np.ndarray
        Feature data for prediction
    y_data : np.ndarray
        True labels for accuracy calculation
    
    Returns:
    --------
    tuple : (predictions, accuracies)
    """
    
    if mcmc_samples is None or len(mcmc_samples) == 0:
        print("No MCMC samples provided")
        return None, None
    
    print(f"Calculating predictions for {len(mcmc_samples)} MCMC samples...")
    
    predictions = []
    accuracies = []
    
    for i, theta in enumerate(mcmc_samples):
        # Make prediction for this sample
        pred = blr_model.predict(theta, X_data)
        predictions.append(pred)
        
        # Calculate accuracy
        accuracy = np.mean(pred == y_data)
        accuracies.append(accuracy)
        
        # Progress indicator
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(mcmc_samples)} samples")
    
    predictions = np.array(predictions)
    accuracies = np.array(accuracies)
    
    print(f"Completed! Calculated predictions for {len(predictions)} samples")
    
    return predictions, accuracies

def analyze_mcmc_likelihoods(likelihoods):
    """
    Analyze MCMC likelihood values
    
    Parameters:
    -----------
    likelihoods : np.ndarray
        Likelihood values for MCMC samples
    """
    
    if likelihoods is None:
        print("No likelihoods to analyze")
        return
    
    print("\n=== MCMC Likelihood Analysis ===")
    print(f"Number of samples: {len(likelihoods)}")
    print(f"Mean log likelihood per sample: {np.mean(likelihoods):.6f}")
    print(f"Std log likelihood per sample: {np.std(likelihoods):.6f}")
    print(f"Min log likelihood per sample: {np.min(likelihoods):.6f}")
    print(f"Max log likelihood per sample: {np.max(likelihoods):.6f}")
    print(f"Median log likelihood per sample: {np.median(likelihoods):.6f}")
    
    # Percentiles
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        value = np.percentile(likelihoods, p)
        print(f"{p}th percentile: {value:.6f}")
    
    # Compare with MCMC reference (-0.52)
    mcmc_reference = -0.52
    diff_from_reference = np.mean(likelihoods) - mcmc_reference
    print(f"\nComparison with MCMC reference (-0.52):")
    print(f"Difference: {diff_from_reference:.6f}")
    print(f"Relative difference: {abs(diff_from_reference) / abs(mcmc_reference) * 100:.2f}%")

def analyze_mcmc_accuracies(accuracies, dataset_name="Dataset"):
    """
    Analyze MCMC prediction accuracies
    
    Parameters:
    -----------
    accuracies : np.ndarray
        Accuracy values for MCMC samples
    dataset_name : str
        Name of the dataset for display
    """
    
    if accuracies is None:
        print("No accuracies to analyze")
        return
    
    print(f"\n=== MCMC Accuracy Analysis ({dataset_name}) ===")
    print(f"Number of samples: {len(accuracies)}")
    print(f"Mean accuracy: {np.mean(accuracies):.6f}")
    print(f"Std accuracy: {np.std(accuracies):.6f}")
    print(f"Min accuracy: {np.min(accuracies):.6f}")
    print(f"Max accuracy: {np.max(accuracies):.6f}")
    print(f"Median accuracy: {np.median(accuracies):.6f}")
    
    # Percentiles
    percentiles = [5, 25, 50, 75, 95]
    for p in percentiles:
        value = np.percentile(accuracies, p)
        print(f"{p}th percentile: {value:.6f}")

def plot_mcmc_analysis(likelihoods, train_accuracies, test_accuracies):
    """
    Plot MCMC analysis results
    
    Parameters:
    -----------
    likelihoods : np.ndarray
        Likelihood values for MCMC samples
    train_accuracies : np.ndarray
        Training accuracies for MCMC samples
    test_accuracies : np.ndarray
        Test accuracies for MCMC samples
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot likelihood distribution
    axes[0, 0].hist(likelihoods, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(x=-0.52, color='red', linestyle='--', alpha=0.8, label='MCMC Reference (-0.52)')
    axes[0, 0].axvline(x=np.mean(likelihoods), color='green', linestyle='-', alpha=0.8, label=f'Mean ({np.mean(likelihoods):.4f})')
    axes[0, 0].set_xlabel('Log Likelihood per Sample')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('MCMC Likelihood Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot training accuracy distribution
    axes[0, 1].hist(train_accuracies, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(x=np.mean(train_accuracies), color='red', linestyle='-', alpha=0.8, label=f'Mean ({np.mean(train_accuracies):.4f})')
    axes[0, 1].set_xlabel('Training Accuracy')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('MCMC Training Accuracy Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot test accuracy distribution
    axes[1, 0].hist(test_accuracies, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].axvline(x=np.mean(test_accuracies), color='red', linestyle='-', alpha=0.8, label=f'Mean ({np.mean(test_accuracies):.4f})')
    axes[1, 0].set_xlabel('Test Accuracy')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('MCMC Test Accuracy Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot likelihood vs accuracy scatter
    axes[1, 1].scatter(likelihoods, test_accuracies, alpha=0.6, s=20)
    axes[1, 1].set_xlabel('Log Likelihood per Sample')
    axes[1, 1].set_ylabel('Test Accuracy')
    axes[1, 1].set_title('Likelihood vs Test Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(likelihoods, test_accuracies)[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                    transform=axes[1, 1].transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('mcmc_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to analyze MCMC results"""
    
    print("=== MCMC Likelihood and Accuracy Analysis ===")
    
    # Load MCMC results
    mcmc_results = load_mcmc_results_from_pickle()
    if mcmc_results is None:
        print("Failed to load MCMC results")
        return None, None, None, None, None
    
    # Extract components
    true_mu, true_A, mcmc_samples = extract_mcmc_components(mcmc_results)
    
    if mcmc_samples is None:
        print("No MCMC samples found")
        return None, None, None, None, None
    
    # Initialize BLR model and load data
    print("\nInitializing BLR model...")
    blr_model = BLR(alpha_prior=1.0, beta_prior=0.01)
    X_train, y_train, X_test, y_test = blr_model.load_libsvm_binary_covertype_data(
        n_samples=None, test_size=0.2, random_state=42
    )
    
    # Calculate likelihoods
    likelihoods = calculate_mcmc_likelihoods(mcmc_samples, blr_model)
    
    # Calculate predictions and accuracies
    train_predictions, train_accuracies = calculate_mcmc_predictions(
        mcmc_samples, blr_model, X_train, y_train
    )
    test_predictions, test_accuracies = calculate_mcmc_predictions(
        mcmc_samples, blr_model, X_test, y_test
    )
    
    # Analyze results
    analyze_mcmc_likelihoods(likelihoods)
    analyze_mcmc_accuracies(train_accuracies, "Training")
    analyze_mcmc_accuracies(test_accuracies, "Test")
    
    # Plot results
    plot_mcmc_analysis(likelihoods, train_accuracies, test_accuracies)
    
    # Summary
    print("\n=== Summary ===")
    print(f"MCMC Likelihood: {np.mean(likelihoods):.6f} ± {np.std(likelihoods):.6f}")
    print(f"MCMC Training Accuracy: {np.mean(train_accuracies):.6f} ± {np.std(train_accuracies):.6f}")
    print(f"MCMC Test Accuracy: {np.mean(test_accuracies):.6f} ± {np.std(test_accuracies):.6f}")
    
    return likelihoods, train_accuracies, test_accuracies, train_predictions, test_predictions

# For use in Jupyter notebook
if __name__ == "__main__":
    likelihoods, train_acc, test_acc, train_pred, test_pred = main()
else:
    print("MCMC analysis functions available:")
    print("  calculate_mcmc_likelihoods(mcmc_samples, blr_model)")
    print("  calculate_mcmc_predictions(mcmc_samples, blr_model, X_data, y_data)")
    print("  analyze_mcmc_likelihoods(likelihoods)")
    print("  analyze_mcmc_accuracies(accuracies, dataset_name)")
    print("  plot_mcmc_analysis(likelihoods, train_accuracies, test_accuracies)")
    print("  main()") 