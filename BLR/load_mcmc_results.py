#!/usr/bin/env python3
"""
Load MCMC results from pickle files
"""

import pickle
import numpy as np
import os
import glob

def load_mcmc_results_from_pickle(results_path='results/', filename_pattern=None):
    """
    Load MCMC results from pickle files
    
    Parameters:
    -----------
    results_path : str
        Path to the results directory
    filename_pattern : str, optional
        Specific filename pattern to search for (e.g., 'mcmc_results_*.pkl')
    
    Returns:
    --------
    dict : Dictionary containing MCMC results
    """
    
    if filename_pattern is None:
        # Default pattern for MCMC results
        pattern = os.path.join(results_path, 'mcmc_results_*.pkl')
    else:
        pattern = os.path.join(results_path, filename_pattern)
    
    files = glob.glob(pattern)
    
    if not files:
        print(f"No MCMC result files found matching pattern: {pattern}")
        return None
    
    print(f"Found {len(files)} MCMC result file(s):")
    for file in files:
        print(f"  {os.path.basename(file)}")
    
    # Load the first file (or you can modify to load multiple files)
    mcmc_file = files[0]
    print(f"\nLoading MCMC results from: {mcmc_file}")
    
    try:
        with open(mcmc_file, 'rb') as f:
            mcmc_results = pickle.load(f)
        
        print("Successfully loaded MCMC results!")
        
        # Print available keys
        print("\nAvailable keys in MCMC results:")
        for key in mcmc_results.keys():
            if isinstance(mcmc_results[key], np.ndarray):
                print(f"  {key}: shape {mcmc_results[key].shape}, dtype {mcmc_results[key].dtype}")
            else:
                print(f"  {key}: {type(mcmc_results[key])}")
        
        return mcmc_results
        
    except Exception as e:
        print(f"Error loading MCMC results: {e}")
        return None

def extract_mcmc_components(mcmc_results):
    """
    Extract specific components from MCMC results
    
    Parameters:
    -----------
    mcmc_results : dict
        MCMC results dictionary
    
    Returns:
    --------
    tuple : (true_mu, true_A, mcmc_samples)
    """
    
    if mcmc_results is None:
        return None, None, None
    
    # Extract true parameters
    true_mu = mcmc_results.get('true_mu', None)
    true_A = mcmc_results.get('true_A', None)
    mcmc_samples = mcmc_results.get('mcmc_samples', None)
    
    # Print information about extracted components
    print("\nExtracted MCMC components:")
    if true_mu is not None:
        print(f"  true_mu: shape {true_mu.shape}, norm {np.linalg.norm(true_mu):.6f}")
    else:
        print("  true_mu: Not found")
    
    if true_A is not None:
        print(f"  true_A: shape {true_A.shape}, condition number {np.linalg.cond(true_A):.2e}")
    else:
        print("  true_A: Not found")
    
    if mcmc_samples is not None:
        print(f"  mcmc_samples: shape {mcmc_samples.shape}")
        print(f"  mcmc_samples: mean norm {np.linalg.norm(np.mean(mcmc_samples, axis=0)):.6f}")
        print(f"  mcmc_samples: std norm {np.linalg.norm(np.std(mcmc_samples, axis=0)):.6f}")
    else:
        print("  mcmc_samples: Not found")
    
    return true_mu, true_A, mcmc_samples

def analyze_mcmc_samples(mcmc_samples):
    """
    Analyze MCMC samples for convergence and quality
    
    Parameters:
    -----------
    mcmc_samples : np.ndarray
        MCMC samples array
    """
    
    if mcmc_samples is None:
        print("No MCMC samples to analyze")
        return
    
    print("\nMCMC Samples Analysis:")
    print(f"  Number of samples: {mcmc_samples.shape[0]}")
    print(f"  Number of parameters: {mcmc_samples.shape[1]}")
    
    # Basic statistics
    mean_samples = np.mean(mcmc_samples, axis=0)
    std_samples = np.std(mcmc_samples, axis=0)
    
    print(f"  Sample mean: min={np.min(mean_samples):.6f}, max={np.max(mean_samples):.6f}")
    print(f"  Sample std: min={np.min(std_samples):.6f}, max={np.max(std_samples):.6f}")
    
    # Check for convergence (Geweke diagnostic approximation)
    if mcmc_samples.shape[0] > 100:
        first_10 = np.mean(mcmc_samples[:100], axis=0)
        last_10 = np.mean(mcmc_samples[-100:], axis=0)
        convergence_diff = np.abs(first_10 - last_10) / (std_samples + 1e-8)
        print(f"  Convergence check (first vs last 100 samples): max diff={np.max(convergence_diff):.6f}")
    
    # Parameter-wise analysis
    print("\nParameter-wise analysis:")
    for i in range(min(5, mcmc_samples.shape[1])):  # Show first 5 parameters
        param_samples = mcmc_samples[:, i]
        print(f"  Param {i}: mean={np.mean(param_samples):.6f}, std={np.std(param_samples):.6f}")

def main():
    """Main function to load and analyze MCMC results"""
    
    # Load MCMC results
    mcmc_results = load_mcmc_results_from_pickle()
    
    if mcmc_results is None:
        print("Failed to load MCMC results")
        return
    
    # Extract components
    true_mu, true_A, mcmc_samples = extract_mcmc_components(mcmc_results)
    
    # Analyze MCMC samples
    analyze_mcmc_samples(mcmc_samples)
    
    # Return the components for further use
    return true_mu, true_A, mcmc_samples

# Example usage in Jupyter notebook:
if __name__ == "__main__":
    true_mu, true_A, mcmc_samples = main()
    
    # You can now use these variables in your analysis
    print("\nVariables available for use:")
    print("  true_mu: MCMC-estimated true mean")
    print("  true_A: MCMC-estimated true precision matrix")
    print("  mcmc_samples: MCMC samples array")
else:
    # For use in Jupyter notebook
    print("MCMC loading functions available:")
    print("  load_mcmc_results_from_pickle()")
    print("  extract_mcmc_components(mcmc_results)")
    print("  analyze_mcmc_samples(mcmc_samples)")
    print("  main()") 