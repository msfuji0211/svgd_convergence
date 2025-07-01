#!/usr/bin/env python3
"""
Compare different KL divergence estimation methods
This script compares Gaussian assumption, KDE, and k-NN methods for KL divergence estimation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

def load_experiment_results(results_path, pattern="svgd_results_*.pkl"):
    """Load all experiment results from the results directory"""
    import glob
    
    results = []
    for filename in glob.glob(os.path.join(results_path, pattern)):
        with open(filename, 'rb') as f:
            result = pickle.load(f)
            results.append(result)
    
    return results

def compare_kl_methods(results):
    """Compare different KL divergence estimation methods"""
    
    print("="*60)
    print("KL DIVERGENCE METHOD COMPARISON")
    print("="*60)
    
    # Extract KL values for comparison
    comparison_data = []
    
    for result in results:
        n_particles = result['n_particles']
        decay_beta = result['decay_beta']
        
        # Gaussian assumption KL (final value)
        kl_gaussian = None
        if result['kl_list'] is not None and len(result['kl_list']) > 0:
            kl_gaussian = result['kl_list'][-1]
            if isinstance(kl_gaussian, np.ndarray):
                kl_gaussian = kl_gaussian.item()
        
        # KDE KL
        kl_kde = result.get('kl_kde', None)
        
        comparison_data.append({
            'n_particles': n_particles,
            'decay_beta': decay_beta,
            'kl_gaussian': kl_gaussian,
            'kl_kde': kl_kde
        })
    
    # Create comparison table
    print("\nKL Divergence Comparison Table:")
    print("-" * 60)
    print(f"{'N_particles':<12} {'Decay_β':<10} {'Gaussian':<12} {'KDE':<12}")
    print("-" * 60)
    
    for data in comparison_data:
        gaussian_str = f"{data['kl_gaussian']:.6f}" if data['kl_gaussian'] is not None else "N/A"
        kde_str = f"{data['kl_kde']:.6f}" if data['kl_kde'] is not None else "N/A"
        
        print(f"{data['n_particles']:<12} {data['decay_beta']:<10} {gaussian_str:<12} {kde_str:<12}")
    
    return comparison_data

def analyze_distribution_assumptions(results):
    """Analyze whether the Gaussian assumption is valid"""
    
    print("\n" + "="*60)
    print("DISTRIBUTION ASSUMPTION ANALYSIS")
    print("="*60)
    
    # Use the first result to get MCMC samples
    if not results:
        print("No results available for analysis")
        return
    
    first_result = results[0]
    mcmc_samples = first_result.get('mcmc_samples', None)
    
    if mcmc_samples is None:
        print("MCMC samples not available for distribution analysis")
        return
    
    print(f"MCMC samples shape: {mcmc_samples.shape}")
    
    # Test for multivariate normality
    from scipy.stats import multivariate_normal, chi2
    
    # Compute sample mean and covariance
    sample_mean = np.mean(mcmc_samples, axis=0)
    sample_cov = np.cov(mcmc_samples.T)
    
    # Mahalanobis distances
    mahal_distances = []
    for sample in mcmc_samples:
        diff = sample - sample_mean
        mahal_dist = np.sqrt(diff @ np.linalg.solve(sample_cov, diff))
        mahal_distances.append(mahal_dist)
    
    mahal_distances = np.array(mahal_distances)
    
    # Chi-square test for multivariate normality
    # Under normality, squared Mahalanobis distances follow chi-square distribution
    squared_mahal = mahal_distances ** 2
    df = mcmc_samples.shape[1]  # degrees of freedom = number of dimensions
    
    # Kolmogorov-Smirnov test
    from scipy.stats import kstest
    ks_stat, ks_pvalue = kstest(squared_mahal, 'chi2', args=(df,))
    
    print(f"\nMultivariate Normality Test:")
    print(f"  Degrees of freedom: {df}")
    print(f"  KS statistic: {ks_stat:.6f}")
    print(f"  KS p-value: {ks_pvalue:.6f}")
    
    if ks_pvalue < 0.05:
        print("  ❌ REJECT: MCMC samples do not follow multivariate normal distribution")
        print("  → KDE or k-NN methods may be more appropriate")
    else:
        print("  ✅ ACCEPT: MCMC samples appear to follow multivariate normal distribution")
        print("  → Gaussian assumption may be reasonable")
    
    # Visual analysis
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Mahalanobis distances histogram vs chi-square
    axes[0, 0].hist(squared_mahal, bins=30, density=True, alpha=0.7, label='MCMC samples')
    x = np.linspace(0, np.max(squared_mahal), 100)
    chi2_pdf = chi2.pdf(x, df)
    axes[0, 0].plot(x, chi2_pdf, 'r-', linewidth=2, label=f'Chi-square({df})')
    axes[0, 0].set_xlabel('Squared Mahalanobis Distance')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Multivariate Normality Test')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Q-Q plot
    from scipy.stats import probplot
    probplot(squared_mahal, dist=chi2, sparams=(df,), plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot vs Chi-square')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. PCA visualization (first 2 components)
    if mcmc_samples.shape[1] > 2:
        pca = PCA(n_components=2)
        mcmc_pca = pca.fit_transform(mcmc_samples)
        axes[1, 0].scatter(mcmc_pca[:, 0], mcmc_pca[:, 1], alpha=0.6, s=10)
        axes[1, 0].set_xlabel('First Principal Component')
        axes[1, 0].set_ylabel('Second Principal Component')
        axes[1, 0].set_title('MCMC Samples (PCA)')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].scatter(mcmc_samples[:, 0], mcmc_samples[:, 1], alpha=0.6, s=10)
        axes[1, 0].set_xlabel('Parameter 1')
        axes[1, 0].set_ylabel('Parameter 2')
        axes[1, 0].set_title('MCMC Samples')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Correlation matrix heatmap
    corr_matrix = np.corrcoef(mcmc_samples.T)
    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_title('Parameter Correlation Matrix')
    axes[1, 1].set_xticks(range(len(corr_matrix)))
    axes[1, 1].set_yticks(range(len(corr_matrix)))
    axes[1, 1].set_xticklabels([f'P{i+1}' for i in range(len(corr_matrix))])
    axes[1, 1].set_yticklabels([f'P{i+1}' for i in range(len(corr_matrix))])
    plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return ks_pvalue

def plot_kl_comparison(comparison_data):
    """Plot comparison of different KL divergence methods"""
    
    print("\n" + "="*60)
    print("KL DIVERGENCE COMPARISON PLOTS")
    print("="*60)
    
    # Filter out None values
    valid_data = [d for d in comparison_data if d['kl_gaussian'] is not None or 
                  d['kl_kde'] is not None]
    
    if not valid_data:
        print("No valid KL divergence data for plotting")
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. KL values by method
    methods = ['kl_gaussian', 'kl_kde']
    method_names = ['Gaussian', 'KDE']
    
    kl_values = {method: [] for method in methods}
    for data in valid_data:
        for method in methods:
            if data[method] is not None:
                kl_values[method].append(data[method])
    
    # Box plot
    kl_data = [kl_values[method] for method in methods if kl_values[method]]
    method_labels = [method_names[i] for i, method in enumerate(methods) if kl_values[method]]
    
    if kl_data:
        axes[0, 0].boxplot(kl_data, labels=method_labels)
        axes[0, 0].set_ylabel('KL Divergence')
        axes[0, 0].set_title('KL Divergence by Method')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. KL vs number of particles
    for method, name in zip(methods, method_names):
        n_particles = []
        kl_vals = []
        for data in valid_data:
            if data[method] is not None:
                n_particles.append(data['n_particles'])
                kl_vals.append(data[method])
        
        if n_particles:
            axes[0, 1].scatter(n_particles, kl_vals, label=name, alpha=0.7)
    
    axes[0, 1].set_xlabel('Number of Particles')
    axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].set_title('KL Divergence vs Number of Particles')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. KL vs decay beta
    for method, name in zip(methods, method_names):
        decay_betas = []
        kl_vals = []
        for data in valid_data:
            if data[method] is not None:
                decay_betas.append(data['decay_beta'])
                kl_vals.append(data[method])
        
        if decay_betas:
            axes[1, 0].scatter(decay_betas, kl_vals, label=name, alpha=0.7)
    
    axes[1, 0].set_xlabel('Decay Beta')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence vs Decay Beta')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Method correlation
    valid_methods = []
    for method, name in zip(methods, method_names):
        if any(data[method] is not None for data in valid_data):
            valid_methods.append((method, name))
    
    if len(valid_methods) >= 2:
        # Create correlation matrix
        method_data = {}
        for method, name in valid_methods:
            method_data[name] = []
            for data in valid_data:
                if data[method] is not None:
                    method_data[name].append(data[method])
        
        # Find common indices
        min_length = min(len(vals) for vals in method_data.values())
        if min_length > 0:
            corr_matrix = np.zeros((len(valid_methods), len(valid_methods)))
            method_names_list = [name for _, name in valid_methods]
            
            for i, (method1, name1) in enumerate(valid_methods):
                for j, (method2, name2) in enumerate(valid_methods):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        # Use only common data points
                        vals1 = method_data[name1][:min_length]
                        vals2 = method_data[name2][:min_length]
                        if len(vals1) == len(vals2) and len(vals1) > 1:
                            corr = np.corrcoef(vals1, vals2)[0, 1]
                            corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0
            
            im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 1].set_xticks(range(len(method_names_list)))
            axes[1, 1].set_yticks(range(len(method_names_list)))
            axes[1, 1].set_xticklabels(method_names_list)
            axes[1, 1].set_yticklabels(method_names_list)
            axes[1, 1].set_title('Method Correlation Matrix')
            
            # Add correlation values as text
            for i in range(len(method_names_list)):
                for j in range(len(method_names_list)):
                    text = axes[1, 1].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                          ha="center", va="center", color="black")
            
            plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('kl_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare KL divergence estimation methods')
    parser.add_argument('--results_path', type=str, default='./results',
                       help='Path to results directory')
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_path}")
    results = load_experiment_results(args.results_path)
    
    if not results:
        print("No results found. Please run experiments first.")
        return
    
    print(f"Loaded {len(results)} experiment results")
    
    # Compare KL methods
    comparison_data = compare_kl_methods(results)
    
    # Analyze distribution assumptions
    p_value = analyze_distribution_assumptions(results)
    
    # Plot comparisons
    plot_kl_comparison(comparison_data)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*60)
    
    if p_value is not None:
        if p_value < 0.05:
            print("❌ The MCMC samples do not follow a multivariate normal distribution.")
            print("   → KDE or k-NN methods are recommended for KL divergence estimation.")
            print("   → The Gaussian assumption may lead to biased results.")
        else:
            print("✅ The MCMC samples appear to follow a multivariate normal distribution.")
            print("   → All three methods (Gaussian, KDE, k-NN) should give similar results.")
            print("   → The Gaussian assumption is reasonable for this case.")
    
    print("\nRecommendations:")
    print("1. Always check distribution assumptions before choosing KL estimation method")
    print("2. Use KDE or k-NN methods when the true distribution is unknown or non-Gaussian")
    print("3. Compare results across methods to assess robustness")
    print("4. Consider computational cost: Gaussian < KDE < k-NN")
    print("5. For high-dimensional problems, k-NN may be more stable than KDE")

if __name__ == "__main__":
    main() 