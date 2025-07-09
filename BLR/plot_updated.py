#!/usr/bin/env python3
"""
Updated plotting script for BLR SVGD experiments with NLL instead of Gaussian KL
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob

def cumulative_mean(values):
    """Compute cumulative mean of values"""
    return np.cumsum(values) / np.arange(1, len(values) + 1)

def load_results(results_path='results/'):
    """Load all result files and organize them"""
    results = {}
    
    # Find all result files
    pattern = os.path.join(results_path, 'svgd_results_*.pkl')
    files = glob.glob(pattern)
    
    for file in files:
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                
            # Extract key information from filename
            filename = os.path.basename(file)
            parts = filename.replace('svgd_results_', '').replace('.pkl', '').split('_')
            
            n_particles = int(parts[0].replace('n', ''))
            beta = float(parts[1].replace('beta', ''))
            n_iter = int(parts[2].replace('iter', ''))
            
            key = f"n{n_particles}_beta{beta}"
            results[key] = data
            print(f"Loaded: {filename} -> {key}")
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return results

def plot_convergence(results):
    """Plot convergence for different numbers of particles"""
    plt.figure(figsize=(20, 12))

    # Colors for different particle counts
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#984ea3']
    particle_counts = [5, 10, 20, 50]

    # Plot NLL convergence
    plt.subplot(2, 3, 1)
    for i, n_particles in enumerate(particle_counts):
        key = f"n{n_particles}_beta1.0"
        if key in results:
            nll_list = results[key].get('nll_list', None)
            if nll_list is not None and len(nll_list) > 0:
                # Handle both 1D and 2D arrays
                if nll_list.ndim == 2:
                    nll_values = nll_list[:, 0].flatten()
                else:
                    nll_values = nll_list.flatten()
                plt.plot(nll_values, label=f'{n_particles} particles', 
                        linewidth=2, color=colors[i % len(colors)])

    plt.xscale('log')
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Negative Log Likelihood', fontsize=14)
    plt.title('NLL Convergence', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot KSD convergence
    plt.subplot(2, 3, 2)
    for i, n_particles in enumerate(particle_counts):
        key = f"n{n_particles}_beta1.0"
        if key in results:
            ksd_list = results[key]['ksd_list']
            if ksd_list is not None and len(ksd_list) > 0:
                ksd_values = ksd_list.flatten()
                plt.plot(ksd_values, label=f'{n_particles} particles', 
                        linewidth=2, color=colors[i % len(colors)])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Kernel Stein Discrepancy', fontsize=14)
    plt.title('KSD Convergence', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot KDE-based KL divergence convergence
    plt.subplot(2, 3, 3)
    for i, n_particles in enumerate(particle_counts):
        key = f"n{n_particles}_beta1.0"
        if key in results:
            kl_kde_list = results[key].get('kl_kde_list', None)
            if kl_kde_list is not None and len(kl_kde_list) > 0:
                valid_indices = ~np.isnan(kl_kde_list.flatten())
                if np.any(valid_indices):
                    kl_kde_values = kl_kde_list.flatten()[valid_indices]
                    iterations = np.arange(len(kl_kde_list))[valid_indices]
                    plt.plot(iterations, kl_kde_values, label=f'{n_particles} particles', 
                            linewidth=2, color=colors[i % len(colors)])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('KL Divergence (KDE)', fontsize=14)
    plt.title('KDE-based KL Divergence Convergence', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot MMD divergence convergence
    plt.subplot(2, 3, 4)
    for i, n_particles in enumerate(particle_counts):
        key = f"n{n_particles}_beta1.0"
        if key in results:
            kl_mmd_list = results[key].get('kl_mmd_list', None)
            if kl_mmd_list is not None and len(kl_mmd_list) > 0:
                valid_indices = ~np.isnan(kl_mmd_list.flatten())
                if np.any(valid_indices):
                    kl_mmd_values = kl_mmd_list.flatten()[valid_indices]
                    iterations = np.arange(len(kl_mmd_list))[valid_indices]
                    plt.plot(iterations, kl_mmd_values, label=f'{n_particles} particles', 
                            linewidth=2, color=colors[i % len(colors)])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('MMD Divergence', fontsize=14)
    plt.title('MMD Divergence Convergence', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot cumulative mean KSD
    plt.subplot(2, 3, 5)
    for i, n_particles in enumerate(particle_counts):
        key = f"n{n_particles}_beta1.0"
        if key in results:
            ksd_list = results[key]['ksd_list']
            if ksd_list is not None and len(ksd_list) > 0:
                ksd_values = ksd_list.flatten()
                cum_ksd = cumulative_mean(ksd_values)
                plt.plot(cum_ksd, label=f'{n_particles} particles', 
                        linewidth=2, color=colors[i % len(colors)])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=14)
    plt.ylabel('Cumulative Mean KSD', fontsize=14)
    plt.title('Cumulative Mean KSD', fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot final values comparison
    plt.subplot(2, 3, 6)
    final_nll = []
    final_kl_kde = []
    final_mmd = []

    for n_particles in particle_counts:
        key = f"n{n_particles}_beta1.0"
        if key in results:
            # Final NLL
            nll_list = results[key].get('nll_list', None)
            if nll_list is not None and len(nll_list) > 0:
                if nll_list.ndim == 2:
                    final_nll.append(nll_list[-1, 0].item())
                else:
                    final_nll.append(nll_list[-1].item())
            else:
                final_nll.append(np.nan)
            
            # Final KDE KL
            kl_kde = results[key].get('kl_kde', None)
            if kl_kde is not None:
                # Convert to scalar if it's a numpy array
                if hasattr(kl_kde, 'item'):
                    final_kl_kde.append(kl_kde.item())
                else:
                    final_kl_kde.append(float(kl_kde))
            else:
                final_kl_kde.append(np.nan)

            # Final MMD
            kl_mmd = results[key].get('kl_mmd', None)
            if kl_mmd is not None:
                # Convert to scalar if it's a numpy array
                if hasattr(kl_mmd, 'item'):
                    final_mmd.append(kl_mmd.item())
                else:
                    final_mmd.append(float(kl_mmd))
            else:
                final_mmd.append(np.nan)

    x = np.arange(len(particle_counts))
    width = 0.25

    plt.bar(x - width, final_nll, width, label='NLL', alpha=0.8)
    plt.bar(x, final_kl_kde, width, label='KDE KL', alpha=0.8)
    plt.bar(x + width, final_mmd, width, label='MMD', alpha=0.8)

    plt.xlabel('Number of Particles', fontsize=14)
    plt.ylabel('Final Value', fontsize=14)
    plt.title('Final Values Comparison', fontsize=16)
    plt.xticks(x, particle_counts)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_nll_comparison(results):
    """Plot NLL comparison across different particle counts and beta values"""
    plt.figure(figsize=(15, 10))
    
    particle_counts = [5, 10, 20, 50]
    beta_values = [0.0, 0.5, 0.67, 1.0]
    colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf']
    
    # Plot NLL convergence for different beta values
    for i, beta in enumerate(beta_values):
        plt.subplot(2, 2, i+1)
        
        for j, n_particles in enumerate(particle_counts):
            key = f"n{n_particles}_beta{beta}"
            if key in results:
                nll_list = results[key].get('nll_list', None)
                if nll_list is not None and len(nll_list) > 0:
                    if nll_list.ndim == 2:
                        nll_values = nll_list[:, 0].flatten()
                    else:
                        nll_values = nll_list.flatten()
                    plt.plot(nll_values, label=f'{n_particles} particles', 
                            linewidth=2, color=colors[j % len(colors)])
        
        plt.xscale('log')
        plt.xlabel('Iterations', fontsize=12)
        plt.ylabel('Negative Log Likelihood', fontsize=12)
        plt.title(f'NLL Convergence (β={beta})', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nll_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    print("Loading results...")
    results = load_results()
    
    if not results:
        print("No results found. Please run experiments first.")
        return
    
    print(f"Loaded {len(results)} result files")
    
    # Plot convergence
    print("Plotting convergence...")
    plot_convergence(results)
    
    # Plot NLL comparison
    print("Plotting NLL comparison...")
    plot_nll_comparison(results)
    
    print("Plots saved as 'convergence_plots.png' and 'nll_comparison.png'")

if __name__ == "__main__":
    main() 