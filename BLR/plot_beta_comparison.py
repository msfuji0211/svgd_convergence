# Plot KSD comparison across different beta values
plt.figure(figsize=(15, 10))

particle_counts = [5, 10, 20, 50]
beta_values = [0.0, 0.5, 0.67, 1.0]
colors = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf']

# Plot KSD convergence for different beta values
for i, beta in enumerate(beta_values):
    plt.subplot(2, 2, i+1)
    
    for j, n_particles in enumerate(particle_counts):
        key = f"n{n_particles}_beta{beta}"
        if key in results:
            ksd_list = results[key].get('ksd_list', None)
            if ksd_list is not None and len(ksd_list) > 0:
                ksd_values = ksd_list.flatten()
                plt.plot(ksd_values, label=f'{n_particles} particles', 
                        linewidth=2, color=colors[j % len(colors)])
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Kernel Stein Discrepancy', fontsize=12)
    plt.title(f'KSD Convergence (β={beta})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot KDE-KL comparison across different beta values
plt.figure(figsize=(15, 10))

# Plot KDE-based KL divergence convergence for different beta values
for i, beta in enumerate(beta_values):
    plt.subplot(2, 2, i+1)
    
    for j, n_particles in enumerate(particle_counts):
        key = f"n{n_particles}_beta{beta}"
        if key in results:
            kl_kde_list = results[key].get('kl_kde_list', None)
            if kl_kde_list is not None and len(kl_kde_list) > 0:
                valid_indices = ~np.isnan(kl_kde_list.flatten())
                if np.any(valid_indices):
                    kl_kde_values = kl_kde_list.flatten()[valid_indices]
                    iterations = np.arange(len(kl_kde_list))[valid_indices]
                    plt.plot(iterations, kl_kde_values, label=f'{n_particles} particles', 
                            linewidth=2, color=colors[j % len(colors)])
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('KL Divergence (KDE)', fontsize=12)
    plt.title(f'KDE-KL Convergence (β={beta})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Plot MMD comparison across different beta values
plt.figure(figsize=(15, 10))

# Plot MMD divergence convergence for different beta values
for i, beta in enumerate(beta_values):
    plt.subplot(2, 2, i+1)
    
    for j, n_particles in enumerate(particle_counts):
        key = f"n{n_particles}_beta{beta}"
        if key in results:
            kl_mmd_list = results[key].get('kl_mmd_list', None)
            if kl_mmd_list is not None and len(kl_mmd_list) > 0:
                valid_indices = ~np.isnan(kl_mmd_list.flatten())
                if np.any(valid_indices):
                    kl_mmd_values = kl_mmd_list.flatten()[valid_indices]
                    iterations = np.arange(len(kl_mmd_list))[valid_indices]
                    plt.plot(iterations, kl_mmd_values, label=f'{n_particles} particles', 
                            linewidth=2, color=colors[j % len(colors)])
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('MMD Divergence', fontsize=12)
    plt.title(f'MMD Convergence (β={beta})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Combined comparison: Final values across different beta values
plt.figure(figsize=(20, 15))

metrics = ['NLL', 'KSD', 'KDE-KL', 'MMD']
metric_keys = ['nll_list', 'ksd_list', 'kl_kde_list', 'kl_mmd_list']

for metric_idx, (metric_name, metric_key) in enumerate(zip(metrics, metric_keys)):
    plt.subplot(2, 2, metric_idx + 1)
    
    for beta_idx, beta in enumerate(beta_values):
        final_values = []
        
        for n_particles in particle_counts:
            key = f"n{n_particles}_beta{beta}"
            if key in results:
                if metric_key == 'nll_list':
                    # Handle NLL
                    metric_list = results[key].get(metric_key, None)
                    if metric_list is not None and len(metric_list) > 0:
                        if metric_list.ndim == 2:
                            final_values.append(metric_list[-1, 0].item())
                        else:
                            final_values.append(metric_list[-1].item())
                    else:
                        final_values.append(np.nan)
                elif metric_key == 'ksd_list':
                    # Handle KSD
                    metric_list = results[key].get(metric_key, None)
                    if metric_list is not None and len(metric_list) > 0:
                        final_values.append(metric_list[-1].item())
                    else:
                        final_values.append(np.nan)
                else:
                    # Handle KDE-KL and MMD
                    metric_list = results[key].get(metric_key, None)
                    if metric_list is not None and len(metric_list) > 0:
                        valid_values = metric_list.flatten()[~np.isnan(metric_list.flatten())]
                        if len(valid_values) > 0:
                            final_values.append(valid_values[-1])
                        else:
                            final_values.append(np.nan)
                    else:
                        final_values.append(np.nan)
            else:
                final_values.append(np.nan)
        
        # Plot bars for this beta value
        x = np.arange(len(particle_counts))
        width = 0.2
        offset = (beta_idx - 1.5) * width
        
        plt.bar(x + offset, final_values, width, 
                label=f'β={beta}', alpha=0.8)
    
    plt.xlabel('Number of Particles', fontsize=12)
    plt.ylabel(f'Final {metric_name}', fontsize=12)
    plt.title(f'Final {metric_name} Comparison', fontsize=14)
    plt.xticks(x, particle_counts)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use log scale for KSD, KDE-KL, and MMD
    if metric_name in ['KSD', 'KDE-KL', 'MMD']:
        plt.yscale('log')

plt.tight_layout()
plt.show() 