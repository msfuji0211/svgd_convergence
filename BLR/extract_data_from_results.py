# Extract data from results for different beta values and particle counts
# This code extracts kl_kde_list, ksd_list, and eig_list from the results dictionary

def extract_data_from_results(results, particle_count=5):
    """
    Extract data from results dictionary for a specific particle count
    
    Parameters:
    -----------
    results : dict
        Results dictionary loaded from pickle files
    particle_count : int
        Number of particles to extract data for
    
    Returns:
    --------
    dict : Dictionary containing extracted data for different beta values
    """
    
    extracted_data = {}
    beta_values = [0.0, 0.5, 0.67, 1.0]
    
    for beta in beta_values:
        key = f"n{particle_count}_beta{beta}"
        if key in results:
            data = results[key]
            
            # Extract KDE-KL (replacing the old Gaussian KL)
            kl_kde_list = data.get('kl_kde_list', None)
            if kl_kde_list is not None:
                # Remove NaN values and get valid data
                valid_kl = kl_kde_list.flatten()[~np.isnan(kl_kde_list.flatten())]
            else:
                valid_kl = np.array([])
            
            # Extract KSD
            ksd_list = data.get('ksd_list', None)
            if ksd_list is not None:
                ksd_data = ksd_list.flatten()
            else:
                ksd_data = np.array([])
            
            # Extract eigenvalues
            eig_list = data.get('eig_list', None)
            if eig_list is not None:
                eig_data = eig_list
            else:
                eig_data = np.array([])
            
            # Store extracted data
            extracted_data[f'beta_{beta}'] = {
                'kl_kde': valid_kl,
                'ksd': ksd_data,
                'eig': eig_data
            }
            
            print(f"Extracted data for {key}:")
            print(f"  KDE-KL shape: {valid_kl.shape}")
            print(f"  KSD shape: {ksd_data.shape}")
            print(f"  Eigenvalues shape: {eig_data.shape}")
        else:
            print(f"Warning: Key {key} not found in results")
    
    return extracted_data

# Extract data for particle count 5
data_5 = extract_data_from_results(results, particle_count=5)

# Create variables in the same format as the original code
# For beta = 1.0
kl_rbf_5 = data_5['beta_1.0']['kl_kde']
ksd_rbf_5 = data_5['beta_1.0']['ksd']
eig_rbf_5 = data_5['beta_1.0']['eig']

# For beta = 0.67
kl_rbf_67_5 = data_5['beta_0.67']['kl_kde']
ksd_rbf_67_5 = data_5['beta_0.67']['ksd']
eig_rbf_67_5 = data_5['beta_0.67']['eig']

# For beta = 0.5
kl_rbf_sqrt_5 = data_5['beta_0.5']['kl_kde']
ksd_rbf_sqrt_5 = data_5['beta_0.5']['ksd']
eig_rbf_sqrt_5 = data_5['beta_0.5']['eig']

# For beta = 0.0
kl_rbf_0_5 = data_5['beta_0.0']['kl_kde']
ksd_rbf_0_5 = data_5['beta_0.0']['ksd']
eig_rbf_0_5 = data_5['beta_0.0']['eig']

# Calculate cumulative means
def cumulative_mean(values):
    """Compute cumulative mean of values"""
    if len(values) == 0:
        return np.array([])
    return np.cumsum(values) / np.arange(1, len(values) + 1)

cum_kl_rbf_5 = cumulative_mean(kl_rbf_5)
cum_ksd_rbf_5 = cumulative_mean(ksd_rbf_5)

cum_kl_rbf_67_5 = cumulative_mean(kl_rbf_67_5)
cum_ksd_rbf_67_5 = cumulative_mean(ksd_rbf_67_5)

cum_kl_rbf_sqrt_5 = cumulative_mean(kl_rbf_sqrt_5)
cum_ksd_rbf_sqrt_5 = cumulative_mean(ksd_rbf_sqrt_5)

cum_kl_rbf_0_5 = cumulative_mean(kl_rbf_0_5)
cum_ksd_rbf_0_5 = cumulative_mean(ksd_rbf_0_5)

# Print summary
print("\nData extraction summary:")
print(f"Beta 1.0: KL-KDE={len(kl_rbf_5)}, KSD={len(ksd_rbf_5)}, EIG={eig_rbf_5.shape}")
print(f"Beta 0.67: KL-KDE={len(kl_rbf_67_5)}, KSD={len(ksd_rbf_67_5)}, EIG={eig_rbf_67_5.shape}")
print(f"Beta 0.5: KL-KDE={len(kl_rbf_sqrt_5)}, KSD={len(ksd_rbf_sqrt_5)}, EIG={eig_rbf_sqrt_5.shape}")
print(f"Beta 0.0: KL-KDE={len(kl_rbf_0_5)}, KSD={len(ksd_rbf_0_5)}, EIG={eig_rbf_0_5.shape}")

# Optional: Extract data for other particle counts
def extract_all_particle_data(results):
    """Extract data for all particle counts"""
    particle_counts = [5, 10, 20, 50]
    all_data = {}
    
    for n_particles in particle_counts:
        all_data[n_particles] = extract_data_from_results(results, particle_count=n_particles)
    
    return all_data

# Uncomment the following line if you want data for all particle counts
# all_particle_data = extract_all_particle_data(results) 