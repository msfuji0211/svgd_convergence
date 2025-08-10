import os
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.metrics.pairwise import linear_kernel as linear
from tqdm.auto import tqdm
from SVGD import SVGD
from model import MVN

import warnings
warnings.filterwarnings('ignore')

def runexp_Gaussian(mu, sig, init_mu, init_sig, n_iter, decay_factor=0.1, beta=1, mode='rbf', n_particles=10, step_size=1e-2, cons=1., seed=10, adagrad=False, lr_decay=True):
    """
    Run SVGD experiment for multivariate normal distribution.
    
    Parameters:
    -----------
    mu: array-like
        True mean of the target distribution
    sig: array-like
        True covariance matrix of the target distribution
    init_mu: array-like
        Initial mean for particle initialization
    init_sig: array-like
        Initial covariance for particle initialization
    n_iter: int
        Number of SVGD iterations
    decay_factor: float
        Learning rate decay factor
    beta: float
        Learning rate decay exponent
    mode: str
        Kernel type ('rbf' or 'linear')
    n_particles: int
        Number of particles
    step_size: float
        Initial learning rate
    cons: float
        Kernel constant
    seed: int
        Random seed for reproducibility
    adagrad: bool
        Whether to use AdaGrad optimization
    lr_decay: bool
        Whether to use learning rate decay
        
    Returns:
    --------
    tuple: (kl_list, ksd_list, eig_list)
        Lists of KL divergence, KSD, and eigenvalues over iterations
    """
    np.random.seed(seed)
    
    # Initialize model and particles
    model = MVN(mu, sig)
    x0 = np.random.multivariate_normal(mean=init_mu, cov=init_sig, size=n_particles)
    
    # Run SVGD optimization
    theta, mse_list, kl_list, ksd_list, fisher_list, eig_list = SVGD().update(
        x0, model.dlnprob, n_iter=n_iter, stepsize=step_size, cons=cons, 
        decay_factor=decay_factor, beta=beta, mode=mode, adagrad=adagrad, 
        lr_decay=lr_decay, verbose=True, true_mu=mu, true_A=sig
    )
    
    return kl_list, ksd_list, eig_list

def cumulative_mean(dist_list):
    """
    Compute cumulative mean of a list of values.
    
    Parameters:
    -----------
    dist_list: array-like
        List of values to compute cumulative mean for
        
    Returns:
    --------
    array-like: Cumulative mean at each position
    """
    cum_sum = np.cumsum(dist_list, 0)
    for i in range(len(cum_sum)):
        cum_sum[i] /= (i+1)
    return cum_sum


def main():
    """Main function to run multivariate normal SVGD experiments."""
    
    # Create results directory
    path = os.getcwd()
    if not os.path.exists(path+'/results/'):
        os.mkdir('results')
    results_path = path + '/results/'
    
    # Define target distribution parameters (2D multivariate normal)
    A = np.array([[1, 0], [0, 1]])  # Identity covariance matrix
    mu = np.array([1, 1])  # Target mean
    
    # Define initial particle distribution
    init_sig = np.array([[1, 0], [0, 1]])  # Initial covariance
    init_mu = np.array([0, 0])  # Initial mean
    
    # Experiment parameters
    n_iter = 10**5  # Number of iterations (100000 seems better)
    stepsize = 1e-2  # Learning rate
    mode = 'rbf'  # Kernel type: 'rbf' or 'linear'
    beta = 0.0  # Learning rate decay exponent [0.0, 0.5, 0.67, 1.0]
    decay_factor = 1.0  # Learning rate decay factor

    # Run experiments with different numbers of particles
    print("Running experiments with 5 particles...")
    kl_5, ksd_5, eig_5 = runexp_Gaussian(mu, A, init_mu, init_sig, n_iter, decay_factor=decay_factor, beta=beta, mode=mode, n_particles=5, step_size=stepsize, adagrad=False, lr_decay=True)
    
    print("Running experiments with 10 particles...")
    kl_10, ksd_10, eig_10 = runexp_Gaussian(mu, A, init_mu, init_sig, n_iter, decay_factor=decay_factor, beta=beta, mode=mode, n_particles=10, step_size=stepsize, adagrad=False, lr_decay=True)
    
    print("Running experiments with 100 particles...")
    kl_100, ksd_100, eig_100 = runexp_Gaussian(mu, A, init_mu, init_sig, n_iter, decay_factor=decay_factor, beta=beta, mode=mode, n_particles=100, step_size=stepsize, adagrad=False, lr_decay=True)
    
    print("Running experiments with 1000 particles...")
    kl_1000, ksd_1000, eig_1000 = runexp_Gaussian(mu, A, init_mu, init_sig, n_iter, decay_factor=decay_factor, beta=beta, mode=mode, n_particles=1000, step_size=stepsize, adagrad=False, lr_decay=True)

    # Save results to files
    print("Saving results...")
    for i, kls in zip([5, 10, 100, 1000], [kl_5, kl_10, kl_100, kl_1000]):
        np.save(results_path+'kl_{0}_{1}_{2}'.format(mode, beta, i), kls)
    for i, ksd in zip([5, 10, 100, 1000], [ksd_5, ksd_10, ksd_100, ksd_1000]):
        np.save(results_path+'ksd_{0}_{1}_{2}'.format(mode, beta, i), ksd)
    for i, eig in zip([5, 10, 100, 1000], [eig_5, eig_10, eig_100, eig_1000]):
        np.save(results_path+'eig_{0}_{1}_{2}'.format(mode, beta, i), eig)
    
    print("All experiments completed and results saved!")

if __name__ == '__main__':
    main()