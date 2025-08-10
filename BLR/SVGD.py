import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.metrics.pairwise import linear_kernel as linear
from tqdm.auto import tqdm
from sklearn.neighbors import KernelDensity, NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


class SVGD():

    def __init__(self):
        pass
    
    def svgd_kernel(self, theta, h = -1, cons=1):
        """
        Compute RBF kernel matrix and its gradient for SVGD.
        
        Parameters:
        -----------
        theta: array-like, shape (n_particles, n_params)
            Current particle positions
        h: float, optional
            Bandwidth parameter. If h < 0, median trick is used.
        cons: float, optional
            Constant multiplier for the kernel
            
        Returns:
        --------
        Kxy: array-like, shape (n_particles, n_particles)
            RBF kernel matrix
        dxkxy: array-like, shape (n_particles, n_particles, n_params)
            Gradient of the kernel matrix with respect to theta
        """
        # Numerical stability: handle NaN/inf values in particle positions
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            print("Warning: theta contains NaN or inf values")
            theta = np.nan_to_num(theta, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Compute pairwise squared distances between all particles
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist)**2
        
        # Clip distances to prevent numerical overflow
        pairwise_dists = np.clip(pairwise_dists, 0, 1e10)
        
        # Automatic bandwidth selection using median trick
        if h < 0:  # median trick for bandwidth selection
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))
            h = np.clip(h, 1e-6, 1e6)

        # Compute RBF kernel matrix
        Kxy = cons * np.exp( -pairwise_dists / h**2 / 2)
        
        # Ensure kernel values are within reasonable bounds
        Kxy = np.clip(Kxy, 1e-10, 1e10)

        # Compute gradient of kernel matrix with respect to theta
        # This implements the SVGD gradient formula: ∇_x K(x,y)
        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        
        # Ensure gradient values are within reasonable bounds
        dxkxy = np.clip(dxkxy, -1e6, 1e6)
        
        return (Kxy, dxkxy)
    
    def svgd_linear(self, theta, cons=1):
        """
        Compute linear kernel matrix and its gradient for SVGD.
        
        Parameters:
        -----------
        theta: array-like, shape (n_particles, n_params)
            Current particle positions
        cons: float, optional
            Constant added to the kernel
            
        Returns:
        --------
        Kxy: array-like, shape (n_particles, n_particles)
            Linear kernel matrix
        dxkxy: array-like, shape (n_particles, n_params)
            Gradient of the kernel matrix
        """
        Kxy = linear(theta, theta) + cons
        dxkxy = theta
        
        return (Kxy, dxkxy)
 
    def update(self, x0, lnprob, n_iter = 1000, stepsize = 1e-3, bandwidth = -1, alpha = 0.9, cons = 1, decay_factor=0.01, beta=1, mode = 'rbf', adagrad=True, lr_decay=False, debug = False, verbose=False, true_mu=None, true_A=None, mcmc_samples=None):
        """
        Update SVGD particles using Stein variational gradient descent.
        
        Parameters:
        -----------
        x0: array-like, shape (n_particles, n_params)
            Initial particle positions
        lnprob: callable
            Function that returns log probability gradients
        n_iter: int, optional
            Number of iterations
        stepsize: float, optional
            Learning rate
        bandwidth: float, optional
            Kernel bandwidth. If < 0, median trick is used
        alpha: float, optional
            Momentum parameter for AdaGrad
        cons: float, optional
            Kernel constant
        decay_factor: float, optional
            Learning rate decay factor
        beta: float, optional
            Learning rate decay exponent
        mode: str, optional
            Kernel type ('rbf' or 'linear')
        adagrad: bool, optional
            Whether to use AdaGrad optimization
        lr_decay: bool, optional
            Whether to use learning rate decay
        debug: bool, optional
            Whether to print debug information
        verbose: bool, optional
            Whether to compute and return tracking metrics
        true_mu: array-like, optional
            True mean for KL divergence calculation
        true_A: array-like, optional
            True precision matrix for KL divergence calculation
        mcmc_samples: array-like, optional
            MCMC samples for KDE/MMD-based divergence estimation
            
        Returns:
        --------
        theta: array-like
            Final particle positions
        Various tracking metrics if verbose=True
        """
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')
        
        likelihood_list = np.zeros([n_iter, 1])
        
        if verbose == True:
            n_particle = x0.shape[0]
            mse_list = np.zeros([n_iter,1])
            kl_list = np.zeros([n_iter,1])
            ksd_list = np.zeros([n_iter,1])
            kl_kde_list = np.zeros([n_iter,1])
            kl_mmd_list = np.zeros([n_iter,1])
            fisher_list = np.zeros([n_iter,1])
            eig_list = np.zeros([3, n_particle], dtype=complex)
            if true_mu is not None and true_A is not None:
                pos_sample = np.random.multivariate_normal(mean=true_mu, cov=true_A, size=1000)
            else:
                pos_sample = None
        
        theta = np.copy(x0) 
        
        fudge_factor = 1e-6
        historical_grad = 0
        eig_count = 0
        for iter in tqdm(range(n_iter)):
            if debug and (iter+1) % 1000 == 0:
                print('iter ' + str(iter+1))
            
            # Handle multiple particles
            if theta.ndim == 1:
                theta = theta.reshape(1, -1)
            
            lnpgrad = lnprob(theta)
            
            try:
                if hasattr(lnprob, 'log_likelihood'):
                    log_probs = []
                    for i in range(theta.shape[0]):
                        theta_i = theta[i]
                        log_prob_i = lnprob.log_likelihood(theta_i)
                        if hasattr(lnprob, 'n_samples'):
                            log_prob_i = log_prob_i / lnprob.n_samples
                        elif hasattr(lnprob, 'X_train'):
                            log_prob_i = log_prob_i / lnprob.X_train.shape[0]
                        log_probs.append(log_prob_i)
                    log_probs = np.array(log_probs)
                elif hasattr(lnprob, 'log_posterior'):
                    log_probs = []
                    for i in range(theta.shape[0]):
                        theta_i = theta[i]
                        log_prob_i = lnprob.log_posterior(theta_i)
                        if hasattr(lnprob, 'n_samples'):
                            log_prob_i = log_prob_i / lnprob.n_samples
                        elif hasattr(lnprob, 'X_train'):
                            log_prob_i = log_prob_i / lnprob.X_train.shape[0]
                        log_probs.append(log_prob_i)
                    log_probs = np.array(log_probs)
                elif hasattr(lnprob, 'log_prob'):
                    log_probs = lnprob.log_prob(theta)
                else:
                    log_probs = -0.5 * np.sum(lnpgrad**2, axis=1)
                
                avg_log_prob = np.mean(log_probs)
                likelihood_list[iter] = avg_log_prob
                
                print(f"Iteration {iter+1}: Average log likelihood per sample = {avg_log_prob:.6f} (min: {np.min(log_probs):.6f}, max: {np.max(log_probs):.6f})")
                
            except Exception as e:
                print(f"Warning: Could not compute likelihood at iteration {iter+1}: {e}")
                likelihood_list[iter] = np.nan
            
            if np.any(np.isnan(lnpgrad)) or np.any(np.isinf(lnpgrad)):
                print(f"Warning: lnpgrad contains NaN or inf values at iteration {iter+1}")
                lnpgrad = np.nan_to_num(lnpgrad, nan=0.0, posinf=1e6, neginf=-1e6)
            
            if mode == 'rbf':
                kxy, dxkxy = self.svgd_kernel(theta, h = -1, cons= cons)
            elif mode == 'linear':
                kxy, dxkxy = self.svgd_linear(theta, cons=cons)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / theta.shape[0]
            
            grad_theta = np.clip(grad_theta, -1e6, 1e6)
            
            if adagrad:
                if iter == 0:
                    historical_grad = historical_grad + grad_theta ** 2
                else:
                    historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
                adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
                theta = theta + stepsize * adj_grad
            else:
                if iter == 0:
                    theta = theta + stepsize * grad_theta
                else:
                    if lr_decay:
                        current_stepsize = (1/(1+(decay_factor*iter**beta)))*stepsize
                    else:
                        current_stepsize = stepsize
                    theta = theta + current_stepsize * grad_theta
            
            if verbose == True:
                kl_list[iter] = self.kl_divergence(theta, true_mu, true_A)
                ksd_list[iter] = self.ksd_distance(theta, lnprob, mode)
                
                try:
                    kl_kde_list[iter] = self.kl_divergence_kde(theta, mcmc_samples, bandwidth='silverman')
                except Exception as e:
                    if iter == 0:
                        print(f"Warning: KDE KL calculation failed: {e}")
                    kl_kde_list[iter] = np.nan
                
                try:
                    kl_mmd_list[iter] = self.kl_divergence_mmd(theta, mcmc_samples, bandwidth='median')
                except Exception as e:
                    if iter == 0:
                        print(f"Warning: MMD calculation failed: {e}")
                    kl_mmd_list[iter] = np.nan
                
                if iter+1 == 1 or (iter+1) == (n_iter/2) or (iter+1) / n_iter == 1:
                    # Add numerical stability for eigenvalue computation
                    try:
                        eig_list[eig_count] = np.linalg.eigvals(kxy)
                    except np.linalg.LinAlgError:
                        print(f"Warning: Could not compute eigenvalues at iteration {iter+1}")
                        eig_list[eig_count] = np.zeros(kxy.shape[0], dtype=complex)
                    eig_count += 1
        
        # Return appropriate values based on verbose flag
        if verbose == True:
            return theta, mse_list, kl_list, ksd_list, fisher_list, np.sort(eig_list,axis=1)[:,::-1], kl_kde_list, kl_mmd_list, likelihood_list
        else:
            # Return empty arrays for tracking variables when verbose=False
            n_particle = x0.shape[0]
            return theta, np.zeros([n_iter,1]), np.zeros([n_iter,1]), np.zeros([n_iter,1]), np.zeros([n_iter,1]), np.zeros([3, n_particle], dtype=complex), np.zeros([n_iter,1]), np.zeros([n_iter,1]), likelihood_list
    
    def MSE(self, theta, true_param):
        avg_theta = np.mean(theta, 0)
        mean_squared_error = mse(avg_theta, true_param)
        return mean_squared_error
    
    def kl_divergence(self, theta, true_mu, true_A):
        """
        Compute KL divergence between SVGD particles and true posterior
        Assumes true_A is the precision matrix (inverse of covariance)
        """
        if true_mu is None or true_A is None:
            return 0.0
        
        mu_theta = np.mean(theta, 0)
        cov_theta = np.cov(theta.T)
        
        eps = 1e-6
        cov_theta += eps * np.eye(cov_theta.shape[0])
        
        try:
            true_cov = np.linalg.inv(true_A)
            
            d = len(true_mu)
            mean_diff = true_mu - mu_theta
            
            kl = 0.5 * (
                np.trace(np.linalg.solve(true_cov, cov_theta)) +
                np.dot(mean_diff, np.linalg.solve(true_cov, mean_diff)) -
                d + np.log(np.linalg.det(true_cov) / np.linalg.det(cov_theta))
            )
            
            return kl
        except np.linalg.LinAlgError:
            return np.mean((mu_theta - true_mu) ** 2)
    
    def kl_divergence_kde(self, theta, mcmc_samples, bandwidth='silverman'):
        """
        Compute KL divergence using Kernel Density Estimation
        This method doesn't assume any specific distribution form
        
        Parameters:
        -----------
        theta: SVGD particles
        mcmc_samples: MCMC samples from true posterior
        bandwidth: bandwidth for KDE ('silverman', 'scott', or float)
        
        Returns:
        --------
        kl: KL divergence estimate
        """
        if mcmc_samples is None or len(mcmc_samples) == 0:
            return 0.0
        
        try:
            n_mcmc = min(len(mcmc_samples), 1000)
            mcmc_subset = mcmc_samples[:n_mcmc]
            
            kde_true = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(mcmc_subset)
            kde_svgd = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(theta)
            
            log_density_true_at_mcmc = kde_true.score_samples(mcmc_subset)
            log_density_svgd_at_mcmc = kde_svgd.score_samples(mcmc_subset)
            
            kl = np.mean(log_density_true_at_mcmc - log_density_svgd_at_mcmc)
            
            kl = max(0.0, kl)
            
            return kl
            
        except Exception as e:
            print(f"Warning: KDE-based KL calculation failed: {e}")
            return 0.0
    

    
    def kl_divergence_mmd(self, theta, mcmc_samples, bandwidth='median'):
        """
        Compute KL divergence approximation using Maximum Mean Discrepancy (MMD)
        This is a more stable alternative when direct KL estimation is problematic
        
        Parameters:
        -----------
        theta: SVGD particles
        mcmc_samples: MCMC samples from true posterior
        bandwidth: bandwidth for RBF kernel ('median', 'silverman', or float)
        
        Returns:
        --------
        mmd: MMD-based divergence estimate
        """
        if mcmc_samples is None or len(mcmc_samples) == 0:
            return 0.0
        
        try:
            # Use a subset of MCMC samples for computational efficiency
            n_mcmc = min(len(mcmc_samples), 1000)
            mcmc_subset = mcmc_samples[:n_mcmc]
            
            # Compute pairwise distances for bandwidth selection
            all_samples = np.vstack([theta, mcmc_subset])
            pairwise_dists = cdist(all_samples, all_samples)
            
            # Select bandwidth
            if bandwidth == 'median':
                h = np.median(pairwise_dists)
            elif bandwidth == 'silverman':
                h = np.median(pairwise_dists) * (len(all_samples) ** (-1.0 / (all_samples.shape[1] + 4)))
            else:
                h = bandwidth
            
            Kxx = rbf(theta, theta, h).mean()
            Kxy = rbf(theta, mcmc_subset, h).mean()
            Kyy = rbf(mcmc_subset, mcmc_subset, h).mean()
            
            mmd = Kxx - 2 * Kxy + Kyy
            
            mmd = max(0.0, mmd)
            
            return mmd
            
        except Exception as e:
            print(f"Warning: MMD-based divergence calculation failed: {e}")
            return 0.0
    
    def rbf_kernel(self, x, y, bandwidth=1.):
        diff = x - y
        squared_distance = np.sum(diff**2)
        if bandwidth < 1:
            bandwidth = np.median(squared_distance)  
            bandwidth = np.sqrt(0.5 * bandwidth / np.log(x.shape[0]+1))
        
        return np.exp(-0.5 * squared_distance / (bandwidth**2))
    
    def mmd_distance(self, theta, pos_sample, bandwidth=1.0):
        Kxx = rbf(theta, theta, bandwidth).mean()
        kxy = rbf(theta, pos_sample, bandwidth).mean()
        kyy = rbf(pos_sample, pos_sample, bandwidth).mean()
        
        return Kxx - 2. * kxy + kyy
    
    def ksd_distance(self, theta, lnprob, mode='rbf', h=-1, cons=1):
        if mode == 'rbf':
            pairwise_dists = cdist(theta, theta)
            if h < 0:
                h = np.median(pairwise_dists)
                h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))
            Kxy = np.exp(- pairwise_dists ** 2 / h)
            p = theta.shape[1]
            score = lnprob(theta)
            diffs = (theta * score).sum(-1) - (theta @ score.T)
            diffs = diffs + diffs.T
            scalars = score @ score.T
            der2 = p - pairwise_dists / h
            stein_kernel = (Kxy * (scalars + diffs / h + der2 / h)).mean()
        elif mode == 'linear':
            Kxy = linear(theta, theta) + cons
            d = theta.shape[1]
            score = lnprob(theta)
            stein_kernel = (score @ score.T * Kxy + 2 * score @ theta.T + d).mean()
            
        return stein_kernel
    
    def approximate_fisher(self, kernel_function, grad):
        try:
            kernel_inv = np.linalg.inv(kernel_function)
        except np.linalg.LinAlgError:
            epsilon = 1e-6
            kernel_inv = np.linalg.inv(kernel_function + epsilon * np.identity(kernel_function.shape[0]))
        return np.linalg.norm(np.matmul(kernel_inv, grad)) ** 2
    
    def save_likelihood_history(self, likelihood_list, filename='likelihood_history.npy'):
        """Save likelihood history to file"""
        try:
            np.save(filename, likelihood_list)
            print(f"Likelihood history saved to {filename}")
        except Exception as e:
            print(f"Warning: Could not save likelihood history: {e}")
    
    def plot_likelihood_convergence(self, likelihood_list, save_path='likelihood_convergence.png'):
        """Plot likelihood convergence over iterations"""
        try:
            import matplotlib.pyplot as plt
            
            if likelihood_list.ndim == 2:
                likelihood_1d = likelihood_list[:, 0]
            else:
                likelihood_1d = likelihood_list
            
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(likelihood_1d, 'b-', linewidth=2, alpha=0.8)
            plt.xlabel('Iteration')
            plt.ylabel('Average Log Likelihood per Sample')
            plt.title('SVGD Likelihood Convergence')
            plt.grid(True, alpha=0.3)
            
            plt.axhline(y=-0.52, color='r', linestyle='--', alpha=0.7, label='MCMC Reference (-0.52)')
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.hist(likelihood_1d, bins=20, alpha=0.7, color='blue', edgecolor='black')
            plt.axvline(x=-0.52, color='r', linestyle='--', alpha=0.7, label='MCMC Reference')
            plt.xlabel('Log Likelihood per Sample')
            plt.ylabel('Frequency')
            plt.title('Distribution of Likelihood Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Likelihood convergence plot saved to {save_path}")
        except ImportError:
            print("Warning: matplotlib not available for plotting")
        except Exception as e:
            print(f"Warning: Could not create likelihood plot: {e}") 