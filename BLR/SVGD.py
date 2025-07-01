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
        RBF kernel for parameters.
        """
        # Add numerical stability checks
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            print("Warning: theta contains NaN or inf values")
            theta = np.nan_to_num(theta, nan=0.0, posinf=1e6, neginf=-1e6)
        
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist)**2
        
        # Add numerical stability
        pairwise_dists = np.clip(pairwise_dists, 0, 1e10)
        
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))
            h = np.clip(h, 1e-6, 1e6)  # Clip h to reasonable range

        # compute the rbf kernel
        Kxy = cons * np.exp( -pairwise_dists / h**2 / 2)
        
        # Add numerical stability
        Kxy = np.clip(Kxy, 1e-10, 1e10)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        
        # Add numerical stability
        dxkxy = np.clip(dxkxy, -1e6, 1e6)
        
        return (Kxy, dxkxy)
    
    def svgd_linear(self, theta, cons=1):
        Kxy = linear(theta, theta) + cons
        dxkxy = theta
        
        return (Kxy, dxkxy)
 
    def update(self, x0, lnprob, n_iter = 1000, stepsize = 1e-3, bandwidth = -1, alpha = 0.9, cons = 1, decay_factor=0.01, c=0.1, beta=1, mode = 'rbf', adagrad=True, lr_decay=False, debug = False, verbose=False, true_mu=None, true_A=None):
        # Check input
        if x0 is None or lnprob is None:
            raise ValueError('x0 or lnprob cannot be None!')
        
        if verbose == True:
            n_particle = x0.shape[0]
            mse_list = np.zeros([n_iter,1])
            kl_list = np.zeros([n_iter,1])
            ksd_list = np.zeros([n_iter,1])
            fisher_list = np.zeros([n_iter,1])
            eig_list = np.zeros([3, n_particle], dtype=complex)
            pos_sample = np.random.multivariate_normal(mean=true_mu, cov=true_A, size=1000)
        
        theta = np.copy(x0) 
        
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        c = 0
        for iter in tqdm(range(n_iter)):
            if debug and (iter+1) % 1000 == 0:
                print('iter ' + str(iter+1))
            
            # Handle multiple particles
            if theta.ndim == 1:
                theta = theta.reshape(1, -1)
            
            lnpgrad = lnprob(theta)
            
            # Add numerical stability for gradients
            if np.any(np.isnan(lnpgrad)) or np.any(np.isinf(lnpgrad)):
                print(f"Warning: lnpgrad contains NaN or inf values at iteration {iter+1}")
                lnpgrad = np.nan_to_num(lnpgrad, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # calculating the kernel matrix
            if mode == 'rbf':
                kxy, dxkxy = self.svgd_kernel(theta, h = -1, cons= cons)
            elif mode == 'linear':
                kxy, dxkxy = self.svgd_linear(theta, cons=cons)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / theta.shape[0]
            
            # Add numerical stability for final gradient
            grad_theta = np.clip(grad_theta, -1e6, 1e6)
            
            # adagrad
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
                        stepsize = (1/(1+(decay_factor*iter**beta)))*c
                    theta = theta + stepsize * grad_theta
            
            if verbose == True:
                kl_list[iter] = self.kl_divergence(theta, true_mu, true_A)
                ksd_list[iter] = self.ksd_distance(theta, lnprob, mode)
                if iter+1 == 1 or (iter+1) == (n_iter/2) or (iter+1) / n_iter == 1:
                    # Add numerical stability for eigenvalue computation
                    try:
                        eig_list[c] = np.linalg.eigvals(kxy)
                    except np.linalg.LinAlgError:
                        print(f"Warning: Could not compute eigenvalues at iteration {iter+1}")
                        eig_list[c] = np.zeros(kxy.shape[0], dtype=complex)
                    c += 1
        
        return theta, mse_list, kl_list, ksd_list, fisher_list, np.sort(eig_list,axis=1)[:,::-1]
    
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
        
        # Add small regularization to avoid singular matrix
        eps = 1e-6
        cov_theta += eps * np.eye(cov_theta.shape[0])
        
        try:
            # true_A is precision matrix, so we need its inverse for covariance
            true_cov = np.linalg.inv(true_A)
            
            # Compute KL divergence
            d = len(true_mu)
            mean_diff = true_mu - mu_theta
            
            # KL divergence formula for multivariate normal
            kl = 0.5 * (
                np.trace(np.linalg.solve(true_cov, cov_theta)) +
                np.dot(mean_diff, np.linalg.solve(true_cov, mean_diff)) -
                d + np.log(np.linalg.det(true_cov) / np.linalg.det(cov_theta))
            )
            
            return kl
        except np.linalg.LinAlgError:
            # Fallback: use simpler distance measure
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
            # Use a subset of MCMC samples for computational efficiency
            n_mcmc = min(len(mcmc_samples), 1000)  # Limit to 1000 samples
            mcmc_subset = mcmc_samples[:n_mcmc]
            
            # Fit KDE to MCMC samples (true posterior)
            kde_true = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(mcmc_subset)
            
            # Fit KDE to SVGD particles
            kde_svgd = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(theta)
            
            # Compute log densities at MCMC sample locations (true distribution)
            log_density_true_at_mcmc = kde_true.score_samples(mcmc_subset)
            log_density_svgd_at_mcmc = kde_svgd.score_samples(mcmc_subset)
            
            # KL divergence: E_p[log(p) - log(q)] where p is true, q is SVGD
            # We compute this expectation over the true distribution (MCMC samples)
            kl = np.mean(log_density_true_at_mcmc - log_density_svgd_at_mcmc)
            
            # Ensure non-negativity (numerical stability)
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
            
            # Compute MMD
            Kxx = rbf(theta, theta, h).mean()
            Kxy = rbf(theta, mcmc_subset, h).mean()
            Kyy = rbf(mcmc_subset, mcmc_subset, h).mean()
            
            mmd = Kxx - 2 * Kxy + Kyy
            
            # Ensure non-negativity
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