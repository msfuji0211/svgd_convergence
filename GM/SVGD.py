import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics.pairwise import rbf_kernel as rbf
from sklearn.metrics.pairwise import linear_kernel as linear
from tqdm.auto import tqdm
from sklearn.neighbors import KernelDensity
import warnings
warnings.filterwarnings('ignore')


class SVGD():

    def __init__(self):
        pass
    
    def svgd_kernel(self, theta, h = -1, cons=1):
        """
        RBF kernel for parameters.
        """
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist)**2
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))

        # compute the rbf kernel
        Kxy = cons * np.exp( -pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)
    
    def svgd_linear(self, theta, cons=1):
        Kxy = linear(theta, theta) + cons
        dxkxy = theta
        
        return (Kxy, dxkxy)
 
    def update(self, x0, lnprob, n_iter = 1000, stepsize = 1e-3, bandwidth = -1, alpha = 0.9, cons = 1, decay_factor=0.01, beta=1, mode = 'rbf', adagrad=True, lr_decay=False, debug = False, verbose=False, true_mu=None, true_A=None, p=1/3):
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
            pos_sample = np.concatenate([np.random.multivariate_normal(mean=true_mu[0:2], cov=np.diag([true_A[0][0], true_A[1][1]]), size=int(1000*(1-p))), np.random.multivariate_normal(mean=true_mu[2:4], cov=np.diag([true_A[2][2], true_A[3][3]]), size=int(1000*p))])
        
        theta = np.copy(x0) 
        
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        eig_count = 0  # Counter for eigenvalue computation
        for iter in tqdm(range(n_iter)):
            if debug and (iter+1) % 1000 == 0:
                print('iter ' + str(iter+1))
            
            lnpgrad = lnprob(theta)
            # calculating the kernel matrix
            if mode == 'rbf':
                kxy, dxkxy = self.svgd_kernel(theta, h = -1, cons= cons)
            elif mode == 'linear':
                kxy, dxkxy = self.svgd_linear(theta, cons=cons)
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]
            
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
                        current_stepsize = (1/(1+(decay_factor*iter**beta)))*stepsize
                    else:
                        current_stepsize = stepsize
                    theta = theta + current_stepsize * grad_theta
            
            if verbose == True:
                kl_list[iter] = self.kl_divergence(theta, pos_sample, p)
                ksd_list[iter] = self.ksd_distance(theta, lnprob, mode)
                if iter+1 == 1 or (iter+1) == (n_iter/2) or (iter+1) / n_iter == 1:
                    eig_list[eig_count] = np.linalg.eigvals(kxy)
                    eig_count += 1
        
        return theta, mse_list, kl_list, ksd_list, fisher_list, np.sort(eig_list,axis=1)[:,::-1]
    
    def MSE(self, theta, true_param):
        avg_theta = np.mean(theta, 0)
        mean_squared_error = mse(avg_theta, true_param)
        return mean_squared_error
    
    def kl_divergence(self, theta, pos_sample, p):
        """
        Approximating KL via kernel density estimation
        """
        kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(pos_sample)
        log_density = kde.score_samples(theta.reshape(-1,2))
        kde2 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(theta.reshape(-1,2))
        log_density2 = kde2.score_samples(theta.reshape(-1,2))
        kl = (np.exp(log_density2)*(log_density2 - log_density)).mean()
        
        return kl
    
    def rbf_kernel(self, x, y, bandwidth=1.):
        diff = x - y
        squared_distance = np.sum(diff**2)
        if bandwidth < 1:
            bandwidth = np.median(squared_distance)  
            bandwidth = np.sqrt(0.5 * bandwidth / np.log(x.shape[0]+1))
        
        return np.exp(-0.5 * squared_distance / (bandwidth**2))
    
    def mmd_distance(self, theta, pos_sample, bandwidth=1.0):
        n_samples = theta.shape[0]
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
            d = theta.shape[1]
            score = lnprob(theta)
            diffs = (theta * score).sum(-1) - (theta @ score.T)
            diffs = diffs + diffs.T
            scalars = score @ score.T
            der2 = d - pairwise_dists / h
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

