import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import bz2
import warnings
warnings.filterwarnings('ignore')

class BLR:
    def __init__(self, X_train=None, y_train=None, X_test=None, y_test=None, 
                 alpha_prior=1.0, beta_prior=0.01):
        """
        Initialize Hierarchical Bayesian Logistic Regression for Binary Classification
        
        Parameters:
        -----------
        X_train: training features
        y_train: training labels (binary: 0 or 1)
        X_test: test features
        y_test: test labels (binary: 0 or 1)
        alpha_prior: prior shape for global precision parameter (Gamma distribution)
        beta_prior: prior rate for global precision parameter (Gamma distribution)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        
        if X_train is not None and y_train is not None:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model parameters"""
        self.n_samples, self.n_features = self.X_train.shape
        self.n_classes = 2  # Binary classification
        self.n_params = self.n_features + 1  # +1 for global precision parameter
        
        print(f"Model initialized: {self.n_features} features, {self.n_classes} classes (binary)")
        print(f"Total parameters: {self.n_params} (including global precision)")
    
    def _sigmoid(self, z):
        """Compute sigmoid function"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def log_prior(self, theta):
        """Compute log prior probability for hierarchical model"""
        # theta = [beta_1, ..., beta_D, log_tau]
        beta = theta[:-1]  # regression coefficients
        log_tau = theta[-1]    # log of global precision parameter
        tau = np.exp(log_tau)  # convert back to tau
        
        # Prior for beta: N(0, 1/sqrt(tau))
        log_prior_beta = -0.5 * np.sqrt(tau) * np.sum(beta**2)
        
        # Prior for log_tau: transformed from Gamma(alpha_prior, beta_prior)
        # Using change of variables: p(log_tau) = p(tau) * |d tau / d log_tau|
        log_prior_log_tau = (self.alpha_prior - 1) * log_tau - self.beta_prior * tau + log_tau
        
        return log_prior_beta + log_prior_log_tau
    
    def log_likelihood(self, theta):
        """Compute log likelihood for binary classification"""
        beta = theta[:-1]  # regression coefficients
        log_tau = theta[-1]    # log of global precision parameter
        # Note: tau is not used in likelihood for logistic regression
        
        # Compute logits
        logits = self.X_train @ beta
        
        # Compute probabilities
        probs = self._sigmoid(logits)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        
        # Binary cross-entropy
        log_likelihood = np.sum(
            self.y_train * np.log(probs) + 
            (1 - self.y_train) * np.log(1 - probs)
        )
        
        return log_likelihood
    
    def log_posterior(self, theta):
        """Compute log posterior probability"""
        return self.log_prior(theta) + self.log_likelihood(theta)
    
    def dlnprob(self, theta):
        """Compute gradient of log posterior probability"""
        # Handle single particle case
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        
        # Handle multiple particles
        n_particles = theta.shape[0]
        gradients = []
        
        for i in range(n_particles):
            theta_i = theta[i]
            beta = theta_i[:-1]  # regression coefficients
            log_tau = theta_i[-1]    # log of global precision parameter
            tau = np.exp(log_tau)  # convert back to tau
            
            # Add numerical stability for tau
            tau = np.clip(tau, 1e-6, 1e6)
            
            # Compute probabilities
            logits = self.X_train @ beta
            probs = self._sigmoid(logits)
            
            # Gradient of log likelihood w.r.t. beta
            residual = self.y_train - probs
            grad_beta_likelihood = self.X_train.T @ residual
            
            # Gradient of log prior w.r.t. beta
            grad_beta_prior = -np.sqrt(tau) * beta
            
            # Gradient of log likelihood w.r.t. log_tau (no contribution from likelihood)
            grad_log_tau_likelihood = 0.0
            
            # Gradient of log prior w.r.t. log_tau
            # Using chain rule: d/d(log_tau) = d/d(tau) * d(tau)/d(log_tau) = d/d(tau) * tau
            grad_tau_prior = (self.alpha_prior - 1) / tau - self.beta_prior - 0.25 * np.sum(beta**2) / np.sqrt(tau)
            grad_log_tau_prior = grad_tau_prior * tau  # chain rule
            
            # Add numerical stability for gradients
            grad_beta_prior = np.clip(grad_beta_prior, -1e6, 1e6)
            grad_log_tau_prior = np.clip(grad_log_tau_prior, -1e6, 1e6)
            
            # Combine gradients
            grad_beta = grad_beta_likelihood + grad_beta_prior
            grad_log_tau = grad_log_tau_likelihood + grad_log_tau_prior
            
            # Final clipping
            grad_beta = np.clip(grad_beta, -1e6, 1e6)
            grad_log_tau = np.clip(grad_log_tau, -1e6, 1e6)
            
            grad_i = np.concatenate([grad_beta, [grad_log_tau]])
            gradients.append(grad_i)
        
        return np.array(gradients)
    
    def predict(self, theta, X):
        """Make predictions"""
        beta = theta[:-1]  # regression coefficients
        logits = X @ beta
        probs = self._sigmoid(logits)
        return (probs > 0.5).astype(int)
    
    def accuracy(self, theta, X, y):
        """Compute accuracy"""
        predictions = self.predict(theta, X)
        return np.mean(predictions == y)
    
    def load_libsvm_binary_covertype_data(self, n_samples=None, test_size=0.2, random_state=42):
        """Load and preprocess libsvm binary UCI Covertype dataset"""
        print("Loading libsvm binary UCI Covertype dataset...")
        
        # Load libsvm format data
        data_file = "covtype.libsvm.binary.scale.bz2"
        try:
            X, y = load_svmlight_file(data_file)
            X = X.toarray()  # Convert sparse matrix to dense array
            
            print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Label range: {np.min(y)} to {np.max(y)}")
            print(f"Unique labels: {np.unique(y)}")
            
            # Ensure labels are binary (0 and 1)
            if len(np.unique(y)) == 2:
                # Map labels to 0 and 1 if they're not already
                unique_labels = np.unique(y)
                if not np.array_equal(unique_labels, [0, 1]):
                    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
                    y = np.array([label_map[label] for label in y])
                    print(f"Mapped labels: {unique_labels} -> [0, 1]")
            else:
                raise ValueError(f"Expected binary labels, but found {len(np.unique(y))} unique values: {np.unique(y)}")
            
        except Exception as e:
            print(f"Error loading libsvm file: {e}")
            print("Falling back to sklearn fetch_covtype...")
            return self.load_binary_covertype_data(n_samples, test_size, random_state)
        
        print(f"Binary labels: {np.unique(y)}")
        print(f"Class distribution: {np.bincount(y)}")
        
        # Subsample for faster computation if specified
        if n_samples is not None and n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y = y[indices]
            print(f"Subsampled to {n_samples} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Note: libsvm data is already scaled, so we don't need to standardize
        print("Note: Using pre-scaled libsvm data (no additional standardization)")
        
        # Update model data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Initialize model parameters
        self._initialize_model()
        
        print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        print(f"Features: {X_train.shape[1]}, Classes: {self.n_classes}")
        print(f"Training class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        
        return X_train, y_train, X_test, y_test
    
    def load_binary_covertype_data(self, n_samples=10000, test_size=0.2, random_state=42):
        """Load and preprocess binary UCI Covertype dataset (fallback method)"""
        print("Loading binary UCI Covertype dataset (fallback method)...")
        data = fetch_covtype()
        X, y = data.data, data.target - 1  # Convert to 0-based indexing
        
        # Convert to binary classification: class 1 vs all others
        y_binary = (y == 1).astype(int)
        
        print(f"Original labels: {np.unique(y)}")
        print(f"Binary labels: {np.unique(y_binary)}")
        print(f"Class distribution: {np.bincount(y_binary)}")
        
        # Subsample for faster computation
        if n_samples < len(X):
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            y_binary = y_binary[indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, random_state=random_state, stratify=y_binary
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Update model data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Initialize model parameters
        self._initialize_model()
        
        print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        print(f"Features: {X_train.shape[1]}, Classes: {self.n_classes}")
        print(f"Training class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        
        return X_train, y_train, X_test, y_test 