
        data {
            int<lower=1> N;           // number of observations
            int<lower=1> D;           // number of features
            array[N] int<lower=0, upper=1> y;  // binary class labels
            matrix[N, D] X;           // feature matrix
            real<lower=0> alpha_prior; // prior shape for global precision
            real<lower=0> beta_prior;  // prior rate for global precision
        }
        
        parameters {
            vector[D] beta;           // regression coefficients
            real<lower=0.1, upper=10.0> tau;  // global precision parameter with tighter bounds
        }
        
        model {
            // Hierarchical prior structure
            // Global precision parameter: Gamma(alpha_prior, beta_prior)
            tau ~ gamma(alpha_prior, beta_prior);
            
            // Regression coefficients: Normal(0, 1/sqrt(tau)) - more stable
            beta ~ normal(0, 1/sqrt(tau));
            
            // Likelihood: Bernoulli with logistic link
            for (n in 1:N) {
                y[n] ~ bernoulli_logit(X[n, :] * beta);
            }
        }
        
        generated quantities {
            vector[N] log_likelihood;
            vector[N] y_pred;
            
            for (n in 1:N) {
                log_likelihood[n] = bernoulli_logit_lpmf(y[n] | X[n, :] * beta);
                y_pred[n] = bernoulli_logit_rng(X[n, :] * beta);
            }
        }
        