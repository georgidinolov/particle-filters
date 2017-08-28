#include <algorithm>
#include <chrono>
#include <cmath>
#include "DataTypes.hpp"
#include <fstream>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <iostream>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc < 9 || argc > 9) {
    printf("You must provide input\n");
    printf("The input is: \n data file list (each file on new line); \noutput directory;\nrelative tolerance for function during mle estimation (as double); \ninitial guess for sigma_x; \ninitial guess for sigma_y; \ninitial guess for rho; \nfile name prefix; \nfile name suffix; \n");
    printf("file names will be PREFIXmle-results-NUMBER_DATA_SET-order-ORDERSUFFIX.csv, stored in the output directory.\n");
    exit(0);
  }

  long unsigned T = 1 * 20 * 6.5 * 3600 * 1000; // one year ms
  long unsigned Delta = 1 * 6.5*3600*1000; // one day in ms
  
  // Values taken from the microstructure paper
  double mu_hat = 1.7e-12;
  double theta_hat = 5.6e-10;
  double alpha_hat = -13;
  double tau_hat = std::sqrt(1.3e-9);

  parameters params;
  params.mu_x = 0; // setting it to zero artificically
  params.mu_y = 0;

  params.alpha_x = alpha_hat + 1.0/2.0*log(Delta);
  params.alpha_y = alpha_hat + 1.0/2.0*log(Delta);

  params.theta_x = exp(-theta_hat * 1.0*Delta);
  params.theta_y = exp(-theta_hat * 1.0*Delta);

  params.tau_x = tau_hat * sqrt((1 - exp(-2*theta_hat*Delta))/(2*theta_hat));
  params.tau_y = tau_hat * sqrt((1 - exp(-2*theta_hat*Delta))/(2*theta_hat));
  params.tau_rho = 0.01;

  params.leverage_x_rho = 0;
  params.leverage_y_rho = 0;

  unsigned N = T/Delta;
  std::vector<observable_datum> ys (N);
  std::vector<stoch_vol_datum> thetas (N);
  
  generate_data(ys,
		thetas,
		params,
		6.5*3600*10);

  unsigned N_particles = 20;
  std::vector<double> log_weights (N_particles);
  for (unsigned i=0; i<N_particles; ++i) {
    log_weights[i] = 0.0;
  }

  const gsl_rng_type * Type;
  gsl_rng_env_setup();
  Type = gsl_rng_default;
  gsl_rng * r_ptr = gsl_rng_alloc(Type);
  
  std::vector<stoch_vol_datum> theta_tm1 = sample_theta_prior(params,
							      N_particles,
							      r_ptr);

  std::vector<stoch_vol_datum> theta_t = theta_tm1;
  std::vector<unsigned> ks = std::vector<unsigned> (N_particles, 1);

  std::vector<unsigned> particle_indeces = std::vector<unsigned> (N_particles);
  std::iota(std::begin(particle_indeces), std::end(particle_indeces), 0);

  std::ofstream mean_levels;
  mean_levels.open("inference.csv");
  mean_levels << "mean_log_sigma_x, var_log_sigma_x,"
	      << "mean_log_sigma_y, var_log_sigma_y,"
	      << "mean_rho_tilde, var_rho_tilde, NA\n";

  double dx = 1.0/256.0;
  double dx_likelihood = 1.0/256.0;
  double rho_basis = 0.0;
  double sigma = 0.3;
  double power = 1.0;
  double std_dev_factor = 0.5;
  
  BivariateGaussianKernelBasis basis_positive =
    BivariateGaussianKernelBasis(dx, 0.0, sigma,power,std_dev_factor);

    // BivariateGaussianKernelBasis basis_negative =
    // BivariateGaussianKernelBasis(dx, -0.6, sigma,power,std_dev_factor);
  
  for (unsigned tt=1; tt<N; ++tt) {
    observable_datum y_t = ys[tt];
    observable_datum y_tm1 = ys[tt-1];
    
    std::vector<stoch_vol_datum> theta_t_mean =
      theta_next_mean(theta_tm1,
  		      y_t,
  		      y_tm1,
  		      params);

    std::vector<double> lls = log_likelihoods_OCHL(y_t,
						   y_tm1,
						   theta_t_mean,
						   params,
						   &basis_positive,
						   dx,
						   dx_likelihood);
    
    auto compare = [&lls, &log_weights](unsigned i, unsigned j)
      { return lls[i] + log_weights[i] < lls[j] + log_weights[j]; };
    
    auto max_index_ptr = std::max_element(std::begin(particle_indeces),
					  std::end(particle_indeces),
					  compare);

    double probs [N_particles];
    for (unsigned index : particle_indeces) {
      probs[index] = exp( lls[index] + log_weights[index] -
				      (lls[*max_index_ptr] +
				       log_weights[*max_index_ptr]) );
    }
    gsl_ran_discrete_t * particle_sampler = gsl_ran_discrete_preproc(N_particles,
								     probs);
    for (unsigned m=0; m<N_particles; ++m) {
      unsigned k = gsl_ran_discrete(r_ptr, particle_sampler);

      theta_t[m] = sample_theta(theta_tm1[k],
				y_t,
				y_tm1,
				params,
				r_ptr);

      double log_new_weight =
	log_likelihood_OCHL(y_t,
			    y_tm1,
			    theta_t[m],
			    params,
			    &basis_positive,
			    dx,
			    dx_likelihood) -
	lls[k];
	// log_likelihood_OCHL(y_t,
	// 		    y_tm1,
	// 		    theta_t_mean[k],
	// 		    params,
	// 		    &basis_positive,
	// 		    dx,
	// 		    dx_likelihood);


      log_weights[m] = log_new_weight;
    }
    
    auto max_weight_iter_ptr = std::max_element(std::begin(log_weights),
						std::end(log_weights));
    // normalizing weights
    std::transform(log_weights.begin(), log_weights.end(), log_weights.begin(),
		   [&max_weight_iter_ptr, &log_weights](double log_weight)
		   {
		     return log_weight - log_weights[*max_weight_iter_ptr];
		   });

    double sum_of_weights = 0.0;
    for (double log_weight : log_weights) {
      sum_of_weights = sum_of_weights + exp(log_weight);
    }

    std::transform(log_weights.begin(), log_weights.end(), log_weights.begin(),
		   [&sum_of_weights, &log_weights](double log_weight)
		   {
		     return log_weight - log(sum_of_weights);
		   });
    
    theta_tm1 = theta_t;
    gsl_ran_discrete_free(particle_sampler);

    std::vector<double> quantiles = compute_quantiles(theta_t,
						  log_weights);
    for (auto& quantile : quantiles) {
      std::cout << quantile << " ";
      mean_levels << quantile << ",";
    }
    std::cout << tt << std::endl;
    mean_levels << "\n";
    
  }
  mean_levels.close();
  
  gsl_rng_free(r_ptr);
  return 0;
}
