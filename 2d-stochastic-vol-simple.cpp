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
#include <omp.h>
#include <vector>



int main(int argc, char *argv[]) {
  if (argc < 9 || argc > 9) {
    printf("You must provide input\n");
    printf("The input is: \n data file list (each file on new line); \noutput directory;\nrelative tolerance for function during mle estimation (as double); \ninitial guess for sigma_x; \ninitial guess for sigma_y; \ninitial guess for rho; \nfile name prefix; \nfile name suffix; \n");
    printf("file names will be PREFIXmle-results-NUMBER_DATA_SET-order-ORDERSUFFIX.csv, stored in the output diectory.\n");
    exit(0);
  }

  omp_set_dynamic(0);

  static int counter = 0;
#pragma omp threadprivate(counter)

  static BivariateGaussianKernelBasis* private_bases;
#pragma omp threadprivate(private_bases)
  
  unsigned n_threads = 3;
  
  long unsigned T = 1 * 100 * 6.5 * 3600 * 1000; // number days in ms
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
  // params.alpha_x = 0;
  // params.alpha_y = 0;

  params.theta_x = exp(-theta_hat * 1.0*Delta);
  params.theta_y = exp(-theta_hat * 1.0*Delta);

  params.tau_x = tau_hat * sqrt((1 - exp(-2*theta_hat*Delta))/(2*theta_hat));
  params.tau_y = tau_hat * sqrt((1 - exp(-2*theta_hat*Delta))/(2*theta_hat));
  params.tau_rho = 0.1;

  params.leverage_x_rho = 0;
  params.leverage_y_rho = 0;

  unsigned N = T/Delta;
  std::vector<observable_datum> ys (N);
  std::vector<stoch_vol_datum> thetas (N);
  
  generate_data(ys,
		thetas,
		params,
		6.5*3600*1);

  unsigned N_particles = 1000;
  std::vector<double> log_weights (N_particles);
  for (unsigned i=0; i<N_particles; ++i) {
    log_weights[i] = 0.0;
  }

  const gsl_rng_type * Type;
  gsl_rng_env_setup();
  Type = gsl_rng_default;
  gsl_rng * r_ptr = gsl_rng_alloc(Type);

  long unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  seed = 10;
  gsl_rng_set(r_ptr, seed);
  
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

  double dx = 1.0/300.0;
  double dx_likelihood = 1.0/32.0;
  double rho_basis = 0.6;
  double sigma_x = 0.3;
  double sigma_y = 0.1;
  double power = 1.0;
  double std_dev_factor = 1.0;
  
  BivariateGaussianKernelBasis basis_positive =
    BivariateGaussianKernelBasis(dx,
				 rho_basis,
				 sigma_x,
				 sigma_y,
				 power,
				 std_dev_factor);

  // BASES COPY FOR THREADS START
  int tid = 0;
  unsigned i = 0;
  
  std::cout << "copying bases vectors for threads as private variables" << std::endl;
  omp_set_num_threads(n_threads);
  auto t1 = std::chrono::high_resolution_clock::now();

  #pragma omp parallel default(none) private(tid, i) shared(basis_positive)
  {
    tid = omp_get_thread_num();
    
    private_bases = new BivariateGaussianKernelBasis();
    (*private_bases) = basis_positive;

    printf("Thread %d: counter %d\n", tid, counter);
  }
  auto t2 = std::chrono::high_resolution_clock::now();    
  std::cout << "OMP duration = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds\n";
  std::cout << "DONE copying bases vectors for threads as private variables" << std::endl;
  std::cout << std::endl;
  // BAsES COPY FOR THREADS END

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
    
    std::vector<double> lls (N_particles);
    
    t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(none) private(i) shared(lls, theta_t, N_particles) firstprivate(y_t, y_tm1, params, dx, dx_likelihood)
    {
#pragma omp for
      for (i=0; i<N_particles; ++i) {
	stoch_vol_datum thetum = theta_t[i];
	double likelihood = log_likelihood_OCHL(y_t,
						y_tm1,
						thetum,
						params,
						private_bases,
						dx,
						dx_likelihood);
	lls[i] = likelihood;
	printf("Thread %d with address %p produces likelihood %f where &params=%p\n",
	       omp_get_thread_num(),
	       private_bases,
	       likelihood,
	       &params);
      }
    }
    t2 = std::chrono::high_resolution_clock::now();    
    std::cout << "OMP duration = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	      << " milliseconds" << std::endl;
    
    for (unsigned i=0; i<lls.size(); ++i) {
      std::cout << "lls[" << i << "] = " << lls[i] << std::endl;
    }
    
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
    std::vector<unsigned> ks (N_particles);
    for (unsigned i=0; i<N_particles; ++i) {
      ks[i] = gsl_ran_discrete(r_ptr, particle_sampler);
    }
    
    
    t1 = std::chrono::high_resolution_clock::now();
    unsigned m=0;
#pragma omp parallel default(none) private(m) shared(lls, theta_t, theta_tm1, N_particles, ks, r_ptr, log_weights) firstprivate(y_t, y_tm1, params, dx, dx_likelihood)
    {
#pragma omp for
      for (m=0; m<N_particles; ++m) {

	unsigned k = ks[m];
      
	theta_t[m] = sample_theta(theta_tm1[k],
				  y_t,
				  y_tm1,
				  params,
				  r_ptr);
	
	double log_new_weight = 0.0;
	if (std::abs(lls[k] - log(1e-16)) <= 1e-16) {
	  log_new_weight = log(1e-32);
	} else {
	  double ll_for_sample = log_likelihood_OCHL(y_t,
						     y_tm1,
						     theta_t[m],
						     params,
						     private_bases,
						     dx,
						     dx_likelihood);
	  log_new_weight =
	    ll_for_sample - 
	    lls[k];
	}
	
	printf("on likelihood %d\n",k);
	log_weights[m] = log_new_weight;
      }
    }
    t2 = std::chrono::high_resolution_clock::now();    
    std::cout << "OMP duration = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	      << " milliseconds" << std::endl;
    
    
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
