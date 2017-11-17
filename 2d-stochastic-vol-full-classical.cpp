#include <algorithm>
#include <chrono>
#include <cmath>
#include "DataTypes.hpp"
#include <fstream>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdio.h>
#include <vector>



int main(int argc, char *argv[]) {
  if (argc < 9 || argc > 9) {
    printf("You must provide input\n");
    printf("The input is: \n\noutput file prefix;\nnumber particles to be used; \ndx_likelihood; \nrho_basis; \nsigma_x for basis; \nsigma_y for basis; \nnumber data points; \noutput file suffix \n");
    printf("It is wise to include the parameter values in the file name. We are using a fixed random seed.\n");
    exit(0);
  }

  std::string output_file_PREFIX = argv[1];
  unsigned N_particles = std::stoi(argv[2]);
  double dx_likelihood = std::stod(argv[3]);
  double rho_basis = std::stod(argv[4]);
  double sigma_x = std::stod(argv[5]);
  double sigma_y = std::stod(argv[6]);
  int number_data_points = std::stod(argv[7]);

  std::string output_file = output_file_PREFIX +
    "-sigma_x-" + argv[5] +
    "-sigma_y-" + argv[6] +
    "-rho-" + argv[4] +
    "-dx-likelihood-" + std::to_string(dx_likelihood) +
    "-nparticles-" + argv[2] + "-" +
    argv[8] + ".csv";

  std::cout << "output_file = " << output_file << std::endl;

  omp_set_dynamic(0);
  omp_set_num_threads(40);

  static int counter = 0;
#pragma omp threadprivate(counter)

  static BivariateGaussianKernelBasis* private_bases;
#pragma omp threadprivate(private_bases)

  long unsigned T = 1 * number_data_points * 6.5 * 3600 * 1000; // number days in ms
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

  long unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  // seed = 10;

  unsigned buffer = 0;
  generate_data(ys,
		thetas,
		params,
		6.5*3600*1,
		10,
		buffer,
		false);

  std::vector<double> log_weights (N_particles);
  for (unsigned i=0; i<N_particles; ++i) {
    log_weights[i] = 0.0;
  }

  const gsl_rng_type * Type;
  gsl_rng_env_setup();
  Type = gsl_rng_default;
  gsl_rng * r_ptr = gsl_rng_alloc(Type);
  gsl_rng_set(r_ptr, seed);


  // PARAMS START
  double mu_hat_std_dev = 1e-11;

  double theta_hat_std_dev = 1e-9;

  double alpha_hat_std_dev = 1;

  double tau_square_hat = 1.3e-9;
  double tau_square_hat_std_dev = 1e-8;

  double rho_mean = 0;
  double rho_std_dev = 0.2;

  double xi_square_mean = 0; // these don't matter
  double xi_square_std_dev = 1;
    StochasticVolatilityPriors priors_x =
      StochasticVolatilityPriors(mu_hat,
				 mu_hat_std_dev,
				 theta_hat,
				 theta_hat_std_dev,
				 alpha_hat,
				 alpha_hat_std_dev,
				 tau_square_hat,
				 tau_square_hat_std_dev,
				 xi_square_mean,
				 xi_square_std_dev,
				 rho_mean,
				 rho_std_dev,
				 Delta);
    StochasticVolatilityPriors priors_y =
      StochasticVolatilityPriors(mu_hat,
				 mu_hat_std_dev,
				 theta_hat,
				 theta_hat_std_dev,
				 alpha_hat,
				 alpha_hat_std_dev,
				 tau_square_hat,
				 tau_square_hat_std_dev,
				 xi_square_mean,
				 xi_square_std_dev,
				 rho_mean,
				 rho_std_dev,
				 Delta);
    StochasticVolatilityPriors priors_rho =
      StochasticVolatilityPriors(mu_hat,
				 mu_hat_std_dev,
				 theta_hat,
				 theta_hat_std_dev,
				 -8.5,
				 1.0,
				 tau_square_hat,
				 tau_square_hat_std_dev,
				 xi_square_mean,
				 xi_square_std_dev,
				 rho_mean,
				 rho_std_dev,
				 Delta);
    RhoPrior prior_rho_leverage = RhoPrior(rho_mean, rho_std_dev);
    // PARAMS END

  std::vector<parameters> params_tm1 =
    sample_parameters_prior(priors_x,
    			    priors_y,
    			    priors_rho,
    			    prior_rho_leverage,
    			    N_particles,
    			    r_ptr);
  std::vector<parameters> params_t = params_tm1;

  std::vector<stoch_vol_datum> theta_tm1 = sample_theta_prior(params,
							      N_particles,
							      r_ptr);
  std::vector<stoch_vol_datum> theta_t = theta_tm1;
  print_params(params);

  std::vector<unsigned> ks = std::vector<unsigned> (N_particles, 1);

  std::vector<unsigned> particle_indeces = std::vector<unsigned> (N_particles);
  std::iota(std::begin(particle_indeces), std::end(particle_indeces), 0);

  std::ofstream mean_levels;
  mean_levels.open(output_file);
  mean_levels << "mean_log_sigma_x, var_log_sigma_x,"
	      << "mean_log_sigma_y, var_log_sigma_y,"
	      << "mean_rho_tilde, var_rho_tilde,"
    //
	      << "mean_mu_x, var_mu_x,"
	      << "mean_mu_y, var_mu_y,"
    //
	      << "mean_alpha_x, var_alpha_x,"
	      << "mean_alpha_y, var_alpha_y,"
	      << "mean_alpha_rho, var_alpha_rho,"
    //
	      << "mean_theta_x_transformed, var_theta_x_transformed,"
	      << "mean_theta_y_transformed, var_theta_y_transformed,"
	      << "mean_theta_rho_transformed, var_theta_rho_transformed,"
    //
	      << "mean_tau_x_transformed, var_tau_x_transformed,"
	      << "mean_tau_y_transformed, var_tau_y_transformed,"
	      << "mean_tau_rho_transformed, var_tau_rho_transformed,"
    //
	      << "mean_leverage_x_rho_transformed, var_leverage_x_rho_transformed,"
	      << "mean_leverage_y_rho_transformed, var_leverage_y_rho_transformed,"
	      << "ess\n";
  mean_levels.close();

  double dx = 1.0/500.0;
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

    double scale_a = 0.9999;

    t1 = std::chrono::high_resolution_clock::now();
    gsl_vector* scaled_mean = compute_parameters_mean(params_tm1,
    						      log_weights);

    gsl_matrix* scaled_cov = compute_parameters_cov(scaled_mean,
    						    params_tm1,
    						    log_weights);
    

    gsl_vector_scale(scaled_mean, (1.0-scale_a));
    gsl_matrix_scale(scaled_cov,  std::pow((1.0-scale_a*scale_a), 2));
    // OUTPUTTING MATRIX START
    FILE * cov_matrix = fopen ("scaled_cov.dat", "wb");
    gsl_matrix_fwrite(cov_matrix, scaled_cov);
    fclose(cov_matrix);
    // OUTPUTTING MATRIX END

    // TO CHECK IF THE COV IS POS DEF START //
    gsl_error_handler_t* old_handler = gsl_set_error_handler_off();
    gsl_matrix* work = gsl_matrix_alloc(13, 13);
    gsl_matrix_memcpy(work, scaled_cov);

    int status = gsl_linalg_cholesky_decomp(work);
    if (status == GSL_EDOM) {
      gsl_vector_view diag_view = gsl_matrix_diagonal(scaled_cov);
      gsl_vector* diag_cpy = gsl_vector_alloc(13);
      gsl_vector_memcpy(diag_cpy, &diag_view.vector);

      gsl_matrix_set_zero(scaled_cov);
      gsl_vector_memcpy(&diag_view.vector, diag_cpy);
      
      // we need to check for non-zero diagonal entries too
      for (unsigned i=0; i<13; ++i) {
	if ( gsl_matrix_get(scaled_cov,i,i) < std::numeric_limits<double>::epsilon() ) {
	  gsl_matrix_set(scaled_cov, i,i,
			 std::pow(gsl_vector_get(scaled_mean,i), 2));
	}
      }
    }
    gsl_set_error_handler(old_handler);
    gsl_matrix_free(work);
    // TO CHECK IF THE COV IS POS DEF END //

    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "\nmean cov computation times = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " milliseconds\n";

    std::vector<stoch_vol_datum> theta_t_mean =
      theta_next_mean(theta_tm1,
  		      y_t,
  		      y_tm1,
  		      params_tm1);

    std::vector<parameters> params_t_mean = 
      parameters_next_mean(params_tm1,
      			   scaled_mean,
      			   scale_a);

    std::vector<double> lls (N_particles);

    t1 = std::chrono::high_resolution_clock::now();
#pragma omp parallel default(none) private(i) shared(lls, theta_t_mean, params_t_mean, N_particles) firstprivate(y_t, y_tm1, params, dx, dx_likelihood)
    {
#pragma omp for
      for (i=0; i<N_particles; ++i) {
	double likelihood = log_likelihood(y_t,
					   y_tm1,
					   theta_t_mean[i],
					   params_t_mean[i]);
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

    // SAMPLING PARAMTERS AND VOLATILITIES START 
    t1 = std::chrono::high_resolution_clock::now();
    unsigned m=0;
#pragma omp parallel default(none) private(m) shared(lls, N_particles, ks, r_ptr, log_weights, theta_tm1, theta_t, params_tm1, params_t, scaled_cov, params_t_mean) firstprivate(y_t, y_tm1, params, dx, dx_likelihood)
    {
#pragma omp for
      for (m=0; m<N_particles; ++m) {

	unsigned k = ks[m];

	gsl_vector * params_t_mean_gsl = parameters_to_reals(params_t_mean[k]);
	MultivariateNormal mvnorm = MultivariateNormal();
	gsl_vector * params_t_sample_gsl = gsl_vector_alloc(13);
	mvnorm.rmvnorm(r_ptr,
		       13,
		       params_t_mean_gsl,
		       scaled_cov,
		       params_t_sample_gsl);
	parameters params_t_sample = reals_to_parameters(params_t_sample_gsl);
	
	params_t[m] = params_t_sample;
	theta_t[m] = sample_theta(theta_tm1[k],
				  y_t,
				  y_tm1,
				  params_t_sample,
				  r_ptr);

	gsl_vector_free(params_t_sample_gsl);
	gsl_vector_free(params_t_mean_gsl);

	double log_new_weight = 0.0;
	if (std::abs(lls[k] - log(1e-16)) <= 1e-16) {
	  log_new_weight = log(1e-32);
	} else {
	  double ll_for_sample = log_likelihood(y_t,
						y_tm1,
						theta_t[m],
						params_t[m]);
	  log_new_weight =
	    ll_for_sample -
	    lls[k];

	  if (std::isinf(log_new_weight) || std::isnan(log_new_weight)) {
	    log_new_weight = -1.0*std::numeric_limits<double>::infinity();
	  }
	}

	printf("on likelihood %d: sigma_x = %f, sigma_y = %f, rho = %f, log_new_weight = %f\n",
	       k,
	       exp(theta_t[m].log_sigma_x),
	       exp(theta_t[m].log_sigma_y),
	       exp(theta_t[m].rho_tilde)/(exp(theta_t[m].rho_tilde) + 1)*2.0-1.0,
	       log_new_weight);
	log_weights[m] = log_new_weight;
      }
    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "OMP duration = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	      << " milliseconds" << std::endl;
    // SAMPLING PARAMTERS AND VOLATILITIES END

    gsl_matrix_free(scaled_cov);
    gsl_vector_free(scaled_mean);

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
    params_tm1 = params_t;
    gsl_ran_discrete_free(particle_sampler);

    std::vector<double> quantiles = compute_quantiles(theta_t,
						      log_weights);
    std::vector<double> structural_quantiles = compute_quantiles(params_t,
								 log_weights);

    mean_levels.open(output_file, std::ios_base::app);
    for (auto& quantile : quantiles) {
      std::cout << quantile << " ";
      mean_levels << quantile << ",";
    }
    for (auto& quantile : structural_quantiles) {
      mean_levels << quantile << ",";
    }

    mean_levels << compute_ESS(log_weights);
    std::cout << tt << " ";
    std::cout << "ess = " << compute_ESS(log_weights) << std::endl;

    mean_levels << "\n";
    mean_levels.close();
  }

  gsl_rng_free(r_ptr);
  return 0;
}
