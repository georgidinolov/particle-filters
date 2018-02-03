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
  if (argc < 6 || argc > 6) {
    printf("You must provide input\n");
    printf("The input is: \n\noutput file prefix;\nnumber particles to be used; \nnumber data points; \nnumber points for interpolator; \nfile name for interpolator; \n");
    printf("It is wise to include the parameter values in the file name. We are using a fixed seed for random number generation.\n");
    exit(0);
  }

  std::string output_file_PREFIX = argv[1];
  unsigned N_particles = std::stoi(argv[2]);
  int number_data_points = std::stod(argv[3]);
  int number_points_for_interpolator = std::stod(argv[4]);
  std::string input_file_name = argv[5];

  std::string output_file = output_file_PREFIX +  ".csv";

  std::cout << "output_file = " << output_file << std::endl;

  omp_set_dynamic(0);
  omp_set_num_threads(30);

  double sigma_2=0.00291518;
  double phi=-1.38676;
  double nu=5;
  double tau_2=5.59013;
  std::vector<double> L = {3.17564,
			   2.85297, 4.22303,
			   3.61173, 4.51855, 13.9496,
			   -1.0784, 4.66776, 13.1338, 6.66533,
			   8.85899, 1.16777, -5.07872, 16.8475, 12.4105,
			   -7.05084, 2.30011, 4.42575, -1.69786, 6.34489, 0.222189,
			   4.85171, 2.84438, -0.105236, -0.875338, 0.803392, 0.906027, 2.41978};

  std::ifstream input_file(input_file_name);
  std::vector<likelihood_point> points_for_kriging =
    std::vector<likelihood_point> (number_points_for_interpolator);
  std::vector<likelihood_point> points_for_interpolation = std::vector<likelihood_point> (1);

  if (input_file.is_open()) {
    for (unsigned i=0; i<points_for_kriging.size(); ++i) {
      input_file >> points_for_kriging[i];
      points_for_kriging[i].likelihood = log(points_for_kriging[i].likelihood);
    }
  }

  parameters_nominal params_for_GP_prior = parameters_nominal();
  params_for_GP_prior.sigma_2 = sigma_2;
  params_for_GP_prior.phi = phi;
  params_for_GP_prior.nu = nu;
  params_for_GP_prior.tau_2 = tau_2;
  params_for_GP_prior.lower_triag_mat_as_vec = L;

  GaussianInterpolator GP_prior = GaussianInterpolator(points_for_interpolation,
						       points_for_kriging,
						       params_for_GP_prior);

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
  seed = 10;

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

  std::vector<parameters> prior_samples = sample_parameters_prior(priors_x,
								  priors_y,
								  priors_rho,
								  prior_rho_leverage,
								  10000,
								  r_ptr);
  gsl_vector * prior_mean = compute_parameters_mean(prior_samples);
  gsl_matrix * prior_cov = compute_parameters_cov(prior_mean,
						  prior_samples);
  std::vector<NormalInverseWishartParameters> NIWkernels_tm1 (N_particles);
  for (unsigned i=0; i<N_particles; ++i) {
    gsl_vector_memcpy( NIWkernels_tm1[i].mu_not, prior_mean );
    gsl_matrix_memcpy( NIWkernels_tm1[i].inverse_scale_mat, prior_cov );
  }
  gsl_vector_free(prior_mean);
  gsl_matrix_free(prior_cov);

  std::vector<NormalInverseWishartParameters> NIWkernels_t (N_particles);
  NIWkernels_t = NIWkernels_tm1;

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

//   // BASES COPY FOR THREADS START
//   double dx = 1.0/500.0;
//   double power = 1.0;
//   double std_dev_factor = 1.0;

//   double sigma_x = 0.5;
//   double sigma_y = 0.2;
//   double rho_basis = 0.8;

//   std::cout << "creating basis" << std::endl;
//   BivariateGaussianKernelBasis basis_positive =
//     BivariateGaussianKernelBasis(dx,
// 				 rho_basis,
// 				 sigma_x,
// 				 sigma_y,
// 				 power,
// 				 std_dev_factor);

//   int tid = 0;
//   unsigned i = 0;

//   std::cout << "copying bases vectors for threads as private variables" << std::endl;
// #pragma omp parallel default(none) private(tid, i) shared(basis_positive)
//   {
//     tid = omp_get_thread_num();

//     private_bases = new BivariateGaussianKernelBasis();
//     (*private_bases) = basis_positive;

//     printf("Thread %d: counter %d\n", tid, counter);
//   }
//   // BASES COPY FOR THREADS END
  
  for (unsigned tt=1; tt<N; ++tt) {
    observable_datum y_t = ys[tt];
    observable_datum y_tm1 = ys[tt-1];

    std::vector<parameters> params_t_mean (params_tm1.size());
    for (unsigned ii = 0; ii<params_t_mean.size(); ++ii) {
      const gsl_vector * current_param_mean = NIWkernels_tm1[ii].mu_not;
      params_t_mean[ii] = reals_to_parameters(current_param_mean);
    }

    std::vector<stoch_vol_datum> theta_t_mean =
      theta_next_mean(theta_tm1,
  		      y_t,
  		      y_tm1,
  		      params_t_mean);

    std::vector<double> lls (N_particles);

    auto t1 = std::chrono::high_resolution_clock::now();
    unsigned i=0;
#pragma omp parallel default(none) private(i) shared(GP_prior, lls, theta_t_mean, params_t_mean, N_particles) firstprivate(y_t, y_tm1, params)
    {
#pragma omp for
      for (i=0; i<N_particles; ++i) {
	likelihood_point lp = log_likelihood_OCHL(y_t,
						  y_tm1,
						  theta_t_mean[i],
						  params_t_mean[i],
						  GP_prior);
	lls[i] = lp.likelihood;
	double variance = GP_prior.prediction_variance(lp);
	printf("Thread %d with address ' ' produces likelihood %f where &params=%p with uncertainty %f\n",
	       omp_get_thread_num(),
	       lp.likelihood,
	       &params,
	       sqrt(variance));
      }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
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
    std::vector<likelihood_point> lps (N_particles);
    
    // SAMPLING PARAMTERS AND VOLATILITIES START
    t1 = std::chrono::high_resolution_clock::now();
    unsigned m=0;
#pragma omp parallel default(none) private(m) shared(lls, N_particles, ks, r_ptr, log_weights, theta_tm1, theta_t, params_tm1, params_t, params_t_mean, NIWkernels_tm1, NIWkernels_t, GP_prior, lps) firstprivate(y_t, y_tm1, params)
    {
#pragma omp for
      for (m=0; m<N_particles; ++m) {

	unsigned k = ks[m];

	MultivariateNormal mvnorm = MultivariateNormal();

	NormalInverseWishartParameters NIWcurrent =
	  NormalInverseWishartParameters();
	NIWcurrent = NIWkernels_tm1[k];
	// Sample Epsilon
	double sample_epsilon_array [13*13];
	gsl_matrix_view sample_epsilon_view =
	  gsl_matrix_view_array(sample_epsilon_array,
				13, 13);
	gsl_matrix * sample_epsilon = &sample_epsilon_view.matrix;
	mvnorm.rinvwishart(r_ptr,
			   NIWcurrent.dimension,
			   NIWcurrent.deg_freedom,
			   NIWcurrent.inverse_scale_mat,
			   sample_epsilon);
	gsl_matrix_scale(sample_epsilon, 1.0/NIWcurrent.lambda);

	double sample_mu_array [13];
	gsl_vector_view sample_mu_view = gsl_vector_view_array(sample_mu_array,
							       13);
	gsl_vector * sample_mu = &sample_mu_view.vector;
	mvnorm.rmvnorm(r_ptr,
		       NIWcurrent.dimension,
		       NIWcurrent.mu_not,
		       sample_epsilon,
		       sample_mu);

	bool sample_within_bounds = check_parameter_bounds(sample_mu);
	long unsigned counter = 0;
	while (!sample_within_bounds && counter < 100) {
	  mvnorm.rmvnorm(r_ptr,
			 NIWcurrent.dimension,
			 NIWcurrent.mu_not,
			 sample_epsilon,
			 sample_mu);
	  counter++;
	  sample_within_bounds = check_parameter_bounds(sample_mu);
	}

	gsl_vector * params_t_sample_gsl = gsl_vector_alloc(13);
	gsl_matrix_scale(sample_epsilon, NIWcurrent.lambda);

	mvnorm.rmvnorm(r_ptr,
		       NIWcurrent.dimension,
		       sample_mu,
		       sample_epsilon,
		       params_t_sample_gsl);

	sample_within_bounds = check_parameter_bounds(params_t_sample_gsl);
	counter = 0;
	while (!sample_within_bounds && counter < 100) {
	  mvnorm.rmvnorm(r_ptr,
			 NIWcurrent.dimension,
			 sample_mu,
			 sample_epsilon,
			 params_t_sample_gsl);
	  counter++;
	  sample_within_bounds = check_parameter_bounds(params_t_sample_gsl);
	}

	parameters params_t_sample = reals_to_parameters(params_t_sample_gsl);

	params_t[m] = params_t_sample;
	theta_t[m] = sample_theta(theta_tm1[k],
				  y_t,
				  y_tm1,
				  params_t_sample,
				  r_ptr);
	
	// mu update
	gsl_vector * mu_not_update = gsl_vector_alloc(13);
	gsl_vector_memcpy(mu_not_update, NIWcurrent.mu_not);
	gsl_vector_scale(mu_not_update, NIWcurrent.lambda);
	gsl_vector_scale(params_t_sample_gsl, 1.0);
	gsl_vector_add(mu_not_update, params_t_sample_gsl);
	gsl_vector_scale(mu_not_update, 1.0/(NIWcurrent.lambda + 1.0));
	gsl_vector_free(NIWcurrent.mu_not);
	NIWcurrent.mu_not = mu_not_update;
	// inverse_scale_matrix update
	gsl_matrix* inverse_scale_matrix_update = gsl_matrix_alloc(13,13);
	for (unsigned ii=0; ii<13; ++ii) {
	  for (unsigned jj=0; jj<13; ++jj) {
	    double entry =
	      gsl_matrix_get(NIWcurrent.inverse_scale_mat, ii,jj) +
	      (NIWcurrent.lambda*1.0)/
			   (NIWcurrent.lambda + 1.0) *
	      (gsl_vector_get(params_t_sample_gsl, ii) - gsl_vector_get(NIWcurrent.mu_not, ii))*
	      (gsl_vector_get(params_t_sample_gsl, jj) - gsl_vector_get(NIWcurrent.mu_not, jj));

	    gsl_matrix_set(inverse_scale_matrix_update,
			   ii, jj, entry);


	  }
	}
	gsl_matrix_free(NIWcurrent.inverse_scale_mat);
	NIWcurrent.inverse_scale_mat = inverse_scale_matrix_update;
	// lambda update
	NIWcurrent.lambda = NIWcurrent.lambda + 1.0;
	NIWcurrent.deg_freedom = NIWcurrent.deg_freedom + 1.0;
	NIWkernels_t[m] = NIWcurrent;

	gsl_vector_free(params_t_sample_gsl);

	double log_new_weight = 0.0;
	if (std::abs(lls[k] - log(1e-16)) <= 1e-16) {
	  log_new_weight = log(1e-32);
	} else {
	  likelihood_point lp_for_sample = log_likelihood_OCHL(y_t,
							       y_tm1,
							       theta_t[m],
							       params_t[m],
							       GP_prior);
	  log_new_weight =
	    lp_for_sample.likelihood -
	    lls[k];

	  lps[m] = lp_for_sample;

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

      std::ofstream lps_for_out;
      lps_for_out.open("likelihood_points_from_filter.csv");
      for (m=0; m<N_particles; ++m) {
	lps_for_out << lps[m];
      }
      lps_for_out.close();

    }
    t2 = std::chrono::high_resolution_clock::now();
    std::cout << "OMP duration = " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
	      << " milliseconds" << std::endl;
    // SAMPLING PARAMTERS AND VOLATILITIES END
    

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
    NIWkernels_tm1 = NIWkernels_t;

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
