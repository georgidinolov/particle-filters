#include <algorithm>
#include "DataTypes.hpp"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>

void print_params(const parameters& params) {
  printf("mu_x = %f \nmu_y = %f\n alpha_x = %f\nalpha_y = %f\nalpha_rho = %f\n theta_x = %f\n theta_y = %f\n theta_rho = %f\n",
	 params.mu_x, params.mu_y, params.alpha_x, params.alpha_y, params.alpha_rho, params.theta_x, params.theta_y, params.theta_rho);
}

double logit(double p) {
  return (log(p/(1.0-p)));
}

double logit_inv(double logit_p) {
  double out = exp(logit_p) / (exp(logit_p) + 1);
  return (out);
}

std::vector<double> epsilons_given_theta(const stoch_vol_datum& theta_current,
					 const observable_datum& y_current,
					 const observable_datum& y_current_m1,
					 const parameters& params)
{
  double sigma_x = exp(theta_current.log_sigma_x);
  double sigma_y = exp(theta_current.log_sigma_y);
  double rho = logit_inv(theta_current.rho_tilde)*2 - 1;

  double epsilon_y = (y_current.y_t - y_current_m1.y_t - params.mu_y) /
    sigma_y;

  double epsilon_x = (y_current.x_t -
		      y_current_m1.x_t -
		      params.mu_x -
		      rho*sigma_x*epsilon_y) /
    (sqrt(1.0 - rho*rho) * sigma_x);

  return std::vector<double> {epsilon_x, epsilon_y};
}

std::vector<stoch_vol_datum> sample_theta_prior(const parameters& params,
						unsigned N_particles,
						gsl_rng *r) {

  std::vector<stoch_vol_datum> output (N_particles);
  for (unsigned i=0; i<N_particles; ++i) {
    stoch_vol_datum Theta;
    Theta.log_sigma_x =
      params.alpha_x +
      gsl_ran_gaussian(r, params.tau_x/sqrt(1.0 - params.theta_x*params.theta_x));

    Theta.log_sigma_y =
      params.alpha_y +
      gsl_ran_gaussian(r, params.tau_y/sqrt(1.0 - params.theta_y*params.theta_y));

    Theta.rho_tilde = logit( (0.6 + 1.0)/2.0 ) +
      gsl_ran_gaussian(r, params.tau_rho);

    output[i] = Theta;
  }

  return output;
}

std::vector<stoch_vol_datum> sample_theta_prior(const std::vector<parameters>& parameters_vector,
						unsigned N_particles,
						gsl_rng *r) {

  std::vector<stoch_vol_datum> output (N_particles);
  for (unsigned i=0; i<N_particles; ++i) {

    parameters params = parameters_vector[i];

    stoch_vol_datum Theta;
    Theta.log_sigma_x =
      params.alpha_x +
      gsl_ran_gaussian(r, params.tau_x/sqrt(1.0 - params.theta_x*params.theta_x));

    Theta.log_sigma_y =
      params.alpha_y +
      gsl_ran_gaussian(r, params.tau_y/sqrt(1.0 - params.theta_y*params.theta_y));

    Theta.rho_tilde = logit( (0.6 + 1.0)/2.0 ) +
      gsl_ran_gaussian(r, params.tau_rho);

    output[i] = Theta;
  }

  return output;
}

std::vector<parameters> sample_parameters_prior(const StochasticVolatilityPriors& prior_x,
						const StochasticVolatilityPriors& prior_y,
						const StochasticVolatilityPriors& prior_rho,
						const RhoPrior& prior_rho_leverage,
						unsigned N_particles,
						gsl_rng *r)
{
  std::vector<parameters> out = std::vector<parameters> (N_particles);

  for (unsigned i=0; i<N_particles; ++i) {
    out[i].mu_x = prior_x.get_mu_prior().get_mu_mean() +
      gsl_ran_gaussian(r, prior_x.get_mu_prior().get_mu_std_dev());

    out[i].mu_y = prior_y.get_mu_prior().get_mu_mean() +
      gsl_ran_gaussian(r, prior_y.get_mu_prior().get_mu_std_dev());
    //
    out[i].alpha_x = prior_x.get_alpha_prior().get_alpha_mean() +
      gsl_ran_gaussian(r, prior_x.get_alpha_prior().get_alpha_std_dev());
    out[i].alpha_y = prior_y.get_alpha_prior().get_alpha_mean() +
      gsl_ran_gaussian(r, prior_y.get_alpha_prior().get_alpha_std_dev());
    out[i].alpha_rho = prior_rho.get_alpha_prior().get_alpha_mean() +
      gsl_ran_gaussian(r, prior_rho.get_alpha_prior().get_alpha_std_dev());
    //
    out[i].theta_x = 0.8 +
      gsl_ran_gaussian(r, 0.2);
		       //		       prior_x.get_theta_prior().get_theta_std_dev());

    out[i].theta_y = gsl_ran_gamma(r,
				   prior_y.get_theta_prior().get_theta_shape(),
				   prior_y.get_theta_prior().get_theta_scale());
    out[i].theta_rho = gsl_ran_gamma(r,
				     prior_rho.get_theta_prior().get_theta_shape(),
				     prior_rho.get_theta_prior().get_theta_scale());
    //
    out[i].tau_x = std::sqrt(gsl_ran_gamma(r,
					   prior_x.get_tau_square_prior().get_tau_square_shape(),
					   prior_x.get_tau_square_prior().get_tau_square_scale()));

    out[i].tau_y = std::sqrt(gsl_ran_gamma(r,
					   prior_y.get_tau_square_prior().get_tau_square_shape(),
					   prior_y.get_tau_square_prior().get_tau_square_scale()));
    out[i].tau_rho = std::sqrt(gsl_ran_gamma(r,
					     prior_rho.get_tau_square_prior().get_tau_square_shape(),
					     prior_rho.get_tau_square_prior().get_tau_square_scale()));
    //
    out[i].leverage_x_rho = prior_rho_leverage.get_rho_mean() +
      gsl_ran_gaussian(r, prior_rho_leverage.get_rho_std_dev());
    while ( std::abs(out[i].leverage_x_rho) >= 1.0 ) {
      out[i].leverage_x_rho = prior_rho_leverage.get_rho_mean() +
	gsl_ran_gaussian(r, prior_rho_leverage.get_rho_std_dev());
    }

    out[i].leverage_y_rho = prior_rho_leverage.get_rho_mean() +
      gsl_ran_gaussian(r, prior_rho_leverage.get_rho_std_dev());
    while ( std::abs(out[i].leverage_y_rho) >= 1.0 ) {
      out[i].leverage_y_rho = prior_rho_leverage.get_rho_mean() +
	gsl_ran_gaussian(r, prior_rho_leverage.get_rho_std_dev());
    }
  }

  return out;
}

std::vector<stoch_vol_datum> theta_next_mean(const std::vector<stoch_vol_datum>& theta_current,
					     const observable_datum& y_current,
					     const observable_datum& y_current_tm1,
					     const parameters& params) {

  std::vector<stoch_vol_datum> out =
    std::vector<stoch_vol_datum> (theta_current.size());

  for (unsigned i=0; i<theta_current.size(); ++i) {
    std::vector<double> epsilons =
      epsilons_given_theta(theta_current[i],
  			   y_current,
  			   y_current_tm1,
  			   params);

    std::vector<double> eta_means {
      params.leverage_x_rho*epsilons[0],
  	params.leverage_y_rho*epsilons[1],
  	0.0};

    out[i].log_sigma_x =
      params.alpha_x +
      params.theta_x*(theta_current[i].log_sigma_x - params.alpha_x) +
      params.tau_x*eta_means[0];

    out[i].log_sigma_y =
      params.alpha_y +
      params.theta_y*(theta_current[i].log_sigma_y - params.alpha_y) +
      params.tau_y*eta_means[1];

    out[i].rho_tilde =
      theta_current[i].rho_tilde + params.tau_rho*eta_means[2];
  }

  return out;
}
std::vector<stoch_vol_datum> theta_next_mean(const std::vector<stoch_vol_datum>& theta_current,
					     const observable_datum& y_current,
					     const observable_datum& y_current_tm1,
					     const std::vector<parameters>& params) {

  std::vector<stoch_vol_datum> out =
    std::vector<stoch_vol_datum> (theta_current.size());

  for (unsigned i=0; i<theta_current.size(); ++i) {
    std::vector<double> epsilons =
      epsilons_given_theta(theta_current[i],
  			   y_current,
  			   y_current_tm1,
  			   params[i]);

    std::vector<double> eta_means {
      params[i].leverage_x_rho*epsilons[0],
  	params[i].leverage_y_rho*epsilons[1],
  	0.0};

    out[i].log_sigma_x =
      params[i].alpha_x +
      params[i].theta_x*(theta_current[i].log_sigma_x - params[i].alpha_x) +
      params[i].tau_x*eta_means[0];

    out[i].log_sigma_y =
      params[i].alpha_y +
      params[i].theta_y*(theta_current[i].log_sigma_y - params[i].alpha_y) +
      params[i].tau_y*eta_means[1];

    out[i].rho_tilde =
      theta_current[i].rho_tilde + params[i].tau_rho*eta_means[2];
  }

  return out;
}

std::vector<parameters> parameters_next_mean(const std::vector<parameters>& parameters_current,
					     const gsl_vector* scaled_mean,
					     double scale_a)
{
  std::vector<parameters> out = std::vector<parameters> (parameters_current.size());

  for (unsigned i=0; i<parameters_current.size(); ++i) {

    const parameters& current_parameter = parameters_current[i];
    gsl_vector* current_parameter_real = parameters_to_reals(current_parameter);

    gsl_vector_scale(current_parameter_real, scale_a);
    std::cout << "alpha_x_current = " << gsl_vector_get(current_parameter_real, 2) << " ";
    std::cout << "alpha_x_scaled_mean = " << gsl_vector_get(scaled_mean, 2) << " ";

    gsl_vector_add(current_parameter_real, scaled_mean);

    out[i] = reals_to_parameters(current_parameter_real);

    gsl_vector_free(current_parameter_real);
  }

  return out;
}

std::vector<double> log_likelihoods(const observable_datum& y_t,
				    const observable_datum& y_tm1,
				    const std::vector<stoch_vol_datum>& theta_t,
				    const parameters& params)
{

  double mean [2] = {y_tm1.x_t + params.mu_x,
		     y_tm1.y_t + params.mu_y};
  double cov [4] = {1.0, 0.0, 0.0, 1.0};
  double x [2] = {y_t.x_t, y_t.y_t};

  gsl_vector_view gsl_mean = gsl_vector_view_array(mean, 2);
  gsl_matrix_view gsl_cov = gsl_matrix_view_array(cov, 2, 2);
  gsl_vector_view gsl_x = gsl_vector_view_array(x, 2);
  MultivariateNormal bivariate_gaussian = MultivariateNormal();

  std::vector<double> out = std::vector<double> (theta_t.size());


  for (unsigned i=0; i<theta_t.size(); ++i) {
    double sigma_x = exp(theta_t[i].log_sigma_x);
    double sigma_y = exp(theta_t[i].log_sigma_y);
    double rho = logit_inv(theta_t[i].rho_tilde)*2-1;

    cov[0] = sigma_x*sigma_x;
    cov[1] = rho*sigma_x*sigma_y;
    cov[2] = rho*sigma_x*sigma_y;
    cov[3] = sigma_y*sigma_y;

    out[i] = bivariate_gaussian.dmvnorm_log(2,
					    &gsl_x.vector,
					    &gsl_mean.vector,
					    &gsl_cov.matrix);
  }

  return out;
}

double log_likelihood(const observable_datum& y_t,
		      const observable_datum& y_tm1,
		      const stoch_vol_datum& theta_t,
		      const parameters& params)
{

  double mean [2] = {y_tm1.x_t + params.mu_x,
		     y_tm1.y_t + params.mu_y};
  double cov [4] = {1.0, 0.0, 0.0, 1.0};
  double x [2] = {y_t.x_t, y_t.y_t};

  gsl_vector_view gsl_mean = gsl_vector_view_array(mean, 2);
  gsl_matrix_view gsl_cov = gsl_matrix_view_array(cov, 2, 2);
  gsl_vector_view gsl_x = gsl_vector_view_array(x, 2);
  MultivariateNormal bivariate_gaussian = MultivariateNormal();

  double sigma_x = exp(theta_t.log_sigma_x);
  double sigma_y = exp(theta_t.log_sigma_y);
  double rho = logit_inv(theta_t.rho_tilde)*2.0-1.0;

  cov[0] = sigma_x*sigma_x;
  cov[1] = rho*sigma_x*sigma_y;
  cov[2] = rho*sigma_x*sigma_y;
  cov[3] = sigma_y*sigma_y;

  double out = bivariate_gaussian.dmvnorm_log(2,
					      &gsl_x.vector,
					      &gsl_mean.vector,
					      &gsl_cov.matrix);
  return out;

}

double log_likelihood(const stoch_vol_datum& theta_t,
		      const stoch_vol_datum& theta_tm1,
		      const parameters& params)
{

  // x likelihood
  double point = (theta_t.log_sigma_x - params.alpha_x) -
    params.theta_x*(theta_tm1.log_sigma_x - params.alpha_x);
  double likelihood_x = log(gsl_ran_gaussian_pdf(point, params.tau_x));

  // y likelihood
  point = (theta_t.log_sigma_y - params.alpha_y) -
    params.theta_y*(theta_tm1.log_sigma_y - params.alpha_y);
  double likelihood_y = log(gsl_ran_gaussian_pdf(point, params.tau_y));

  // rho likelihood
  point = (theta_t.rho_tilde - params.alpha_rho) -
    params.theta_rho*(theta_tm1.rho_tilde - params.alpha_rho);
  double likelihood_rho = log(gsl_ran_gaussian_pdf(point, params.tau_rho));

  double out = likelihood_x;
  return out;
}

std::vector<double> log_likelihoods_OCHL(const observable_datum& y_t,
					 const observable_datum& y_tm1,
					 const std::vector<stoch_vol_datum>& theta_t,
					 const parameters& params,
					 BivariateGaussianKernelBasis * basis,
					 double dx,
					 double dx_likelihood)
{
  double x [2] = {y_t.x_t, y_t.y_t};
  gsl_vector_view gsl_x = gsl_vector_view_array(x, 2);


  std::vector<double> out = std::vector<double> (theta_t.size());

  for (unsigned i=0; i<theta_t.size(); ++i) {
    double sigma_x = exp(theta_t[i].log_sigma_x);
    double sigma_y = exp(theta_t[i].log_sigma_y);
    double rho = logit_inv(theta_t[i].rho_tilde)*2-1;

    double Lx = y_t.b_x - y_t.a_x;
    double Ly = y_t.b_y - y_t.a_y;

    BivariateSolver solver = BivariateSolver(basis,
					     sigma_x/Lx,
					     sigma_y/Ly,
					     rho,
					     y_t.a_x/Lx,
					     y_t.x_tm1/Lx,
					     y_t.b_x/Lx,
					     y_t.a_y/Ly,
					     y_t.y_tm1/Ly,
					     y_t.b_y/Ly,
					     1.0,
					     dx);

    x[0] = y_t.x_t / Lx;
    x[1] = y_t.y_t / Ly;

    double likelihood = solver.numerical_likelihood_extended(&gsl_x.vector,
							     dx_likelihood);
    likelihood = likelihood/( std::pow(Lx, 3) * std::pow(Ly, 3) );

    std::cout << theta_t[i].log_sigma_x << " "
	      << theta_t[i].log_sigma_y << " "
	      << theta_t[i].rho_tilde << std::endl;

    if (!std::signbit(likelihood)) {
      out[i] = log(likelihood);
      printf("\nFor rho=%f, data %d produces likelihood %f.\n", rho, i, out[i]);
      printf("sigma_x=%f; sigma_y=%f; rho=%f; ax=%f; x_T=%f; bx=%f; ay=%f; y_T=%f; by=%f;\n",
	     sigma_x,
	     sigma_y,
	     rho,
	     y_t.a_x - y_t.x_tm1,
	     y_t.x_t - y_t.x_tm1,
	     y_t.b_x - y_t.x_tm1,
	     y_t.a_y - y_t.y_tm1,
	     y_t.y_t - y_t.y_tm1,
	     y_t.b_y - y_t.y_tm1);
    } else {
      out[i] = log(1e-16);
      printf("\nFor rho=%f, data %d produces neg likelihood.\n", rho, i);
      printf("sigma_x=%f; sigma_y=%f; rho=%f; ax=%f; x_T=%f; bx=%f; ay=%f; y_T=%f; by=%f;\n",
	     sigma_x,
	     sigma_y,
	     rho,
	     y_t.a_x - y_t.x_tm1,
	     y_t.x_t - y_t.x_tm1,
	     y_t.b_x - y_t.x_tm1,
	     y_t.a_y - y_t.y_tm1,
	     y_t.y_t - y_t.y_tm1,
	     y_t.b_y - y_t.y_tm1);
    }
  }

  return out;

}

double log_likelihood_OCHL(const observable_datum& y_t,
			   const observable_datum& y_tm1,
			   const stoch_vol_datum& theta_t,
			   const parameters& params,
			   BivariateGaussianKernelBasis * basis,
			   double dx,
			   double dx_likelihood)
{
  double x [2] = {y_t.x_t, y_t.y_t};
  gsl_vector_view gsl_x = gsl_vector_view_array(x, 2);

  double out = 0.0;

  double sigma_x = exp(theta_t.log_sigma_x);
  double sigma_y = exp(theta_t.log_sigma_y);
  double rho = logit_inv(theta_t.rho_tilde)*2-1;

  double Lx = y_t.b_x - y_t.a_x;
  double Ly = y_t.b_y - y_t.a_y;
  double likelihood = 0;

  if (!std::signbit(rho)) {
    BivariateSolver solver = BivariateSolver(basis,
					     sigma_x/Lx,
					     sigma_y/Ly,
					     rho,
					     y_t.a_x/Lx,
					     y_t.x_tm1/Lx,
					     y_t.b_x/Lx,
					     y_t.a_y/Ly,
					     y_t.y_tm1/Ly,
					     y_t.b_y/Ly,
					     1.0,
					     dx);

    x[0] = y_t.x_t/Lx;
    x[1] = y_t.y_t/Ly;

    likelihood = solver.numerical_likelihood_extended(&gsl_x.vector,
						      dx_likelihood);
  } else {
    BivariateSolver solver = BivariateSolver(basis,
					     sigma_x/Lx,
					     sigma_y/Ly,
					     -rho,
					     -y_t.b_x/Lx,
					     -y_t.x_tm1/Lx,
					     -y_t.a_x/Lx,
					     y_t.a_y/Ly,
					     y_t.y_tm1/Ly,
					     y_t.b_y/Ly,
					     1.0,
					     dx);

    x[0] = -y_t.x_t/Lx;
    x[1] = y_t.y_t/Ly;

    likelihood = solver.numerical_likelihood_extended(&gsl_x.vector,
						      dx_likelihood);
  }

  if ( (likelihood - 20.0) > std::numeric_limits<double>::epsilon() ) {
    double tau_x = 0;
    double tau_y = 0;

    if ( std::signbit(sigma_x/Lx-sigma_y/Ly)) {
      tau_x = sigma_y/Ly;
      tau_y = sigma_x/Lx;
    } else {
      tau_x = sigma_x/Lx;
      tau_y = sigma_y/Ly;
    }

    double t_tilde = 1.0*tau_x*tau_x;
    double sigma_y_tilde = tau_y/tau_x;

    printf("\nLikelihood for point abnormally large: %f, t_tilde = %f, sigma_y_tilde = %f \n",
	   likelihood, t_tilde, sigma_y_tilde);
  }

  likelihood = likelihood / (std::pow(Lx,3) * std::pow(Ly,3));
  if (!std::signbit(likelihood)) {
    // printf("\nFor rho=%f, data produces likelihood %f.\n", rho, likelihood);
    // printf("sigma_x=%f; sigma_y=%f; rho=%f; ax=%f; x_T=%f; bx=%f; ay=%f; y_T=%f; by=%f;\n",
    // 	   sigma_x,
    // 	   sigma_y,
    // 	   rho,
    // 	   y_t.a_x - y_t.x_tm1,
    // 	   y_t.x_t - y_t.x_tm1,
    // 	   y_t.b_x - y_t.x_tm1,
    // 	   y_t.a_y - y_t.y_tm1,
    // 	   y_t.y_t - y_t.y_tm1,
    // 	   y_t.b_y - y_t.y_tm1);

    out = log(likelihood);
  } else {
    // printf("\nFor rho=%f, data produces neg likelihood.\n", rho);
    // printf("sigma_x=%f; sigma_y=%f; rho=%f; ax=%f; x_T=%f; bx=%f; ay=%f; y_T=%f; by=%f;\n",
    // 	   sigma_x,
    // 	   sigma_y,
    // 	   rho,
    // 	   y_t.a_x - y_t.x_tm1,
    // 	   y_t.x_t - y_t.x_tm1,
    // 	   y_t.b_x - y_t.x_tm1,
    // 	   y_t.a_y - y_t.y_tm1,
    // 	   y_t.y_t - y_t.y_tm1,
    // 	   y_t.b_y - y_t.y_tm1);
    out = log(1e-32);
  }
  return out;
}

likelihood_point log_likelihood_OCHL(const observable_datum& y_t,
				     const observable_datum& y_tm1,
				     const stoch_vol_datum& theta_t,
				     const parameters& params,
				     GaussianInterpolator& GP_prior)
{
  double x [2] = {y_t.x_t, y_t.y_t};
  gsl_vector_view gsl_x = gsl_vector_view_array(x, 2);

  double out = 0.0;

  double sigma_x = exp(theta_t.log_sigma_x);
  double sigma_y = exp(theta_t.log_sigma_y);
  double rho = logit_inv(theta_t.rho_tilde)*2-1;

  double Lx = y_t.b_x - y_t.a_x;
  double Ly = y_t.b_y - y_t.a_y;
  double log_likelihood = 0;

  double sigma_xi = sigma_x / Lx;
  double sigma_eta = sigma_y / Ly;

  double xi_T = y_t.x_t / Lx;
  double a_xi = y_t.a_x / Lx;
  double b_xi = y_t.b_x / Lx;
  double xi_0 = y_t.x_tm1 / Lx;

  double eta_T = y_t.y_t / Ly;
  double a_eta = y_t.a_y / Ly;
  double b_eta = y_t.b_y / Ly;
  double eta_0 = y_t.y_tm1 / Ly;

  double t_tilde = 1.0*sigma_xi*sigma_xi;
  double sigma_y_tilde = sigma_eta/sigma_xi;

  if (sigma_xi < sigma_eta) {
    t_tilde = 1.0*sigma_eta*sigma_eta;
    sigma_y_tilde = sigma_xi/sigma_eta;
  }

  xi_T = xi_T - a_xi;
  xi_0 = xi_0 - a_xi;

  eta_T = eta_T - a_eta;
  eta_0 = eta_0 - a_eta;
  
  likelihood_point lp = likelihood_point(xi_0,
					 eta_0,
					 //
					 xi_T,
					 eta_T,
					 //
					 sigma_y_tilde,
					 t_tilde,
					 rho,
					 0.0);
  log_likelihood = GP_prior(lp);
  if (std::isnan(log_likelihood)) {
    log_likelihood = log(0.0);
  }
        
  lp.likelihood = log_likelihood - (3*log(Lx) + 3*log(Ly));
  
  return lp;
}

std::vector<double> log_likelihood_OCHL_2(const observable_datum& y_t,
					const observable_datum& y_tm1,
					const stoch_vol_datum& theta_t,
					const parameters& params,
					GaussianInterpolator& GP_prior)
{
  double x [2] = {y_t.x_t, y_t.y_t};
  gsl_vector_view gsl_x = gsl_vector_view_array(x, 2);

  double sigma_x = exp(theta_t.log_sigma_x);
  double sigma_y = exp(theta_t.log_sigma_y);
  double rho = logit_inv(theta_t.rho_tilde)*2-1;

  double Lx = y_t.b_x - y_t.a_x;
  double Ly = y_t.b_y - y_t.a_y;
  double log_likelihood = 0;

  double sigma_xi = sigma_x / Lx;
  double sigma_eta = sigma_y / Ly;

  double xi_T = y_t.x_t / Lx;
  double a_xi = y_t.a_x / Lx;
  double b_xi = y_t.b_x / Lx;
  double xi_0 = y_t.x_tm1 / Lx;

  double eta_T = y_t.y_t / Ly;
  double a_eta = y_t.a_y / Ly;
  double b_eta = y_t.b_y / Ly;
  double eta_0 = y_t.y_tm1 / Ly;

  double t_tilde = 1.0*sigma_xi*sigma_xi;
  double sigma_y_tilde = sigma_eta/sigma_xi;

  if (sigma_xi < sigma_eta) {
    t_tilde = 1.0*sigma_eta*sigma_eta;
    sigma_y_tilde = sigma_xi/sigma_eta;
  }

  xi_T = xi_T - a_xi;
  xi_0 = xi_0 - a_xi;

  eta_T = eta_T - a_eta;
  eta_0 = eta_0 - a_eta;
  
  likelihood_point lp = likelihood_point(xi_0,
					 eta_0,
					 //
					 xi_T,
					 eta_T,
					 //
					 sigma_y_tilde,
					 t_tilde,
					 rho,
					 0.0);
  log_likelihood = GP_prior(lp);
  lp.likelihood = log_likelihood;
  lp.print_point();
  
  printf("Uncertainty in interpolation = %f\n",
	 sqrt(GP_prior.prediction_variance(lp)));
  
  lp.likelihood = log_likelihood - (3*log(Lx) + 3*log(Ly));

  std::vector<double> out = std::vector<double> (10);
  out[0] = y_t.x_t - y_t.a_x;
  out[1] = y_t.y_t - y_t.a_y;
  //
  out[2] = y_t.x_tm1 - y_t.a_x;
  out[3] = y_t.y_tm1 - y_t.a_y;
  //
  out[4] = Lx;
  out[5] = Ly;
  //
  out[6] = theta_t.log_sigma_x;
  out[7] = theta_t.log_sigma_y;
  out[8] = rho;
  out[9] = lp.likelihood;

  return out;
}

stoch_vol_datum sample_theta(const stoch_vol_datum& theta_current,
			     const observable_datum& y_current,
			     const observable_datum& y_current_m1,
			     const parameters& params,
			     gsl_rng * r) {

  std::vector<double> epsilons = epsilons_given_theta(theta_current,
						      y_current,
						      y_current_m1,
						      params);

  std::vector<double> eta_means {params.leverage_x_rho*epsilons[0],
				   params.leverage_y_rho*epsilons[1],
				   0.0};

  std::vector<double> eta_vars {1.0-params.leverage_x_rho*params.leverage_x_rho,
				  1.0-params.leverage_y_rho*params.leverage_y_rho,
				  1.0};

  std::vector<double> etas {eta_means[0] + gsl_ran_gaussian(r, sqrt(eta_vars[0])),
      eta_means[1] + gsl_ran_gaussian(r, sqrt(eta_vars[1])),
      eta_means[2] + gsl_ran_gaussian(r, sqrt(eta_vars[2]))};

  std::vector<stoch_vol_datum> out =
    theta_next_mean(std::vector<stoch_vol_datum> {theta_current},
		    y_current,
		    y_current_m1,
		    params);

  out[0].log_sigma_x = out[0].log_sigma_x + params.tau_x*etas[0];
  out[0].log_sigma_y = out[0].log_sigma_y + params.tau_y*etas[1];

  // FIXING RHO TO CORRECT VAL
  out[0].rho_tilde = out[0].rho_tilde + params.tau_rho*etas[2];
  // sout[0].rho_tilde = logit((0.6 + 1.0)/2.0);

  return out[0];
}

parameters sample_parameters(const gsl_vector* scaled_mean,
			     const gsl_matrix* scaled_cov,
			     const parameters& params,
			     gsl_rng * r,
			     double a) {

  double phi_array [13] = {params.mu_x,
			   params.mu_y,
			   params.alpha_x,
			   params.alpha_x,
			   params.alpha_rho,
			   log(params.theta_x),
			   log(params.theta_y),
			   log(params.theta_rho),
			   log(params.tau_x),
			   log(params.tau_y),
			   log(params.tau_rho),
			   logit( (params.leverage_x_rho+1)/2.0 ),
			   logit( (params.leverage_y_rho+1)/2.0 )};

  double out_array [13];
  gsl_vector_view out = gsl_vector_view_array(out_array, 13);

  gsl_vector_view phi = gsl_vector_view_array(phi_array, 13);
  gsl_vector_scale(&phi.vector, a);
  gsl_vector_add(&phi.vector, scaled_mean);

  MultivariateNormal mvnorm = MultivariateNormal();

  mvnorm.rmvnorm(r,
		 13,
		 &phi.vector,
		 scaled_cov,
		 &out.vector);

  parameters out_params;
  // out_params.mu_x = gsl_vector_get(&out.vector, 0);
  // out_params.mu_y = gsl_vector_get(&out.vector, 1);
  out_params.mu_x = 0.0;
  out_params.mu_y = 0.0;
  //
  out_params.alpha_x = gsl_vector_get(&out.vector, 2);
  out_params.alpha_y = gsl_vector_get(&out.vector, 3);
  out_params.alpha_rho = gsl_vector_get(&out.vector, 4);
  //
  out_params.theta_x = exp(gsl_vector_get(&out.vector, 5));
  out_params.theta_y = exp(gsl_vector_get(&out.vector, 6));
  out_params.theta_rho = exp(gsl_vector_get(&out.vector, 7));
  //
  out_params.tau_x = exp(gsl_vector_get(&out.vector, 8));
  out_params.tau_y = exp(gsl_vector_get(&out.vector, 9));
  out_params.tau_rho = exp(gsl_vector_get(&out.vector, 10));
  //
  out_params.leverage_x_rho = 2.0*logit_inv(gsl_vector_get(&out.vector, 11)) - 1.0;
  out_params.leverage_y_rho = 2.0*logit_inv(gsl_vector_get(&out.vector, 12)) - 1.0;
  // out_params.leverage_x_rho = 0.0;
  // out_params.leverage_y_rho = 0.0;

  return out_params;
}

double compute_ESS(const std::vector<double>& log_weights)
{
  unsigned N = log_weights.size();
  double sum_weights = 0;
  double sum_sq_weights = 0;
  for (const double& log_weight : log_weights) {
    sum_sq_weights += exp(2*log_weight) * 1.0/N;
    sum_weights += exp(log_weight) * 1.0/N;
  }
  double var_weights = sum_sq_weights - sum_weights*sum_weights;
  double mean_weights = sum_weights;

  return (1.0*N) / (1.0 + var_weights/(sum_weights*sum_weights));
}

std::vector<double> compute_quantiles(const std::vector<stoch_vol_datum>& theta_t,
				      const std::vector<double>& log_weights)
{
  double weighted_sum_log_sigma_x = 0.0;
  double weighted_sum_sq_log_sigma_x = 0.0;

  double weighted_sum_log_sigma_y = 0.0;
  double weighted_sum_sq_log_sigma_y = 0.0;

  double weighted_sum_rho_tilde = 0.0;
  double weighted_sum_sq_rho_tilde = 0.0;

  for (unsigned i=0; i<theta_t.size(); ++i) {
    weighted_sum_log_sigma_x = weighted_sum_log_sigma_x +
      exp(log_weights[i])*theta_t[i].log_sigma_x;
    //
    weighted_sum_sq_log_sigma_x = weighted_sum_sq_log_sigma_x +
      exp(log_weights[i])*(theta_t[i].log_sigma_x*theta_t[i].log_sigma_x);


    weighted_sum_log_sigma_y = weighted_sum_log_sigma_y +
      exp(log_weights[i])*theta_t[i].log_sigma_y;
    //
    weighted_sum_sq_log_sigma_y = weighted_sum_sq_log_sigma_y +
      exp(log_weights[i])*(theta_t[i].log_sigma_y*theta_t[i].log_sigma_y);


    weighted_sum_rho_tilde = weighted_sum_rho_tilde +
      exp(log_weights[i])*theta_t[i].rho_tilde;
    //
    weighted_sum_sq_rho_tilde = weighted_sum_sq_rho_tilde +
      exp(log_weights[i])*(theta_t[i].rho_tilde*theta_t[i].rho_tilde);

  }

  double log_sigma_x_mean = weighted_sum_log_sigma_x;
  double log_sigma_x_var = weighted_sum_sq_log_sigma_x -
    log_sigma_x_mean*log_sigma_x_mean;

  double log_sigma_y_mean = weighted_sum_log_sigma_y;
  double log_sigma_y_var = weighted_sum_sq_log_sigma_y -
    log_sigma_y_mean*log_sigma_y_mean;

  double rho_tilde_mean = weighted_sum_rho_tilde;
  double rho_tilde_var = weighted_sum_sq_rho_tilde -
    rho_tilde_mean*rho_tilde_mean;

  return std::vector<double>
    {log_sigma_x_mean, log_sigma_x_var,
	log_sigma_y_mean, log_sigma_y_var,
	rho_tilde_mean, rho_tilde_var};
}

std::vector<double> compute_quantiles(const std::vector<parameters>& params_t,
				      const std::vector<double>& log_weights)
{
  unsigned params_length = params_t[0].parameters_number;
  std::vector<double> sum (params_length, 0.0);
  std::vector<double> sum_sq (params_length, 0.0);

  for (unsigned i=0; i<params_t.size(); ++i) {
    const parameters& current_parameters = params_t[i];
    gsl_vector* current_parameter_transformed = parameters_to_reals(current_parameters);
    gsl_vector* current_parameter_transformed_sq = gsl_vector_alloc(params_length);
    gsl_vector_memcpy(current_parameter_transformed_sq, current_parameter_transformed);
    gsl_vector_mul(current_parameter_transformed_sq, current_parameter_transformed);

    for (unsigned j=0; j<params_length; ++j) {
      sum[j] = sum[j] + exp(log_weights[i])*gsl_vector_get(current_parameter_transformed,j);
      sum_sq[j] = sum_sq[j] + exp(log_weights[i])*gsl_vector_get(current_parameter_transformed_sq,j);
    }

    gsl_vector_free(current_parameter_transformed);
    gsl_vector_free(current_parameter_transformed_sq);
  }

  std::vector<double> out (2*params_length);

  for (unsigned i=0; i<params_length; ++i) {
    out[2*i] = sum[i];
    out[2*i+1] = sum_sq[i] - sum[i]*sum[i];
  }

  return out;
}

std::vector<observable_datum> read_data_from_csv(std::string file)
{
  std::ifstream data_file(file);
  std::string value;
  std::vector<observable_datum> out (0);

  if (data_file.is_open()) {
    // go through the header
    for (unsigned i=0; i<14; ++i) {
      if (i<13) {
	std::getline(data_file,value,',');
      } else {
	std::getline(data_file,value);
      }
    }

    // first values is date
    while (std::getline(data_file, value, ',')) {

      observable_datum current_datum;

      // second value is FTSE open
      std::getline(data_file, value, ',');
      double ftse_open = log(std::stod(value));
      current_datum.x_tm1 = ftse_open;

      // third value is FTSE high
      std::getline(data_file, value, ',');
      double ftse_high = log(std::stod(value));
      current_datum.b_x = ftse_high;

      // fourth value is FTSE low
      std::getline(data_file, value, ',');
      double ftse_low = log(std::stod(value));
      current_datum.a_x = ftse_low;

      // fifth value is FTSE close
      std::getline(data_file, value, ',');
      double ftse_close = log(std::stod(value));
      current_datum.x_t = ftse_close;

      std::getline(data_file, value, ',');
      std::getline(data_file, value, ',');
      std::getline(data_file, value, ',');

      // ninth value is SPY open
      std::getline(data_file, value, ',');
      double spy_open = log(std::stod(value));
      current_datum.y_tm1 = spy_open;

      // tenth value is SPY high
      std::getline(data_file, value, ',');
      double spy_high = log(std::stod(value));
      current_datum.b_y = spy_high;

      // 11th value is SPY low
      std::getline(data_file, value, ',');
      double spy_low = log(std::stod(value));
      current_datum.a_y = spy_low;

      // 12th value is SPY close
      std::getline(data_file, value, ',');
      double spy_close = log(std::stod(value));
      current_datum.y_t = spy_close;

      std::getline(data_file, value, ',');
      std::getline(data_file, value);

      out.push_back(current_datum);
    }
  }

  for (unsigned i=0; i<out.size(); ++i) {
    std::cout << out[i] << std::endl;
  }

  return out;
}

 void generate_data(std::vector<observable_datum>& ys,
		    std::vector<stoch_vol_datum>& thetas,
		    const parameters& params,
		    unsigned order,
		    long unsigned seed,
		    unsigned buffer,
		    bool SWITCH_XY)
 {
   std::ofstream output;
   output.open("data.csv");

  unsigned N = ys.size() + buffer;
  std::vector<observable_datum> ys_long (N);
  std::vector<stoch_vol_datum> thetas_long (N);

  double rho_t = 0;
  double rho_tilde_t = logit((rho_t+1)/2);

  double x_t = log(100);
  double y_t = log(100);

  double log_sigma_x_t = params.alpha_x;
  double log_sigma_y_t = params.alpha_y;

  std::cout << "N+1 = " << N+1 << std::endl;
  std::cout << "log_sigma_x_t = " << log_sigma_x_t << std::endl;

  // setnames(innovations, seq(1,5), c("epsilon_x", "epsilon_y",
  //                                   "eta_x", "eta_y", "eta_rho"))


  //  double * output = new double [(N+1)*9];
    // output = data_table(x = as_numeric(rep(NA,N)),
    //                      y = as_numeric(rep(NA,N)),
    //                      log_sigma_x = as_numeric(rep(NA,N)),
    //                      log_sigma_y = as_numeric(rep(NA,N)),
    //                      rho_tilde = as_numeric(rep(NA,N)))




  ys_long[0].x_tm1 = x_t;
  ys_long[0].y_tm1 = y_t;

  thetas_long[0].log_sigma_x = log_sigma_x_t;
  thetas_long[0].log_sigma_y = log_sigma_y_t;
  thetas_long[0].rho_tilde = rho_tilde_t;

  // output[1, c("x",
  //             "y",
  //             "log_sigma_x",
  //             "log_sigma_y",
  //             "rho_tilde") := as_list(c(x_t,
  //                                       y_t,
  //                                       log_sigma_x_t,
  //                                       log_sigma_y_t,
  //                                       rho_tilde_t))]

  const gsl_rng_type * Type;
  gsl_rng_env_setup();
  Type = gsl_rng_default;
  gsl_rng * r_ptr = gsl_rng_alloc(Type);
  gsl_rng_set(r_ptr, seed);

  for (unsigned i=0; i<N; ++i) {
    double log_sigma_x_tp1 = params.alpha_x +
      params.theta_x*(log_sigma_x_t - params.alpha_x) +
      gsl_ran_gaussian(r_ptr, params.tau_x);

    double log_sigma_y_tp1 = params.alpha_y +
      params.theta_y*(log_sigma_y_t - params.alpha_y) +
      gsl_ran_gaussian(r_ptr, params.tau_y);

    double rho_tp1 = 0.6*sin(2.0*M_PI/256.0*i);
    if (i < 50) {
      rho_tp1 = 0.7;
    } else {
      rho_tp1 = -0.7;
    }
    // rho_tp1 = 0.6;
    double rho_tilde_tp1 = logit( (rho_tp1 + 1.0)/2.0 );

    // rho_tilde_tp1 = rho_tilde_t + tau_rho*innovations[i-1,eta_rho]
    // rho_tp1 = 2*logit_inv(rho_tilde_tp1) - 1

    BrownianMotion BM = BrownianMotion(i,
				       order,
				       rho_t,
				       exp(log_sigma_x_t),
				       exp(log_sigma_y_t),
				       x_t,
				       y_t,
				       1.0);
    double x_tp1 = BM.get_x_T();
    double y_tp1 = BM.get_y_T();

    ys_long[i].x_t = x_tp1;
    ys_long[i].a_x = BM.get_a();
    ys_long[i].b_x = BM.get_b();
    // // // //
    ys_long[i].y_t = y_tp1;
    ys_long[i].a_y = BM.get_c();
    ys_long[i].b_y = BM.get_d();

    if (i < N-1) {
      ys_long[i+1].x_tm1 = x_tp1;
      ys_long[i+1].y_tm1 = y_tp1;
      // // // // //
      thetas_long[i+1].log_sigma_x = log_sigma_x_tp1;
      thetas_long[i+1].log_sigma_y = log_sigma_y_tp1;
      thetas_long[i+1].rho_tilde = rho_tilde_tp1;
    }

    x_t = x_tp1;
    y_t = y_tp1;
    log_sigma_x_t = log_sigma_x_tp1;
    log_sigma_y_t = log_sigma_y_tp1;
    rho_tilde_t = rho_tilde_tp1;
    rho_t = rho_tp1;


  }

  gsl_rng_free(r_ptr);

  for (unsigned i=0; i<ys.size(); ++i) {
    if (SWITCH_XY) {
      ys[i].x_tm1 = ys_long[i+buffer].y_tm1;
      ys[i].x_t = ys_long[i+buffer].y_t;
      ys[i].a_x = ys_long[i+buffer].a_y;
      ys[i].b_x = ys_long[i+buffer].b_y;

      ys[i].y_tm1 = ys_long[i+buffer].x_tm1;
      ys[i].y_t = ys_long[i+buffer].x_t;
      ys[i].a_y = ys_long[i+buffer].a_x;
      ys[i].b_y = ys_long[i+buffer].b_x;

      thetas[i].log_sigma_x = thetas_long[i+buffer].log_sigma_y;
      thetas[i].log_sigma_y = thetas_long[i+buffer].log_sigma_x;
      thetas[i].rho_tilde = thetas_long[i+buffer].rho_tilde;
    } else {
      ys[i] = ys_long[i+buffer];
      thetas[i] = thetas_long[i+buffer];
    }
  }

  // FIRST ROW
  output << "ax, x, bx, ay, y, by, log.sigma.x, log.sigma.y, rho.tilde \n";
  output << "NA" << "," << ys[0].x_tm1 << "," << "NA" << ","
	 << "NA" << "," << ys[0].y_tm1 << "," << "NA" << ","
	 << params.alpha_x << ","
	 << params.alpha_y << ","
	 << logit((0.0+1.0)/2) << "\n";

  for (unsigned i=0; i<ys.size(); ++i) {
      output << ys[i].a_x << "," << ys[i].x_t << "," << ys[i].b_x << ","
	     << ys[i].a_y << "," << ys[i].y_t << "," << ys[i].b_y << ","
	     << thetas[i].log_sigma_x << ","
	     << thetas[i].log_sigma_y << ","
	     << thetas[i].rho_tilde << "\n";
  }

   output.close();
 }


gsl_vector* compute_parameters_mean(const std::vector<parameters>& params_t)
{
  unsigned N = params_t.size();
  gsl_vector * out = gsl_vector_calloc(13);

  for (unsigned i=0; i<params_t.size(); ++i) {
    double summand [] = {1.0/N*params_t[i].mu_x,
			 1.0/N*params_t[i].mu_y,
			 //
			 1.0/N*params_t[i].alpha_x,
			 1.0/N*params_t[i].alpha_y,
			 1.0/N*params_t[i].alpha_rho,
			 //
			 1.0/N*log(params_t[i].theta_x),
			 1.0/N*log(params_t[i].theta_y),
			 1.0/N*log(params_t[i].theta_rho),
			 //
			 1.0/N*log(params_t[i].tau_x),
			 1.0/N*log(params_t[i].tau_y),
			 1.0/N*log(params_t[i].tau_rho),
			 //
			 1.0/N* logit( (params_t[i].leverage_x_rho+1.0)/2.0 ),
			 1.0/N* logit( (params_t[i].leverage_y_rho+1.0)/2.0 )};

    gsl_vector_view gsl_summand = gsl_vector_view_array(summand, 13);
    gsl_vector_add(out, &gsl_summand.vector);
  }

  return out;
}

gsl_matrix* compute_parameters_cov(const gsl_vector* mean,
				   const std::vector<parameters>& params_t)
{
  unsigned N = params_t.size();

  gsl_matrix * covariance = gsl_matrix_alloc(13,13);
  gsl_matrix_set_zero(covariance);

  for (unsigned i=0; i<params_t.size(); ++i) {
    std::vector<double> summand {params_t[i].mu_x-gsl_vector_get(mean, 0),
	params_t[i].mu_y-gsl_vector_get(mean, 1),
	//
	params_t[i].alpha_x-gsl_vector_get(mean, 2),
	params_t[i].alpha_y-gsl_vector_get(mean, 3),
	params_t[i].alpha_rho-gsl_vector_get(mean, 4),
	//
	log(params_t[i].theta_x)-gsl_vector_get(mean, 5),
	log(params_t[i].theta_y)-gsl_vector_get(mean, 6),
	log(params_t[i].theta_rho)-gsl_vector_get(mean, 7),
	//
	log(params_t[i].tau_x)-gsl_vector_get(mean, 8),
	log(params_t[i].tau_y)-gsl_vector_get(mean, 9),
	log(params_t[i].tau_rho)-gsl_vector_get(mean, 10),
	//
	logit( (params_t[i].leverage_x_rho+1.0)/2.0 ) - gsl_vector_get(mean, 11),
	logit( (params_t[i].leverage_y_rho+1.0)/2.0 ) - gsl_vector_get(mean, 12) };

    for (unsigned j=0; j<13; ++j) {
      for (unsigned k=0; k<13; ++k) {

	double current_entry = gsl_matrix_get(covariance,j,k) +
	  summand[j]*summand[k] * (1.0/N);

	gsl_matrix_set(covariance,
		       j,k,
		       current_entry);
      }
    }
  }

  // TO CHECK IF THE COV IS POS DEF START //
  gsl_error_handler_t* old_handler = gsl_set_error_handler_off();
  gsl_matrix* work = gsl_matrix_alloc(13, 13);
  gsl_matrix_memcpy(work, covariance);

  int status = gsl_linalg_cholesky_decomp(work);
  if (status == GSL_EDOM) {
    gsl_vector_view diag_view = gsl_matrix_diagonal(covariance);
    gsl_vector* diag_cpy = gsl_vector_alloc(13);
    gsl_vector_memcpy(diag_cpy, &diag_view.vector);

    gsl_matrix_set_zero(covariance);
    gsl_vector_memcpy(&diag_view.vector, diag_cpy);

    // we need to check for non-zero diagonal entries too
    for (unsigned i=0; i<13; ++i) {
      if ( gsl_matrix_get(covariance, i,i) < std::numeric_limits<double>::epsilon() ||
	   std::isnan(gsl_matrix_get(covariance, i,i))) {

	double candidates[] = {std::pow(gsl_vector_get(mean,i), 2),
			       std::numeric_limits<double>::epsilon() * 10};

	gsl_matrix_set(covariance, i,i,
		       *std::max_element(candidates, candidates+2));
      }
    }
  }

  gsl_set_error_handler(old_handler);
  gsl_matrix_free(work);
  // TO CHECK IF THE COV IS POS DEF END //

  return covariance;
}

void print_vector(const gsl_vector* mean,
		  unsigned size)
{
  std::cout << "mean = c(";
  for (unsigned i=0; i<size; ++i) {
    if (i < size-1) {
      std::cout << gsl_vector_get(mean, i) << ",";
    } else {
      std::cout << gsl_vector_get(mean, i) << ");";
    }
  }
}

void print_matrix(const gsl_matrix* cov,
		  unsigned size)
{
  std::cout << "covData = c(";
  for (unsigned i=0; i<size; ++i) {
    for (unsigned j=0; j<size; ++j) {
      if (j < size-1) {
	std::cout << gsl_matrix_get(cov, i, j) << ",";
      } else {
	std::cout << gsl_matrix_get(cov, i, j) << ",";
      }
    }
  }
  std::cout << std::endl;
}


gsl_vector* parameters_to_reals(const parameters& params)
{
  gsl_vector* out = gsl_vector_alloc(13);

  gsl_vector_set(out, 0, params.mu_x);
  gsl_vector_set(out, 1, params.mu_y);
  // //
  gsl_vector_set(out, 2, params.alpha_x);
  gsl_vector_set(out, 3, params.alpha_y);
  gsl_vector_set(out, 4, params.alpha_rho);
  // //
  gsl_vector_set(out, 5, log(params.theta_x));
  gsl_vector_set(out, 6, log(params.theta_y));
  gsl_vector_set(out, 7, log(params.theta_rho));
  // //
  gsl_vector_set(out, 8, log(params.tau_x));
  gsl_vector_set(out, 9, log(params.tau_y));
  gsl_vector_set(out, 10, log(params.tau_rho));
  //
  gsl_vector_set(out, 11, logit( (params.leverage_x_rho+1.0)/2.0 ));
  gsl_vector_set(out, 12, logit( (params.leverage_y_rho+1.0)/2.0 ));

  return out;
}

parameters reals_to_parameters(const gsl_vector* params)
{
  parameters out;
  out.mu_x = gsl_vector_get(params, 0);
  out.mu_y = gsl_vector_get(params, 1);
  // //
  out.alpha_x = gsl_vector_get(params, 2);
  out.alpha_y = gsl_vector_get(params, 3);
  out.alpha_rho = gsl_vector_get(params, 4);
  // //
  out.theta_x = exp(gsl_vector_get(params, 5));
  out.theta_y = exp(gsl_vector_get(params, 6));
  out.theta_rho = exp(gsl_vector_get(params, 7));
  // //
  out.tau_x = exp(gsl_vector_get(params, 8));
  out.tau_y = exp(gsl_vector_get(params, 9));
  out.tau_rho = exp(gsl_vector_get(params, 10));
  //
  out.leverage_x_rho = logit_inv(gsl_vector_get(params, 11))*2.0 - 1.0;
  out.leverage_y_rho = logit_inv(gsl_vector_get(params, 12))*2.0 - 1.0;

  return out;
}

bool check_parameter_bounds(const parameters& params) {
  if ( std::abs(params.theta_x) >= 1.0 ||
       std::abs(params.theta_x) >= 1.0 ||
       std::abs(params.theta_rho) >= 1.0 ||
       std::abs(params.leverage_x_rho) >= 0.99 ||
       std::abs(params.leverage_y_rho) >= 0.99 ) { 
    return false ;
  } else {
    return true;
  }
}

bool check_parameter_bounds(const gsl_vector * params) {
  parameters params_nominal = reals_to_parameters(params);
  return check_parameter_bounds(params_nominal);
}

void print_matrix(const gsl_matrix* mat, unsigned size_x, unsigned size_y) {
 for (unsigned i=0; i<size_x; ++i) {
    for (unsigned j=0; j<size_y; ++j) {
      if (j < size_y-1) {
	std::cout << gsl_matrix_get(mat, i,j) << " ";
      } else {
	std::cout << gsl_matrix_get(mat, i,j) << "\n";
      }
    }
  }
  std::cout << std::endl;
}
