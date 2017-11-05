#include "BivariateSolver.hpp"
#include <cmath>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <vector>

struct parameters {
  double mu_x;
  double mu_y;
  double alpha_x;
  double alpha_y;
  double theta_x;
  double theta_y;
  double tau_x;
  double tau_y;
  double tau_rho;
  double leverage_x_rho;
  double leverage_y_rho;
};

struct observable_datum {
  double x_tm1;
  double x_t;
  double a_x;
  double b_x;
  //
  double y_tm1;
  double y_t;
  double a_y;
  double b_y;
};

struct stoch_vol_datum {
  double log_sigma_x;
  double log_sigma_y;
  double rho_tilde;
};

double logit(double p);
double logit_inv(double logit_p);

std::vector<double> epsilons_given_theta(const stoch_vol_datum& theta_current,
					   const observable_datum& y_current,
					   const observable_datum& y_current_m1,
					   const parameters& params);

std::vector<stoch_vol_datum> sample_theta_prior(const parameters& params,
						unsigned N_particles,
						gsl_rng *r);

std::vector<stoch_vol_datum> theta_next_mean(const std::vector<stoch_vol_datum>& theta_current,
					       const observable_datum& y_current,
					       const observable_datum& y_current_tm1,
					       const parameters& params);

std::vector<double> log_likelihoods(const observable_datum& y_t,
				    const observable_datum& y_tm1,
				    const std::vector<stoch_vol_datum>& theta_t,
				    const parameters& params);

double log_likelihood(const observable_datum& y_t,
		      const observable_datum& y_tm1,
		      const stoch_vol_datum& theta_t,
		      const parameters& params);

std::vector<double> log_likelihoods_OCHL(const observable_datum& y_t,
					 const observable_datum& y_tm1,
					 const std::vector<stoch_vol_datum>& theta_t,
					 const parameters& params,
					 BivariateGaussianKernelBasis * basis,
					 double dx,
					 double dx_likelihood);

double log_likelihood_OCHL(const observable_datum& y_t,
			   const observable_datum& y_tm1,
			   const stoch_vol_datum& theta_t,
			   const parameters& params,
			   BivariateGaussianKernelBasis * basis,
			   double dx,
			   double dx_likelihood);

stoch_vol_datum sample_theta(const stoch_vol_datum& theta_current,
			     const observable_datum& y_current,
			     const observable_datum& y_current_m1,
			     const parameters& params,
			     gsl_rng * r);

double compute_ESS(const std::vector<double>& log_weights);
std::vector<double> compute_quantiles(const std::vector<stoch_vol_datum>& theta_t,
				      const std::vector<double>& log_weights);

void generate_data(std::vector<observable_datum>& ys,
 		   std::vector<stoch_vol_datum>& thetas,
		   const parameters& params,
		   unsigned order,
		   long unsigned seed);
