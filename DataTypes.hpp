#include "BivariateSolver.hpp"
#include <cmath>
#include "GaussianInterpolator.hpp"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_vector.h>
#include "PriorTypes.hpp"
#include <vector>

struct NormalInverseWishartParameters {
  unsigned dimension;
  //
  gsl_vector* mu_not;
  gsl_matrix* inverse_scale_mat;
  //
  double lambda;
  double deg_freedom;

  NormalInverseWishartParameters()
    : dimension(13),
      lambda(1.0),
      deg_freedom(13)
  {
    mu_not = gsl_vector_alloc(dimension);
    inverse_scale_mat = gsl_matrix_calloc(dimension, dimension);
  };

  ~NormalInverseWishartParameters() {
    gsl_vector_free(mu_not);
    gsl_matrix_free(inverse_scale_mat);
  };

  NormalInverseWishartParameters& operator=(const NormalInverseWishartParameters &rhs)
  {
    if (this==&rhs) {
      return *this;
    } else {
      dimension = rhs.dimension;
      lambda = rhs.lambda;
      deg_freedom = rhs.deg_freedom;

      gsl_vector_free(mu_not);
      gsl_matrix_free(inverse_scale_mat);

      mu_not = gsl_vector_alloc(dimension);
      inverse_scale_mat = gsl_matrix_alloc(dimension, dimension);

      gsl_vector_memcpy(mu_not, rhs.mu_not);
      gsl_matrix_memcpy(inverse_scale_mat, rhs.inverse_scale_mat);
      return *this;
    }
  }

  NormalInverseWishartParameters(const NormalInverseWishartParameters& rhs)
  {
    dimension = rhs.dimension;
    lambda = rhs.lambda;
    deg_freedom = rhs.deg_freedom;

    gsl_vector_free(mu_not);
    gsl_matrix_free(inverse_scale_mat);

    mu_not = gsl_vector_alloc(dimension);
    gsl_vector_memcpy(mu_not, rhs.mu_not);
    //
    inverse_scale_mat = gsl_matrix_alloc(dimension,dimension);
    gsl_matrix_memcpy(inverse_scale_mat, rhs.inverse_scale_mat);
  }
};

struct parameters {
  double mu_x;
  double mu_y;
  //
  double alpha_x;
  double alpha_y;
  double alpha_rho;
  //
  double theta_x;
  double theta_y;
  double theta_rho;
  //
  double tau_x;
  double tau_y;
  double tau_rho;
  //
  double leverage_x_rho;
  double leverage_y_rho;
  //
  unsigned parameters_number = 13;

};
void print_params(const parameters& params);

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

  friend std::ostream& operator << (std::ostream& stream, const observable_datum& datum)
  {
    stream << "x_tm1 = " << datum.x_tm1 << "\n";
    stream << "x_t = " << datum.x_t << "\n";
    stream << "a_x = " << datum.a_x << "\n";
    stream << "b_x = " << datum.b_x << "\n";
    //
    stream << "y_tm1 = " << datum.y_tm1 << "\n";
    stream << "y_t = " << datum.y_t << "\n";
    stream << "a_y = " << datum.a_y << "\n";
    stream << "b_y = " << datum.b_y << "\n";
    return stream;
  }
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
std::vector<stoch_vol_datum> sample_theta_prior(const std::vector<parameters>& params,
						unsigned N_particles,
						gsl_rng *r);

std::vector<parameters> sample_parameters_prior(const StochasticVolatilityPriors& priors_x,
						const StochasticVolatilityPriors& prior_y,
						const StochasticVolatilityPriors& prior_rho,
						const RhoPrior& prior_rho_leverage,
						unsigned N_particles,
						gsl_rng *r);

std::vector<stoch_vol_datum> theta_next_mean(const std::vector<stoch_vol_datum>& theta_current,
					       const observable_datum& y_current,
					       const observable_datum& y_current_tm1,
					       const parameters& params);
std::vector<stoch_vol_datum> theta_next_mean(const std::vector<stoch_vol_datum>& theta_current,
					     const observable_datum& y_current,
					     const observable_datum& y_current_tm1,
					     const std::vector<parameters>& params_vector);

std::vector<parameters> parameters_next_mean(const std::vector<parameters>& parameters_current,
					     const gsl_vector* scaled_mean,
					     double scale_a);

std::vector<double> log_likelihoods(const observable_datum& y_t,
				    const observable_datum& y_tm1,
				    const std::vector<stoch_vol_datum>& theta_t,
				    const parameters& params);

double log_likelihood(const observable_datum& y_t,
		      const observable_datum& y_tm1,
		      const stoch_vol_datum& theta_t,
		      const parameters& params);

double log_likelihood(const stoch_vol_datum& theta_t,
		      const stoch_vol_datum& theta_tm1,
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

likelihood_point log_likelihood_OCHL(const observable_datum& y_t,
				     const observable_datum& y_tm1,
				     const stoch_vol_datum& theta_t,
				     const parameters& params,
				     GaussianInterpolator& GP_prior);

stoch_vol_datum sample_theta(const stoch_vol_datum& theta_current,
			     const observable_datum& y_current,
			     const observable_datum& y_current_m1,
			     const parameters& params,
			     gsl_rng * r);

parameters sample_parameters(const gsl_vector* scaled_mean,
			     const gsl_matrix* scaled_cov,
			     const parameters& params,
			     gsl_rng * r,
			     double a);

double compute_ESS(const std::vector<double>& log_weights);
std::vector<double> compute_quantiles(const std::vector<stoch_vol_datum>& theta_t,
				      const std::vector<double>& log_weights);

std::vector<double> compute_quantiles(const std::vector<parameters>& params_t,
				      const std::vector<double>& log_weights);

std::vector<observable_datum> read_data_from_csv(std::string file);

void generate_data(std::vector<observable_datum>& ys,
 		   std::vector<stoch_vol_datum>& thetas,
		   const parameters& params,
		   unsigned order,
		   long unsigned seed,
		   unsigned buffer,
		   bool SWITCH_XY);

gsl_vector*  compute_parameters_mean(const std::vector<parameters>& params_t);
gsl_matrix*  compute_parameters_cov(const gsl_vector* mean,
				    const std::vector<parameters>& params_t);
void print_vector(const gsl_vector* mean,
		  unsigned size);
void print_matrix(const gsl_matrix* cov,
		  unsigned size);

gsl_vector* parameters_to_reals(const parameters& params);
parameters reals_to_parameters(const gsl_vector* params);

bool check_parameter_bounds(const parameters& params);
bool check_parameter_bounds(const gsl_vector * params);

void print_matrix(const gsl_matrix* mat, unsigned size_x, unsigned size_y);
