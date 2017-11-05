#include "2DBrownianMotionPath.hpp"
#include "DataTypes.hpp"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <iostream>
#include <fstream>
#include <limits>

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
  
  BivariateSolver solver = BivariateSolver(basis,
					   sigma_x/Lx,
					   sigma_y/Lx,
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
  
  double likelihood = solver.numerical_likelihood_extended(&gsl_x.vector,
							   dx_likelihood);
  if ( (likelihood - 15.0) > std::numeric_limits<double>::epsilon() ) {
    printf("\nLikelihood for point abnormally large: %f \n", likelihood);
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
  out[0].rho_tilde = out[0].rho_tilde + params.tau_rho*etas[2];

  return out[0];
}

double compute_ESS(const std::vector<double>& log_weights)
{
  double sum_sq_weights = 0;
  for (const double& log_weight : log_weights) {
    sum_sq_weights += exp(2*log_weight);
  }
  
  return 1.0 /sum_sq_weights;
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

 void generate_data(std::vector<observable_datum>& ys,
		    std::vector<stoch_vol_datum>& thetas,
		    const parameters& params,
		    unsigned order,
		    long unsigned seed)
 {
   std::ofstream output;
   output.open("data.csv");
  
  unsigned N = ys.size();

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


  // FIRST ROW
  output << "ax, x, bx, ay, y, by, log.sigma.x, log.sigma.y, rho.tilde \n";
  output << "NA" << "," << x_t << "," << "NA" << ","
	 << "NA" << "," << y_t << "," << "NA" << ","
	 << log_sigma_x_t << ","
	 << log_sigma_y_t << ","
	 << rho_tilde_t << "\n";
  
  ys[0].x_tm1 = x_t;
  ys[0].y_tm1 = y_t;

  thetas[0].log_sigma_x = log_sigma_x_t;
  thetas[0].log_sigma_y = log_sigma_y_t;
  thetas[0].rho_tilde = rho_tilde_t;

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
    if (i < 130) {
      rho_tp1 = -0.7;
    } else {
      rho_tp1 = 0.7;
    }
    rho_tp1 = 0.6;
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

    ys[i].x_t = x_tp1;
    ys[i].a_x = BM.get_a();
    ys[i].b_x = BM.get_b();
    // // // // 
    ys[i].y_t = y_tp1;
    ys[i].a_y = BM.get_c();
    ys[i].b_y = BM.get_d();

    if (i < N-1) {
      ys[i+1].x_tm1 = x_tp1;
      ys[i+1].y_tm1 = y_tp1;
      // // // // //
      thetas[i+1].log_sigma_x = log_sigma_x_tp1;
      thetas[i+1].log_sigma_y = log_sigma_y_tp1;
      thetas[i+1].rho_tilde = rho_tilde_tp1;
    }

    x_t = x_tp1;
    y_t = y_tp1;
    log_sigma_x_t = log_sigma_x_tp1;
    log_sigma_y_t = log_sigma_y_tp1;
    rho_tilde_t = rho_tilde_tp1;
    rho_t = rho_tp1;

    output << ys[i].a_x << "," << x_t << "," << ys[i].b_x << ","
	   << ys[i].a_y << "," << y_t << "," << ys[i].b_y << ","
	   << log_sigma_x_t << ","
	   << log_sigma_y_t << ","
	   << rho_tilde_t << "\n";
  }
  
  gsl_rng_free(r_ptr);
  
   output.close();
 }
