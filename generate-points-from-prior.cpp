#include "DataTypes.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <iostream>
#include <sstream>
#include <limits>
#include "nlopt.hpp"
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <vector>

int main(int argc, char *argv[]) {
  if (argc < 3 || argc > 3) {
    printf("You must provide input\n");
    printf("The input is: \n\nnumber data points;\nfile prefix;\n");
    exit(0);
  }

  unsigned N = std::stoi(argv[1]);
  std::string file_prefix = argv[2];
  // ############################################# //
  long unsigned T = 1 * N * 6.5 * 3600 * 1000; // number days in ms
  long unsigned Delta = 1 * 6.5*3600*1000; // one day in ms

  // Values taken from the microstructure paper
  double mu_hat = 1.7e-12;
  double theta_hat = 5.6e-10;
  double alpha_hat = -13;
  double tau_hat = std::sqrt(1.3e-9);

  parameters params;
  params.mu_x = 0; // setting it to zero artificically
  params.mu_y = 0;

  // params.alpha_x = alpha_hat + 1.0/2.0*log(Delta);
  // params.alpha_y = alpha_hat + 1.0/2.0*log(Delta);
  params.alpha_x = 0;
  params.alpha_y = 0;

  params.theta_x = exp(-theta_hat * 1.0*Delta);
  params.theta_y = exp(-theta_hat * 1.0*Delta);

  params.tau_x = tau_hat * sqrt((1 - exp(-2*theta_hat*Delta))/(2*theta_hat));
  params.tau_y = tau_hat * sqrt((1 - exp(-2*theta_hat*Delta))/(2*theta_hat));
  params.tau_rho = 0.1;

  params.leverage_x_rho = 0;
  params.leverage_y_rho = 0;

  std::vector<BrownianMotion> BMs (N);

  long unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  seed = 10;

  unsigned buffer = 0;
  generate_data(BMs,
		params,
		6.5*3600*1,
		seed,
		buffer);
  //  ############################################# //
  std::vector<BrownianMotion> points_for_kriging (N);
  for (unsigned i=0; i<N; ++i) {
    std::cout << "x0=" << BMs[i].get_x_0() << std::endl;
    std::cout << "xT=" << BMs[i].get_x_T() << std::endl;
    points_for_kriging[i] = BMs[i];
  }

  std::string output_file_name = file_prefix +
    "-number-points-" + argv[1] +
    ".csv";

  std::ofstream output_file;
  output_file.open(output_file_name);
  
  for (unsigned i=0; i<N; ++i) {
    output_file << points_for_kriging[i];
  }
  output_file.close();

//     parameters_nominal params = parameters_nominal();
//     GaussianInterpolator GP_prior = GaussianInterpolator(points_for_integration,
// 							 points_for_kriging,
// 							 params);
//     // for (unsigned i=0; i<points_for_kriging.size(); ++i) {
//     //   for (unsigned j=0; j<points_for_kriging.size(); ++j) {
//     // 	std::cout << gsl_matrix_get(GP_prior.Cinv, i,j) << " ";
//     //   }
//     //   std::cout << std::endl;
//     // }

//     double integral = 0;
//     for (unsigned i=0; i<points_for_integration.size(); ++i) {
//       double add = GP_prior(points_for_integration[i]);
//       integral = integral +
// 	add;
//       if (add < 0) {
// 	std::cout << points_for_integration[i] << std::endl;
//       }
//     }
//     std::cout << "Integral = " << integral << std::endl;

//     // MultivariateNormal mvtnorm = MultivariateNormal();
//     // std::cout << mvtnorm.dmvnorm(N,y,mu,C) << std::endl;

//     nlopt::opt opt(nlopt::LN_NELDERMEAD, 32);
//     //    std::vector<double> lb =

//     // std::vector<double> x = params.as_vector();
//     // std::cout << optimization_wrapper(x, NULL, &points_for_kriging) << std::endl;

//     gsl_rng_free(r_ptr_local);


    return 0;
}
