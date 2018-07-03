#include <iostream>
#include <stdio.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <limits>

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

int
main (void)
{
  unsigned size = 13;
  gsl_matrix * a = gsl_matrix_alloc (size, size);
  gsl_matrix * work = gsl_matrix_alloc (size, size);
  gsl_matrix * b = gsl_matrix_alloc (size, size);
  gsl_matrix_set_zero(b);
  
  FILE * f = fopen ("/soe/gdinolov/PDE-solvers/scaled_cov.dat", "rb");
  gsl_matrix_fread (f, a);
  fclose (f);
  print_matrix(a, size, size);
  gsl_matrix_memcpy(work, a);

  gsl_error_handler_t* old_handler = gsl_set_error_handler_off();
  int status = gsl_linalg_cholesky_decomp(work);
  if (status == GSL_EDOM) {
    printf("Cannot decompose\n");
    gsl_vector_view a_diag = gsl_matrix_diagonal(a);
    gsl_vector_view b_diag = gsl_matrix_diagonal(b);
    gsl_vector_memcpy(&b_diag.vector, &a_diag.vector);
  }
  gsl_set_error_handler(NULL);

  for (unsigned i=0; i<size; ++i) {
    if ( gsl_matrix_get(b,i,i) < std::numeric_limits<double>::epsilon() ) {
      printf("detected small diagonal entry\n");
      gsl_matrix_set(b, i,i, 10.0);
    }
  }
  print_matrix(b, size, size);
  gsl_linalg_cholesky_decomp(b);
  return (0);
}
