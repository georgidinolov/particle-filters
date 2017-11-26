cc_binary(
	name = "2d-stochastic-vol-full-beta",
	srcs = ["2d-stochastic-vol-full-beta.cpp"],
	copts = ["-Isrc/multivariate-normal",
	      	 "-Isrc/finite-element-igraph",
		 "-Isrc/igraph-0.7.1/include",
		 "-Isrc/SV-with-leverage/src",
		 "-Isrc/nlopt/api",
		 "-fopenmp",
		 "-O3"],
 	deps = [":DataTypes"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas", "-fopenmp"],		 
)

cc_binary(
	name = "2d-stochastic-vol-classical-beta",
	srcs = ["2d-stochastic-vol-classical-beta.cpp"],
	copts = ["-Isrc/multivariate-normal",
	      	 "-Isrc/finite-element-igraph",
		 "-Isrc/igraph-0.7.1/include",
		 "-Isrc/SV-with-leverage/src",
		 "-Isrc/nlopt/api",
		 "-fopenmp",
		 "-O3"],
 	deps = [":DataTypes"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas", "-fopenmp"],		 
)

cc_binary(
	name = "2d-stochastic-vol-full-classical",
	srcs = ["2d-stochastic-vol-full-classical.cpp"],
	copts = ["-Isrc/multivariate-normal",
	      	 "-Isrc/finite-element-igraph",
		 "-Isrc/igraph-0.7.1/include",
		 "-Isrc/SV-with-leverage/src",
		 "-Isrc/nlopt/api",
		 "-fopenmp",
		 "-O3"],
 	deps = [":DataTypes"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas", "-fopenmp"],		 
)

cc_binary(
	name = "2d-stochastic-vol-full-tmp-real-data",
	srcs = ["2d-stochastic-vol-full-tmp-real-data.cpp"],
	copts = ["-Isrc/multivariate-normal",
	      	 "-Isrc/finite-element-igraph",
		 "-Isrc/igraph-0.7.1/include",
		 "-Isrc/SV-with-leverage/src",
		 "-Isrc/nlopt/api",
		 "-fopenmp",
		 "-O3"],
 	deps = [":DataTypes"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas", "-fopenmp"],		 
)

cc_binary(
	name = "2d-stochastic-vol-full-tmp",
	srcs = ["2d-stochastic-vol-full-tmp.cpp"],
	copts = ["-Isrc/multivariate-normal",
	      	 "-Isrc/finite-element-igraph",
		 "-Isrc/igraph-0.7.1/include",
		 "-Isrc/SV-with-leverage/src",
		 "-Isrc/nlopt/api",
		 "-fopenmp",
		 "-O3"],
 	deps = [":DataTypes"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas", "-fopenmp"],		 
)

cc_binary(
	name = "2d-stochastic-vol-full",
	srcs = ["2d-stochastic-vol-full.cpp"],
	copts = ["-Isrc/multivariate-normal",
	      	 "-Isrc/finite-element-igraph",
		 "-Isrc/igraph-0.7.1/include",
		 "-Isrc/SV-with-leverage/src",
		 "-fopenmp",
		 "-O3"],
 	deps = [":DataTypes"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas", "-fopenmp"],		 
)

cc_binary(
	name = "2d-stochastic-vol-simple",
	srcs = ["2d-stochastic-vol-simple.cpp"],
	copts = ["-Isrc/multivariate-normal",
	      	 "-Isrc/finite-element-igraph",
		 "-Isrc/SV-with-leverage/src",	
		 "-Isrc/igraph-0.7.1/include",
		 "-fopenmp",
		 "-O3"],
 	deps = [":DataTypes",
	        "//src/SV-with-leverage/src:prior-types"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas", "-fopenmp"],		 
)

cc_binary(
	name = "2d-stochastic-vol-classical",
	srcs = ["2d-stochastic-vol-classical.cpp"],
	deps = [":DataTypes"],
	copts = ["-Isrc/multivariate-normal",
	      	 "-Isrc/finite-element-igraph",
		 "-Isrc/igraph-0.7.1/include",
		 "-fopenmp",
		 "-O3"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas", "-fopenmp"],		 
)

cc_library(
	name = "DataTypes",
	hdrs = ["DataTypes.hpp"],
	srcs = ["DataTypes.cpp"],
	copts = ["-Isrc/multivariate-normal",
		 "-Isrc/brownian-motion",
		 "-Isrc/finite-element-igraph",
		 "-Isrc/igraph-0.7.1/include",
		 "-Isrc/SV-with-leverage/src",
		 "-Isrc/nlopt/api",
		 "-O3"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas"],
	deps = ["//src/multivariate-normal:multivariate-normal",
		"//src/brownian-motion:2d-brownian-motion",
		"//src/finite-element-igraph:bivariate-solver",
		"//src/SV-with-leverage/src:prior-types"],
)

cc_binary(
	name = "test-pos-def-matrix",
	srcs = ["test-pos-def-matrix.cpp"],
	linkopts = ["-lm", "-lgsl", "-lgslcblas", "-fopenmp"],
)