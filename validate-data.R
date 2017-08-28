library("data.table");

sim.data <- fread(input = "data.csv",
                  header = TRUE,
                  sep = ",")
sample.data <- sim.data;

post.mean <- fread(input = "inference.csv",
                   header = TRUE,
                   sep = ",")

par(mfrow=c(3,2))
plot(sim.data[, x], type = "l")
plot(sim.data[, y], type = "l")
##
plot(sim.data[, log.sigma.x], type = "l",
     ylim = c(min(post.mean[, mean_log_sigma_x] - 2*sqrt(post.mean[, var_log_sigma_x])),
              max(post.mean[, mean_log_sigma_x] + 2*sqrt(post.mean[, var_log_sigma_x])) ))
lines(post.mean[, mean_log_sigma_x], col = "blue")
lines(post.mean[, mean_log_sigma_x] - 2*sqrt(post.mean[, var_log_sigma_x]),
      col="blue", lty = "dashed")
lines(post.mean[, mean_log_sigma_x] + 2*sqrt(post.mean[, var_log_sigma_x]),
      col="blue", lty = "dashed")
##
plot(sim.data[, log.sigma.y], type = "l",
          ylim = c(min(post.mean[, mean_log_sigma_y] - 2*sqrt(post.mean[, var_log_sigma_y])),
              max(post.mean[, mean_log_sigma_y] + 2*sqrt(post.mean[, var_log_sigma_y])) ))
lines(post.mean[, mean_log_sigma_y], col = "blue")
lines(post.mean[, mean_log_sigma_y] - 2*sqrt(post.mean[, var_log_sigma_y]),
      col="blue", lty = "dashed")
lines(post.mean[, mean_log_sigma_y] + 2*sqrt(post.mean[, var_log_sigma_y]),
      col="blue", lty = "dashed")
##
plot(sim.data[, rho.tilde], type = "l",
     ylim = c(min( post.mean[, mean_rho_tilde] - 2*sqrt(post.mean[, var_rho_tilde])  ),
               max( post.mean[, mean_rho_tilde] + 2*sqrt(post.mean[, var_rho_tilde]) )))

lines(post.mean[, mean_rho_tilde],
      col = "blue")

lines(post.mean[, mean_rho_tilde] - 2*sqrt(post.mean[, var_rho_tilde]),
      col="blue", lty = "dashed")
lines(post.mean[, mean_rho_tilde] + 2*sqrt(post.mean[, var_rho_tilde]),
      col="blue", lty = "dashed")
