library("data.table");

sim.data <- fread(input = "../../data.csv",
                  header = TRUE,
                  sep = ",")
sample.data <- sim.data;

post.mean <- fread(input = "../../inference-ochl-sigma_x-0.3-sigma_y-0.2-rho-0.0-dx-like-0.1.csv",
                   header = TRUE,
                   sep = ",")

pdf("ochl.pdf")
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
     ylim = c(min(c(min(post.mean[, mean_log_sigma_y] - 2*sqrt(post.mean[, var_log_sigma_y])),
                    min(sample.data[, log.sigma.y]))),
              max(c(max(post.mean[, mean_log_sigma_y] + 2*sqrt(post.mean[, var_log_sigma_y])),
                    max(sample.data[, log.sigma.y])))));
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
dev.off();
