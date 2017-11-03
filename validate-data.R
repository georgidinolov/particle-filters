library("data.table");
file.name = "inference-ochl-sigma_x-0.30-sigma_y-0.30-rho-0.0-dx-likelihood-0.003900-nparticles-100-new-layout-old-bounds"

sim.data <- fread(input = "../../data.csv",
                  header = TRUE,
                  sep = ",");
sample.data <- sim.data;

post.mean <- fread(input = paste("../../", file.name, ".csv", sep=""),
                   header = TRUE,
                   sep = ",");

pdf(paste(file.name, ".pdf", sep=""));
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
