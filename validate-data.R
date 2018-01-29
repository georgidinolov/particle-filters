rm(list=ls())
library("data.table");
library("latex2exp");
## file.name = "test-classical-beta-nparticles-80-theta-x";
file.name.1 = "test-full-tmp-real-data-sigma_x-0.50-sigma_y-0.50-rho-0.0-dx-likelihood-0.003905-nparticles-10-ALL";
    ## "test-full-tmp-sigma_x-0.30-sigma_y-0.10-rho-0.7-dx-likelihood-0.003905-nparticles-80-ALL"
file.name.2 = "test-full-beta-sigma_x-0.40-sigma_y-0.40-rho-0.8-dx-likelihood-0.003905-nparticles-20-ALL"

sim.data <- fread(input = "../../data.csv",
                  header = TRUE,
                  sep = ",");
sample.data <- sim.data;

post.mean.1 <- fread(input = paste("../../", file.name.1, ".csv", sep=""),
                   header = TRUE,
                   sep = ",");
post.mean.2 <- fread(input = paste("../../", file.name.2, ".csv", sep=""),
                   header = TRUE,
                   sep = ",");

pdf(paste(file.name, ".pdf", sep=""));
par(mfrow=c(3,2))
plot(sim.data[, x], type = "l")
plot(sim.data[, y], type = "l")
dev.off()
##
pdf("log-sigma-x.pdf", 6, 3);
par(mar=c(2,4,1,1))
plot(sim.data[-1, log.sigma.x], type = "l", lwd=2,
     ylim = c(min(post.mean.2[, mean_log_sigma_x] - 2*sqrt(post.mean.2[, var_log_sigma_x])),
              max(post.mean.2[, mean_log_sigma_x] + 2*sqrt(post.mean.2[, var_log_sigma_x]))),
     xlab = "", ylab = TeX("$\\log(\\sigma)_x)"))
lines(post.mean.1[, mean_log_sigma_x], col = "blue")
lines(post.mean.1[, mean_log_sigma_x] - 2*sqrt(post.mean.1[, var_log_sigma_x]),
      col="blue", lty = "dashed")
lines(post.mean.1[, mean_log_sigma_x] + 2*sqrt(post.mean.1[, var_log_sigma_x]),
      col="blue", lty = "dashed")

lines(post.mean.2[, mean_log_sigma_x], col = "red")
lines(post.mean.2[, mean_log_sigma_x] - 2*sqrt(post.mean.2[, var_log_sigma_x]),
      col="red", lty = "dashed")
lines(post.mean.2[, mean_log_sigma_x] + 2*sqrt(post.mean.2[, var_log_sigma_x]),
      col="red", lty = "dashed")
dev.off();
##
pdf("log-sigma-y.pdf", 6, 3);
par(mar=c(2,4,1,1))
plot(sim.data[-1, log.sigma.y], type = "l",  lwd=2,
     ylim = c(min(c(min(post.mean.1[, mean_log_sigma_y] - 2*sqrt(post.mean.1[, var_log_sigma_y])),
                    min(sample.data[, log.sigma.y]))),
              max(c(max(post.mean.1[, mean_log_sigma_y] + 2*sqrt(post.mean.1[, var_log_sigma_y])),
                    max(sample.data[, log.sigma.y])))),
          xlab = "", ylab = TeX("$\\log(\\sigma)_y)"));
lines(post.mean.1[, mean_log_sigma_y], col = "blue")
lines(post.mean.1[, mean_log_sigma_y] - 2*sqrt(post.mean.1[, var_log_sigma_y]),
      col="blue", lty = "dashed")
lines(post.mean.1[, mean_log_sigma_y] + 2*sqrt(post.mean.1[, var_log_sigma_y]),
      col="blue", lty = "dashed")

lines(post.mean.2[, mean_log_sigma_y], col = "red")
lines(post.mean.2[, mean_log_sigma_y] - 2*sqrt(post.mean.2[, var_log_sigma_y]),
      col="red", lty = "dashed")
lines(post.mean.2[, mean_log_sigma_y] + 2*sqrt(post.mean.2[, var_log_sigma_y]),
      col="red", lty = "dashed");
dev.off();
##
pdf("logit-rho.pdf", 6, 3);
par(mar=c(2,4,1,1))
plot(sim.data[, rho.tilde], type = "l",  lwd=2,
     ylim = c(min( post.mean.1[, mean_rho_tilde] - 2*sqrt(post.mean.1[, var_rho_tilde])  ),
              max( post.mean.1[, mean_rho_tilde] + 2*sqrt(post.mean.1[, var_rho_tilde]))),
     xlab = "", ylab = TeX("$logit((\\rho+1)/2)"));

lines(post.mean.1[, mean_rho_tilde],
      col = "blue")
lines(post.mean.1[, mean_rho_tilde] - 2*sqrt(post.mean.1[, var_rho_tilde]),
      col="blue", lty = "dashed")
lines(post.mean.1[, mean_rho_tilde] + 2*sqrt(post.mean.1[, var_rho_tilde]),
      col="blue", lty = "dashed")

lines(post.mean.2[, mean_rho_tilde],
      col = "red")
lines(post.mean.2[, mean_rho_tilde] - 2*sqrt(post.mean.2[, var_rho_tilde]),
      col="red", lty = "dashed")
lines(post.mean.2[, mean_rho_tilde] + 2*sqrt(post.mean.2[, var_rho_tilde]),
      col="red", lty = "dashed")
dev.off();
