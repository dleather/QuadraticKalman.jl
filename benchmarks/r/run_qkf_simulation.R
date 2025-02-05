library(jsonlite)
library(tidyverse)

# Source the R code
source("augmented_model.R")
source("checking_arguments.R")
source("loglik.R")
source("predict.R")
source("update.R")
source("quadratic_kalman_filtering.R")
source("quadratic_kalman_smoothing.R")
source("unconditional_moments.R")
source("update.R")

# Read the simulated data
X <- read.csv("../simulated_data/simulated_states.csv")
Y <- read.csv("../simulated_data/simulated_measurements.csv")
params <- fromJSON("../simulated_data/simulated_params.json")

# Convert parameters from JSON format to R matrices/arrays
mu <- matrix(unlist(params$mu), ncol = 1)
Phi <- matrix(unlist(params$Phi), nrow = 2, byrow = TRUE)
Omega <- matrix(unlist(params$Omega), nrow = 2, byrow = TRUE)
a <- matrix(unlist(params$a), ncol = 1)
B <- matrix(unlist(params$B), nrow = 2, byrow = TRUE)
D <- matrix(unlist(params$D), nrow = 2, byrow = TRUE)
alpha <- matrix(unlist(params$alpha), nrow = 2, byrow = TRUE)

# Convert C array (Need to hand code)
C1 <- matrix(c(0.2, 0.1, 0.1, 0.0), nrow = 2, ncol = 2, byrow = TRUE)
C2 <- matrix(c(0.0, 0.1, 0.1, 0.2), nrow = 2, ncol = 2, byrow = TRUE)
C <- array(c(C1, C2), dim = c(2, 2, 2))
# Convert data to matrices
Y <- as.matrix(Y)
X <- as.matrix(X)

# Get dimensions
n <- nrow(Phi)  # state dimension
m <- nrow(B)    # measurement dimension
T <- nrow(Y)    # time periods

# Initialize filter using unconditional moments
init <- initialize.filter(mu, Phi, Omega)
Z0 <- init$Z0
V0 <- init$V0

# Run the Quadratic Kalman Filter
qkf_results <- QKF(
    Z0 = Z0,
    V0 = V0,
    Mu.t = mu,
    Phi.t = Phi,
    Omega.t = Omega,
    A.t = a,
    B.t = B,
    C.t = C,
    D.t = D,
    observable = t(Y[2:100,]))

# Run the Quadratic Kalman Smoother
qks_results <- QKS(qkf_results)

# Extract filtered and smoothed states
filtered_states <- t(qkf_results$Z.updated[1:n,])  # First n rows contain state means
smoothed_states <- t(qks_results$Z.smoothed[1:n,])

# Extract smoothed state variances (diagonal elements of covariance matrices)
filtered_vars <- array(0, dim = c(T-1, n))
for (t in 1:T-1) {
    V_t <- matrix(qkf_results$P.updated[1:n, 1:n, t], n, n)
    filtered_vars[t,] <- diag(V_t)
}

smoothed_vars <- array(0, dim = c(T-1, n))
for (t in 1:T-1) {
    V_t <- matrix(qks_results$P.smoothed[1:n, 1:n, t], n, n)
    smoothed_vars[t,] <- diag(V_t)
}

# Get log-likelihood vector
loglik_vector <- qkf_results$loglik.vector

# Calculate confidence intervals (95%)
z_score <- 1.96  # for 95% confidence interval
smoothed_ci_lower <- smoothed_states - z_score * sqrt(smoothed_vars)
smoothed_ci_upper <- smoothed_states + z_score * sqrt(smoothed_vars)

# Calculate RMSEs
rmse_filtered <- sqrt(colMeans((X[2:T,] - filtered_states)^2))
rmse_smoothed <- sqrt(colMeans((X[2:T,] - smoothed_states)^2))

cat("RMSE for filtered states:\n")
cat("State 1:", rmse_filtered[1], "\n")
cat("State 2:", rmse_filtered[2], "\n")
cat("\nRMSE for smoothed states:\n")
cat("State 1:", rmse_smoothed[1], "\n")
cat("State 2:", rmse_smoothed[2], "\n")

library(ggplot2)
library(scales)  # For better color handling

# Create data frame for plotting (keeping your existing code)
T_bar = T - 1
plot_data <- data.frame(
    Time = rep(1:T_bar, 6),
    Value = c(X[,1], X[,2],
              filtered_states[,1], filtered_states[,2],
              smoothed_states[,1], smoothed_states[,2]),
    State = rep(rep(c("State 1", "State 2"), each = T-1), 3),
    Type = rep(c("True", "Filtered", "Smoothed"), each = 2*T)
)

# Confidence interval data frame (keeping your existing code)
ci_data <- data.frame(
    Time = rep(1:T, 2),
    Lower = c(smoothed_ci_lower[,1], smoothed_ci_lower[,2]),
    Upper = c(smoothed_ci_upper[,1], smoothed_ci_upper[,2]),
    State = rep(c("State 1", "State 2"), each = T)
)

# Create the enhanced plot
ggplot() +
    # Add confidence bands with improved aesthetics
    geom_ribbon(data = ci_data,
                aes(x = Time, ymin = Lower, ymax = Upper),
                fill = "grey90", alpha = 0.75) +
    # Add lines with better colors and clear distinctions
    geom_line(data = plot_data,
              aes(x = Time, y = Value, color = Type, linetype = Type),
              size = 0.8) +
    # Set custom colors that are visually distinct and professional
    scale_color_manual(values = c(
        "True" = "#2563eb",      # Blue
        "Filtered" = "#dc2626",  # Red
        "Smoothed" = "#059669"   # Green
    )) +
    # Set custom line types
    scale_linetype_manual(values = c(
        "True" = "solid",
        "Filtered" = "longdash",
        "Smoothed" = "dotted"
    )) +
    # Facet the plots with improved spacing
    facet_wrap(~State, scales = "free_y", nrow = 2) +
    # Apply a clean theme with improvements
    theme_minimal() +
    theme(
        # Improve plot background and grid
        panel.background = element_rect(fill = "white", color = NA),
        panel.grid.major = element_line(color = "grey95"),
        panel.grid.minor = element_line(color = "grey98"),
        
        # Enhance text elements
        plot.title = element_text(size = 14, face = "bold", margin = margin(b = 20)),
        plot.caption = element_text(size = 8, color = "grey30", margin = margin(t = 10)),
        
        # Improve legend
        legend.position = "top",
        legend.box.margin = margin(b = 10),
        legend.background = element_rect(fill = "white", color = NA),
        
        # Adjust strip (facet) appearance
        strip.text = element_text(face = "bold", size = 10),
        strip.background = element_rect(fill = "grey95"),
        
        # Add padding
        plot.margin = margin(t = 20, r = 20, b = 20, l = 20)
    ) +
    # Improve labels
    labs(
        title = "Comparison of True, Filtered, and Smoothed States",
        y = "Value",
        x = "Time",
        caption = "Grey bands show 95% confidence intervals for smoothed states",
        color = "State Type",
        linetype = "State Type"
    )

# Save results for Julia comparison
results_for_julia <- list(
    filtered_states = filtered_states,
    filtered_vars = filtered_vars,
    smoothed_states = smoothed_states,
    smoothed_vars = smoothed_vars,
    loglik_vector = loglik_vector
)

saveRDS(qkf_results, "../r_results/qkf_results.rds")
saveRDS(qks_results, "../r_results/qks_results.rds")
