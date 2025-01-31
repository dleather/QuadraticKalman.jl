# Plotting Guide

```julia
using QuadraticKalman, Plots

# After running filter/smoother
kf_truth_plot = kalman_filter_truth_plot(true_states, filter_results)
plot(kf_truth_plot)


ks_truth_plot = kalman_smoother_truth_plot(true_states, smoother_results) 
plot(ks_truth_plot)


kf_plot = kalman_filter_plot(filter_results)
plot(kf_plot)

ks_plot = kalman_smoother_plot(smoother_results)
plot(ks_plot)
```