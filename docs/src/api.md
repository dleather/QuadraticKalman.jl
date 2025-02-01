# API Reference

## Types

### Main Types
```@docs
QKData
QKModel
StateParams
MeasParams
Moments
AugStateParams
QKFOutput
FilterOutput
SmootherOutput
```


### Plotting Types
```@docs
KalmanFilterTruthPlot
KalmanSmootherTruthPlot 
KalmanFilterPlot
KalmanSmootherPlot
```


## Functions    

### Filtering and Smoothing
```@docs
qkf_filter
qkf_filter!
qkf_smoother
qkf_smoother!
qkf
```

### Convenience Functions
```@docs
get_measurement
qkf_negloglik
model_to_params
params_to_model
```


## Plotting API

```@docs
kalman_filter_truth_plot
kalman_smoother_truth_plot
kalman_filter_plot
kalman_smoother_plot
```
