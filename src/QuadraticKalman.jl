module QuadraticKalman

using LinearAlgebra, Parameters, SpecialFunctions, DifferentiableEigen, 
    LogExpFunctions, SparseArrays, Random, RecipesBase

#=
Original Model Specification:
---------------------------
State Equation (Eq 1a):
    Xₜ = μ + ΦXₜ₋₁ + Ωεₜ 
    - Xₜ is an N×1 state vector that can be partially observable
    - μ is an N×1 constant vector
    - Φ is an N×N transition matrix
    - Ω is an N×N matrix scaling the noise
    - εₜ is an N×1 standard normal noise vector

Observation Equation (Eq 1b):
    Yₜ = A + BXₜ + αYₜ₋₁ + ∑ₖ₌₁ᵐ eₖXₜ'C⁽ᵏ⁾Xₜ + D ηₜ
    - Yₜ is an M×1 fully observable measurement vector
    - A is an M×1 constant vector
    - B is an M×N observation matrix
    - α captures autoregressive effects
    - C⁽ᵏ⁾ matrices capture quadratic effects
    - D scales time-varying observation noise
    - ηₜ is an M×1 standard normal noise vector

Covariance Matrices:
    Σ = ΩΩ' (N×N state covariance)
    V = DD' (M×M observation covariance)

Augmented State-Space Form:
--------------------------
State Vector:
    Zₜ := [Xₜ, vec(XₜXₜ')] 
    - Augmented (N + N²)×1 state vector combining linear and quadratic terms

Augmented State Equation:
    Zₜ = μ̃ + Φ̃Zₜ₋₁ + Ω̃ₜ₋₁ξₜ
    - μ̃, Φ̃ are augmented versions of original parameters
    - Ω̃ₜ₋₁ captures state-dependent volatility

Augmented Observation Equation:
    Yₜ = A + B̃Zₜ + D ηₜ
    - B̃ combines linear and quadratic observation effects
=#
# Import types
include("types/parameters.jl")
include("types/data.jl")
include("types/outputs.jl")

# Import matrix utilities
include("matrix_utils/commutation.jl")
include("matrix_utils/selection.jl")
include("matrix_utils/numerical.jl")

# Import core functionality
include("core/augmented_moments.jl")
include("core/likelihood.jl")
include("core/filter.jl")
include("core/smoother.jl")

# Import recipes
include("plotting/recipes.jl")

# Export public types and functions
export QKData, QKModel, QKFOutput, FilterOutput, SmootherOutput     # Types
export qkf_filter, qkf_filter!, qkf_smoother, qkf_smoother!, qkf    # Functions
export get_measurement                                              # Convenience function
export kalman_filter_truth_plot, kalman_smoother_truth_plot, kalman_filter_plot
export kalman_smoother_plot, KalmanFilterPlot, KalmanSmootherPlot
export KalmanFilterTruthPlot, KalmanSmootherTruthPlot
export params_to_model, qkf_negloglik, model_to_params
export StateParams, MeasParams, Moments, AugStateParams

end # module