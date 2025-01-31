using Test
import QuadraticKalman as QK
using LinearAlgebra, Random
using CSV, DataFrames
using RData, JSON
using Aqua, Plots
@testset "QuadraticKalman.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(QK)
    end
    include("test_data.jl")
    include("test_params.jl")
    include("test_matrix_utils.jl")
    include("test_augmented_moments_and_likelihood.jl")
    include("test_core_filter.jl")
    include("test_core_smoother.jl")
    include("test_plots.jl")
    include("test_model_params_conversion.jl")
    #include("test_end_to_end.jl")
    try
        include("test_r_comparison.jl")  # If you have R-based comparisons
    catch err
        @warn "Skipping R comparison tests: $err"
    end
end
