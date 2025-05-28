using Test
import QuadraticKalman as QK
using LinearAlgebra, Random
using CSV, DataFrames
using RData, JSON
using Aqua, Plots
@testset "QuadraticKalman.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(QK, ambiguities=false, stale_deps = False)
        Aqua.test_ambiguities(QK)
        Aqua.test_unbound_args(QK)
        Aqua.test_undefined_exports(QK)
        Aqua.test_project_extras(QK)
        Aqua.test_deps_compat(QK)
        Aqua.test_piracies(QK)
        Aqua.test_persistent_tasks(QK)
    end
    include("test_data.jl")
    include("test_params.jl")
    include("test_matrix_utils.jl")
    include("test_augmented_moments_and_likelihood.jl")
    include("test_core_filter.jl")
    include("test_core_smoother.jl")
    include("test_plots.jl")
    include("test_model_params_conversion.jl")
    include("test_likelihood_edge_cases.jl")
    include("test_logpdf_mvn.jl")
    include("test_qkf_negloglik.jl")
    include("test_mutating_filter.jl")
    include("test_end_to_end.jl")
    if VERSION >= v"1.11.0"
        include("test_r_comparison.jl")
    else
        @info "Skipping R comparison tests on Julia < 1.11.0"
    end
end
