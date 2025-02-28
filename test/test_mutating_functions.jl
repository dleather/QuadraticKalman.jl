using Pkg
import QuadraticKalman as QK
using Random, Test, LinearAlgebra  # Add LinearAlgebra

@testset "Mutating Filter Functions" begin
    # Setup test data
    Random.seed!(123)
    N, M, T = 2, 1, 10
    
    # Create model parameters
    state = QK.StateParams(N, [0.1, 0.2], [0.5 0.1; 0.1 0.3], [0.1 0.0; 0.0 0.1])
    
    # Fix MeasParams constructor call to match expected types
    meas = QK.MeasParams(
        M, N,
        [0.0],             # a vector
        Matrix([1.0 1.0]), # ensure this is a matrix
        [Matrix([0.1 0.05; 0.05 0.1]) for _ in 1:M], # M-length vector of N×N matrices
        Matrix([0.1]'),    # ensure this is a matrix
        Matrix([0.0]')     # ensure this is a matrix
    )
    model = QK.QKModel(state, meas)
    
    # Create data
    Y = randn(M, T)
    data = QK.QKData(Y)
    
    # First run the non-mutating version to get a reference
    reference_results = QK.qkf_filter(data, model)
    
    # Now test the mutating version directly
    # The mutating version takes data and model as arguments, not a FilterOutput
    mutating_results = QK.qkf_filter!(data, model)
    
    # Verify results are populated
    @test any(mutating_results.Z_tt .!= 0)
    @test any(mutating_results.P_tt .!= 0)
    @test any(mutating_results.ll_t .!= 0)
    
    # Verify that both versions produce similar results
    @test isapprox(reference_results.ll_t, mutating_results.ll_t, rtol=1e-5)
    @test isapprox(reference_results.Z_tt, mutating_results.Z_tt, rtol=1e-5)
    
    # Test smoother with the filter results
    reference_smoother = QK.qkf_smoother(reference_results, model)
    
    # Test the mutating smoother
    # Create a pre-allocated output for the smoother
    P = N + N^2  # Augmented state dimension
    smoother_results = QK.SmootherOutput(
        zeros(P, T),  # Z_smooth
        zeros(P, P, T) # P_smooth
    )
    
    # Run the smoother with the filter results
    QK.qkf_smoother!(smoother_results, reference_results, model)
    
    # Verify smoother results are populated
    @test any(smoother_results.Z_smooth .!= 0)
    @test any(smoother_results.P_smooth .!= 0)
    
    # Verify that both smoother versions produce similar results
    @test isapprox(reference_smoother.Z_smooth, smoother_results.Z_smooth, rtol=1e-5)
    @test isapprox(reference_smoother.P_smooth, smoother_results.P_smooth, rtol=1e-5)
end


