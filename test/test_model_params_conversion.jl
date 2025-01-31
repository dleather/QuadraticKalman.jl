using QuadraticKalman
using Test, LinearAlgebra

@testset "Model-Parameter Conversion Tests" begin
    # Set up test parameters
    N = 2  # Number of states 
    M = 2  # Number of measurements

    # Generate test model parameters
    Phi = [0.5 0.1; 0.1 0.3]
    mu = [0.1, 0.2]
    Sigma = [0.6 0.15; 0.15 0.4]
    Omega = cholesky(Sigma).L

    A = [0.0, 0.0]
    B = [1.0 0.0; 0.0 1.0]
    C = [[0.2 0.1; 0.1 0.0],
         [0.0 0.1; 0.1 0.2]]
    V = [0.2 0.0; 0.0 0.2]
    D = cholesky(V).L
    alpha = zeros(M, M)

    # Create test model
    model = QKModel(N, M, mu, Phi, Omega, A, B, C, D, alpha)

    # Test model -> params -> model conversion
    @testset "Round-trip conversion" begin
        params = model_to_params(model)
        model_from_params = params_to_model(params, N, M)

        # Test state parameters
        @test model.state.mu ≈ model_from_params.state.mu
        @test model.state.Phi ≈ model_from_params.state.Phi
        @test model.state.Omega ≈ model_from_params.state.Omega
        @test model.state.N == model_from_params.state.N
        @test model.state.Sigma ≈ model_from_params.state.Sigma

        # Test measurement parameters
        @test model.meas.A ≈ model_from_params.meas.A
        @test model.meas.B ≈ model_from_params.meas.B
        @test all(model.meas.C[i] ≈ model_from_params.meas.C[i] for i in 1:M)
        @test model.meas.D ≈ model_from_params.meas.D
        @test model.meas.alpha ≈ model_from_params.meas.alpha
        @test model.meas.M == model_from_params.meas.M
        @test model.meas.V ≈ model_from_params.meas.V

        # Test model moments
        @test model.moments.state_mean ≈ model_from_params.moments.state_mean
        @test model.moments.state_cov ≈ model_from_params.moments.state_cov
        @test model.moments.aug_mean ≈ model_from_params.moments.aug_mean
        @test model.moments.aug_cov ≈ model_from_params.moments.aug_cov

        # Test augmented state parameters
        @test model.aug_state.mu_aug ≈ model_from_params.aug_state.mu_aug
        @test model.aug_state.Phi_aug ≈ model_from_params.aug_state.Phi_aug
        @test model.aug_state.B_aug ≈ model_from_params.aug_state.B_aug
        @test model.aug_state.H_aug ≈ model_from_params.aug_state.H_aug
        @test model.aug_state.G_aug ≈ model_from_params.aug_state.G_aug
        @test model.aug_state.Lambda ≈ model_from_params.aug_state.Lambda
        @test model.aug_state.L1 ≈ model_from_params.aug_state.L1
        @test model.aug_state.L2 ≈ model_from_params.aug_state.L2
        @test model.aug_state.L3 ≈ model_from_params.aug_state.L3
        @test model.aug_state.P ≈ model_from_params.aug_state.P
    end

    @testset "Parameter vector length" begin
        params = model_to_params(model)
        expected_length = N + N^2 + N*(N+1)÷2 + M + M*N + M*N^2 + M*(M+1)÷2 + M^2
        @test length(params) == expected_length
    end

    @testset "Invalid parameter vector" begin
        params = model_to_params(model)
        wrong_params = vcat(params, 0.0)  # Add extra parameter
        @test_throws AssertionError params_to_model(wrong_params, N, M)
    end
end
