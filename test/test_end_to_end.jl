using Test
import QuadraticKalman as QK
using LinearAlgebra, Random

# Only use CSV and DataFrames if they're available
const has_csv = try
    using CSV, DataFrames
    true
catch
    false
end

@testset "End-to-End Tests" begin
    @testset "Basic Workflow" begin
        # Setup test environment
        Random.seed!(42)
        
        # Define model dimensions
        N = 2  # Number of states
        M = 2  # Number of measurements
        T = 100  # Number of time periods
        
        # Generate stable state transition parameters
        Phi = [0.5 0.1; 0.1 0.3]  # Autoregressive matrix
        mu = [0.1, 0.2]  # State drift vector
        Sigma = [0.6 0.15; 0.15 0.4]  # State noise covariance matrix
        Omega = cholesky(Sigma).L  # Scale for state noise
        
        # Generate measurement parameters
        A = [0.0, 0.0]  # Measurement drift vector
        B = [1.0 0.0; 0.0 1.0]  # Measurement matrix
        C = [[0.2 0.1; 0.1 0.0], [0.0 0.1; 0.1 0.2]]  # Quadratic effect matrices
        V = [0.2 0.0; 0.0 0.2]  # Measurement noise covariance matrix
        D = cholesky(V).L  # Scale for measurement noise
        alpha = zeros(M, M)  # Measurement autoregressive matrix
        
        # Simulate states
        X = zeros(N, T)
        X[:,1] = (I - Phi) \ mu  # Start from unconditional mean
        for t in 1:(T-1)
            shock = randn(N)
            X[:,t+1] = mu + Phi * X[:,t] + Omega * shock
        end
        
        # Simulate observations
        Y = zeros(M, T)
        for t in 1:T
            noise = randn(M)
            xt = X[:,t]
            
            # Linear terms
            Y[:,t] = A + B * xt
            
            if t > 1
                Y[:,t] += alpha * Y[:,t-1]
            end
            
            # Quadratic terms
            for i in 1:M
                Y[i,t] += xt' * C[i] * xt
            end
            
            # Add measurement noise
            Y[:,t] += D * noise
        end
        
        # Create model and data objects
        model = QK.QKModel(N, M, mu, Phi, Omega, A, B, C, D, alpha)
        data = QK.QKData(Y)
        
        # Run filter and smoother
        filter_results = QK.qkf_filter(data, model)
        smoother_results = QK.qkf_smoother(filter_results, model)
        
        # Test filter results
        @testset "Filter results structure" begin
            @test length(filter_results.ll_t) == T-1
            @test size(filter_results.Z_tt) == (N + N^2, T)
            @test !any(isnan, filter_results.Z_tt)
            @test !any(isnan, filter_results.ll_t)
        end
        
        # Test smoother results
        @testset "Smoother results structure" begin
            @test size(smoother_results.Z_smooth) == (N + N^2, T)
            @test !any(isnan, smoother_results.Z_smooth)
        end
        
        # Test parameter conversion
        @testset "Parameter conversion" begin
            params = QK.model_to_params(model)
            model_reconstructed = QK.params_to_model(params, N, M)
            
            # Test state parameters
            @test model_reconstructed.state.N == model.state.N
            @test model_reconstructed.state.mu == model.state.mu
            @test model_reconstructed.state.Phi == model.state.Phi
            @test model_reconstructed.state.Omega == model.state.Omega
            @test model_reconstructed.state.Sigma == model.state.Sigma
            
            # Test measurement parameters
            @test model_reconstructed.meas.M == model.meas.M
            @test model_reconstructed.meas.A == model.meas.A
            @test model_reconstructed.meas.B == model.meas.B
            @test model_reconstructed.meas.C == model.meas.C
            @test model_reconstructed.meas.D == model.meas.D
            @test model_reconstructed.meas.alpha == model.meas.alpha
            
            # Test augmented state parameters
            @test model_reconstructed.aug_state.mu_aug == model.aug_state.mu_aug
            @test model_reconstructed.aug_state.Phi_aug == model.aug_state.Phi_aug
            @test model_reconstructed.aug_state.B_aug == model.aug_state.B_aug
            @test model_reconstructed.aug_state.H_aug == model.aug_state.H_aug
            @test model_reconstructed.aug_state.G_aug == model.aug_state.G_aug
            @test model_reconstructed.aug_state.Lambda == model.aug_state.Lambda
            @test model_reconstructed.aug_state.L1 == model.aug_state.L1
            @test model_reconstructed.aug_state.L2 == model.aug_state.L2
            @test model_reconstructed.aug_state.L3 == model.aug_state.L3
            @test model_reconstructed.aug_state.P == model.aug_state.P
            
            # Test moments
            @test model_reconstructed.moments.state_mean == model.moments.state_mean
            @test model_reconstructed.moments.state_cov == model.moments.state_cov
            @test model_reconstructed.moments.aug_mean == model.moments.aug_mean
            @test model_reconstructed.moments.aug_cov == model.moments.aug_cov
            
        end
        
        # Test negative log-likelihood function
        @testset "Negative log-likelihood" begin
            # Define params here to ensure it's in scope
            params = QK.model_to_params(model)
            nll = QK.qkf_negloglik(params, data, N, M)
            @test nll isa Real
            @test !isnan(nll)
            @test !isinf(nll)
        end
    end
    
    @testset "Error Handling" begin
        # Test with invalid data
        N, M = 2, 2
        invalid_Y = zeros(M, 1)  # Only one time point, should fail
        
        # Create valid model parameters
        mu = [0.0, 0.0]
        Phi = [0.5 0.0; 0.0 0.5]
        Omega = [0.1 0.0; 0.0 0.1]
        A = [0.0, 0.0]
        B = [1.0 0.0; 0.0 1.0]
        C = [[0.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 0.0]]
        D = [0.1 0.0; 0.0 0.1]
        alpha = zeros(M, M)
        
        model = QK.QKModel(N, M, mu, Phi, Omega, A, B, C, D, alpha)
        
        # Test that QKData constructor throws error for invalid data
        @testset "Invalid data detection" begin
            @test_throws ArgumentError QK.QKData(invalid_Y)
        end
        
        # Test with NaN values - only if the implementation checks for NaNs
        @testset "NaN detection" begin
            Y_with_nan = ones(M, 3)
            Y_with_nan[1,2] = NaN
            
            @test_throws AssertionError QK.QKData(Y_with_nan)
        end
    end
    
    if has_csv
        @testset "Integration with CSV Data" begin
            # Create temporary directory for test data
            temp_dir = mktempdir()
            
            try
                # Generate simple test data
                N, M = 1, 1
                T = 10
                # Fix the cumsum issue by specifying dims
                Y = cumsum(randn(M, T), dims=2)  # Simple random walk
                
                @testset "CSV data export" begin
                    # Save to CSV
                    Y_df = DataFrame(Y', :auto)
                    rename!(Y_df, [Symbol("y$i") for i in 1:M])
                    csv_path = joinpath(temp_dir, "test_data.csv")
                    CSV.write(csv_path, Y_df)
                    @test isfile(csv_path)
                end
                
                @testset "CSV data import" begin
                    # Read back and create QKData
                    csv_path = joinpath(temp_dir, "test_data.csv")
                    data_df = CSV.read(csv_path, DataFrame)
                    Y_loaded = Matrix(data_df)'
                    
                    # Create data object
                    data = QK.QKData(Y_loaded)
                    @test data.M == M
                    @test data.T_bar == T-1
                end
                
                @testset "Filter with CSV data" begin
                    # Create data object
                    csv_path = joinpath(temp_dir, "test_data.csv")
                    data_df = CSV.read(csv_path, DataFrame)
                    Y_loaded = Matrix(data_df)'
                    data = QK.QKData(Y_loaded)
                    
                    # Create simple model
                    mu = [0.0]
                    Phi = reshape([0.9], 1, 1)
                    Omega = reshape([0.1], 1, 1)
                    A = [0.0]
                    B = reshape([1.0], 1, 1)
                    C = [reshape([0.0], 1, 1)]
                    D = reshape([0.1], 1, 1)
                    alpha = reshape([0.0], 1, 1)
                    
                    model = QK.QKModel(N, M, mu, Phi, Omega, A, B, C, D, alpha)
                    
                    # Run filter
                    results = QK.qkf_filter(data, model)
                    
                    # Basic checks
                    @test length(results.ll_t) == T-1
                    @test size(results.Z_tt) == (N + N^2, T)
                end
            finally
                # Clean up
                rm(temp_dir, recursive=true)
            end
        end
    else
        @info "CSV package not available, skipping CSV integration tests"
    end
    
    @testset "Performance" begin
        # Generate larger test data
        N, M = 2, 2
        T = 1000
        
        # Simple AR(1) model parameters
        mu = zeros(N)
        Phi = 0.9 * Matrix(I, N, N)
        Omega = 0.1 * Matrix(I, N, N)
        A = zeros(M)
        B = Matrix{Float64}(I, M, N)  # Ensure correct type
        C = [zeros(N, N) for _ in 1:M]  # Vector of matrices
        D = 0.1 * Matrix{Float64}(I, M, M)
        alpha = zeros(M, M)
        
        # Simulate data
        Random.seed!(123)
        X = zeros(N, T)
        for t in 2:T
            X[:,t] = mu + Phi * X[:,t-1] + Omega * randn(N)
        end
        
        Y = zeros(M, T)
        for t in 1:T
            Y[:,t] = A + B * X[:,t] + D * randn(M)
        end
        
        model = QK.QKModel(N, M, mu, Phi, Omega, A, B, C, D, alpha)
        data = QK.QKData(Y)
        
        # Test performance
        @testset "Filter execution time" begin
            filter_time = @elapsed QK.qkf_filter(data, model)
            @test filter_time < 10.0  # Should complete in reasonable time
        end
        
        # Test memory allocation
        @testset "Filter memory allocation" begin
            filter_allocs = @allocated QK.qkf_filter(data, model)
            @test filter_allocs < 100_000_000  # Reasonable memory usage
        end
    end
end