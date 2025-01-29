using Test
import QuadraticKalman as QK
using LinearAlgebra, Random
@testset "Quadratic Kalman Filter Tests" begin
    
    @testset "Full Model Construction" begin
        N, M = 2, 1
        mu = [0.1, 0.2]  # Vector is fine for mu
        Phi = [0.5 0.1; 0.1 0.5]  # 2×2 matrix
        Omega = [0.1 0.0; 0.0 0.1]  # 2×2 matrix
        A = [0.0]  # 1×1 matrix
        B = [1.0 1.0]  # 1×2 matrix
        C = [ones(2,2)]  # Array of 2×2 matrices
        D = reshape([0.1], 1, 1)  # 1×1 matrix
        alpha = reshape([0.2], 1, 1)  # 1×1 matrix
        
        # Create state and measurement parameters
        state = QK.StateParams(N, mu, Phi, Omega)
        meas = QK.MeasParams(M, N, A, B, C, D, alpha)
        
        # Construct full model
        model = QK.QKModel(state, meas)
        
        @test model.state.N == N
        @test model.meas.M == M
    end
    
    @testset "Filter Prediction Steps" begin
        # Setup minimal example
        N, M = 1, 1
        state = QK.StateParams(N, [0.0], reshape([0.5], 1, 1), reshape([0.1], 1, 1))
        meas = QK.MeasParams(M, N, [0.0], reshape([1.0], 1, 1), [reshape([1.0], 1, 1)], reshape([0.1], 1, 1), reshape([0.0], 1, 1))
        model = QK.QKModel(state, meas)
        
        Z_tt = [0.1, 0.1]  # Current state
        
        # Test state prediction
        Z_ttm1 = QK.predict_Z_ttm1(Z_tt, model)
        @test length(Z_ttm1) == N + N^2  # Augmented state dimension
        
        # Test measurement prediction
        Y = [0.1, 0.2]  # Some measurement data
        Y_ttm1 = QK.predict_Y_ttm1(Z_ttm1, Y, model, 1)
        @test length(Y_ttm1) == M
    end
    
    @testset "Filter Update Steps" begin
        N, M = 1, 1
        state = QK.StateParams(N, [0.0], reshape([0.5], 1, 1), reshape([0.1], 1, 1))
        meas = QK.MeasParams(M, N, [0.0], reshape([1.0], 1, 1), [reshape([1.0], 1, 1)], reshape([0.1], 1, 1), reshape([0.0], 1, 1))
        model = QK.QKModel(state, meas)
        
        # Setup minimal state and covariance
        K_t = reshape([0.5, 0.5], 2, 1)  # Some Kalman gain
        Y_t = 0.1                   # Measurement
        Y_ttm1 = 0.0               # Predicted measurement
        Z_ttm1 = [0.0, 0.0]        # Predicted state
        
        # Test state update
        Z_tt_new = QK.update_Z_tt(K_t, Y_t, Y_ttm1, Z_ttm1, 1)
        @test length(Z_tt_new) == N + N^2
    end
    
    @testset "Full Filter Integration" begin
        
        Random.seed!(123)

        T_bar = 100
        N, M = 1, 1
        Y = cumsum(randn(T_bar + 1))  # Random walk plus noise


        # Create state and measurement parameters
        state = QK.StateParams(N, [0.0], reshape([0.95], 1, 1), reshape([0.1], 1, 1))
        meas = QK.MeasParams(M, N, [0.0], reshape([1.0], 1, 1), [reshape([1.0], 1, 1)], reshape([0.1], 1, 1), reshape([0.0], 1, 1))
        model = QK.QKModel(state, meas)

        data = QK.QKData(Y = Y, M=M, T_bar=T_bar)

        # Test both filter versions
        result = QK.qkf_filter(data, model)

        # Basic sanity checks
        @test length(result.ll_t) == T_bar
        @test size(result.Z_tt) == (N + N^2, T_bar + 1)
        @test size(result.P_tt) == (N + N^2, N + N^2, T_bar + 1)
        @test size(result.Z_ttm1) == (N + N^2, T_bar)
        @test size(result.P_ttm1) == (N + N^2, N + N^2, T_bar)
        @test size(result.K_t) == (N + N^2, M, T_bar)
        @test length(result.Y_ttm1) == T_bar
        @test size(result.M_ttm1) == (M, M, T_bar)

        # Check output types
        @test eltype(result.ll_t) == Float64
        @test eltype(result.Z_tt) == Float64
        @test eltype(result.P_tt) == Float64
    end
end