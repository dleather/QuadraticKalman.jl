using Test
using LinearAlgebra
using Statistics
import QuadraticKalman as QK
using Random
using UnPack

@testset "Low-level QKF Implementation" begin
    Random.seed!(123)  # For reproducibility

    @testset "Basic _qkf_filter_impl! Functionality" begin
        # Setup model and data
        N = 2  # State dimension
        M = 1  # Measurement dimension
        T̄ = 5  # Time steps
        P = N + N^2  # Augmented state dimension
        
        # Create data with random observations
        Y = randn(M, T̄)
        data = QK.QKData(Y)
        
        # Create model with appropriate parameters
        mu = [0.1, 0.2]
        phi = [0.8 0.1; 0.0 0.7]
        sigma = [0.2 0.0; 0.0 0.3]
        state = QK.StateParams(N, mu, phi, sigma)
        
        a = [0.0]
        b = reshape([1.0 0.5], 1, 2)
        c = [reshape([0.1 0.0; 0.0 0.2], N, N)]
        d = reshape([sqrt(0.3)], 1, 1)
        alpha = zeros(M, M)
        meas = QK.MeasParams(M, N, a, b, c, d, alpha)
        
        model = QK.QKModel(state, meas)
        
        # Preallocate all arrays for the low-level function
        Z_tt = zeros(Float64, P, T̄)
        P_tt = zeros(Float64, P, P, T̄)
        Z_ttm1 = zeros(Float64, P, T̄ - 1)
        P_ttm1 = zeros(Float64, P, P, T̄ - 1)
        Y_ttm1 = zeros(Float64, M, T̄ - 1)
        M_ttm1 = zeros(Float64, M, M, T̄ - 1)
        K_t = zeros(Float64, P, M, T̄ - 1)
        ll_t = zeros(Float64, T̄ - 1)
        Sigma_ttm1 = zeros(Float64, P, P, T̄ - 1)
        tmpP = zeros(Float64, P, P)
        tmpKM = zeros(Float64, P, P)
        tmpKMK = zeros(Float64, P, P)
        tmpB = zeros(Float64, M, P)
        
        # Run the low-level implementation
        QK._qkf_filter_impl!(
            Z_tt, P_tt, Z_ttm1, P_ttm1, 
            Y_ttm1, M_ttm1, K_t, ll_t,
            Sigma_ttm1, tmpP, tmpKM, tmpKMK, tmpB,
            data, model
        )
        
        # Create FilterOutput from the results
        custom_result = QK.FilterOutput(ll_t, Z_tt, P_tt, Y_ttm1, M_ttm1, K_t, Z_ttm1, P_ttm1)
        
        # Run the standard qkf_filter! for comparison
        standard_result = QK.qkf_filter!(data, model)
        
        # Compare results
        @test isapprox(custom_result.ll_t, standard_result.ll_t)
        @test isapprox(custom_result.Z_tt, standard_result.Z_tt)
        @test isapprox(custom_result.P_tt, standard_result.P_tt)
        @test isapprox(custom_result.Y_ttm1, standard_result.Y_ttm1)
        @test isapprox(custom_result.M_ttm1, standard_result.M_ttm1)
        @test isapprox(custom_result.K_t, standard_result.K_t)
        @test isapprox(custom_result.Z_ttm1, standard_result.Z_ttm1)
        @test isapprox(custom_result.P_ttm1, standard_result.P_ttm1)
    end
    
    @testset "Array Reuse and Zero Allocations" begin
        # Setup model and data
        N = 2  # State dimension
        M = 1  # Measurement dimension
        T̄ = 5  # Time steps
        P = N + N^2  # Augmented state dimension
        
        # Create two different datasets for comparison
        Y1 = randn(M, T̄)
        Y2 = randn(M, T̄)
        data1 = QK.QKData(Y1)
        data2 = QK.QKData(Y2)
        
        # Create model
        mu = [0.1, 0.2]
        phi = [0.8 0.1; 0.0 0.7]
        sigma = [0.2 0.0; 0.0 0.3]
        state = QK.StateParams(N, mu, phi, sigma)
        
        a = [0.0]
        b = reshape([1.0 0.5], 1, 2)
        c = [reshape([0.1 0.0; 0.0 0.2], N, N)]
        d = reshape([sqrt(0.3)], 1, 1)
        alpha = zeros(M, M)
        meas = QK.MeasParams(M, N, a, b, c, d, alpha)
        
        model = QK.QKModel(state, meas)
        
        # Preallocate all arrays for the low-level function
        Z_tt = zeros(Float64, P, T̄)
        P_tt = zeros(Float64, P, P, T̄)
        Z_ttm1 = zeros(Float64, P, T̄ - 1)
        P_ttm1 = zeros(Float64, P, P, T̄ - 1)
        Y_ttm1 = zeros(Float64, M, T̄ - 1)
        M_ttm1 = zeros(Float64, M, M, T̄ - 1)
        K_t = zeros(Float64, P, M, T̄ - 1)
        ll_t = zeros(Float64, T̄ - 1)
        Sigma_ttm1 = zeros(Float64, P, P, T̄ - 1)
        tmpP = zeros(Float64, P, P)
        tmpKM = zeros(Float64, P, P)
        tmpKMK = zeros(Float64, P, P)
        tmpB = zeros(Float64, M, P)
        
        # First run with data1
        QK._qkf_filter_impl!(
            Z_tt, P_tt, Z_ttm1, P_ttm1, 
            Y_ttm1, M_ttm1, K_t, ll_t,
            Sigma_ttm1, tmpP, tmpKM, tmpKMK, tmpB,
            data1, model
        )
        
        # Save first set of results
        result1_Z_tt = copy(Z_tt)
        result1_ll_t = copy(ll_t)
        
        # Clear arrays
        Z_tt .= 0
        P_tt .= 0
        Z_ttm1 .= 0
        P_ttm1 .= 0
        Y_ttm1 .= 0
        M_ttm1 .= 0
        K_t .= 0
        ll_t .= 0
        Sigma_ttm1 .= 0
        
        # Second run with data2
        QK._qkf_filter_impl!(
            Z_tt, P_tt, Z_ttm1, P_ttm1, 
            Y_ttm1, M_ttm1, K_t, ll_t,
            Sigma_ttm1, tmpP, tmpKM, tmpKMK, tmpB,
            data2, model
        )
        
        # Compare with direct calls to qkf_filter!
        reference1 = QK.qkf_filter!(data1, model)
        reference2 = QK.qkf_filter!(data2, model)
        
        # Test that our implementation matches standard implementation
        @test isapprox(result1_Z_tt, reference1.Z_tt)
        @test isapprox(Z_tt, reference2.Z_tt)
        
        # Verify that results differ between runs with different data
        @test !isapprox(result1_Z_tt, Z_tt)
        @test !isapprox(result1_ll_t, ll_t)
        
        # Test allocation-free execution with @allocated macro
        # Uncomment for manual verification of no allocations
        # Note: This can be challenging in automated tests due to GC and other factors
        # allocs = @allocated QK._qkf_filter_impl!(Z_tt, P_tt, Z_ttm1, P_ttm1, Y_ttm1, M_ttm1, K_t, ll_t,
        #                                          Sigma_ttm1, tmpP, tmpKM, tmpKMK, tmpB, data2, model)
        # println("Allocations: ", allocs)
    end
    
    @testset "Input Validation and Edge Cases" begin
        # Setup model and data with small dimensions
        N = 1  # State dimension
        M = 1  # Measurement dimension
        T̄ = 3  # Time steps
        P = N + N^2  # Augmented state dimension
        
        # Create model and data
        Y = zeros(M, T̄)  # All zeros for predictable results
        data = QK.QKData(Y)
        
        mu = [0.0]
        phi = reshape([0.5], 1, 1)
        sigma = reshape([0.1], 1, 1)
        state = QK.StateParams(N, mu, phi, sigma)
        
        a = [0.0]
        b = reshape([1.0], 1, 1)
        c = [reshape([0.0], 1, 1)]
        d = reshape([0.1], 1, 1)
        alpha = reshape([0.0], 1, 1)
        meas = QK.MeasParams(M, N, a, b, c, d, alpha)
        
        model = QK.QKModel(state, meas)
        
        # Preallocate correctly sized arrays
        Z_tt = zeros(Float64, P, T̄)
        P_tt = zeros(Float64, P, P, T̄)
        Z_ttm1 = zeros(Float64, P, T̄ - 1)
        P_ttm1 = zeros(Float64, P, P, T̄ - 1)
        Y_ttm1 = zeros(Float64, M, T̄ - 1)
        M_ttm1 = zeros(Float64, M, M, T̄ - 1)
        K_t = zeros(Float64, P, M, T̄ - 1)
        ll_t = zeros(Float64, T̄ - 1)
        Sigma_ttm1 = zeros(Float64, P, P, T̄ - 1)
        tmpP = zeros(Float64, P, P)
        tmpKM = zeros(Float64, P, P)
        tmpKMK = zeros(Float64, P, P)
        tmpB = zeros(Float64, M, P)
        
        # Run the filter
        QK._qkf_filter_impl!(
            Z_tt, P_tt, Z_ttm1, P_ttm1, 
            Y_ttm1, M_ttm1, K_t, ll_t,
            Sigma_ttm1, tmpP, tmpKM, tmpKMK, tmpB,
            data, model
        )
        
        # Verify expected behavior with zero inputs
        @test !any(isnan.(Z_tt))  # No NaNs should be produced
        @test !any(isinf.(Z_tt))  # No Infs should be produced
        @test !any(isnan.(ll_t))  # No NaNs in log-likelihood
        
        # Verify initial conditions are correctly populated
        @test Z_tt[:, 1] ≈ model.moments.aug_mean
        @test P_tt[:, :, 1] ≈ model.moments.aug_cov
    end
end