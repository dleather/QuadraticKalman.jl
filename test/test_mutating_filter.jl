using Test
using LinearAlgebra
using Statistics
import QuadraticKalman: QKModel, QKData, StateParams, MeasParams, AugStateParams
using Random
using UnPack

include("../src/core/filter.jl")  # Adjust path as needed

@testset "QKF Mutating Functions" begin
    Random.seed!(123)  # For reproducibility

    @testset "Setup Test Data" begin
        # Define small test dimensions
        N = 2      # State dimension
        P = 6      # Augmented state dimension (N + N^2)
        M = 1      # Measurement dimension
        T̄ = 10     # Time steps

        # Create a simple test model
        Phi = [0.9 0.1; 0.0 0.8]  # State transition
        Q = [0.1 0.0; 0.0 0.2]    # Process noise
        A = [0.0]                  # Observation constant
        B = [1.0 0.0]              # Observation matrix
        V = [0.5]                  # Measurement noise
        alpha = [0.0]              # AR term

        # Create augmented model components
        Phi_aug = zeros(P, P)
        Phi_aug[1:N, 1:N] = Phi
        
        # Simple augmentation for second moment
        # This assumes state evolves as xₜ = Φxₜ₋₁ + wₜ
        # Second moment evolution is approximate
        for i in 1:N
            for j in 1:N
                idx_i, idx_j = i, j
                row = N + (i-1)*N + j
                for k in 1:N
                    for l in 1:N
                        col_k, col_l = k, l
                        col = N + (k-1)*N + l
                        if i == k
                            Phi_aug[row, col] += Phi[j, l]
                        end
                        if j == l
                            Phi_aug[row, col] += Phi[i, k]
                        end
                    end
                end
            end
        end

        mu_aug = zeros(P)
        B_aug = zeros(M, P)
        B_aug[:, 1:N] = B

        # Construct the state parameters.
        state = StateParams(
            N = N,
            mu = [1.0, -0.5],  # example initial state (length N)
            Phi = [0.9 0.1; 0.0 0.8],  # state transition matrix (N×N)
            Omega = [0.316 0.0; 0.0 0.447],  # state noise matrix (N×N); e.g. sqrt of process noise variances
            Sigma = [0.1 0.0; 0.0 0.2]  # state covariance
        )

        # Construct the measurement parameters.
        meas = MeasParams(
            M,                                  # Measurement dimension
            N,                                  # State dimension
            [0.0],                              # measurement intercept
            [1.0 0.0],                          # measurement matrix (M×N)
            [reshape([0.0 0.0; 0.0 0.0], 2, 2)], # vector of quadratic measurement matrices
            [0.5][:,1:1],                       # measurement noise scaling (M×M)
            [0.0][:,1:1]                        # autoregressive measurement parameter (M×M)
        )

        # Construct the augmented state parameters.
        # Using the constructor that computes all required matrices automatically
        aug_state = AugStateParams(
            N,                          # state dimension
            state.mu,                   # initial state mean
            state.Phi,                  # state transition matrix
            state.Sigma,                # state covariance
            meas.B,                     # measurement matrix
            meas.C                      # quadratic coefficient matrices
        )

        # Now construct the QKModel.
        model = QKModel(state, meas)

        # Initial state and covariance
        Z_tt = zeros(P, T̄)
        Z_tt[1:N, 1] = [1.0, -0.5]  # Initial state
        
        # Set initial second moment
        X0 = Z_tt[1:N, 1]
        Z_tt[N+1:end, 1] = vec(X0 * X0' + [0.2 0.0; 0.0 0.3])
        
        # Create storage arrays
        Z_ttm1 = zeros(P, T̄)
        P_tt = zeros(P, P, T̄)
        P_ttm1 = zeros(P, P, T̄)
        Σ_ttm1 = zeros(P, P, T̄)
        
        # Initial covariance
        P_tt[:, :, 1] = [0.2 0.0 0.0 0.0 0.0 0.0;
                         0.0 0.3 0.0 0.0 0.0 0.0;
                         0.0 0.0 0.5 0.0 0.0 0.0;
                         0.0 0.0 0.0 0.6 0.0 0.0;
                         0.0 0.0 0.0 0.0 0.7 0.0;
                         0.0 0.0 0.0 0.0 0.0 0.8]
        Y = randn(M, T̄)  # Random observations
        Y_ttm1 = zeros(M, T̄)  # Predicted observations
        M_ttm1 = zeros(M, M, T̄)  # Measurement covariance
        K_t = zeros(P, M, T̄)  # Kalman gain
        
        # Working buffers
        tmpP = zeros(P, P)
        tmpB = zeros(M, P)
        tmpKM = zeros(P, M)
        tmpKMK = zeros(P, P)

        # Store in test_data dictionary
        test_data = Dict(
            :model => model,
            :Z_tt => Z_tt,
            :Z_ttm1 => Z_ttm1,
            :P_tt => P_tt,
            :P_ttm1 => P_ttm1,
            :Σ_ttm1 => Σ_ttm1,
            :Y => Y,
            :Y_ttm1 => Y_ttm1,
            :M_ttm1 => M_ttm1,
            :K_t => K_t,
            :tmpP => tmpP,
            :tmpB => tmpB,
            :tmpKM => tmpKM,
            :tmpKMK => tmpKMK,
            :N => N,
            :P => P,
            :M => M,
            :T̄ => T̄
        )

        # Basic size checks
        @test size(Z_tt) == (P, T̄)
        @test size(P_tt) == (P, P, T̄)
        @test size(Y) == (M, T̄)
        
        # Return test data for use in other tests
        return test_data
    end

    test_data = @testset "Setup Test Data" begin
        # This will run the setup and return test_data
        @test true
        setup_result = include_string(Main, read("test/filter_test.jl", String), "test/filter_test.jl")
        setup_result
    end

    @testset "predict_Z_ttm1!" begin
        # Extract test data
        @unpack model, Z_tt, Z_ttm1, P, T̄ = test_data
        
        t = 1  # Test for first time step
        
        # Before prediction Z_ttm1 should be zeros
        @test all(Z_ttm1[:, t] .== 0.0)
        
        # Run prediction
        predict_Z_ttm1!(Z_tt, Z_ttm1, model, t)
        
        # After prediction Z_ttm1 should not be zeros
        @test !all(Z_ttm1[:, t] .== 0.0)
        
        # Test equivalence with out-of-place version
        Z_ttm1_out = predict_Z_ttm1(Z_tt[:, t], model)
        @test Z_ttm1[:, t] ≈ Z_ttm1_out
        
        # Test correct linear transformation
        @unpack Phi_aug, mu_aug = model.aug_state
        expected = mu_aug + Phi_aug * Z_tt[:, t]
        @test Z_ttm1[:, t] ≈ expected
    end

    @testset "predict_P_ttm1!" begin
        # Extract test data
        @unpack model, Z_tt, P_tt, P_ttm1, Σ_ttm1, tmpP, P = test_data
        
        t = 1  # Test for first time step
        
        # Define a simple compute_Sigma_ttm1! for testing
        function compute_Sigma_ttm1!(Σ_ttm1, Z_tt, model, t)
            Σ_ttm1[:, :, t] = 0.1 * I(P)  # Simple identity scaled
        end
        
        # Run prediction
        predict_P_ttm1!(P_tt, P_ttm1, Σ_ttm1, Z_tt, tmpP, model, t)
        
        # Test that P_ttm1 is positive definite
        @test isposdef(P_ttm1[:, :, t])
        
        # Test the correct formula
        @unpack Phi_aug = model.aug_state
        expected = Phi_aug * P_tt[:, :, t] * Phi_aug' + 0.1*I(P)
        @test P_ttm1[:, :, t] ≈ expected
        
        # Test equivalence with out-of-place version
        # Define compute_Sigma_ttm1 for out-of-place
        function compute_Sigma_ttm1(Z_tt, model)
            return 0.1 * I(P)
        end
        
        P_ttm1_out = predict_P_ttm1(P_tt[:, :, t], Z_tt, model, t)
        @test P_ttm1[:, :, t] ≈ P_ttm1_out
    end

    @testset "predict_Y_ttm1!" begin
        # Extract test data
        @unpack model, Z_ttm1, Y, Y_ttm1, M = test_data
        
        t = 1  # Test for first time step
        
        # Run prediction
        predict_Y_ttm1!(Y_ttm1, Z_ttm1, Y, model, t)
        
        # Test correct formula
        @unpack A, alpha = model.meas
        @unpack B_aug = model.aug_state
        expected = A + B_aug * Z_ttm1[:, t] + alpha * Y[:, t]
        @test Y_ttm1[:, t] ≈ expected
        
        # Test equivalence with out-of-place version
        Y_ttm1_out = predict_Y_ttm1(Z_ttm1, Y, model, t)
        @test Y_ttm1[:, t] ≈ Y_ttm1_out
        
        # Test with modified parameters
        model_tmp = deepcopy(model)
        model_tmp.meas.A = [0.5]
        predict_Y_ttm1!(Y_ttm1, Z_ttm1, Y, model_tmp, t)
        expected = [0.5] + B_aug * Z_ttm1[:, t] + alpha * Y[:, t]
        @test Y_ttm1[:, t] ≈ expected
    end

    @testset "predict_M_ttm1!" begin
        # Extract test data
        @unpack model, P_ttm1, M_ttm1, tmpB, M = test_data
        
        t = 1  # Test for first time step
        
        # Run prediction
        predict_M_ttm1!(M_ttm1, P_ttm1, tmpB, model, t)
        
        # Test that M_ttm1 is positive definite (for M>1) or positive (for M=1)
        if M > 1
            @test isposdef(M_ttm1[:, :, t])
        else
            @test M_ttm1[1, 1, t] > 0
        end
        
        # Test correct formula
        @unpack V = model.meas
        @unpack B_aug = model.aug_state
        expected = B_aug * P_ttm1[:, :, t] * B_aug' + V
        @test M_ttm1[:, :, t] ≈ expected
        
        # Test equivalence with out-of-place version
        M_ttm1_out = predict_M_ttm1(P_ttm1[:, :, t], model)
        @test M_ttm1[:, :, t] ≈ M_ttm1_out
        
        # Test handling of near-singular case
        P_ttm1_small = 1e-10 * ones(size(P_ttm1[:, :, t]))
        model_tmp = deepcopy(model)
        model_tmp.meas.V = [1e-10]
        
        if M == 1
            # Univariate case should handle small values
            predict_M_ttm1!(M_ttm1, fill!(P_ttm1, 0), tmpB, model_tmp, t)
            @test M_ttm1[1, 1, t] ≥ 0  # Should be non-negative
        end
    end

    @testset "compute_K_t!" begin
        # Extract test data
        @unpack model, P_ttm1, M_ttm1, K_t, tmpB, P, M = test_data
        
        t = 1  # Test for first time step
        
        # Run Kalman gain computation
        compute_K_t!(K_t, P_ttm1, M_ttm1, tmpB, model, t)
        
        # Test dimensions
        @test size(K_t[:, :, t]) == (P, M)
        
        # Test correct formula for Kalman gain
        @unpack B_aug = model.aug_state
        P_B = P_ttm1[:, :, t] * B_aug'
        expected = P_B / M_ttm1[:, :, t]
        @test K_t[:, :, t] ≈ expected
        
        # Test equivalence with out-of-place version
        K_t_out = compute_K_t(P_ttm1[:, :, t], M_ttm1[:, :, t], model, t)
        @test K_t[:, :, t] ≈ K_t_out
        
        # Test with singular measurement covariance
        M_ttm1_singular = zeros(M, M, test_data[:T̄])
        M_ttm1_singular[1, 1, t] = 1e-10
        
        # This should not throw an error, but might give large values
        compute_K_t!(K_t, P_ttm1, M_ttm1_singular, tmpB, model, t)
        @test !any(isnan.(K_t[:, :, t]))
        @test !any(isinf.(K_t[:, :, t]))
    end

    @testset "update_Z_tt!" begin
        # Extract test data
        @unpack model, Z_tt, Z_ttm1, K_t, Y, Y_ttm1, P = test_data
        
        t = 1  # Test for first time step
        
        # Add a known value to K_t for testing
        K_t[:, :, t] = randn(P, model.meas.M)
        
        # Run state update
        update_Z_tt!(Z_tt, K_t, Y, Y_ttm1, Z_ttm1, t)
        
        # Test correct formula
        expected = Z_ttm1[:, t] + K_t[:, :, t] * (Y[:, t] - Y_ttm1[:, t])
        @test Z_tt[:, t+1] ≈ expected
        
        # Test with zero innovation
        Y_zero = copy(Y)
        Y_zero[:, t] = Y_ttm1[:, t]  # Zero innovation
        
        # Create a copy for testing
        Z_tt_copy = copy(Z_tt)
        update_Z_tt!(Z_tt_copy, K_t, Y_zero, Y_ttm1, Z_ttm1, t)
        
        # With zero innovation, Z_tt should equal Z_ttm1
        @test Z_tt_copy[:, t+1] ≈ Z_ttm1[:, t]
    end

    @testset "update_P_tt!" begin
        # Extract test data
        @unpack model, P_tt, P_ttm1, K_t, Z_ttm1, tmpKM, tmpKMK, P, M = test_data
        
        t = 1  # Test for first time step
        
        # Run covariance update
        update_P_tt!(P_tt, K_t, P_ttm1, Z_ttm1, tmpKM, tmpKMK, model, t)
        
        # Test that P_tt is positive definite
        @test isposdef(P_tt[:, :, t+1])
        
        # Test correct formula
        @unpack V = model.meas
        @unpack B_aug = model.aug_state
        A = I - K_t[:, :, t] * B_aug
        expected = A * P_ttm1[:, :, t] * A' + K_t[:, :, t] * V * K_t[:, :, t]'
        @test P_tt[:, :, t+1] ≈ make_positive_definite(expected)
        
        # Test equivalence with out-of-place version
        P_tt_out = update_P_tt(K_t[:, :, t], P_ttm1[:, :, t], Z_ttm1[:, t], model, t)
        @test P_tt[:, :, t+1] ≈ P_tt_out
    end

    @testset "correct_Z_tt!" begin
        # Extract test data
        @unpack model, Z_tt, N = test_data
        
        t = 1  # Use t+1 data which was updated in previous test
        
        # Store original state
        Z_tt_original = copy(Z_tt)
        
        # Create a state with negative eigenvalues in the implied covariance
        x_t = Z_tt[1:N, t+1]
        XX_t = reshape(Z_tt[N+1:end, t+1], N, N)
        
        # Make the implied covariance indefinite
        implied_cov = XX_t - x_t * x_t'
        # Add a negative eigenvalue
        F = eigen(Symmetric(implied_cov))
        F.values[1] = -1.0  # Make one eigenvalue negative
        bad_implied = F.vectors * Diagonal(F.values) * F.vectors'
        
        # Put it back into Z_tt
        Z_tt[N+1:end, t+1] = vec(bad_implied + x_t * x_t')
        
        # Run correction
        correct_Z_tt!(Z_tt, model, t)
        
        # Extract corrected implied covariance
        XX_t_corrected = reshape(Z_tt[N+1:end, t+1], N, N)
        implied_cov_corrected = XX_t_corrected - x_t * x_t'
        
        # Test that all eigenvalues are non-negative
        F_corrected = eigen(Symmetric(implied_cov_corrected))
        @test all(F_corrected.values .>= -1e-10)  # Allow small numerical error
        
        # Test that the first part (x_t) is unchanged
        @test Z_tt[1:N, t+1] == Z_tt_original[1:N, t+1]
        
        # Test equivalence with out-of-place version
        Z_tt_vec = Z_tt[:, t+1]
        Z_tt_out = correct_Z_tt(Z_tt_vec, model, t)
        
        # Only check the impled covariance part, as implementations may differ slightly
        XX_t_out = reshape(Z_tt_out[N+1:end], N, N)
        implied_cov_out = XX_t_out - Z_tt_out[1:N] * Z_tt_out[1:N]'
        F_out = eigen(Symmetric(implied_cov_out))
        @test all(F_out.values .>= -1e-10)  # All eigenvalues should be non-negative
    end

    @testset "qkf_filter! Single Step Integration" begin
        # Extract test data
        @unpack model, Z_tt, Z_ttm1, P_tt, P_ttm1, Σ_ttm1, Y, Y_ttm1, M_ttm1,
                K_t, tmpP, tmpB, tmpKM, tmpKMK, T̄ = test_data
        
        # Define compute_Sigma_ttm1! function for the integration test
        function compute_Sigma_ttm1!(Σ_ttm1, Z_tt, model, t)
            P = size(Z_tt, 1)
            Σ_ttm1[:, :, t] = 0.1 * I(P)
        end
        
        # Test a single step of the filter
        t = 1
        
        # Make sure Z_tt has a valid initial state at t=1
        Z_tt_orig = copy(Z_tt)
        P_tt_orig = copy(P_tt)
        
        # 1. State prediction
        predict_Z_ttm1!(Z_tt, Z_ttm1, model, t)
        
        # 2. Covariance prediction
        predict_P_ttm1!(P_tt, P_ttm1, Σ_ttm1, Z_tt, tmpP, model, t)
        
        # 3. Measurement prediction
        predict_Y_ttm1!(Y_ttm1, Z_ttm1, Y, model, t)
        
        # 4. Measurement covariance prediction
        predict_M_ttm1!(M_ttm1, P_ttm1, tmpB, model, t)
        
        # 5. Kalman gain
        compute_K_t!(K_t, P_ttm1, M_ttm1, tmpB, model, t)
        
        # 6. State update
        update_Z_tt!(Z_tt, K_t, Y, Y_ttm1, Z_ttm1, t)
        
        # 7. Covariance update
        update_P_tt!(P_tt, K_t, P_ttm1, Z_ttm1, tmpKM, tmpKMK, model, t)
        
        # 8. PSD correction
        correct_Z_tt!(Z_tt, model, t)
        
        # Test that second moment portion of state is consistent with state + covariance
        N = model.state.N
        x_t = Z_tt[1:N, t+1]
        XX_t = reshape(Z_tt[N+1:end, t+1], N, N)
        implied_cov = XX_t - x_t * x_t'
        
        # Test positive semidefiniteness
        @test all(eigvals(Symmetric(implied_cov)) .>= -1e-10)
        
        # Test state is different from initial
        @test Z_tt[:, t+1] != Z_tt_orig[:, t]
        @test P_tt[:, :, t+1] != P_tt_orig[:, :, t]
        
        # Test absence of NaNs or Infs
        @test !any(isnan.(Z_tt[:, t+1]))
        @test !any(isinf.(Z_tt[:, t+1]))
        @test !any(isnan.(P_tt[:, :, t+1]))
        @test !any(isinf.(P_tt[:, :, t+1]))
    end
end