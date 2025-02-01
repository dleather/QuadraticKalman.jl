# test_augmented_moments_and_likelihood.jl
using Test
import QuadraticKalman as QK
using LinearAlgebra, Random
@testset "Augmented Moments & Likelihood Tests" begin

    # Set seed for reproducibility
    Random.seed!(1234)

    @testset "compute_mu_aug tests" begin
        @testset "Scalar case" begin
            # If μ is scalar and Σ is also a scalar, compute_μ̃(μ, Σ) 
            # should return a 2-element vector: [μ, μ^2 + Σ].
            μ_scalar = 2.0
            Σ_scalar = 1.5
            result = QK.compute_mu_aug(μ_scalar, Σ_scalar)
            @test result ≈ [2.0, 2.0^2 + 1.5]  # [2, 5.5]
        end

        @testset "Multivariate case" begin
            # N=2
            mu_vec = [1.0, 2.0]
            Sigma_mat = [1.0 0.2;
                        0.2 2.0]
            # compute_mu_aug should return a vector of length N + N^2 = 2 + 4 = 6.
            result = QK.compute_mu_aug(mu_vec, Sigma_mat)
            @test length(result) == 6

            # Check first part is just μ
            @test result[1:2] ≈ mu_vec
            # Check next part is vec(μμ' + Σ)
            # i.e. vectorize( [1.0 2.0; 2.0 4.0] + [1.0 0.2; 0.2 2.0] ) = ...
            # mmᵗ = [1*1 1*2; 2*1 2*2] = [1 2; 2 4]
            # sum  = [2  2.2; 2.2 6]
            # vec( sum ) in column-major = [2, 2.2, 2.2, 6]
            expected_tail = vec([2.0  2.2;
                                 2.2  6.0])
            @test result[3:end] ≈ expected_tail
        end
    end

    @testset "compute_Phi_aug tests" begin
        @testset "Scalar case" begin
            # For scalar μ, Φ, the formula is:
            # Φ̃ = [Φ  0; 2μΦ  Φ^2]
            mu_s = 1.5
            Phi_s = 0.8
                Phi_aug = QK.compute_Phi_aug(mu_s, Phi_s)
            # Should be a 2x2:
            @test size(Phi_aug) == (2,2)
            @test Phi_aug ≈ [0.8  0.0;
                            2*1.5*0.8  0.8^2]
        end

        @testset "Multivariate case" begin
            # N=2
            mu_vec = [1.0, 2.0]
            Phi_mat = [0.8  0.1;
                     0.05 0.9]
            result = QK.compute_Phi_aug(mu_vec, Phi_mat)
            # Should be (N+N^2) x (N+N^2) = 6 x 6 for N=2
            @test size(result) == (6,6)
        end
    end

    @testset "compute_L1, L2, L3 tests" begin
        # We'll do a small N=2 example
        Sigma_mat = [1.0  0.3;
                     0.3 0.8]
        Lambda = QK.compute_Lambda(2)  # e.g. 4×4 commutation
        L1_mat = QK.compute_L1(Sigma_mat, Lambda)
        L2_mat = QK.compute_L2(Sigma_mat, Lambda)
        L3_mat = QK.compute_L3(Sigma_mat, Lambda)

        # Just check that they have expected sizes. 
        # The exact shape may depend on your definitions. 
        # L1 and L2 are N³ x N matrices. L3 is a N⁴ x N² matrix.
        @test size(L1_mat,1) == 8
        @test size(L1_mat,2) == 2

        @test size(L2_mat,1) == 8
        @test size(L2_mat,2) == 2

        @test size(L3_mat,1) == 16
        @test size(L3_mat,2) == 4

    end

    @testset "Likelihood: log_pdf_normal" begin
        # 1) Known normal values
        #    μ=0, σ²=1, x=0 => log pdf = -0.5*log(2π)
        val = QK.log_pdf_normal(0.0, 1.0, 0.0)
        @test isapprox(val, -0.5*log(2*π); atol=1e-10)

        # 2) Negative variance => returns -Inf
        val_bad = QK.log_pdf_normal(0.0, -1e-8, 0.0)
        @test val_bad == -Inf

        # 3) Very small variance => test clamp or numeric behavior
        val_small = QK.log_pdf_normal(1.0, 1e-12, 1.0)
        # Should be large but finite negative number
        @test isfinite(val_small)
    end

    @testset "Likelihood: compute_loglik" begin
        @testset "Univariate case" begin
            # Suppose Y=0.0, Ypred=0.0, Mpred=1×1 => σ²=1.0
            # => log_pdf ~ -0.5*log(2π)
            Yactual = 0.0
            Ypred   = 0.0
            Mpred   = [1.0]  # 1x1 matrix
            ll = QK.compute_loglik(Yactual, Ypred, Mpred)
            @test isapprox(ll, -0.5*log(2π); atol=1e-10)
        end

        @testset "Multivariate case" begin
            # Suppose we have 2D data. Y= [0.0; 0.0], Ypred= [0.0; 0.0], cov= I(2).
            # log-likelihood => -0.5*(2*log(2π) + 0.0) = -log(2π)
            Y2 = [0.0 0.0; 0.0 0.0]
            Y2_pred = [0.0 0.0; 0.0 0.0]
            M2pred = 1.0 * Matrix(I, 2, 2)  # 2×2 identity
            # Using the time-indexed version: compute_loglik(data, mean, cov, t)
            # or the simpler form compute_loglik(Y2, Y2_pred, M2pred).
            # We'll assume a function signature like:

            ll2 = QK.compute_loglik(Y2, Y2_pred, M2pred, 1) 
            # or if you have a "no-index" version:
            # ll2 = compute_loglik(Y2, Y2_pred, M2pred)
            @test isapprox(ll2, -log(2π); atol=1e-10)
        end
    end

end