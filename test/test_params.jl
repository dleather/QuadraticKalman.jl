using Test
import QuadraticKalman as QK
using LinearAlgebra

@testset "QKModel Tests" begin

    # Test StateParams
    @testset "StateParams Tests" begin
        N = 2
        mu = [1.0, 2.0]
        Phi = [0.9 0.05; 0.0 0.8]
        Omega = [1.0 0.2; 0.2 1.0]

        sp = QK.StateParams(N, mu, Phi, Omega; check_stability=true)
        @test sp.N == N
        @test sp.mu == mu
        @test sp.Phi ≈ Phi
        @test sp.Omega == Omega
        @test sp.Sigma ≈ Omega * Omega'

        # Test stable/unstable
        Phi_unstable = [1.1 0.0; 0.0 1.2]
        @test_throws AssertionError QK.StateParams(N, mu, Phi_unstable, Omega; check_stability=true)

        # Test dimension mismatch
        @test_throws AssertionError QK.StateParams(N, [1.0], Phi, Omega)  # mu has length 1, needs 2
        @test_throws AssertionError QK.StateParams(N, mu, Phi[1:1, :], Omega)  # Phi not NxN
        @test_throws AssertionError QK.StateParams(N, mu, Phi, rand(3, 3))    # Omega not NxN

        # Test UniformScaling for Omega or Phi
        sp_I = QK.StateParams(N, mu, I, I)  # Should handle
        @test sp_I.Sigma == I * I'
    end

    # Test MeasParams
    @testset "MeasParams Tests" begin
        N = 2
        M = 3
        A = [1.0, 2.0, 3.0]
        B = [1.0 0.0; 0.5 2.0; 3.0 4.0]
        C = [
            [0.1 0.0; 0.0 0.1],
            [0.2 0.1; 0.1 0.2],
            [0.0 0.0; 0.0 0.3]
        ]
        D = [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0]
        alpha = [0.9 0.1; 0.1 0.9; 0.2 0.8]

        mp = QK.MeasParams(M, N, A, B, C, D, alpha)
        @test mp.M == M
        @test mp.A == A
        @test mp.B == B
        @test mp.C == C
        @test mp.D == D
        @test mp.V == D * D'
        @test mp.alpha == alpha

        # Dimension mismatch checks
        @test_throws AssertionError QK.MeasParams(M, N, A[1:2], B, C, D, alpha)  # length(A) != M
        @test_throws AssertionError QK.MeasParams(M, N, A, B[:,1:1], C, D, alpha) # B not MxN
        bad_C = [C[1], C[2]]  # length != M
        @test_throws AssertionError QK.MeasParams(M, N, A, B, bad_C, D, alpha)
        bad_Cdim = copy(C)
        bad_Cdim[1] = [1.0 2.0 3.0; 1.0 2.0 3.0]
        @test_throws AssertionError QK.MeasParams(M, N, A, B, bad_Cdim, D, alpha) # not NxN
        @test_throws AssertionError QK.MeasParams(M, N, A, B, C, D[:,1:2], alpha) # D not MxM

        # Test UniformScaling for D
        mp_I = QK.MeasParams(M, N, A, B, C, I, alpha)
        @test mp_I.D == Matrix(I, M, M)
        @test mp_I.V == mp_I.D * mp_I.D'
    end

    # Test AugStateParams
    @testset "AugStateParams Tests" begin
        N = 2
        mu = [1.0, 2.0]
        Phi = [0.9 0.05; 0.0 0.8]
        Omega = [1.0 0.2; 0.2 1.0]
        Sigma = Omega * Omega' # just for test

        B = [1.0 0.0; 0.5 2.0; 3.0 4.0]  # M=3 example
        C = [
            [0.1 0.0; 0.0 0.1],
            [0.2 0.1; 0.1 0.2],
            [0.0 0.0; 0.0 0.3]
        ]

        aug = QK.AugStateParams(N, mu, Phi, Sigma, B, C)
        @test length(aug.mu_aug) == N + N^2
        @test size(aug.Phi_aug, 1) == N + N^2
        @test size(aug.Phi_aug, 2) == N + N^2
        @test size(aug.B_aug, 1) == size(B,1)
        @test size(aug.H_aug, 1) == (N*(N+3)) / 2
        @test size(aug.H_aug, 2) == N * (N + 1)
        @test size(aug.G_aug, 1) == N * (N+1)
        @test size(aug.G_aug, 2) == ((N*(N+3))/2)
        @test size(aug.Lambda,1) == N^2
        @test size(aug.L1,1) == N^3
        @test size(aug.L1,2) == N
        @test size(aug.L2,1) == N^3
        @test size(aug.L2,2) == N
        @test size(aug.L3,1) == N^4
        @test size(aug.L3,2) == N^2
        @test aug.P == N + N^2
    end

    # Test Moments
    @testset "Moments Tests" begin
        # Create mock data
        N = 2
        mu = [1.0, 2.0]
        Phi = [0.9 0.05; 0.0 0.8]
        Omega = [1.0 0.2; 0.2 1.0]
        Sigma = Omega * Omega'
        mu_aug = vcat(mu, vec(Sigma))
        Phi_aug = [Phi zeros(2,4); zeros(4,2) 0.5 * I(4)]  # just for demonstration
        L1 = ones(N^3, N)
        L2 = ones(N^3, N)
        L3 = ones(N^4, N^2)
        Lambda = 1.0 * Matrix(I, N^2, N^2)

        mom = QK.Moments(mu, Phi, Sigma, mu_aug, Phi_aug, L1, L2, L3, Lambda)
        @test length(mom.state_mean) == N
        @test size(mom.state_cov) == (N, N)
        P = N + N^2
        @test length(mom.aug_mean) == P
        @test size(mom.aug_cov) == (P, P)
    end

    # ---------------------------------------------------------------------
    # 5) Test the main QKModel constructor
    # ---------------------------------------------------------------------
    @testset "QKModel Constructor Tests" begin
        N = 2
        M = 3
        mu = [1.0, 2.0]
        Phi = [0.8 0.1; 0.2 0.7]
        Omega = [1.0 0.0; 0.0 1.0]
        A = [1.0, 2.0, 3.0]
        B = [1.0 0.0; 0.5 2.0; 3.0 4.0]
        C = [
            [0.1 0.0; 0.0 0.1],
            [0.2 0.1; 0.1 0.2],
            [0.0 0.0; 0.0 0.3]
        ]
        D = [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 1.0]
        alpha = [0.9 0.1; 0.1 0.9; 0.2 0.8]

        model = QK.QKModel(N, M, mu, Phi, Omega, A, B, C, D, alpha; check_stability=true)
        
        @test model.state.N == N
        @test model.meas.M == M
        @test length(model.aug.mu_aug) == N + N^2
        @test length(model.moments.state_mean) == N
        @test size(model.moments.state_cov) == (N, N)

        # check_stability with an unstable Phi
        Phi_unstable = [1.01 0.0; 0.0 0.99]
        @test_throws AssertionError QK.QKModel(N, M, mu, Phi_unstable, Omega,
                                            A, B, C, D, alpha; check_stability=true)
    end
end
