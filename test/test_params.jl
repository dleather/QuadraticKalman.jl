using Test
using QuadraticKalman
using LinearAlgebra

@testset "QKParams Tests" begin
    @testset "Constructor checks" begin
        N, M = 2, 1
        μ = rand(N)
        Φ = 0.8 .* I(N) # stable
        Ω = 0.5 .* I(N)
        A = rand(M)
        B = rand(M, N)
        C = [rand(N, N) for _ in 1:M]
        D = 1.0 .* I(M)
        α = 0.1 .* I(M)
        Δt = 1.0

        params = QKParams(N, M, μ, Φ, Ω, A, B, C, D, α, Δt)
        @test params.Σ ≈ Ω * Ω'
        @test spectral_radius(params.Φ) < 1

        # Dimension mismatch
        Φ_bad = rand(N, N+1)
        @test_throws AssertionError QKParams(N, M, μ, Φ_bad, Ω, A, B, C, D, α, Δt)
    end
end