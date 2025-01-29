using Test
import QuadraticKalman as QK
using LinearAlgebra, Random
@testset "Matrix Utils Tests" begin
    @testset "selection_matrix" begin
        S = QK.selection_matrix(3)
        @test size(S) == (6, 9)
        # Test extracting lower-triangular elements from e.g. 3×3
        A = [1. 2. 3.; 4. 5. 6.; 7. 8. 9.]
        tA = tril(A)[:]
        @test S * A[:] == [tA[i] for i in 1:length(tA) if tA[i] != 0]
    end

    @testset "make_positive_definite" begin
        A = [1.0 0.2; 0.2 1.0]
        # already PD
        A_pd = QK.make_positive_definite(A)
        @test isposdef(A_pd)

        # Slightly negative eigenvalue
        B = [ 1.0 0.99
              0.99 0.98 ]
        @test !isposdef(B)
        B_pd = QK.make_positive_definite(B)
        @test isposdef(B_pd)
    end

    @testset "spectral_radius" begin
        M = [0.5 0.0; 0.0 0.9]
        @test QK.spectral_radius(M) ≈ 0.9
    end
end

