using Test
import QuadraticKalman as QK
using LinearAlgebra

@testset "Likelihood Edge Cases" begin
    # Test log_pdf_normal with negative variance
    @test QK.log_pdf_normal(0.0, -1.0, 0.0) == -Inf
    
    # Test log_pdf_normal with zero variance
    @test QK.log_pdf_normal(0.0, 0.0, 0.0) == -Inf
    
    # Test compute_loglik with small variance
    small_var = 1e-13
    Y = 1.0
    Ypred = 1.0
    Mpred = reshape([small_var], 1, 1)
    @test QK.compute_loglik(Y, Ypred, Mpred) ≈ QK.log_pdf_normal(Ypred, 1e-12, Y)
    
    # Test multivariate case with near-singular covariance
    t = 1
    k = 2
    Y = ones(k, 2)
    Y_pred = ones(k, 2)
    Sigma_pred = cat(Matrix(1e-5I, k, k), Matrix(1e-5I, k, k), dims=3)
    @test !isnan(QK.logpdf_mvn(Y_pred, Sigma_pred, Y, t))
end