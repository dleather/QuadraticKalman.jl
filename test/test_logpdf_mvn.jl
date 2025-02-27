using Test
import QuadraticKalman as QK
using LinearAlgebra

@testset "Multivariate Normal Log PDF" begin
    # 2D case with identity covariance
    t = 1
    k = 2
    Y = zeros(k, 2)
    Y_pred = zeros(k, 2)
    Sigma_pred = cat(Matrix{Float64}(I, k, k), Matrix{Float64}(I, k, k), dims=3)
    expected_logpdf = -k/2 * log(2π) # For standard normal at x=mu
    @test QK.logpdf_mvn(Y_pred, Sigma_pred, Y, t) ≈ expected_logpdf
    
    # Test with non-zero mean and observation
    Y[:, 2] = [1.0, 1.0]
    Y_pred[:, 1] = [0.0, 0.0]
    expected_logpdf = -k/2 * log(2π) - 1.0 # For ||x-mu||^2 = 2
    @test QK.logpdf_mvn(Y_pred, Sigma_pred, Y, t) ≈ expected_logpdf
    
    # Test with non-identity covariance
    Sigma_pred[:,:,1] = [2.0 0.5; 0.5 1.0]
    C = cholesky(Symmetric(Sigma_pred[:,:,1]))
    diff = Y[:,2] - Y_pred[:,1]
    solved = C \ diff
    log_det = 2 * sum(log, diag(C.U))
    expected = -k/2 * log(2π) - 0.5 * (log_det + dot(diff, solved))
    @test QK.logpdf_mvn(Y_pred, Sigma_pred, Y, t) ≈ expected
end


