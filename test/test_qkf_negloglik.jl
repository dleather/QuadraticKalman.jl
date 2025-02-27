using Test
import QuadraticKalman as QK
using LinearAlgebra

@testset "Negative Log-Likelihood" begin
    # Create simple test data
    N, M = 1, 1
    T_bar = 10
    Y = reshape(randn(T_bar + 1), T_bar + 1, 1)
    data = QK.QKData(Y = Y', M=M, T_bar=T_bar)
    
    # Create parameters that correspond to a simple model
    params = [0.0, 0.5, 0.1, 0.0, 1.0, 1.0, 0.1, 0.0]
    
    # Test the negative log-likelihood function
    ll = QK.qkf_negloglik(params, data, N, M)
    @test !isnan(ll)
    @test !isinf(ll)
    
    # Test with slightly different parameters and verify the likelihood changes
    params2 = params .* 1.1
    ll2 = QK.qkf_negloglik(params2, data, N, M)
    @test ll != ll2
end



