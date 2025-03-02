using Test, BenchmarkTools
import QuadraticKalman as QK

# Original function for reference:
function compute_cond_cov_state_aug_original(Z, L1, L2, L3, Lambda, Sigma, mu, Phi_aug)
    N = length(mu)
    mu_Phi_z = (mu .+ Phi_aug[1:N,:] * Z)
    Sigma_11 = Sigma
    Sigma_12 = reshape(L1 * mu_Phi_z, (N^2, N))'
    Sigma_21 = Sigma_12'
    Sigma_22 = reshape(
        L3*(kron(mu, mu) .+ Phi_aug[N+1:end,:] * Z) .+
        kron(I(N^2), I + Lambda) * vec(kron(Sigma, Sigma)),
        (N^2, N^2)
    )
    return [Sigma_11 Sigma_12; Sigma_21 Sigma_22]
end

# A simple check with random data (choose N=2 or 3 for a quick test):
N = 2
Z1      = rand(N)
Z       = [Z1; (Z1*Z1')[:]]
L1     = rand(N^3, N)      # or whatever dims you actually use
L2     = rand(N^3, N^2)    # etc.
L3     = rand(N^4, N^2)
Lambda = rand(N^2, N^2)
Sigma  = rand(N, N)
mu     = rand(N)
Phi_aug= rand(N+N^2, N+N^2)

orig = QK.compute_cond_cov_state_aug(Z,L1,L2,L3,Lambda,Sigma,mu,Phi_aug)
new  = QK.compute_cond_cov_state_aug_noalloc(Z,L1,L2,L3,Lambda,Sigma,mu,Phi_aug)
new2 = QK.compute_cond_cov_state_aug_noalloc_allloops(Z,L1,L2,L3,Lambda,Sigma,mu,Phi_aug)

@test isapprox(orig, new; rtol=1e-12)
@test isapprox(orig, new2; rtol=1e-12)

@btime QK.compute_cond_cov_state_aug($Z,$L1,$L2,$L3,$Lambda,$Sigma,$mu,$Phi_aug);
@btime QK.compute_cond_cov_state_aug_noalloc($Z,$L1,$L2,$L3,$Lambda,$Sigma,$mu,$Phi_aug);