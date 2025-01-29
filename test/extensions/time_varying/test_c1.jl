# This is a test of all conditional moments of the integrated process [y(t), z(t)]
# Case 1: θ_y ≠ θ_z, 2 * θ_y ≠ θ_z, 0.5 * θ_y ≠ θ_z,
using Pkg; Pkg.activate("QuadraticKalman\\")
using Revise, QuadraticKalman, Plots, Statistics, StochasticDiffEq, LinearAlgebra, Random,
    FLoops, Enzyme, DifferentiationInterface, Distributions

Random.seed!(123124)

# Set script parameters
N_boot = 10_000
N_run = 300
N_div = 1_000

# Set Model parameters
θ_z_init = 0.5
σ_z_init = 1.0
θ_y_init = 2.0
σ_y_init = 0.05
ξ₀_init = 2.0 
ξ₁_init = 1.0
α_init = 0.0
t0 = 0.0
T = 0.25
Δt = T - t0

# Set initial conditions
v = 1.8484311498467427
u = 2.0945468980493716
z = [v, u, v^2, u*v, u*v, u^2]

#Define functions
function define_problem_ou_bridge(θ_z, σ_z, u, v, Δt)
    function drift_I1!(du, u, p, t)
        du[1] =  -θ_z * (-coth(-θ_z * (Δt-t))*u[1] + (v / sinh(-θ_z * (Δt-t))))
    end

    function diffusion_I1!(du,u,p,t)
        du[1] = σ_z
    end

    u0 = [u]
    prob = SDEProblem(drift_I1!, diffusion_I1!, u0, (0.0, Δt))
    eprob = EnsembleProblem(prob)
    return eprob
end

function simulate_integrals(eprob, N_run, N_boot,N_div, θ_y, Δt)
    estimates = zeros(N_boot, N_run)
    estimates_sq = zeros(N_boot, N_run)
    @floop for j = 1:N_run
        esol = solve(eprob, EM(); dt = Δt/N_div, trajectories = N_boot)
        y = zeros(length(esol[1].u), N_boot)
        ysq = zeros(length(esol[1].u), N_boot)
        Δ = Δt / N_div
        Δd2 = Δ / 2.0
        N_sim = length(0:Δ:Δt)
        for i = 1:N_boot
            y[:,i] = exp.(θ_y .* esol[i].t) .* 
                [esol[i].u[t][1] for t = 1:length(esol[i].t)]

            ysq[:,i] = exp.(θ_y .* esol[i].t) .* 
                [esol[i].u[t][1] for t = 1:length(esol[i].t)].^2
            for k = 2:N_sim
                estimates[i,j] += y[k,i]
                estimates_sq[i,j] += ysq[k,i]
                estimates[i,j] += y[k-1,i]
                estimates_sq[i,j] += ysq[k-1,i]

            end
            estimates[i,j] *= Δd2
            estimates_sq[i,j] *= Δd2

        end
    end
    return estimates, estimates_sq
end


#Simulate I₁, and I₂
epob = define_problem_ou_bridge(θ_z_init, σ_z_init, u, v, Δt)
I1_sim, I2_sim =  simulate_integrals(eprob, N_run, N_boot, N_div,θ_y_init, Δt)

#Test E[I₁]
dist_EI1 = [mean(I1_sim[:,i]) for i = 1:N_run]
sim_EI1 = mean(dist_EI1)
se_EI1 = std(dist_EI1)
cmpt_EI1 = compute_mean_I1_aug(z, t0, T, θ_z_init, θ_y_init)
@assert abs(sim_EI1 - cmpt_EI1) < 2 .* se_EI1

#Test E[I₁^2]
dist_EI1_sq = [mean(I1_sim[:,i].^2) for i = 1:N_run]
sim_EI1_sq = mean(dist_EI1_sq)
cmpt_EI1_sq = compute_EI1squv(u, v, t0, T, θ_z_init, θ_y_init, σ_z_init)


#Test  V[I₁]
dist_VI1 = [var(I1_sim[:,i]) for i = 1:N_run]
sim_VI1 = mean(dist_VI1)
se_VI1 = std(dist_VI1)
cmpt_VI1 = compute_var_I1_aug(z, t0, T, θ_z_init, θ_y_init, σ_z_init)
@assert abs(sim_VI1 - cmpt_VI1) < 2 .* se_VI1



# Test mean and variance of I₂
sim_EI2 = mean(I2_sim)


# Test covariance between I₁ and I₂

# Test mean and variance of y



# Test gradiant and hessian of y(t)



