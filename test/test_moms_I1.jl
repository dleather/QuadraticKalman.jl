################################################################################
##### Test completed - for case 1 on 7/182024, with script parameters: #########
# Random.seed!(123125)
#
# Set script parameters
# N_boot = 1_000_000
# N_run = 300
# N_div = 100
################################################################################
using Pkg; Pkg.activate("QuadraticKalman\\")
using Revise, QuadraticKalman, Plots, Statistics, LinearAlgebra, Random,
    DifferentialEquations

Random.seed!(123125)

# Set script parameters
N_boot = 1_000_000
N_run = 300
N_div = 100

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

function drift_I1!(du, u, p, t)
    #return -θ_z * uu[1] + 2.0 * θ_z * (v * exp(θ_z * (Δt - t)) - uu[1]) / (exp(2.0 * θ_z * (Δt - t)) - 1.0)
    du[1] = exp(θ_y_init * (t - t0))*u[2]
    du[2] = θ_z_init * (-coth(θ_z_init * (T - t)) * u[2] + (v / (sinh(θ_z_init * (T - t)))))
end

function diffusion_I1!(du,u,p,t)
    du[1] = sqrt(eps())
    du[2] = σ_z_init
end
u0 = [0.0, u]
prob = SDEProblem(drift_I1!, diffusion_I1!, u0, (0.0, Δt))
eprob = EnsembleProblem(prob)

sol = solve(eprob, EulerHeun(), dt = Δt/100; trajectories = N_boot)

function simulate_I1(eprob, N_run, N_boot, N_div, Δt)
    mean_I1 = zeros(N_run)
    var_I1 = zeros(N_run)

    time_taken = @elapsed for j = 1:N_run
        esol = solve(eprob, EulerHeun(), dt = Δt/100; trajectories = N_boot)
        y = zeros(N_boot)
        for i = 1:N_boot
            y[i] = esol[i].u[end][1]
        end
        mean_I1[j] = mean(y)
        var_I1[j] = var(y)
        println(j)

    end
    println("Time taken: ", time_taken)
    return mean_I1, var_I1
end
#eprob, prob = define_problem_I1(θ_z_init, σ_z_init, u, v, Δt)

mean_I1, var_I1 = simulate_I1(eprob, N_run, N_boot, N_div, Δt)

histogram(mean_I1, label = "Mean y(t)", title = "Histogram of Mean y(t)")
vline!([0.635334])

histogram(var_I1, label = "Variance y(t)", title = "Histogram of Variance y(t)")
vline!([0.0021794])