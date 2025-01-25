################################################################################
##### Test passed - for case 1 on 7/20/2024, with script parameters:
# Random.seed!(123125)
#
# Set script parameters
# N_boot = 1_000_000
# N_run = 300
# N_div = 100
################################################################################
using Pkg; Pkg.activate("QuadraticKalman\\")
using Revise, QuadraticKalman, Plots, Statistics, LinearAlgebra, Random,
    DifferentialEquations, Distributions, HypothesisTests

Random.seed!(4658274234)

# Set script parameters
N_boot = 1_000_000
N_run = 300
N_div = 100

# Set Model parameters
θ_z_init = 0.5
σ_z_init = 1.0
θ_y_init = 2.0
σ_y_init = 0.05
ξ₀_init = 1.0
ξ₁_init = 0.5
α_init = 0.2
t0 = 0.1
T = 0.25
Δt = T - t0
y0 = 1.0


# Set initial conditions
v = 1.8484311498467427
u = 2.0945468980493716
z = [v, u, v^2, u*v, u*v, u^2]

true_mean = compute_mean_y(u, v, y0, Δt, α_init, θ_y_init, θ_z_init, σ_z_init,
    ξ₀_init, ξ₁_init)
true_var = compute_var_y(u, v, Δt, θ_y_init, θ_z_init, σ_z_init, σ_y_init, ξ₀_init, ξ₁_init)
function drift_y!(du, u, p, t)
    #return -θ_z * uu[1] + 2.0 * θ_z * (v * exp(θ_z * (Δt - t)) - uu[1]) / (exp(2.0 * θ_z * (Δt - t)) - 1.0)
    du[1] = θ_y_init*(α_init - u[1]) + ξ₀_init * u[2] + ξ₁_init * u[2]^2 
    du[2] = θ_z_init * (-coth(θ_z_init * (T - t)) * u[2] + (v / (sinh(θ_z_init * (T - t)))))
end

function diffusion_y!(du,u,p,t)
    du[1] = σ_y_init
    du[2] = σ_z_init
end
u0 = [y0, u]
prob = SDEProblem(drift_y!, diffusion_y!, u0, (t0, T))
eprob = EnsembleProblem(prob)
#sol = solve(eprob, EM(); dt = Δt/400, trajectories = N_boot)
#plot(sol[1])
function simulate_y(eprob, N_run, N_boot, N_div, Δt)
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

mean_y, var_y = simulate_y(eprob, N_run, N_boot, N_div, Δt)

histogram(mean_y, label = "Mean y(t)", title = "Histogram of Mean y(t)")
vline!([true_mean])

histogram(var_y, label = "Variance y(t)", title = "Histogram of Variance y(t)")
vline!([true_var])

t_test = OneSampleTTest(mean_y, true_mean)
p_value_mean = pvalue(t_test)
#test variancer
n = length(var_y)
chi2_statistic = (n - 1) * mean(var_y) / true_var
p_value_var = 1 - cdf(Chisq(n-1),chi2_statistic)

alpha = 0.05
combined_p_value = min(1,2*min(p_value_var,p_value_mean))

if combined_p_value < alpha
    println("Reject the null hypothesis. The computation may be wrong.")
else
    println("Fail to reject the null hypothesis. The computation is likely correct.")
end

println("Combined p-value: $combined_p_value")