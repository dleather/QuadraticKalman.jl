using Pkg; Pkg.activate("QuadraticKalman\\")
using Revise, QuadraticKalman, Plots, Statistics, StochasticDiffEq, LinearAlgebra, Random,
    FLoops, Enzyme, DifferentiationInterface, Distributions, DifferentialEquations

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


s_star = 0.1
t_star = 0.2

function compue_mean_oub(t, u, v, t0, T, θ)
    return u * (sinh(θ * (T - t)) / sinh(θ* (T - t0))) +
    v * (sinh(θ * (t - t0)) / sinh(θ * (T - t0)))
end

function compute_var_oub(t, t0, T, θ, σ)
    return ((σ^2) / θ) * sinh(θ * (T-t)) * sinh(θ * (t - t0)) /
        sinh(θ * (T - t0))
end

function compute_2m_oub(t, u, v, t0, T, θ, σ)
    return compute_var_oub(t, t0, T, θ, σ) +
        compue_mean_oub(t, u, v, t0, T, θ)^2
end

function compute_cov_oub(t, s, u, v, t0, T, θ, σ)
    if t >= s
        return ((σ^2) / θ) * sinh(θ * (s - t0)) * sinh(θ * (T - t)) /
        sinh(θ * (T - t0))
    else
        return ((σ^2) / θ) * sinh(θ * (t - t0)) * sinh(θ * (T - s)) /
        sinh(θ * (T - t0))
    end
end
cmpt_mean_y_t =  compue_mean_oub(t_star, u, v, t0, T, θ_z_init)
cmpt_mean_y_s =  compue_mean_oub(s_star, u, v, t0, T, θ_z_init)
cmpt_var_y_t = compute_var_oub(t_star, t0, T, θ_z_init, σ_z_init)
cmpt_var_y_s = compute_var_oub(s_star, t0, T, θ_z_init, σ_z_init)
cmpt_2m_y_t = compute_2m_oub(t_star, u, v, t0, T, θ_z_init, σ_z_init)
cmpt_2m_y_s = compute_2m_oub(s_star, u, v, t0, T, θ_z_init, σ_z_init)
cmpt_cov_y_ts = compute_cov_oub(t_star, s_star, u, v, t0, T, θ_z_init, σ_z_init)
#Define functions
function define_problem_ou_bridge(θ_z, σ_z, u, v, Δt)
    function drift_I1(uu, p, t)
        #return -θ_z * uu[1] + 2.0 * θ_z * (v * exp(θ_z * (Δt - t)) - uu[1]) / (exp(2.0 * θ_z * (Δt - t)) - 1.0)
        return θ_z * (-coth(θ_z * (Δt - t)) * uu[1] + (v / (sinh(θ_z * (Δt - t)))))
    end

    function diffusion_I1(uu,p,t)
        return σ_z
    end
    u0 = u
    prob = SDEProblem(drift_I1, diffusion_I1, u0, (0.0, Δt))
    eprob = EnsembleProblem(prob)
    return eprob, prob
end

function simulate_oub(eprob, N_run, N_boot, N_div, Δt, t_star, s_star)
    mean_y_t = zeros(N_run)
    mean_y_s = zeros(N_run)
    mean_ysq_t = zeros(N_run)
    mean_ysq_s = zeros(N_run)
    cov_y_ts = zeros(N_run)
    var_y_t = zeros(N_run)
    var_y_s = zeros(N_run)

    time_taken = @elapsed for j = 1:N_run
        esol = solve(eprob, EulerHeun(), dt = 0.25 / 100; trajectories = N_boot)
        y = zeros(2, N_boot)
        for i = 1:N_boot
            y[1,i] = esol[i](t_star)[1]
            y[2,i] = esol[i](s_star)[1]
        end
        mean_y_t[j] = mean(y[1,:])
        mean_y_s[j] = mean(y[2,:])
        mean_ysq_t[j] = mean(y[1,:].^2)
        mean_ysq_s[j] = mean(y[2,:].^2)
        var_y_t[j] = mean((y[1,:] .- cmpt_mean_y_t).^2)
        var_y_s[j] = mean((y[2,:] .- cmpt_mean_y_s).^2)
        cov_y_ts[j] = mean((y[1,:] .- cmpt_mean_y_t) .* (y[2,:] .- cmpt_mean_y_s))
        println(j)

    end
    println("Time taken: ", time_taken)
    return mean_y_t, mean_y_s, mean_ysq_t, mean_ysq_s, var_y_t, var_y_s, cov_y_ts
end

eprob, prob = define_problem_ou_bridge(θ_z_init, σ_z_init, u, v, Δt)

mean_y_t, mean_y_s, mean_ysq_t, mean_ysq_s, var_y_t, var_y_s, cov_y_ts = simulate_oub(eprob, N_run, N_boot, N_div, Δt, t_star, s_star)

histogram(mean_y_t, label = "Mean y(t)", title = "Histogram of Mean y(t)")
vline!([cmpt_mean_y_t], label = "Analytical Mean y(t)")

histogram(mean_y_s, label = "Mean y(s)", title = "Histogram of Mean y(s)")
vline!([cmpt_mean_y_s], label = "Analytical Mean y(s)")

histogram(mean_ysq_t, label = "Mean y(t)^2", title = "Histogram of Mean y(t)^2")
vline!([cmpt_2m_y_t])

histogram(mean_ysq_s, label = "Mean y(s)^2", title = "Histogram of Mean y(s)^2")
vline!([cmpt_2m_y_s])

histogram(var_y_t, label = "Var y(t)", title = "Histogram of Var y(t)")
vline!([cmpt_var_y_t])

histogram(var_y_s, label = "Var y(s)", title = "Histogram of Var y(s)")
vline!([cmpt_var_y_s])

histogram(cov_y_ts, label = "Cov y(t), y(s)", title = "Histogram of Cov y(t), y(s)")
vline!([cmpt_cov_y_ts])