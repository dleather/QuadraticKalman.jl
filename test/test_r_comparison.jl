using Test
using CSV
using DataFrames
using QuadraticKalman
using RData
using JSON
using Statistics

@testset "Compare to R results" begin
    # Load R results
    r_filter = load("test/r_results/qkf_results.rds")
    r_smoother = load("test/r_results/qks_results.rds")

    # Load test data and parameters
    Y = CSV.read("test/simulated_data/simulated_measurements.csv", DataFrame) |> Matrix
    X = CSV.read("test/simulated_data/simulated_states.csv", DataFrame) |> Matrix
    
    # Load params in JSON
    params = JSON.parsefile("test/simulated_data/simulated_params.json")
    
    # Unload params
    n = 2
    m = 2
    mu = Float64.(params["mu"])
    Phi = Float64.(reduce(vcat, params["Phi"])) |> x -> reshape(x, n, n)
    Omega = Float64.(reduce(vcat, params["Omega"])) |> x -> reshape(x, n, n)
    a = Float64.(params["a"])
    B = Float64.(reduce(vcat, params["B"])) |> x -> reshape(x, m, n)
    C = [Float64.(reduce(vcat, c)) |> x -> reshape(x, n, n) for c in params["C"]]
    D = Float64.(reduce(vcat, params["D"])) |> x -> reshape(x, m, m)
    alpha = zeros(m, m)
    
    # Create model with parameters matching R
    model = QKModel(
        2, 2, mu, Phi, Omega, a, B, C, D, alpha
    )

    # Run filter and smoother
    data = QKData(Y')
    results = qkf(data, model)

    # Compare filter results 
    @test isapprox(results.filter.Z_tt[:,2:end], Matrix(r_filter["Z.updated"]), rtol=1e-4)
    @test isapprox(results.filter.ll_t, r_filter["loglik.vector"], rtol=1e-4)
    
    # Compare smoother results
    @test mean(abs.(results.smoother.Z_smooth[[1,2,3,4,6],2:end] .-Matrix(r_smoother["Z.smoothed"]))) <= 1e-4

    # Test covariance matrices if available
    if hasproperty(r_filter, :P11_tt)
        for t in 1:size(Y,1)
            P_tt_R = [
                r_filter.P11_tt[t] r_filter.P12_tt[t];
                r_filter.P12_tt[t] r_filter.P22_tt[t]
            ]
            @test isapprox(results.filter.P_tt[:,:,t], P_tt_R, rtol=1e-4)
        end
    end

    if hasproperty(r_smoother, :P11_smooth)
        for t in 1:size(Y,1)
            P_smooth_R = [
                r_smoother.P11_smooth[t] r_smoother.P12_smooth[t];
                r_smoother.P12_smooth[t] r_smoother.P22_smooth[t]
            ]
            @test isapprox(results.smoother.P_smooth[:,:,t], P_smooth_R, rtol=1e-4)
        end
    end
end
