using Test
import QuadraticKalman as QK
using LinearAlgebra, Random, Plots, Random
gr()
Random.seed!(123)

@testset "Plot Recipes" begin
    # Set up test data
    N, T = 2, 10
    true_states = randn(N, T)
    ll_t = randn(T-1)
    P_tt = cat([Matrix(0.1I, N, N) for _ in 1:T]..., dims=3)
    Z_tt = randn(N, T)
    filter_results = QK.FilterOutput(ll_t, Z_tt, P_tt)
    
    smoother_results = QK.SmootherOutput(
        (Z_smooth = randn(N, T),
        P_smooth = cat([Matrix(0.1I, N, N) for _ in 1:T]..., dims=3))
    )

    @testset "Truth Comparison Plots" begin
        # Test filter truth plot
        p1 = Plots.plot(QK.KalmanFilterTruthPlot(true_states, filter_results))
        @test length(p1.series_list) == N*3  # 3 series per state
        @test p1[1].attr[:title] == "Kalman Filter Performance"
        
        # Test smoother truth plot
        p2 = plot(QK.KalmanSmootherTruthPlot(true_states, smoother_results))
        @test p2.attr[:layout] == (N, 1)
    end

    @testset "Estimate Plots" begin
        # Test filter estimate plot
        p3 = plot(QK.KalmanFilterPlot(filter_results))
        @test p3.attr[:size] == (1000, 350)
        @test p3[1].series_list[1][:seriestype] == :path
        
        # Test smoother estimate plot
        p4 = plot(QK.KalmanSmootherPlot(smoother_results))
        @test p4[1].attr[:title] == "Kalman Smoother Estimates"
        @test length(p4.series_list) == N  # 2 series per state
    end

    @testset "Helper Functions" begin
        # Test convenience functions
        @test QK.kalman_filter_truth_plot(true_states, filter_results) isa QK.KalmanFilterTruthPlot
        @test QK.kalman_smoother_plot(smoother_results) isa QK.KalmanSmootherPlot
        
        # Test plotting from helper functions
        @test plot(QK.kalman_filter_truth_plot(true_states, filter_results)) isa Plots.Plot
        @test plot(QK.kalman_smoother_truth_plot(true_states, smoother_results)) isa Plots.Plot
    end

    @testset "Edge Cases" begin
        # Test single state system
        single_state = randn(1, 5)
        single_filter = QK.FilterOutput(
            (ll_t = randn(4),
            Z_tt = randn(1, 5),
            P_tt = cat([fill(0.1, 1, 1) for _ in 1:5]..., dims=3),
            Y_ttm1 = Vector{Float64}(undef, 0),
            M_ttm1 = Array{Float64,3}(undef, 0, 0, 0),
            K_t = Array{Float64,3}(undef, 0, 0, 0),
            Z_ttm1 = Matrix{Float64}(undef, 0, 0),
            P_ttm1 = Array{Float64,3}(undef, 0, 0, 0))
        )
        p5 = plot(QK.KalmanFilterTruthPlot(single_state, single_filter))
        @test p5.attr[:layout] == (1, 1)
    end

    @testset "Edge Cases for Plots" begin
        # Test with single state
        single_state = randn(1, 5)
        single_P = cat([[0.1 0.05; 0.05 0.1] for _ in 1:5]..., dims=3)
        single_filter = QK.FilterOutput(randn(4), vcat(single_state, single_state.^2), single_P)
        p = plot(QK.KalmanFilterPlot(single_filter))
        @test p isa Plots.Plot
        
        # Test with very small variance instead of zero
        small_var_P = cat([fill(1e-10, 2, 2) for _ in 1:5]..., dims=3)
        small_var_filter = QK.FilterOutput(randn(4), randn(2, 5), small_var_P)
        p = plot(QK.KalmanFilterPlot(small_var_filter))
        @test p isa Plots.Plot
    end
end