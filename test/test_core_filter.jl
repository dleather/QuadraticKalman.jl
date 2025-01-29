# test_core_filter.jl
using Test
using QuadraticKalman  # or whatever your module is called

@testset "QKF Filter Tests" begin

    ############################################################################
    # 1. Unit Tests for State Prediction Functions
    ############################################################################

    @testset "predict_Zₜₜ₋₁! & predict_Zₜₜ₋₁" begin
        # We'll do a tiny example with P=3 (imagine N=1 => P=1+1^2=2).
        # Suppose μ̃=[1,0,0], Φ̃=I(3), just to see if predict_Z works.
        using LinearAlgebra

        μ̃ = [1.0, 0.0, 0.0]
        Φ̃ = Matrix{Float64}(I, 3,3)
        # build a fake QKParams that just has μ̃, Φ̃
        dummy_params = QKParams(N=1, M=1,
                                μ=zeros(1), Φ=Matrix(I,1,1), Ω=Matrix(I,1,1),
                                A=zeros(1), B=Matrix(I,1,1), C=[Matrix(I,1,1)],
                                D=Matrix(I,1,1), α=Matrix(I,1,1), Δt=1.0,
                                Σ=Matrix(I,1,1), V=Matrix(I,1,1), e=[], μ̃=μ̃, Φ̃=Φ̃,
                                Λ=zeros(1,1), L1=Matrix(I,1,1), L2=Matrix(I,1,1), L3=Matrix(I,1,1),
                                μᵘ=zeros(1), Σᵘ=Matrix(I,1,1),
                                μ̃ᵘ=zeros(3), Σ̃ᵘ=Matrix(I,3,3),
                                B̃=Matrix(I,1,3), H̃=Matrix(I,3,3), G̃=Matrix(I,3,3), P=3
        )

        # a single time step, Ztt, Zttm1 both (3 x T̄), let T̄=2 => columns=2
        Ztt = zeros(3, 2)       # all zeros
        Ztt[:,1] = [10, 20, 30] # let's store something at column 1
        Zttm1 = similar(Ztt)

        # call the in-place version
        predict_Zₜₜ₋₁!(Ztt, Zttm1, dummy_params, 1)
        # => Zttm1[:,1] = μ̃ + Φ̃*Ztt[:,1]
        @test Zttm1[:,1] ≈ μ̃ .+ Ztt[:,1]  # i.e. [11,20,30]

        # also check the pure functional version with a Vector
        current_state = Ztt[:,1]
        Z_pred = predict_Zₜₜ₋₁(current_state, dummy_params)
        @test Z_pred ≈ [11.0, 20.0, 30.0]
    end


    ############################################################################
    # 2. Unit Tests for predict_Pₜₜ₋₁! & predict_Pₜₜ₋₁
    ############################################################################

    @testset "predict_Pₜₜ₋₁! & predict_Pₜₜ₋₁" begin
        # We'll define a small P=2, with Φ̃=some 2x2,
        # Then confirm Pₜₜ₋₁ = Φ̃ * Pₜₜ * Φ̃' + Σₜₜ₋₁
        # We need a function compute_Σₜₜ₋₁! or similar that your code calls,
        # but for the test, we can mimic it or do a partial test.

        using LinearAlgebra
        Φ̃ = [0.9 0.0;
              0.0 0.95]
        # put that in dummy_params
        dummy_params = QKParams(...    # same approach
            Φ̃=Φ̃,
            # also pass whatever else is needed
        )

        # Suppose Pₜₜ[:,:,1] is 2x2 identity => Pₜₜ=I,
        Pₜₜ = zeros(2,2,2)
        Pₜₜ[:,:,1] = Matrix(I,2,2)
        # We also have Σₜₜ₋₁ => let's define it as 0.01I
        Σₜₜ₋₁ = zeros(2,2,2)
        Σₜₜ₋₁[:,:,1] = 0.01 .* I(2)

        # We'll call predict_Pₜₜ₋₁! with t=1
        Pₜₜ₋₁ = zeros(2,2,2)
        # pass a tmpP, plus some Zₜₜ (which your code uses for compute_Σₜₜ₋₁!)
        Zₜₜ = zeros(2,2)
        tmpP = zeros(2,2)

        predict_Pₜₜ₋₁!(Pₜₜ, Pₜₜ₋₁, Σₜₜ₋₁, Zₜₜ, tmpP, dummy_params, 1)
        # => Pₜₜ₋₁[:,:,1] = ensure_positive_definite(Φ̃ * I * Φ̃' + 0.01I)

        expected = Φ̃ * Matrix(I,2,2) * Φ̃' .+ 0.01 .* I(2)
        # ensure posdef => expected is diagonal [0.9^2+0.01, 0.95^2+0.01]
        @test Pₜₜ₋₁[:,:,1] ≈ expected

        # Now check the functional version
        P_in  = I(2)
        P_out = predict_Pₜₜ₋₁(P_in, Zₜₜ[:,1], dummy_params, 1)
        @test P_out ≈ expected
    end


    ############################################################################
    # 3. Unit Tests for Measurement Prediction
    ############################################################################

    @testset "predict_Yₜₜ₋₁! & predict_Yₜₜ₋₁" begin
        # Suppose M=2, P=3. We have A= [1,2], B̃= a 2x3, α=2x2
        # We'll do a single step: Yₜₜ₋₁[:, t] = A + B̃ * Zₜₜ₋₁[:,t] + α * Y[:, t].
        using LinearAlgebra

        A = [1.0, 2.0]
        B̃ = [1.0 0.0 2.0;
              0.5 0.5 1.0]
        α = Matrix(I,2,2) # 2x2 identity
        # build dummy_params
        dummy_params = QKParams(..., A=A, B̃=B̃, α=α, ...)

        # define Zₜₜ₋₁= (3x2), Y= (2x2?), let's do T̄=2
        Zₜₜ₋₁ = zeros(3,2)
        Zₜₜ₋₁[:,1] = [10,20,30]
        Y = zeros(2,2)
        Y[:,1] = [100,200]

        Yₜₜ₋₁_mat = zeros(2,2) # store predicted measurement
        predict_Yₜₜ₋₁!(Yₜₜ₋₁_mat, Zₜₜ₋₁, Y, dummy_params, 1)
        # => Yₜₜ₋₁[:,1] = A + B̃*[10,20,30] + I(2)* [100,200]
        # B̃*[10,20,30] = [1*10 + 0*20 + 2*30; 0.5*10 +0.5*20 +1*30 ] = [10+60, 5+10+30] = [70,45]
        # => [1+70, 2+45] = [71,47], + alpha*[100,200] => [100,200]
        # => final [171,247]
        @test Yₜₜ₋₁_mat[:,1] ≈ [171,247]

        # Also test the functional version predict_Yₜₜ₋₁(Zₜₜ₋₁, Y, params, t)
        y_pred = predict_Yₜₜ₋₁(Zₜₜ₋₁, Y, dummy_params, 1)
        @test y_pred ≈ [171,247]
    end


    ############################################################################
    # 4. Unit Tests for Kalman Gain (compute_Kₜ!)
    ############################################################################

    @testset "compute_Kₜ! & compute_Kₜ" begin
        # Suppose univariate M=1 => B̃ is 1x3, Pₜₜ₋₁ is 3x3, Mₜₜ₋₁ is 1x1
        # K = Pₜₜ₋₁ * B̃' / (B̃ *Pₜₜ₋₁ *B̃' + R)
        B̃ = [1.0 2.0 3.0]  # (1x3)
        # let's do a small Pₜₜ₋₁= I(3)
        Pₜₜ₋₁ = zeros(3,3,1); Pₜₜ₋₁[:,:,1] = I(3)
        # Mₜₜ₋₁= 1x1 => e.g. 4.0
        Mₜₜ₋₁ = zeros(1,1,1); Mₜₜ₋₁[1,1,1] = 4.0
        # Gains => dimension is (3x1xT̄)
        Kₜ = zeros(3,1,1)
        # build dummy_params
        dummy_params = QKParams(..., B̃=B̃, M=1, ...)

        compute_Kₜ!(Kₜ, Pₜₜ₋₁, Mₜₜ₋₁, nothing, dummy_params, 1)
        # => Kₜ[:, :, 1] .= I(3)* B̃' / 4.0 = B̃'/4 => B̃' = (3x1)
        @test Kₜ[:,1,1] ≈ [1/4, 2/4, 3/4]

        # pure function:
        K_mat = compute_Kₜ(I(3), [4.0], dummy_params, 1)
        @test K_mat ≈ reshape([1/4,2/4,3/4], 3,1)
    end


    ############################################################################
    # 5. Unit Tests for update_Zₜₜ! and update_Pₜₜ!
    ############################################################################

    @testset "update_Zₜₜ! & update_Pₜₜ!" begin
        # We'll do univariate again, P=2, M=1, K is 2x1
        # Zₜₜ₋₁ is (2x1) => say [10,20], measurement= y= 5, predicted meas=2 => innovation=3
        # => Zₜₜ= Zₜₜ₋₁ + K*(y - ypred)

        # Z
        Zₜₜ₋₁ = zeros(2,2)
        Zₜₜ₋₁[:,1] = [10,20]
        Yₜ = fill(5.0,2);  # we only use Yₜ[t=1] => 5
        Yₜₜ₋₁ = fill(2.0,2)

        # K => e.g. [0.1;0.2]
        Kₜ = zeros(2,1,2)
        Kₜ[:,:,1] = [0.1;0.2]

        # In-place
        Zₜₜ = similar(Zₜₜ₋₁)
        QuadraticKalman.update_Zₜₜ!(Zₜₜ, Kₜ, Yₜ, Yₜₜ₋₁, Zₜₜ₋₁, zeros(2), nothing, 1)
        # => Zₜₜ[:,2] = [10,20] + [0.1,0.2]* (5-2)=3 => [10+0.3, 20+0.6]= [10.3,20.6]
        @test Zₜₜ[:,2] ≈ [10.3,20.6]

        # Cov update => typically: Pₜₜ= (I-KB) Pₜₜ₋₁ (I-KB)' + K R K'
        # We'll do a small manual check. Suppose B= (1x2). Let's do an example.
        # The snippet is a partial check.

        # We'll rely on your "update_Pₜₜ!" function code. 
        # For simplicity, we'll just pass some small P and see if it runs w/o error.
        Pₜₜ₋₁ = zeros(2,2,2)
        Pₜₜ₋₁[:,:,1] = I(2)
        Pₜₜ = similar(Pₜₜ₋₁)
        Mₜₜ₋₁ = nothing
        tmpKM= zeros(2,1); tmpKMK=zeros(2,2)
        # define B? or is read from params
        # We'll just skip the dimension check. This demonstrates calling it.
        QuadraticKalman.update_Pₜₜ!(Pₜₜ, Kₜ, Mₜₜ₋₁, Pₜₜ₋₁, Zₜₜ₋₁, tmpKM, tmpKMK, nothing, 1)
        @test isnothing(Pₜₜ) == false
        # you might do partial numeric checks if your code uses B & V from params.
    end


    ############################################################################
    # 6. PSD Correction (correct_Zₜₜ!)
    ############################################################################

    @testset "correct_Zₜₜ!" begin
        # Suppose N=1 => the state is [x; x^2], so dimension = 2. For N=2 => dimension=6. 
        # We'll test a quick scenario: N=2 => length=2+2^2=6
        # The function extracts x=Zₜₜ[1:2], XX'=Zₜₜ[3:6], then does implied_cov=XX' - x x'
        # clamps negatives, reconstruct.

        N=2
        Zₜₜ = zeros(6,2)  # 2 columns => T̄=1
        # col 2 is the time=1 updated state. x= [1,2], XX'= ~ [1 2;2 4]
        # but let's make it slightly indefinite
        x = [1.0, 2.0]
        XX = [1.0 2.0; 2.0 3.5]  # not quite 4 => negative e-vals?
        Zₜₜ[1:2,2] = x
        Zₜₜ[3:6,2] = vec(XX)

        # call correct_Zₜₜ!
        dummy_params = QKParams(..., N=N)
        correct_Zₜₜ!(Zₜₜ, dummy_params, 1)
        # => check that the sub-block is now PSD.

        # we can parse out the new block
        x_post = Zₜₜ[1:2,2]
        XX_post = reshape(Zₜₜ[3:6,2], 2,2)
        implied_cov = XX_post .- x_post*x_post'
        @test isposdef(Symmetric(implied_cov))
    end


    ############################################################################
    # 7. Integration Test: qkf_filter! or qkf_filter
    ############################################################################

    @testset "Integration: qkf_filter!" begin
        # We'll do a simple scenario with N=1 => P=2, M=1 => 1D measurement, T̄=4 steps
        # We'll build a toy QKData with Y= [NaN, 1,2,3,4] or something, ignoring Y[1].
        # We'll build QKParams with small random or known stable Φ, etc.
        # Then run qkf_filter! and verify it returns a FilterOutput with the correct shapes.

        using Random
        rng = MersenneTwister(1234)

        T̄ = 4
        Y = [0.0, 1.0, 2.0, 3.0, 4.0]  # length= T̄+1 => 5
        data = QKData(Y)
        # build minimal QKParams => N=1 => P=2
        # we won't test correctness deeply here, just shape
        N=1; M=1;
        μ̃ᵘ = [0.0, 1.0]
        Σ̃ᵘ = Matrix(I,2,2)*1e-2
        # fill the rest...
        my_params = QKParams(..., # fill all needed fields
                             N=1, M=1, P=2, μ̃ᵘ=μ̃ᵘ, Σ̃ᵘ=Σ̃ᵘ, ...)

        # call qkf_filter!
        result = qkf_filter!(data, my_params)
        # check shapes
        @test length(result.llₜ) == T̄
        @test size(result.Zₜₜ) == (2, T̄+1)
        @test size(result.Pₜₜ) == (2,2, T̄+1)
        @test size(result.Yₜₜ₋₁) == (T̄,)
        @test size(result.Mₜₜ₋₁) == (1,1, T̄)
        @test size(result.Kₜ)    == (2,1, T̄)
        @test size(result.Zₜₜ₋₁) == (2, T̄)
        @test size(result.Pₜₜ₋₁) == (2,2, T̄)
        @test size(result.Σₜₜ₋₁) == (2,2, T̄)
    end

    @testset "Integration: qkf_filter (out-of-place)" begin
        # same scenario, but calls the out-of-place version
        Y = [0.0, 1.0, 2.0, 3.0, 4.0]
        data = QKData(Y)
        N=1; M=1; T̄=4;
        μ̃ᵘ = [0.0, 1.0]
        Σ̃ᵘ = Matrix(I,2,2)*1e-2
        my_params = QKParams(..., N=1, M=1, P=2, μ̃ᵘ=μ̃ᵘ, Σ̃ᵘ=Σ̃ᵘ, ...)

        result = qkf_filter(data, my_params)
        @test length(result.llₜ) == T̄
        @test size(result.Zₜₜ) == (2, T̄+1)
        @test size(result.Pₜₜ) == (2,2, T̄+1)
    end

end
