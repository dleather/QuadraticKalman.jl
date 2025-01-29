@testset "QKF Smoother Tests" begin
    # Helper function to create test data
    function create_test_data(P::Int, T_bar::Int)
        # Create sample data with known structure
        Z = randn(P, T_bar + 1)
        P_mat = [Matrix(0.9I, P, P) for _ in 1:(T_bar + 1)]
        P_array = cat(P_mat..., dims=3)
        
        Z_pred = randn(P, T_bar)
        P_pred_mat = [Matrix(1.0I, P, P) for _ in 1:T_bar]
        P_pred_array = cat(P_pred_mat..., dims=3)
        
        # Simple versions of the special matrices
        H_aug = Matrix(1.0I, P, P)
        G_aug = Matrix(1.0I, P, P)
        Φ_aug = Matrix(1.0I, P, P)
        
        return Z, P_array, Z_pred, P_pred_array, H_aug, G_aug, Φ_aug
    end

    @testset "Basic Functionality" begin
        # Test with small dimensions
        P, T_bar = 2, 5
        Z, P_array, Z_pred, P_pred_array, H_aug, G_aug, Φ_aug = create_test_data(P, T_bar)
        
        # Test in-place version
        Z_inplace = copy(Z)
        P_inplace = copy(P_array)
        @test_nowarn QK.qkf_smoother!(Z_inplace, P_inplace, Z_pred, P_pred_array, 
                                  T_bar, G_aug, H_aug, Φ_aug, P)
        
        # Test out-of-place version
        Z_smooth, P_smooth = QK.qkf_smoother(Z, P_array, Z_pred, P_pred_array, 
                                        T_bar, G_aug, H_aug, Φ_aug, P)
        

        # Check that results are different from inputs (smoothing should modify values)
        @test any(Z_smooth .!= Z)
        @test any(P_smooth[1,1,:] .< P_array[1,1,:])

        # Additional check: Verify that smoothed covariances are positive definite
        for t in 1:(T_bar+1)
            @test isposdef(P_smooth[:,:,t])
        end
    end

    @testset "Dimension Checks" begin
        P, T_bar = 3, 4
        Z, P_array, Z_pred, P_pred_array, H_aug, G_aug, Φ_aug = create_test_data(P, T_bar)
        
        # Test with wrong dimensions
        Z_bad = randn(P, T_bar)  # Missing one column
        @test_throws AssertionError QK.qkf_smoother!(Z_bad, P_array, Z_pred, P_pred_array, 
                                                 T_bar, G_aug, H_aug, Φ_aug, P)
        

        P_bad = P_array[:, :, 1:T_bar]  # Missing one time slice
        @test_throws AssertionError QK.qkf_smoother!(Z, P_bad, Z_pred, P_pred_array, 
                                                 T_bar, G_aug, H_aug, Φ_aug, P)
    end


    @testset "Numerical Stability" begin
        # Test with larger dimensions
        P, T_bar = 10, 20
        Z, P_array, Z_pred, P_pred_array, H_aug, G_aug, Φ_aug = create_test_data(P, T_bar)
        
        # Add some numerical complexity
        for t in 1:T_bar
            P_array[:,:,t] *= 1e-3
            P_pred_array[:,:,t] *= 1e-3
        end
        
        # Should still work without numerical issues
        @test_nowarn QK.qkf_smoother!(copy(Z), copy(P_array), Z_pred, P_pred_array, 
                                  T_bar, G_aug, H_aug, Φ_aug, P)
    end


    @testset "Special Cases" begin
        # Test minimal case
        P, T_bar = 1, 2
        Z, P_array, Z_pred, P_pred_array, H_aug, G_aug, Φ_aug = create_test_data(P, T_bar)

        @test_nowarn QK.qkf_smoother!(copy(Z), copy(P_array), Z_pred, P_pred_array, 
                                  T_bar, G_aug, H_aug, Φ_aug, P)
        

        # Test with non-identity special matrices
        H_aug = [1.0 0.5; 0.5 1.0]
        G_aug = [1.0 -0.5; -0.5 1.0]
        Φ_aug = [0.9 0.1; 0.1 0.9]
        
        P = 2
        Z, P_array, Z_pred, P_pred_array, _, _, _ = create_test_data(P, T_bar)
        
        @test_nowarn QK.qkf_smoother!(Z, P_array, Z_pred, P_pred_array, 
                                  T_bar, G_aug, H_aug, Φ_aug, P)
    end


    @testset "Type Stability" begin
        # Test with different floating point types
        for T in [Float32, Float64]
            P, T_bar = 2, 3
            Z = randn(T, P, T_bar + 1)
            
            # Ensure positive definiteness by using proper covariance matrix structure
            P_array = Array{T, 3}(undef, P, P, T_bar + 1)
            for t in 1:(T_bar + 1)
                P_array[:,:,t] = Matrix{T}(I, P, P)
            end
            
            Z_pred = randn(T, P, T_bar)
            P_pred_array = Array{T, 3}(undef, P, P, T_bar)
            for t in 1:T_bar
                P_pred_array[:,:,t] = Matrix{T}(I, P, P)
            end
            
            # Use identity matrices directly instead of scaling
            H_aug = Matrix{T}(I, P, P)
            G_aug = Matrix{T}(I, P, P)
            Phi_aug = Matrix{T}(I, P, P)

            # Test that output types match input types
            Z_smooth, P_smooth = QK.qkf_smoother(Z, P_array, Z_pred, P_pred_array, 
                                            T_bar, G_aug, H_aug, Phi_aug, P)
            @test eltype(Z_smooth) == T
            @test eltype(P_smooth) == T
        end
    end
end