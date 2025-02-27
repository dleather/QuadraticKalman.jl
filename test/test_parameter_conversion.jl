using Test
import QuadraticKalman as QK
using LinearAlgebra

@testset "Parameter Conversion" begin
    N, M = 2, 1
    
    # Calculate expected parameter counts
    n_mu = N                     # 2
    n_Phi = N^2                  # 4
    n_Dstate = div(N*(N+1), 2)   # 3
    n_A = M                      # 1
    n_B = M * N                  # 2
    n_C = M * N^2                # 4
    n_Dmeas = div(M*(M+1), 2)    # 1
    n_alpha = M^2                # 1
    # Total: 18
    
    # Create a parameter vector with exactly 18 elements
    params = [
        # State parameters
        0.1, 0.2,           # mu (N=2)
        0.5, 0.1, 0.1, 0.5, # Phi (N×N=4)
        0.1, 0.0, 0.1,      # D_state (N(N+1)/2=3)
        
        # Measurement parameters
        0.0,                # A (M=1)
        1.0, 0.0,           # B (M×N=2)
        1.0, 0.0, 0.0, 1.0, # C (M×N²=4)
        0.1,                # D_meas (M(M+1)/2=1)
        0.0                 # alpha (M²=1)
    ]
    
    @test length(params) == 18  # Verify we have exactly 18 parameters
    
    # Convert to model
    model = QK.params_to_model(params, N, M)
    
    # Check model components match expected values
    @test model.state.mu ≈ [0.1, 0.2]
    @test model.state.Phi ≈ [0.5 0.1; 0.1 0.5]
    
    # Convert back to parameters
    params2 = QK.model_to_params(model)
    
    # Check round-trip conversion
    @test length(params) == length(params2)
    @test params ≈ params2
end

