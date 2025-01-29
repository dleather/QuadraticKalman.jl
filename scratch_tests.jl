using Revise
using LinearAlgebra, Random, Test
import QuadraticKalman as QK

using LinearAlgebra

μ̃ = [1.0, 0.0, 0.0]
Φ̃ = Matrix{Float64}(I, 3,3)
# build a fake QKParams that just has μ̃, Φ̃
dummy_params = QK.QKParams(1, 1, zeros(1), 0.8 .* Matrix(I,1,1), 1.0 .* Matrix(I,1,1),
                        zeros(1), 1.0 .* Matrix(I,1,1), [1.0 .* Matrix(I,1,1)],
                        1.0 .* Matrix(I,1,1), 1.0 .* Matrix(I,1,1), 1.0)

# a single time step, Ztt, Zttm1 both (3 x T̄), let T̄=2 => columns=2
Ztt = zeros(3, 2)       # all zeros
Ztt[:,1] = [10, 20, 30] # let's store something at column 1
Zttm1 = similar(Ztt)

# call the in-place version
QK.predict_Zₜₜ₋₁!(Ztt, Zttm1, dummy_params, 1)
# => Zttm1[:,1] = μ̃ + Φ̃*Ztt[:,1]
@test Zttm1[:,1] ≈ μ̃ .+ Ztt[:,1]  # i.e. [11,20,30]

# also check the pure functional version with a Vector
current_state = Ztt[:,1]
Z_pred = predict_Zₜₜ₋₁(current_state, dummy_params)
@test Z_pred ≈ [11.0, 20.0, 30.0]