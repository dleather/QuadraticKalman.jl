module QuadraticKalman

using LinearAlgebra, Parameters, SpecialFunctions, Zygote, DifferentiableEigen, LogExpFunctions

#Model:
# (Eq 1a) Xₜ = μ + ΦXₜ₋₁ + Ωεₜ (State Equation; Can be Partially-Observable; N x 1) 
# (Eq 1b) Yₜ = A + BXₜ + αYₜ₋₁ + ∑ₖ₌₁ᵐ eₖXₜ'C⁽ᵏ⁾Xₜ + D_t ηₜ (Observation Equation; Fully Observable; M x 1)
# Σ = ΩΩ' (State Covariance; N x N)
# V = DD' (Observation Covariance; M x M)
# ϵₜ ~ N(0, I) (State Noise; N x 1)
# ηₜ ~ N(0, I) (Observation Noise; M x 1)
# 

#Augmented Model:
# Zₜ := [Xₜ, vec(XₜXₜ')] (Augmented State; N + N² x 1)
# Zₜ = μ̃ + Φ̃Zₜ₋₁ + Ω̃ₜ₋₁ξₜ (Augmented State Equation; N + N² x 1)
# Yₜ = A + B̃Zₜ + D_t ηₜ (Observation Equation; Fully Observable; M x 1)

@with_kw struct QKParams{T <: Real, T2 <: Real} 
    # Deep Parameters
    N::Int
    M::Int
    μ::Vector{T}
    Φ::Matrix{T}
    Ω::Matrix{T}
    A::Vector{T}
    B::Matrix{T}
    C::Vector{Matrix{T}}
    α::Matrix{T}
    Δt::T2
    # V is now a function of weights: V(z) = wc + wu * u + wv * v + wu2 * u^2 + wuv * u*v + wv2 * v^2
    wc::T
    wu::T
    wv::T
    wuu::T
    wuv::T
    wvv::T
    # Precomputed Parameters
    Σ::Matrix{T}
    e::Vector{Vector{T}}
    μ̃::Vector{T}
    Φ̃::Matrix{T}
    Λ::Matrix{T2}
    L1::Matrix{T}
    L2::Matrix{T}
    L3::Matrix{T}
    μᵘ::Vector{T}
    Σᵘ::Matrix{T}
    μ̃ᵘ::Vector{T}
    Σ̃ᵘ::Matrix{T}
    B̃::Matrix{T}
    H̃::Matrix{T}
    G̃::Matrix{T}
    P::Int
end

# If D is a function
function QKParams(N::Int, M::Int, 
    μ::Vector{T}, 
    Φ::Matrix{T},
    Ω::Matrix{T}, 
    A::Vector{T}, 
    B::Matrix{T}, 
    C::Vector{<:AbstractMatrix{T}},
    wc::T, wu::T, wv::T, wuu::T, wuv::T, wvv::T,
    α::Matrix{T}, 
    Δt::T2) where {T <: Real, T2 <: Real}

    # Enforce Restrictions
    # @assert size(Φ) == (N, N)
    # @assert spectral_radius(Φ) < 1
    # @assert size(Ω) == (N, N)
    # @assert length(A) == M
    # @assert size(B) == (M, N)
    # @assert size(D) == (M, M)
    # @assert length(μ) == N
    # @assert length(C) == M
    # @assert Δt > 0
    # for i in 1:M
    #     @assert size(C[i]) == (N, N)
    # end

    # Precompute
    Σ = Ω * Ω'
    e = compute_e(M, T)
    Λ = compute_Λ(N)
    μ̃ = compute_μ̃(μ, Σ)
    Φ̃ = compute_Φ̃(μ, Φ)
    L1 = compute_L1(Σ, Λ)
    L2 = compute_L2(Σ, Λ)
    L3 = compute_L3(Σ, Λ)
    μᵘ = compute_μᵘ(μ, Φ)
    Σᵘ = compute_Σᵘ(Φ, Σ)
    μ̃ᵘ = compute_μᵘ(μ̃, Φ̃)
    Σ̃ᵘ = compute_Σ̃ᵘ(μ̃ᵘ, L1, L2, L3, Λ, Σ, μ, Φ̃)
    B̃ = compute_B̃(B, C)
    H = selection_matrix(N, T)
    G = duplication_matrix(N, T)
    H̃ = compute_H̃(N, H)
    G̃ = compute_G̃(N, G)
    P = N + N^2

    return QKParams(N, M, μ, Φ, Ω, A, B, C, α, Δt, wc, wu, wv, wuu, wuv, wvv,
        Σ, e, μ̃, Φ̃, Λ, L1, L2, L3, μᵘ, Σᵘ, μ̃ᵘ, Σ̃ᵘ, B̃, H̃, G̃, P)
end
    #=
    function get_e_old(k::Int, M::Int, ::ype{<:Real}=Float64)
    # e is the column selection vector of size m whose components are 0 except the kᵗʰ one,
    # which is equal to 1.
        #@assert 1 <= k <= M
        e = zeros(T, M)
        e[k] = one(T)
        return e
    end
    =#

    function get_e(k::Int, M::Int, ::Type{T}=Float64) where T <: Real
        return [i == k ? one(T) : zero(T) for i in 1:M]
    end
    
    function compute_e(M::Int, ::Type{T}=Float64) where T <: Real
    # e is the column selection vector of size m whose components are 0 except the kᵗʰ one,
    # which is equal to 1.
        return [get_e(k, M, T) for k in 1:M]
    end
    #=
    function compute_Λ_old(N::Int, ::Type{<:Real}=Float64)
    # Λₙ being the n² × x² matrix, partioned in n × n blocks, with the (i, j) block being
    # eᵢeⱼ'.
        e = compute_e(N, T)
        Λ = zeros(T, N^2, N^2)
        for i in 1:N
            for j in 1:N
                i0 = (i-1) * N + 1
                i1 = i * N
                j0 = (j-1) * N + 1
                j1 = j * N
                Λ[i0:i1, j0:j1] .= e[j] * e[i]'
            end
        end
        return Λ
    end
    =#
    function compute_Λ(N::Int, ::Type{T}=Float64) where T <: Real
        e = compute_e(N, T)
       
        Λ = [
            Matrix(kron(e[i], e[j])')  # Convert to Matrix explicitly
            for i in 1:N, j in 1:N
        ]
       
        return reduce(vcat, Λ)
    end

    function compute_Λ(e::AbstractVector{AbstractVector{T}}, N::Int) where T <: Real
    # Λₙ being the n² × x² matrix, partioned in n × n blocks, with the (i, j) block being
    # eᵢeⱼ'.
        M = length(e)
        Λ = zeros(T, N^2, N^2)
        for i in 1:N
            for j in 1:N
                i0 = (i-1) * N + 1
                i1 = i * N
                j0 = (j-1) * N + 1
                j1 = j * N
                Λ[i0:i1, j0:j1] .= e[j] * e[i]'
            end
        end
        return Λ
    end

    #=
    function compute_μ̃_old(μ::AbstractVector{T}, Σ::AbstractMatrix{T})
         # Prop. 3.1: μ̃ = [μ, vec(μμ′ + Σ)]
         N = length(μ)
         μ̃ = zeros(T, N + N^2)
         μ̃[1:N] .= μ
         μ̃[N+1:end] .= vec(μ*μ' + Σ)
         return μ̃
     end
    =#

    function compute_μ̃(μ::Real, Σ::Real) 
        return [μ; μ^2 + Σ]
    end

    function compute_μ̃(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where T <: Real
        # Prop. 3.1: μ̃ = [μ, vec(μμ′ + Σ)]

        first_part = μ
        second_part = vec(μ * μ' + Σ)
        
        # Concatenate the two parts
        μ̃ = vcat(first_part, second_part)
        
        return μ̃
    end

    #= 
    function compute_Φ̃_old(μ::AbstractVector{T}, Φ::AbstractMatrix{T})
         # Prop. 3.1: Φ̃ = [Φ 0.0; (μ⊗Φ + Φ⊗μ) \ Φ⊗Φ]
         N = length(μ)
         P = N + N^2
         Φ̃ = zeros(T, P, P)
         Φ̃[1:N, 1:N] .= Φ
         Φ̃[N+1:end, 1:N] .= kron(μ, Φ) .+ kron(Φ, μ)
         Φ̃[N+1:end, N+1:end] .= kron(Φ, Φ)
         return Φ̃
     end
    =#
    function compute_Φ̃(μ::Real, Φ::Real)
        # Prop. 3.1: Φ̃ = [Φ 0.0; (μ⊗Φ + Φ⊗μ), Φ⊗Φ]
        return [Φ 0.0; 2.0*μ*Φ Φ^2]
    end

    function compute_Φ̃(μ::AbstractVector{T}, Φ::AbstractMatrix{T}) where T <: Real
        N = length(μ)
        P = N + N^2
    
        # Construct the top-left part (N x N)
        top_left = copy(Φ)
    
        # Construct the top-right part (N x N^2)
        top_right = zeros(eltype(Φ), N, N^2)
    
        # Construct the bottom part (N^2 x P)
        bottom = [
            let 
                (i, j) = divrem(row - 1, N) .+ 1
                if col <= N
                    μ[j] * Φ[i,col] + μ[i] * Φ[j,col]
                else
                    k, l = divrem(col - N - 1, N) .+ 1
                    Φ[i,k] * Φ[j,l]
                end
            end
            for row in 1:N^2, col in 1:P
        ]
    
        # Combine all parts
        Φ̃ = [
            [top_left top_right]
            reshape(bottom, N^2, P)
        ]
    
        return Φ̃
    end

    function compute_L1(Σ::AbstractMatrix{T1}, Λ::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}
    # LHS coefficient from Proposition 3.2
    #   L₁ = [Σ ⊗ (Iₙ² + Λₙ)] ⊗ [vec(Iₙ) ⊗ Iₙ]
        N = size(Σ, 1)
        return kron(Σ, I + Λ) * kron(vec(I(N)), I(N))
    end

    function compute_L1(Σ::Real, Λ::AbstractMatrix{T}) where T <: Real
        # LHS coefficient from Proposition 3.2
        #   L₁ = [Σ ⊗ (Iₙ² + Λₙ)] ⊗ [vec(Iₙ) ⊗ Iₙ]
            N = size(Σ, 1)
            return Σ * (1.0 + Λ[1]) 
        end

    function compute_L2(Σ::Real, Λ::AbstractMatrix{T}) where T <: Real
        return (1.0 + Λ[1]) * Σ * Λ[1] 
    end
    
    function compute_L2(Σ::AbstractMatrix{T1}, Λ::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}
        N = size(Σ, 1)
        return kron(I + Λ, Σ) * kron(I(N), Λ) * kron(vec(I(N)), I(N))
    end
    function compute_L3(Σ::Real, Λ::AbstractMatrix{T}) where T <: Real
        L3 = (1.0 + Λ[1]) * (1.0 + Λ[1]) * Λ[1] * Σ
        return L3
    end
    function compute_L3(Σ::AbstractMatrix{T1}, Λ::AbstractMatrix{T2}) where {T1 <: Real, T2 <: Real}
        N = size(Σ, 1)
        L3 = kron(I + Λ, I + Λ) * kron(I(N), kron(Λ, I(N))) * kron(vec(Σ), I(N^2))
        return L3
    end
    
    #=
    function compute_ν(L1::AbstractMatrix{T}, L2::AbstractMatrix{T}, L3::AbstractMatrix{T},
                       Λ::AbstractMatrix{T}, Σ::AbstractMatrix{T},
                       μ::AbstractVector{T})
        #Prop. 3.2
        N = size(Σ, 1)
        P = N + N^2
        ν = zeros(T, P^2)
        ν[1:N^2] .= vec(Σ)
        ν[(N^2 + 1):(N^2 + N^3)] .= L1 * μ
        ν[((N^2 + N^3) + 1):(N^2 + 2*N^3)] .= L2 * μ
        ν[((N^2 + 2*N^3) + 1):end] .= L3 * kron(μ, μ) .+
            kron(I(N^2), I + Λ) * vec(kron(Σ, Σ))

        return ν
    end
    
    function compute_Ψ(L1::AbstractMatrix{T}, L2::AbstractMatrix{T}, L3::AbstractMatrix{T},
                       Φ̃::AbstractMatrix{T}, N::Int)
        P = size(Φ̃, 1)
        Ψ = zeros(T, P^2, P)
        Ψ[(N^2 + 1):(N^2 + N^3), :] .= L1 * Φ̃[1:N, :]
        Ψ[((N^2 + N^3) + 1):(N^2 + 2*N^3), :] .= L2 * Φ̃[1:N, :]
        Ψ[((N^2 + 2*N^3) + 1):end, :] .= L3 * Φ̃[(N+1):end, :]
        return Ψ
    end
    function compute_μᵘ(lags::Int, μ::AbstractVector{T}, Φ::AbstractMatrix{T})
        N = length(μ)

        if μ[1:(N - lags)] == zeros(T, N - lags)
            return zeros(T, N)
        else
            if lags > 0
                tmp_μ = (I - Φ[1:(N - lags),:1:(N - lags)]) \ μ[1:(N - lags)]
                return vcat(tmp_μ, zeros(T, lags))
            else
                return (I - Φ)\μ
            end
        end
    end
    =#

    function compute_μᵘ(μ::Real, Φ::Real) 

        return μ / (1.0 - Φ) 
 
    end

    function compute_μᵘ(μ::AbstractVector{T}, Φ::AbstractMatrix{T}) where T <: Real

        return (I - Φ)\μ
    end

    #=
    function compute_μᵘ(μ::AbstractVector{T}, Φ::AbstractMatrix{T})
        N = length(μ)
        
        # Smooth approximation of the condition
        ε = sqrt(eps(T))  # Small positive number
        condition = sum(abs2, μ) / N
        
        # Smooth transition between zero and (I - Φ)\μ
        zero_result = zeros(T, N)
        nonzero_result = (I - Φ) \ μ
        
        # Smooth interpolation
        result = (condition / (condition + ε)) * nonzero_result +
                 (ε / (condition + ε)) * zero_result
        
        return result
    end
    =#

    function compute_Σᵘ(Φ::Real, Σ::Real)
        return Σ / (1.0 - Φ^2) 
    end
    function compute_Σᵘ(Φ::AbstractMatrix{T}, Σ::AbstractMatrix{T}) where T <: Real
        N = size(Σ, 1)
        return reshape((I - kron(Φ, Φ))\vec(Σ), (N, N))
    end

    #=
    function compute_Σᵘ(Φ::AbstractMatrix{T}, Σ::AbstractMatrix{T})
        N = size(Σ, 1)
        vec_Σ = vec(Σ)
        
        # Solve the system without explicitly forming kron(Φ, Φ)
        function apply_kron_phi(v)
            V = reshape(v, N, N)
            return vec(Φ * V * Φ')
        end
        
        # Create a LinearMap object with explicit type
        A = LinearMap{<:Real}(x -> x - apply_kron_phi(x), N^2, issymmetric=true, ismutating=false)
        
        # Use cg with explicit type
        vec_Σᵘ = cg(A, vec_Σ; maxiter=100N^2)
        
        Σᵘ = reshape(vec_Σᵘ, N, N)
        return make_symmetric(make_positive_definite(Σᵘ))
    end
    =#

    function compute_Σ̃condZ(Z::AbstractVector{T}, Σ::Real, μ::Real, Φ::Real) where T <: Real
        
        u = μ + Φ * Z[1]
        Σ̃_11 = Σ
        Σ̃_12 = 2.0 * Σ * u
        Σ̃_21 = 2.0 * Σ * u
        Σ̃_22 = 4.0 * u^2 * Σ + 2.0 * Σ^2

        return [Σ̃_11 Σ̃_12; Σ̃_21 Σ̃_22]

    end

    function compute_Σ̃condZ(Z::AbstractVector{T1} ,L1::AbstractMatrix{T2},
        L2::AbstractMatrix{T3}, L3::AbstractMatrix{T4}, Λ::AbstractMatrix{T5},
        Σ::AbstractMatrix{T6}, μ::AbstractVector{T7}, Φ̃::AbstractMatrix{T8}) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real, T8 <: Real}
        N = length(μ)
        μΦz = (μ + Φ̃[1:N,:] * Z)
        Σ̃_11 = Σ
        Σ̃_12 = reshape(L1 * μΦz, (N, N^2))
        Σ̃_21 = reshape(L2 * μΦz, (N^2, N))
        Σ̃_22 = reshape(L3 * (kron(μ, μ) + Φ̃[N+1:end,:] * Z) +
                       kron(I(N^2), I + Λ) * vec(kron(Σ, Σ)), (N^2, N^2))

        return [Σ̃_11 Σ̃_12; Σ̃_21 Σ̃_22]

    end

    #=
    function compute_Σ̃condZ(Z::AbstractVector{T}, L1::AbstractMatrix{T}, L2::AbstractMatrix{T}, L3::AbstractMatrix{T}, 
                        Λ::AbstractMatrix{T}, Σ::AbstractMatrix{T}, μ::AbstractVector{T}, Φ̃::AbstractMatrix{T})
        N = length(μ)
        
        # Compute Σ̃_11
        Σ̃_11 = Σ
        
        # Compute Σ̃_12
        temp = μ + Φ̃[1:N,:] * Z
        Σ̃_12 = reshape(L1 * temp, (N, N^2))
        
        # Compute Σ̃_21
        Σ̃_21 = reshape(L2 * temp, (N^2, N))
        
        # Compute Σ̃_22
        temp2 = kron(μ, μ) + Φ̃[N+1:end,:] * Z
        Σ̃_22_part1 = reshape(L3 * temp2, (N^2, N^2))
        
        function apply_kron_sigma(v)
            V = reshape(v, N, N)
            return vec(Σ * V * Σ)
        end
        
        Σ̃_22_part2 = reshape(apply_kron_sigma(vec(I + Λ)), (N^2, N^2))
        
        Σ̃_22 = Σ̃_22_part1 + Σ̃_22_part2
        
        # Combine all parts
        Σ̃ = [Σ̃_11 Σ̃_12; Σ̃_21 Σ̃_22]
        
        return make_symmetric(make_positive_definite(make_symmetric(Σ̃)))
    end
    =#

    function compute_Σ̃condX(X::AbstractVector{T}, params::QKParams{T,T2}) where {T <: Real, T2 <: Real}
        @unpack μ, Φ, Σ, Λ, N = params
        N = length(μ)
        Γ = compute_Γₜ₋₁_old(X, μ, Φ)
        Σ̃_11 = Σ
        Σ̃_12 = Σ * Γ' 
        Σ̃_21 = Γ * Σ
        Σ̃_22 = Γ * Σ * Γ' + (I + Λ) * kron(Σ, Σ)
        return [Σ̃_11 Σ̃_12; Σ̃_21 Σ̃_22]

    end
    function compute_Γₜ₋₁(Z::AbstractVector{T}, μ::AbstractVector{T},
        Φ::AbstractMatrix{T}) where T <: Real
        N = length(μ)
        tmp = μ + Φ * Z
        return kron(I(N), tmp) + kron(tmp, I(N)) 
    end

    #=
    function compute_Γₜ₋₁_mul(v::AbstractVector{T}, Z::AbstractVector{T}, μ::AbstractVector{T}, Φ::AbstractMatrix{T})
        N = length(μ)
        tmp = μ + Φ * Z
        V = reshape(v, N, N)
        return vec(tmp * V' + V * tmp')
    end
    
    function compute_Σ̃condX(X::AbstractVector{T}, params::QKParams{T,T2})
        μ = params.μ
        Φ = params.Φ
        Σ = params.Σ
        Λ = params.Λ
        N = length(μ)
    
        Σ̃_11 = Σ
    
        function Σ̃_12_mul(v)
            return Σ * compute_Γₜ₋₁_mul(v, X, μ, Φ)
        end
    
        function Σ̃_21_mul(v)
            return compute_Γₜ₋₁_mul(Σ * v, X, μ, Φ)
        end
    
        function apply_kron_sigma(v)
            V = reshape(v, N, N)
            return vec(Σ * V * Σ)
        end
    
        function Σ̃_22_mul(v)
            return compute_Γₜ₋₁_mul(Σ̃_21_mul(v), X, μ, Φ) + (I + Λ) * apply_kron_sigma(v)
        end
    
        # Return a lazy representation of the matrix
        return LinearMap{<:Real}(v -> [
            Σ̃_11 * v[1:N] + Σ̃_12_mul(v[N+1:end]);
            Σ̃_21_mul(v[1:N]) + Σ̃_22_mul(v[N+1:end])
        ], N + N^2, N + N^2)
    end
    =#
    function compute_Σ̃ᵘ(Z::AbstractVector{T}, Σ::Real, μ::Real, Φ::Real, Φ̃::AbstractMatrix{T}) where T <: Real

        P = size(Φ̃, 1)
        return reshape((I - kron(Φ̃, Φ̃)) \ 
            vec(compute_Σ̃condZ(Z, Σ, μ, Φ)), (P, P))
    end

    function compute_Σ̃ᵘ(μ̃ᵘ::AbstractVector{T1}, L1::AbstractMatrix{T2}, L2::AbstractMatrix{T3},
        L3::AbstractMatrix{T4}, Λ::AbstractMatrix{T5}, Σ::AbstractMatrix{T6},
        μ::AbstractVector{T7}, Φ̃::AbstractMatrix{T8}) where {T1 <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real, T7 <: Real, T8 <: Real}

        P = size(Φ̃, 1)
        return reshape((I - kron(Φ̃, Φ̃)) \ vec(compute_Σ̃condZ(μ̃ᵘ, L1, L2, L3, Λ, Σ, μ, Φ̃)),
                       (P, P))
    end

    #=
    function compute_Σ̃ᵘ(μ̃ᵘ::AbstractVector{T}, L1::AbstractMatrix{T}, L2::AbstractMatrix{T},
        L3::AbstractMatrix{T}, Λ::AbstractMatrix{T}, Σ::AbstractMatrix{T},
        μ::AbstractVector{T}, Φ̃::AbstractMatrix{T})
        P = size(Φ̃, 1)
        N = length(μ)

        # Compute Σ̃condZ lazily
        Σ̃condZ = compute_Σ̃condZ_lazy(μ̃ᵘ, L1, L2, L3, Λ, Σ, μ, Φ̃)

        # Define the linear operator (I - kron(Φ̃, Φ̃))
        function apply_I_minus_kron_Φ̃(v)
        V = reshape(v, P, P)
        return vec(V - Φ̃ * V * Φ̃')
        end

        # Solve the system using an iterative method
        Σ̃ᵘ_vec = cg(v -> apply_I_minus_kron_Φ̃(v), vec(Σ̃condZ(I)))

        # Reshape and apply make_symmetric and make_positive_definite
        Σ̃ᵘ = reshape(Σ̃ᵘ_vec, P, P)
        return make_symmetric(make_positive_definite(make_symmetric(Σ̃ᵘ)))
    end

    function compute_Σ̃condZ_lazy(Z::AbstractVector{T}, L1::AbstractMatrix{T}, L2::AbstractMatrix{T},
                  L3::AbstractMatrix{T}, Λ::AbstractMatrix{T}, Σ::AbstractMatrix{T},
                  μ::AbstractVector{T}, Φ̃::AbstractMatrix{T})
        N = length(μ)

        function Σ̃condZ(v)
            v1, v2 = v[1:N], v[N+1:end]

            # Compute Σ̃_11 * v1
            result1 = Σ * v1

            # Compute Σ̃_12 * v2
            temp = μ + Φ̃[1:N,:] * Z
            result2 = reshape(L1 * temp, (N, N^2)) * v2

            # Compute Σ̃_21 * v1
            result3 = reshape(L2 * temp, (N^2, N)) * v1

            # Compute Σ̃_22 * v2
            temp2 = kron(μ, μ) + Φ̃[N+1:end,:] * Z
            result4_part1 = reshape(L3 * temp2, (N^2, N^2)) * v2

            function apply_kron_sigma(x)
                X = reshape(x, N, N)
                return vec(Σ * X * Σ)
            end

            result4_part2 = apply_kron_sigma((I + Λ) * reshape(v2, N, N))
            result4 = result4_part1 + vec(result4_part2)

            return [result1 + result2; result3 + result4]
        end

        return Σ̃condZ
    end
    =#
    function vech(mat::AbstractMatrix{T}) where T <: Real
        n = size(mat, 1)
        if n != size(mat, 2)
            throw(ArgumentError("Input matrix must be square"))
        end
        
        if !issymmetric(mat)
            throw(ArgumentError("Input matrix must be symmetric"))
        end
        
        indices = [(i, j) for j in 1:n for i in j:n]
        
        return [mat[i, j] for (i, j) in indices]
    end
    #=
    function vech(mat::AbstractMatrix{T})
        n = size(mat, 1)
        if n != size(mat, 2)
            throw(ArgumentError("Input matrix must be square"))
        end
        
        # Avoid explicit symmetry check for AD-friendliness
        result = Vector{<:Real}(undef, (n * (n + 1)) ÷ 2)
        k = 1
        for j in 1:n
            for i in j:n
                result[k] = (mat[i, j] + mat[j, i]) / 2  # Ensure symmetry
                k += 1
            end
        end
        
        return result
    end
    =#
    function issymmetric(mat::AbstractMatrix{T}) where T <: Real
        return all(mat .≈ mat')
    end
    
    #=
    function compute_B̃_old(B::AbstractMatrix{T}, C::Vector{Matrix{T}})

        M, N = size(B)
        B̃ = zeros(T, M, N + N^2)
        B̃[:, 1:N] .= B
        for i in 1:M
            B̃[i, (N + 1):end] .= vec(C[i])
        end

        return B̃
    end
    =#
    function compute_B̃(B::Real, C::Real)
        return [B C]
    end
    function compute_B̃(B::AbstractMatrix{T}, C::Vector{<:AbstractMatrix{T}}) where T <: Real
        M, N = size(B)
       
        # Ensure all matrices in C have the correct size
        if any(size.(C) .!= Ref((N, N)))
            throw(DimensionMismatch("All matrices in C must be N x N"))
        end
       
        # Create the C part of B̃ using array comprehension
        C_part = [C[i][j, k] for i in 1:M, j in 1:N, k in 1:N]
       
        # Reshape C_part to be M x N^2
        C_part = reshape(C_part, M, N^2)
       
        # Combine B and C_part
        B̃ = hcat(B, C_part)
       
        return B̃
    end
    function compute_B̃_old(B::AbstractMatrix{T}, C::Vector{Matrix{T}}) where T <: Real
        M, N = size(B)
        
        # Ensure all matrices in C have the correct size
        if any(size.(C) .!= Ref((N, N)))
            throw(DimensionMismatch("All matrices in C must be N x N"))
        end
        
        # Create the C part of B̃ using array comprehension
        C_part = [C[i][j, k] for i in 1:M, j in 1:N, k in 1:N]
        
        # Reshape C_part to be M x N^2
        C_part = reshape(C_part, M, N^2)
        
        # Combine B and C_part
        B̃ = hcat(B, C_part)
        
        return B̃
    end
    #=
    function compute_B̃(B::AbstractMatrix{T}, C::AbstractVector{<:AbstractMatrix{AbstractFloat}})
        M, N = size(B)
        B̃ = Matrix{<:Real}(undef, M, N + N^2)
        
        @views begin
            B̃[:, 1:N] .= B
            for i in 1:M
                B̃[i, (N + 1):end] .= vec(C[i])
            end
        end
        
        return B̃
    end
    =#

    #Sparse Version
    #=
    function selection_matrix(n::Int, ::Type{<:Real}=Float64) where T
        p = (n * (n + 1)) ÷ 2
        row = Int[]
        col = Int[]
        val = T[]
        k = 1
        for j in 1:n
            for i in j:n
                idx = (i-1) * n + j
                push!(row, k)
                push!(col, idx)
                push!(val, 1)
                k += 1
            end
        end
        return sparse(row, col, val, p, n^2)
    end
    =#
    function selection_matrix(n::Int, ::Type{T}=Float64) where T <: Real
        p = (n * (n + 1)) ÷ 2
        
        result = [
            (k == ((i-1) * n + j) && i >= j) ? one(T) : zero(T)
            for k in 1:p, i in 1:n, j in 1:n
        ]
        
        return reshape(result, (p, n^2))
    end
    #=
    function selection_matrix(n::Int, ::Type{<:Real}=Float64) where T
        p = (n * (n + 1)) ÷ 2
        total_elements = p  # Number of non-zero elements
        
        row = Vector{Int}(undef, total_elements)
        col = Vector{Int}(undef, total_elements)
        val = ones(T, total_elements)
        
        k = 1
        for j in 1:n
            for i in j:n
                idx = (i-1) * n + j
                row[k] = k
                col[k] = idx
                k += 1
            end
        end
        
        return sparse(row, col, val, p, n^2)
    end
    
    function duplication_matrix(n::Int, ::Type{<:Real}=Float64)
        p = (n * (n + 1)) ÷ 2
        row = Int[]
        col = Int[]
        val = T[]
        k = 1
        for j in 1:n
            for i in j:n
                idx1 = (i-1) * n + j
                idx2 = (j-1) * n + i
                push!(row, idx1)
                push!(col, k)
                push!(val, one(T))
                if i != j
                    push!(row, idx2)
                    push!(col, k)
                    push!(val, one(T))
                end
                k += 1
            end
        end
        return sparse(row, col, val, n^2, p)
    end
    =#
    function duplication_matrix(n::Int, ::Type{T}=Float64) where T <: Real
        p = (n * (n + 1)) ÷ 2
        
        result = [
            ((i-1) * n + j <= k && (j-1) * n + i <= k) ? one(T) : zero(T)
            for i in 1:n, j in 1:n, k in 1:p
        ]
        
        return reshape(result, (n^2, p))
    end
    #=
    function duplication_matrix(n::Int, ::Type{<:Real}=Float64)
        p = (n * (n + 1)) ÷ 2
        total_elements = n^2  # Upper bound on number of non-zero elements
        
        row = Vector{Int}(undef, total_elements)
        col = Vector{Int}(undef, total_elements)
        val = Vector{<:Real}(undef, total_elements)
        
        count = 0
        k = 1
        for j in 1:n
            for i in j:n
                idx1 = (i-1) * n + j
                idx2 = (j-1) * n + i
                
                count += 1
                row[count] = idx1
                col[count] = k
                val[count] = one(T)
                
                if i != j
                    count += 1
                    row[count] = idx2
                    col[count] = k
                    val[count] = one(T)
                end
                
                k += 1
            end
        end
        
        # Trim excess allocated space
        resize!(row, count)
        resize!(col, count)
        resize!(val, count)
        
        return sparse(row, col, val, n^2, p)
    end
    =#
    #function spectral_radius(A::AbstractMatrix{T})
    #    @assert size(A, 1) == size(A, 2) "Matrix must be square"
    #    eigenvalues = eigen(A).values
    #    return maximum(abs.(eigenvalues))
    #end
    function spectral_radius(A::AbstractMatrix{T}, num_iterations::Int=100) where T <: Real
        v = normalize(randn(size(A,1))) 
        for _ in 1:num_iterations
            v = normalize(A * v)
        end
        return sqrt((A * v)' * (A * v))
    end
    #=
    function spectral_radius(A::AbstractMatrix{T})
        n = size(A, 1)
        if n != size(A, 2)
            throw(DimensionMismatch("Matrix must be square"))
        end
        
        # Use eigvals instead of eigen for efficiency
        eigenvalues = eigvals(A)
        
        # Avoid broadcasting for better AD compatibility
        max_abs_eigval = zero(T)
        for λ in eigenvalues
            abs_λ = abs(λ)
            if abs_λ > max_abs_eigval
                max_abs_eigval = abs_λ
            end
        end
        
        return max_abs_eigval
    end

    =#
    #H = selection_matrix(N)
    #H̃ = (Iₙ 0; 0 H)
    function compute_H̃_old(N::Int, H::AbstractMatrix{T}) where T <: Real
        H̃ = zeros(T, Int((N * (N + 3)) / 2), N * (N + 1))
        H̃[1:N, 1:N] = Matrix{<:Real}(I, N, N)
        H̃[(N+1):end, (N+1):end] = H
    
        return H̃
    end

    function compute_H̃(N::Int, H::AbstractMatrix{T}) where T <: Real
        n_rows = Int((N * (N + 3)) / 2)
        n_cols = N * (N + 1)
        # Create the identity matrix part
        top_left = Matrix{T}(I, N, N)
        
        # Create the H part
        bottom_right = H
        
        # Create the zero matrices
        top_right = zeros(T, N, n_cols - N)
        bottom_left = zeros(T, n_rows - N, N)
        
        # Combine all parts
        H̃ = [
            top_left    top_right;
            bottom_left bottom_right
        ]
        
        return H̃
    end

    #G = duplication_matrix(N)
    #G̃ = (Iₙ 0; 0 G)

    function compute_G̃_old(N::Int, G::AbstractMatrix{T}) where T <: Real
        G̃ = zeros(T, N * (N + 1), Int((N * (N + 3)) / 2))
        G̃[1:N, 1:N] = Matrix{<:Real}(I, N, N)
        G̃[(N+1):end, (N+1):end] = G
    
        return G̃
    end

    function compute_G̃(N::Int, G::AbstractMatrix{T}) where T <: Real
        n_rows = N * (N + 1)
        n_cols = Int((N * (N + 3)) / 2)
        # Create the identity matrix part
        top_left = Matrix{T}(I, N, N)
        
        # Create the G part
        bottom_right = G
        
        # Create the zero matrices
        top_right = zeros(T, N, n_cols - N)
        bottom_left = zeros(T, n_rows - N, N)
        
        # Combine all parts
        G̃ = [
            top_left    top_right;
            bottom_left bottom_right
        ]
        
        return G̃
    end
    #=
    function compute_H̃(N::Int, H::AbstractSparseMatrix{<:Real})
        P = N * (N + 1) ÷ 2
        rows = Vector{Int}()
        cols = Vector{Int}()
        vals = Vector{<:Real}()
    
        # Identity part
        append!(rows, 1:N)
        append!(cols, 1:N)
        append!(vals, ones(T, N))
    
        # H part
        H_rows, H_cols, H_vals = findnz(H)
        append!(rows, N .+ H_rows)
        append!(cols, N .+ H_cols)
        append!(vals, H_vals)
    
        return sparse(rows, cols, vals, N + P, N * (N + 1))
    end
    
    function compute_G̃(N::Int, G::AbstractSparseMatrix{<:Real})
        P = N * (N + 1) ÷ 2
        rows = Vector{Int}()
        cols = Vector{Int}()
        vals = Vector{<:Real}()
    
        # Identity part
        append!(rows, 1:N)
        append!(cols, 1:N)
        append!(vals, ones(T, N))
    
        # G part
        G_rows, G_cols, G_vals = findnz(G)
        append!(rows, N .+ G_rows)
        append!(cols, N .+ G_cols)
        append!(vals, G_vals)
    
        return sparse(rows, cols, vals, N * (N + 1), N + P)
    end
    =#
    #Data Structures
    @with_kw struct QKData{Tm <: Real, N}
        Y::Array{Tm, N}

        #Precomputed Parameters
        M::Int
        T̄::Int 
    end

    function QKData(Y::Array{T, N}) where {T <: Real, N}
        if N == 1  # Y is a vector
            M = 1
            Tp1 = length(Y)
        else  # Y is a matrix (or higher-dimensional array, though that's not expected)
            M, Tp1 = size(Y)
        end

        return QKData{T, N}(Y = Y, M = M, T̄ = Tp1 - 1) 
    end
    
    function correct_Zₜₜ!(Zₜₜ::AbstractMatrix{T1}, params::QKParams{T,T2},
                          t::Int) where {T1 <: Real, T <: Real, T2 <: Real}

        @unpack N = params
        Ztt = Zₜₜ[:, t + 1]
        Zttcp = Ztt[1:N] * Ztt[1:N]'

        implied_ZZp = reshape(Ztt[N+1:end], (N, N))
        implied_cov = implied_ZZp - Zttcp

        eig_vals, eig_vecs = eigen(implied_cov)

        #replace negative eigenvalues with 0
        eig_vals[eig_vals .< 0.0] .= 0.0
        Zₜₜ[N+1:end, t + 1] = vec(eig_vecs * Diagonal(eig_vals) * eig_vecs' + Zttcp)


    end

    function correct_Zₜₜ(Zₜₜ::AbstractVector{T1}, params::QKParams{T,T2}, t::Int) where {T1 <: Real, T <: Real, T2 <: Real}
        @unpack N = params
        Ztt = Zₜₜ
        Zttcp = Ztt[1:N] * Ztt[1:N]'
        implied_ZZp = reshape(Ztt[N+1:end], (N, N))
        implied_cov = implied_ZZp - Zttcp
        
        # Smooth approximation of max(0, x)
        #smooth_max(x, α=T(0.01)) = (x + √(x^2 + α^2)) / T(2)
        
        #eig_vals, eig_vecs = eigen(Hermitian(implied_cov))
        #eig_vals_corrected = smooth_max.(eig_vals)
        #corrected_ZZp = vec(eig_vecs * Diagonal(eig_vals_corrected) * eig_vecs' + Zttcp)

        ε = sqrt(eps(eltype(Ztt)))
        corrected_cov = implied_cov + ε * I

        corrected_ZZp = vec(corrected_cov + Zttcp)
        
        return vcat(Ztt[1:N], corrected_ZZp)
    end
    #Quadratic Kalman Filter
    function qkf_filter!(data::QKData{T1, 1}, params::QKParams{T,T2}) where {T1 <: Real, T <: Real, T2 <: Real}

        @unpack T̄, Y, M = data
        @unpack N, μ̃ᵘ, Σ̃ᵘ, P = params

        # Predfine Matrix
        Zₜₜ =  zeros(T, P, T̄ + 1)
        Zₜₜ₋₁ = zeros(T, P, T̄)
        Pₜₜ = zeros(T, P, P, T̄ + 1)
        Pₜₜ₋₁ = zeros(T, P, P, T̄)
        Σₜₜ₋₁ = zeros(T, P, P, T̄)
        #vecΣₜₜ₋₁ = zeros(T, P^2, T̄)
        Kₜ = zeros(T, P, M, T̄)
        tmpP = zeros(T, P, P)
        tmpB = zeros(T, M, P)
        Yₜₜ₋₁ = zeros(T, T̄)
        Mₜₜ₋₁ = zeros(T, M, M, T̄)
        tmpPB = zeros(T, P, M)
        llₜ = zeros(Float64, T̄)
        tmpϵ = zeros(T, M)
        tmpKM = zeros(T, P, M)
        tmpKMK = zeros(T, P, P)

        Yₜ = Vector{T}(undef, T̄)
        copyto!(Yₜ, 1, Y, 2, T̄)

        #Initalize: Z₀₀ = μ̃ᵘ, P₀₀ = Σ̃ᵘ
        Zₜₜ[:, 1] .= μ̃ᵘ
        Pₜₜ[:, :, 1] .= Σ̃ᵘ
        
        # Loop over time
        for t in 1:T̄

            # State Prediction: Zₜₜ₋₁ = μ̃ + Φ̃Zₜ₋₁ₜ₋₁, Pₜₜ₋₁ = Φ̃Pₜ₋₁ₜ₋₁Φ̃' + Σ̃(Zₜ₋₁ₜ₋₁)
            predict_Zₜₜ₋₁!(Zₜₜ, Zₜₜ₋₁, params, t)
            predict_Pₜₜ₋₁!(Pₜₜ, Pₜₜ₋₁, Σₜₜ₋₁, Zₜₜ, tmpP, params, t)

            # Observation Prediction: Yₜₜ₋₁ = A + B̃Zₜₜ₋₁, Mₜₜ₋₁ = B̃Pₜₜ₋₁B̃' + V
            predict_Yₜₜ₋₁!(Yₜₜ₋₁, Zₜₜ₋₁, Y, params, t)
            predict_Mₜₜ₋₁!(Mₜₜ₋₁, Pₜₜ₋₁, Zₜₜ₋₁, tmpB, params, t)

            # Kalman Gain: Kₜ = Pₜₜ₋₁B̃′/Mₜₜ₋₁
            compute_Kₜ!(Kₜ, Pₜₜ₋₁, Mₜₜ₋₁, tmpPB, params, t)
            println(Kₜ[:,:,t])

            # Update States: Zₜₜ = Zₜₜ₋₁ + Kₜ(Yₜ - Yₜₜ₋₁); Pₜₜ = Pₜₜ₋₁ - KₜMₜₜ₋₁Kₜ'
            update_Zₜₜ!(Zₜₜ, Kₜ, Yₜ, Yₜₜ₋₁, Zₜₜ₋₁, tmpϵ, params, t)
            update_Pₜₜ!(Pₜₜ, Kₜ, Mₜₜ₋₁, Pₜₜ₋₁, Zₜₜ₋₁, tmpKM, tmpKMK, params, t)

            #Correct for update
            correct_Zₜₜ!(Zₜₜ, params, t)

            #Compute Log Likelihood
            compute_loglik!(llₜ, Yₜ, Yₜₜ₋₁, Mₜₜ₋₁, t)
        end

        return (llₜ = llₜ, Zₜₜ = Zₜₜ, Pₜₜ = Pₜₜ,  Yₜₜ₋₁ = Yₜₜ₋₁, Mₜₜ₋₁ = Mₜₜ₋₁, Kₜ = Kₜ, Zₜₜ₋₁ = Zₜₜ₋₁,
                Pₜₜ₋₁ = Pₜₜ₋₁, Σₜₜ₋₁ = Σₜₜ₋₁)

    end

    #=
    function qkf_filter(data::QKData{T, 1}, params::QKParams{T,T2})
        @unpack T̄, Y, M = data
        @unpack N, μ̃ᵘ, Σ̃ᵘ, P = params
    
        Zₜₜ = zeros(T, P, T̄ + 1)
        Zₜₜ₋₁ = zeros(T, P, T̄)
        Pₜₜ = zeros(T, P, P, T̄ + 1)
        Pₜₜ₋₁ = zeros(T, P, P, T̄)
        Kₜ = zeros(T, P, M, T̄)
        Yₜₜ₋₁ = zeros(T, T̄)
        Mₜₜ₋₁ = zeros(T, M, M, T̄)
        llₜ = zeros(Float64, T̄)
        Yₜ = Y[2:end]
    
        # Initialize
        Zₜₜ[:, 1] = μ̃ᵘ
        Pₜₜ[:, :, 1] = Σ̃ᵘ
    
        for t in 1:T̄
            # State Prediction
            Zₜₜ₋₁[:, t] = predict_Zₜₜ₋₁(Zₜₜ[:, t], params)
            Pₜₜ₋₁[:, :, t] = predict_Pₜₜ₋₁(Pₜₜ[:, :, t], Zₜₜ[:, t], params, t)
    
            # Observation Prediction
            Yₜₜ₋₁[t] = predict_Yₜₜ₋₁(Zₜₜ₋₁[:, t], Y, params, t)
            Mₜₜ₋₁[:, :, t] = predict_Mₜₜ₋₁(Pₜₜ₋₁[:, :, t], Zₜₜ₋₁[:, t], params, t)
    
            # Kalman Gain
            Kₜ[:, :, t] = compute_Kₜ(Pₜₜ₋₁[:, :, t], Mₜₜ₋₁[:, :, t], params, t)
    
            # Update States
            Zₜₜ[:, t+1] = update_Zₜₜ(Kₜ[:, :, t], Yₜ[t], Yₜₜ₋₁[t], Zₜₜ₋₁[:, t], params, t)
            Pₜₜ[:, :, t+1] = update_Pₜₜ(Kₜ[:, :, t], Mₜₜ₋₁[:, :, t], Pₜₜ₋₁[:, :, t], Zₜₜ₋₁[:, t], params, t)
    
            # Correct for update
            Zₜₜ[:, t+1] = correct_Zₜₜ(Zₜₜ[:, t+1], params, t)
    
            # Compute Log Likelihood
            llₜ[t] = compute_loglik(Yₜ[t], Yₜₜ₋₁[t], Mₜₜ₋₁[:, :, t])
        end
    
        return (llₜ = llₜ, Zₜₜ = Zₜₜ, Pₜₜ = Pₜₜ, Yₜₜ₋₁ = Yₜₜ₋₁, Mₜₜ₋₁ = Mₜₜ₋₁, Kₜ = Kₜ, Zₜₜ₋₁ = Zₜₜ₋₁, Pₜₜ₋₁ = Pₜₜ₋₁)
    end
    =#
    function qkf_filter(data::QKData{T1, 1}, params::QKParams{T,T2})  where {T1 <: Real, T <: Real, T2 <: Real}
        @unpack T̄, Y, M = data
        @unpack N, μ̃ᵘ, Σ̃ᵘ, P = params
        Y_concrete = Vector{T1}(vec(Y))  # Convert to vector if it's not already
        Yₜ = @view Y_concrete[2:end]
    
        Zₜₜ = zeros(T, P, T̄ + 1)
        Pₜₜ = zeros(T, P, P, T̄ + 1)
        Zₜₜ₋₁ = zeros(T, P, T̄)
        Pₜₜ₋₁ = zeros(T, P, P, T̄)
        Kₜ = zeros(T, P, M, T̄)
        Yₜₜ₋₁ = zeros(T, T̄)
        Mₜₜ₋₁ = zeros(T, M, M, T̄)
        llₜ = zeros(T, T̄)
    
        # Initialize
        Zₜₜ[:, 1] = μ̃ᵘ
        Pₜₜ[:, :, 1] = Σ̃ᵘ
    
        for t in 1:T̄
            Zₜₜ₋₁[:, t] = predict_Zₜₜ₋₁(Zₜₜ[:, t], params)
            Pₜₜ₋₁[:, :, t] = predict_Pₜₜ₋₁(Pₜₜ[:, :, t], Zₜₜ[:, t], params, t)
    
            Yₜₜ₋₁[t] = predict_Yₜₜ₋₁(Zₜₜ₋₁[:, t], Y, params, t)
            Mₜₜ₋₁[:, :, t] = predict_Mₜₜ₋₁(Pₜₜ₋₁[:, :, t], Zₜₜ₋₁[:, t], params, t)
    
            Kₜ[:, :, t] = compute_Kₜ(Pₜₜ₋₁[:, :, t], Mₜₜ₋₁[:, :, t], params, t)
    
            Zₜₜ[:, t + 1] = update_Zₜₜ(Kₜ[:, :, t], Yₜ[t], Yₜₜ₋₁[t], Zₜₜ₋₁[:, t], params, t)
            Pₜₜ[:, :, t + 1] = update_Pₜₜ(Kₜ[:, :, t], Mₜₜ₋₁[:, :, t], Pₜₜ₋₁[:, :, t], Zₜₜ₋₁[:, t], params, t)
    
            Zₜₜ[:, t + 1] = correct_Zₜₜ(Zₜₜ[:, t + 1], params, t)
    
            llₜ[t] = compute_loglik(Yₜ[t], Yₜₜ₋₁[t], Mₜₜ₋₁[:, :, t])
        end
    
        return (llₜ = copy(llₜ), Zₜₜ = copy(Zₜₜ), Pₜₜ = copy(Pₜₜ), Yₜₜ₋₁ = copy(Yₜₜ₋₁),
            Mₜₜ₋₁ = copy(Mₜₜ₋₁), Kₜ = copy(Kₜ), Zₜₜ₋₁ = copy(Zₜₜ₋₁), Pₜₜ₋₁ = copy(Pₜₜ₋₁))
    end

    function qkf_filter!(data::QKData{T, 2}, params::QKParams{T,T2}) where {T <: Real, T2 <: Real}

        @unpack T̄, Y, M = data
        @unpack N, μ̃ᵘ, Σ̃ᵘ, P = params

        # Predfine Matrix
        Zₜₜ = Matrix{<:Real}(undef, P, T̄ + 1)
        Pₜₜ = Array{T, 3}(undef, P, P, T̄ + 1)
        Zₜₜ₋₁ = Matrix{<:Real}(undef, P, T̄)
        Pₜₜ₋₁ = Array{T, 3}(undef, P, P, T̄)
        Kₜ = Array{T, 3}(undef, P, M, T̄)
        Yₜₜ₋₁ = Vector{<:Real}(undef, T̄)
        Mₜₜ₋₁ = Array{T, 3}(undef, M, M, T̄)
        llₜ = Vector{<:Real}(undef, T̄)

        #Initalize: Z₀₀ = μ̃ᵘ, P₀₀ = Σ̃ᵘ
        Zₜₜ[:, 1] = μ̃ᵘ
        Pₜₜ[:, :, 1] = Σ̃ᵘ
        
        # Loop over time
        for t in 1:T̄
            Zₜₜ₋₁[:, t] = predict_Zₜₜ₋₁(Zₜₜ[:, t], params)
            Pₜₜ₋₁[:, :, t] = predict_Pₜₜ₋₁(Pₜₜ[:, :, t], Zₜₜ[:, t], params, t)
    
            Yₜₜ₋₁[t] = predict_Yₜₜ₋₁(Zₜₜ₋₁[:, t], Y, params, t)
            Mₜₜ₋₁[:, :, t] = predict_Mₜₜ₋₁(Pₜₜ₋₁[:, :, t], Zₜₜ₋₁[:, t], params, t)
    
            Kₜ[:, :, t] = compute_Kₜ(Pₜₜ₋₁[:, :, t], Mₜₜ₋₁[:, :, t], params, t)
    
            Zₜₜ[:, t + 1] = update_Zₜₜ(Kₜ[:, :, t], Yₜ[t], Yₜₜ₋₁[t], Zₜₜ₋₁[:, t], params, t)
            Pₜₜ[:, :, t + 1] = update_Pₜₜ(Kₜ[:, :, t], Mₜₜ₋₁[:, :, t], Pₜₜ₋₁[:, :, t], Zₜₜ₋₁[:, t], params, t)
    
            Zₜₜ[:, t + 1] = correct_Zₜₜ(Zₜₜ[:, t + 1], params, t)
    
            llₜ[t] = compute_loglik(Yₜ[t], Yₜₜ₋₁[t], Mₜₜ₋₁[:, :, t])
        end

        return (llₜ = llₜ, Zₜₜ = Zₜₜ, Pₜₜ = Pₜₜ,  Yₜₜ₋₁ = Yₜₜ₋₁, Mₜₜ₋₁ = Mₜₜ₋₁, Kₜ = Kₜ, Zₜₜ₋₁ = Zₜₜ₋₁,
                Pₜₜ₋₁ = Pₜₜ₋₁, Σₜₜ₋₁ = Σₜₜ₋₁)

    end
    #=
    function qkf_filter(data::QKData{T, 2}, params::QKParams{T,T2})
        @unpack T̄, Y, M = data
        @unpack N, μ̃ᵘ, Σ̃ᵘ, P = params
    
        Yₜ = Y[:, 2:end]
    
        # Initialize: Z₀₀ = μ̃ᵘ, P₀₀ = Σ̃ᵘ
        Zₜₜ_init = hcat(μ̃ᵘ, zeros(T, P, T̄))
        Pₜₜ_init = cat(Σ̃ᵘ, zeros(T, P, P, T̄), dims=3)
    
        function step(t, state)
            (Zₜₜ, Pₜₜ, Zₜₜ₋₁, Pₜₜ₋₁, Σₜₜ₋₁, Kₜ, Yₜₜ₋₁, Mₜₜ₋₁, llₜ) = state
    
            # State Prediction
            Zₜₜ₋₁_t = predict_Zₜₜ₋₁(Zₜₜ[:, t], params)
            Pₜₜ₋₁_t = predict_Pₜₜ₋₁(Pₜₜ[:, :, t], Zₜₜ[:, t], params, t)
            
            # Compute Σₜₜ₋₁
            Σₜₜ₋₁_t = compute_Σₜₜ₋₁(Zₜₜ, params, t)
    
            # Observation Prediction
            Yₜₜ₋₁_t = predict_Yₜₜ₋₁(Zₜₜ₋₁_t, Y, params, t)
            Mₜₜ₋₁_t = predict_Mₜₜ₋₁(Pₜₜ₋₁_t, Zₜₜ₋₁_t, params, t)
    
            # Kalman Gain
            Kₜ_t = compute_Kₜ(Pₜₜ₋₁_t, Mₜₜ₋₁_t, params, t)
    
            # Update States
            Zₜₜ_next = update_Zₜₜ(Kₜ_t, Yₜ[:, t], Yₜₜ₋₁_t, Zₜₜ₋₁_t, params, t)
            Pₜₜ_next = update_Pₜₜ(Kₜ_t, Mₜₜ₋₁_t, Pₜₜ₋₁_t, Zₜₜ₋₁_t, params, t)
    
            # Correct for update
            Zₜₜ_next = correct_Zₜₜ(Zₜₜ_next, params, t)
    
            # Compute Log Likelihood
            llₜ_t = compute_loglik(Yₜ[:, t], Yₜₜ₋₁_t, Mₜₜ₋₁_t, t)
    
            Zₜₜ_new = hcat(Zₜₜ[:, 1:t], Zₜₜ_next)
            Pₜₜ_new = cat(Pₜₜ[:, :, 1:t], Pₜₜ_next, dims=3)
            Zₜₜ₋₁_new = hcat(Zₜₜ₋₁, Zₜₜ₋₁_t)
            Pₜₜ₋₁_new = cat(Pₜₜ₋₁, Pₜₜ₋₁_t, dims=3)
            Σₜₜ₋₁_new = cat(Σₜₜ₋₁, Σₜₜ₋₁_t, dims=3)
            Kₜ_new = cat(Kₜ, Kₜ_t, dims=3)
            Yₜₜ₋₁_new = hcat(Yₜₜ₋₁, Yₜₜ₋₁_t)
            Mₜₜ₋₁_new = cat(Mₜₜ₋₁, Mₜₜ₋₁_t, dims=3)
            llₜ_new = vcat(llₜ, llₜ_t)
    
            (Zₜₜ_new, Pₜₜ_new, Zₜₜ₋₁_new, Pₜₜ₋₁_new, Σₜₜ₋₁_new, Kₜ_new, Yₜₜ₋₁_new, Mₜₜ₋₁_new, llₜ_new)
        end
    
        init_state = (Zₜₜ_init, Pₜₜ_init, zeros(T, P, 0), zeros(T, P, P, 0), zeros(T, P, P, 0), 
                      zeros(T, P, M, 0), zeros(T, M, 0), zeros(T, M, M, 0), Float64[])
    
        final_state = foldl(step, 1:T̄, init=init_state)
    
        (Zₜₜ, Pₜₜ, Zₜₜ₋₁, Pₜₜ₋₁, Σₜₜ₋₁, Kₜ, Yₜₜ₋₁, Mₜₜ₋₁, llₜ) = final_state
    
        return (llₜ = llₜ, Zₜₜ = Zₜₜ, Pₜₜ = Pₜₜ, Yₜₜ₋₁ = Yₜₜ₋₁, Mₜₜ₋₁ = Mₜₜ₋₁, Kₜ = Kₜ, Zₜₜ₋₁ = Zₜₜ₋₁,
                Pₜₜ₋₁ = Pₜₜ₋₁, Σₜₜ₋₁ = Σₜₜ₋₁)
    end
    =#

    function qkf_filter(data::QKData{T1, N}, params::QKParams{T,T2}) where {T1 <:Real, T <: Real, T2 <:Real, N}
        @unpack T̄, Y, M = data
        @assert length(Y) == T̄ + 1 "Y should have T̄ + 1 observations"
    
        # Initialize (use Y[1] here if needed)
        Zₜₜ = [params.μ̃ᵘ]
        Pₜₜ = [params.Σ̃ᵘ]
        llₜ = T[]
    
        for t in 1:T̄
            # Prediction step
            Zₜₜ₋₁ = predict_Zₜₜ₋₁(last(Zₜₜ), params)
            Pₜₜ₋₁ = predict_Pₜₜ₋₁(last(Pₜₜ), last(Zₜₜ), params, t)
    
            # Observation prediction
            Yₜₜ₋₁ = predict_Yₜₜ₋₁(Zₜₜ₋₁, Y, params, t)
            Mₜₜ₋₁ = predict_Mₜₜ₋₁(Pₜₜ₋₁, Zₜₜ₋₁, params, t)
    
            # Kalman gain
            Kₜ = compute_Kₜ(Pₜₜ₋₁, Mₜₜ₋₁, params, t)
    
            # Update step
            Zₜₜ_new = update_Zₜₜ(Kₜ, Y[t+1], Yₜₜ₋₁, Zₜₜ₋₁, params, t)
            Pₜₜ_new = update_Pₜₜ(Kₜ, Mₜₜ₋₁, Pₜₜ₋₁, Zₜₜ₋₁, params, t)
    
            # Correct for update
            Zₜₜ_corrected = correct_Zₜₜ(Zₜₜ_new[:,1], params, t)
    
            # Compute log-likelihood
            ll = compute_loglik(Y[t+1], Yₜₜ₋₁, Mₜₜ₋₁)
    
            push!(Zₜₜ, Zₜₜ_corrected)
            push!(Pₜₜ, Pₜₜ_new)
            push!(llₜ, ll)
    
           # println("Step $t: Zₜₜ₋₁ = $Zₜₜ₋₁, Yₜₜ₋₁ = $Yₜₜ₋₁, ll = $ll")
        end
    
        return (llₜ = llₜ, Zₜₜ = hcat(Zₜₜ...), Pₜₜ = cat(Pₜₜ..., dims=3))
    end
    #=
    function qkf_filter(data::QKData{T, 2}, params::QKParams{T,T2})
        @unpack T̄, Y, M = data
        @unpack N, μ̃ᵘ, Σ̃ᵘ, P = params
    
        Zₜₜ = zeros(T, P, T̄ + 1)
        Zₜₜ₋₁ = zeros(T, P, T̄)
        Pₜₜ = zeros(T, P, P, T̄ + 1)
        Pₜₜ₋₁ = zeros(T, P, P, T̄)
        Σₜₜ₋₁ = zeros(T, P, P, T̄)
        Kₜ = zeros(T, P, M, T̄)
        Yₜₜ₋₁ = zeros(T, M, T̄)
        Mₜₜ₋₁ = zeros(T, M, M, T̄)
        Yₜ = Y[:, 2:end]
        llₜ = zeros(Float64, T̄)
    
        # Initialize: Z₀₀ = μ̃ᵘ, P₀₀ = Σ̃ᵘ
        Zₜₜ[:, 1] = μ̃ᵘ
        Pₜₜ[:, :, 1] = Σ̃ᵘ
    
        for t in 1:T̄
            # State Prediction
            Zₜₜ₋₁[:, t] = predict_Zₜₜ₋₁(Zₜₜ[:, t], params)
            Pₜₜ₋₁[:, :, t] = predict_Pₜₜ₋₁(Pₜₜ[:, :, t], Zₜₜ[:, t], params, t)
            
            # Compute Σₜₜ₋₁
            Σₜₜ₋₁[:, :, t] = compute_Σₜₜ₋₁(Zₜₜ, params, t)
    
            # Observation Prediction
            Yₜₜ₋₁[:, t] = predict_Yₜₜ₋₁(Zₜₜ₋₁[:, t], Y, params, t)
            Mₜₜ₋₁[:, :, t] = predict_Mₜₜ₋₁(Pₜₜ₋₁[:, :, t], Zₜₜ₋₁[:, t], params, t)
    
            # Kalman Gain
            Kₜ[:, :, t] = compute_Kₜ(Pₜₜ₋₁[:, :, t], Mₜₜ₋₁[:, :, t], params, t)
    
            # Update States
            Zₜₜ[:, t+1] = update_Zₜₜ(Kₜ[:, :, t], Yₜ[:, t], Yₜₜ₋₁[:, t], Zₜₜ₋₁[:, t], params, t)
            Pₜₜ[:, :, t+1] = update_Pₜₜ(Kₜ[:, :, t], Mₜₜ₋₁[:, :, t], Pₜₜ₋₁[:, :, t], Zₜₜ₋₁[:, t], params, t)
    
            # Correct for update
            Zₜₜ[:, t+1] = correct_Zₜₜ(Zₜₜ[:, t+1], params, t)
    
            # Compute Log Likelihood
            llₜ[t] = compute_loglik(Yₜ[:, t], Yₜₜ₋₁[:, t], Mₜₜ₋₁[:, :, t], t)
        end
    
        return (llₜ = llₜ, Zₜₜ = Zₜₜ, Pₜₜ = Pₜₜ, Yₜₜ₋₁ = Yₜₜ₋₁, Mₜₜ₋₁ = Mₜₜ₋₁, Kₜ = Kₜ, Zₜₜ₋₁ = Zₜₜ₋₁, 
                Pₜₜ₋₁ = Pₜₜ₋₁, Σₜₜ₋₁ = Σₜₜ₋₁)
    end
    =#

    function log_pdf_normal(μ::Real, σ2::Real, x::Real)
        return -0.5 * (log(2.0 * π) + log(σ2) + (x - μ)^2 / σ2)
    end

    function compute_loglik!(llₜ::AbstractVector{T}, Yₜ::AbstractVector{T},
        Yₜₜ₋₁::AbstractVector{T}, Mₜₜ₋₁::AbstractArray{<:Real}, t::Int) where {T <: Real}

        σ² = Mₜₜ₋₁[1,1,t]
        #if σ² < 0.0
        #    @warn "Negative variance at t = $t"
        #end
        μ = Yₜₜ₋₁[t]
        x = Yₜ[t]
        
        llₜ[t] = log_pdf_normal(μ, σ², x)

    end

    function compute_loglik(Yₜ::Real, Yₜₜ₋₁::Real,
        Mₜₜ₋₁::AbstractMatrix{T}) where T <: Real

        σ² = Mₜₜ₋₁[1,1]
        μ = Yₜₜ₋₁
        x = Yₜ
        if σ² < 0.0
            tmp = 0.0
        end

        return log_pdf_normal(μ, σ², x)

    end
    function logpdf_mvn(Yₜₜ₋₁::AbstractMatrix{T}, Mₜₜ₋₁::AbstractArray{<:Real},
        Yₜ::AbstractMatrix{T}, t::Int) where T <: Real

        μ = @view Yₜₜ₋₁[:, t]
        Σ = @view Mₜₜ₋₁[:, :, t]
        x = @view Yₜ[:, t]
        
        k = length(μ)
        
        # Cholesky decomposition
        C = cholesky(Symmetric(Σ))
        
        # Calculate (x - μ)
        diff = x .- μ
        
        # Solve the system (Σ^-1 * diff) without explicit inversion
        solved = C \ diff
        
        # Calculate log(det(Σ)) using the Cholesky factor
        log_det_Σ = 2 * sum(log, diag(C.U))
        
        # Precompute constant term
        const_term = -0.5 * k * (log(2π) + 1)
        
        # Calculate the log-pdf
        logpdf = const_term - 0.5 * (log_det_Σ + dot(diff, solved))
        
        return logpdf
    end

    function compute_loglik!(llₜ::AbstractVector{T}, Yₜ::AbstractMatrix{T},
        Yₜₜ₋₁::AbstractMatrix{T}, Mₜₜ₋₁::AbstractArray{<:Real}, t::Int) where T <: Real

        llₜ[t] = logpdf_mvn(Yₜₜ₋₁, Mₜₜ₋₁, Yₜ, t)
        

    end

    function compute_loglik(Yₜ::AbstractMatrix{T}, Yₜₜ₋₁::AbstractMatrix{T}, 
        Mₜₜ₋₁::AbstractArray{<:Real}, t::Int) where T <: Real

        return logpdf_mvn(Yₜₜ₋₁, Mₜₜ₋₁, Yₜ, t)
    end

    #function qkf_smoother(params::QKParams{T,T2})
        

    #end

    function update_Pₜₜ!(Pₜₜ::AbstractArray{Real, 3}, Kₜ::AbstractArray{Real, 3},
        Mₜₜ₋₁::AbstractArray{Real, 3}, Pₜₜ₋₁::AbstractArray{Real, 3}, Zₜₜ₋₁::AbstractArray{Real, 2},
        tmpKM::AbstractMatrix{Real}, tmpKMK::AbstractMatrix{Real}, params::QKParams{T,T2},
        t::Int) where {T <: Real, T2 <: Real}
        
        @unpack M, P, B̃, wc, wu, wv, wuu, wuv, wvv = params
        zt = Zₜₜ₋₁[:, t]
        V_tmp = [wc + wu * z[2] + wv * z[1] + wuu * z[6] + wvv * z[3] + 
            (wuv / 2.0) * (z[4] + z[5])]
        #=
        #KₜMₜₜ₋₁Kₜ'
        for p = 1:P
            for m = 1:M
                tmpKM[p, m] = zero(T)
                for k = 1:M
                    tmpKM[p, m] += Kₜ[p, k, t] * Mₜₜ₋₁[k, m, t]
                end
            end
        end
        
        #KₜMₜₜ₋₁Kₜ'
        for p = 1:P
            for q = 1:P
                tmpKMK[p, q] = zero(T)
                for m = 1:M
                    tmpKMK[p, q] += tmpKM[p, m] * Kₜ[q, m, t]
                end
            end
        end

        #Pₜₜ = Pₜₜ₋₁ - KₜMₜₜ₋₁Kₜ'
        for p = 1:P
            for q = 1:P
                Pₜₜ[p, q, t + 1] += Pₜₜ₋₁[p, q, t]
                Pₜₜ[p, q, t + 1] -= tmpKMK[p, q]
            end
        end
        =#
        #Pₜₜ[:, :, t + 1] = (I - Kₜ[:, :, t] * B̃) * Pₜₜ₋₁[:, :, t] * transpose(I - Kₜ[:, :, t] * B̃) +
        #    Kₜ[:, :, t] * V_tmp * Kₜ[:, :, t]'
        A = I - Kₜ[:, :, t] * B̃
        #P = A * P * A.' + K * R * K.';
        Pₜₜ[:, :, t + 1] =
            make_positive_definite(A * Pₜₜ₋₁[:, :, t] * A' +
                                    Kₜ[:, :, t] * V_tmp * Kₜ[:, :, t]'
                                    )

    end

    function update_Pₜₜ(Kₜ::AbstractMatrix{T5},
        Mₜₜ₋₁::AbstractMatrix{T6},
        Pₜₜ₋₁::AbstractMatrix{T3},
        Zₜₜ₋₁::AbstractVector{T4},
        params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real, T3 <: Real, T4 <: Real, T5 <: Real, T6 <: Real}

        @unpack M, P, B̃, wc, wu, wv, wuu, wvv, wuv = params
        zt = Zₜₜ₋₁
        V_tmp = [wc + wu * zt[2] + wv * zt[1] + wuu * zt[6] + wvv * zt[3] + 
            (wuv / 2.0) * (zt[4] + zt[5])]

        A = I - Kₜ * B̃

        P = A * Pₜₜ₋₁ * A' + Kₜ * V_tmp * Kₜ'

        #if any(Diagonal(new_Pₜₜ) .< 0.0)
        #println("Negative Variance")
        #end

        if !isposdef(P)
            new_Pₜₜ = make_positive_definite(P)
        else
            new_Pₜₜ = P
        end
        return new_Pₜₜ
    end

    function update_Zₜₜ!(Zₜₜ::AbstractMatrix{T}, Kₜ::AbstractArray{Real, 3}, 
            Yₜ::AbstractMatrix{T1}, Yₜₜ₋₁::AbstractMatrix{T}, Zₜₜ₋₁::AbstractMatrix{T},
            tmpϵ::AbstractVector{T}, params::QKParams{T,T2}, t::Int) where {T1 <: Real, T <: Real, T2 <: Real}
        # Zₜₜ = Zₜₜ₋₁ + Kₜ(Yₜ - Yₜₜ₋₁)

        @unpack M, P = params
        #=
        for m = 1:M
            tmpϵ[m] = zero(T)
            tmpϵ[m] += Yₜ[m, t]
            tmpϵ[m] -= Yₜₜ₋₁[m, t]
        end
        
        for p = 1:P
            # Zₜₜ = Zₜₜ₋₁ ⋯
            Zₜₜ[p, t + 1] += Zₜₜ₋₁[p, t]
            # ⋯ + Kₜ(Yₜ - Yₜₜ₋₁)
            for m = 1:M
                Zₜₜ[p, t + 1] += Kₜ[p, m, t] * tmpϵ[m]
            end
        end
        =#
        Zₜₜ[:, t + 1] = Zₜₜ₋₁[:, t] + Kₜ[:, :, t] * (Yₜ[:, t] - Yₜₜ₋₁[:, t])


    end

    function update_Zₜₜ(Kₜ::AbstractArray{Real, 3},
        Yₜ::AbstractMatrix{T1},
        Yₜₜ₋₁::AbstractMatrix{T1},
        Zₜₜ₋₁::AbstractMatrix{T1},
        params::QKParams{T,T2},
        t::Int) where {T1 <: Real, T <: Real, T2 <: Real}

        @unpack M, P = params

        return Zₜₜ₋₁[:, t] + Kₜ[:, :, t] * (Yₜ[:, t] - Yₜₜ₋₁[:, t])
        
    end

    function update_Zₜₜ!(Zₜₜ::AbstractMatrix{T}, Kₜ::AbstractArray{Real, 3}, 
        Yₜ::AbstractVector{T1}, Yₜₜ₋₁::AbstractVector{T}, Zₜₜ₋₁::AbstractMatrix{T},
        tmpϵ::AbstractVector{T}, params::QKParams{T,T2}, t::Int) where {T1 <: Real, T <: Real, T2 <: Real}

        # Zₜₜ = Zₜₜ₋₁ + Kₜ(Yₜ - Yₜₜ₋₁)

        @unpack M, P = params
        #=
        for m = 1:M
            tmpϵ[m] = zero(T)
            tmpϵ[m] += Yₜ[m, t]
            tmpϵ[m] -= Yₜₜ₋₁[m, t]
        end
        
        for p = 1:P
            # Zₜₜ = Zₜₜ₋₁ ⋯
            Zₜₜ[p, t + 1] += Zₜₜ₋₁[p, t]
            # ⋯ + Kₜ(Yₜ - Yₜₜ₋₁)
            for m = 1:M
                Zₜₜ[p, t + 1] += Kₜ[p, m, t] * tmpϵ[m]
            end
        end
        =#
        Zₜₜ[:, t + 1] = Zₜₜ₋₁[:, t] + Kₜ[:, :, t] * (Yₜ[t] - Yₜₜ₋₁[t])


    end

    function update_Zₜₜ(Kₜ::AbstractArray{Real, 3},
        Yₜ::AbstractVector{T1},
        Yₜₜ₋₁::AbstractVector{T},
        Zₜₜ₋₁::AbstractMatrix{T},
        params::QKParams{T,T2},
        t::Int) where {T1 <: Real, T <: Real, T2 <: Real}


        @unpack M, P = params

        return Zₜₜ₋₁[:, t] + Kₜ[:, :, t] * (Yₜ[t] - Yₜₜ₋₁[t])
        
    end

    function update_Zₜₜ!(Zₜₜ::AbstractMatrix{T}, Kₜ::AbstractArray{Real, 3}, 
        Yₜ::AbstractMatrix{T1}, Yₜₜ₋₁::AbstractVector{T}, Zₜₜ₋₁::AbstractMatrix{T},
        tmpϵ::AbstractVector{T}, params::QKParams{T,T2}, t::Int) where {T1 <: Real, T <: Real, T2 <: Real}

        # Zₜₜ = Zₜₜ₋₁ + Kₜ(Yₜ - Yₜₜ₋₁)

        @unpack M, P = params
        #=
        for m = 1:M
            tmpϵ[m] = zero(T)
            tmpϵ[m] += Yₜ[m, t]
            tmpϵ[m] -= Yₜₜ₋₁[m, t]
        end
        
        for p = 1:P
            # Zₜₜ = Zₜₜ₋₁ ⋯
            Zₜₜ[p, t + 1] += Zₜₜ₋₁[p, t]
            # ⋯ + Kₜ(Yₜ - Yₜₜ₋₁)
            for m = 1:M
                Zₜₜ[p, t + 1] += Kₜ[p, m, t] * tmpϵ[m]
            end
        end
        =#
        Zₜₜ[:, t + 1] = Zₜₜ₋₁[:, t] + Kₜ[:, :, t] * (Yₜ[:,t] - Yₜₜ₋₁[t])


    end

    function update_Zₜₜ(Kₜ::AbstractArray{Real, 3},
        Yₜ::AbstractMatrix{T1},
        Yₜₜ₋₁::AbstractVector{T},
        Zₜₜ₋₁::AbstractMatrix{T},
        params::QKParams{T,T2},
        t::Int) where {T1 <: Real, T <: Real, T2 <: Real}


        @unpack M, P = params

        return Zₜₜ₋₁[:, t] + Kₜ[:, :, t] * (Yₜ[:,t] - Yₜₜ₋₁[t])
    end

    function update_Zₜₜ!(Zₜₜ::AbstractMatrix{T}, Kₜ::AbstractArray{Real, 3}, 
        Yₜ::AbstractVector{T1}, Yₜₜ₋₁::AbstractMatrix{T}, Zₜₜ₋₁::AbstractMatrix{T},
        tmpϵ::AbstractVector{T}, params::QKParams{T,T2}, t::Int) where {T1 <: Real, T <: Real, T2 <: Real}

        # Zₜₜ = Zₜₜ₋₁ + Kₜ(Yₜ - Yₜₜ₋₁)

        @unpack M, P = params
        #=
        for m = 1:M
            tmpϵ[m] = zero(T)
            tmpϵ[m] += Yₜ[m, t]
            tmpϵ[m] -= Yₜₜ₋₁[m, t]
        end
        
        for p = 1:P
            # Zₜₜ = Zₜₜ₋₁ ⋯
            Zₜₜ[p, t + 1] += Zₜₜ₋₁[p, t]
            # ⋯ + Kₜ(Yₜ - Yₜₜ₋₁)
            for m = 1:M
                Zₜₜ[p, t + 1] += Kₜ[p, m, t] * tmpϵ[m]
            end
        end
        =#
        Zₜₜ[:, t + 1] = Zₜₜ₋₁[:, t] + Kₜ[:, :, t] * (Yₜ[t] - Yₜₜ₋₁[:,t])


    end

    function update_Zₜₜ(Kₜ::AbstractMatrix{T},
        Yₜ::Real,
        Yₜₜ₋₁::Real,
        Zₜₜ₋₁::AbstractVector{T},
        params::QKParams{T,T2},
        t::Int) where {T <: Real, T2 <: Real}


        @unpack M, P = params

        return Zₜₜ₋₁ + Kₜ * (Yₜ - Yₜₜ₋₁)
    end
   
    function compute_Kₜ!(Kₜ::AbstractArray{Real, 3}, Pₜₜ₋₁::AbstractArray{Real, 3},
            Mₜₜ₋₁::AbstractArray{Real, 3}, tmpPB::AbstractMatrix{T},
            params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}

        @unpack B̃, M, P = params

        #for i = 1:P
        #    for j = 1:M
        #        tmpPB[i, j] = zero(T)
        ##        for k = 1:P
        #            tmpPB[i, j] += Pₜₜ₋₁[i, k, t] * B̃[j, k]
        #        end
        #    end
        #end

        #Kₜ[:, :, t] .= tmpPB / Mₜₜ₋₁[:, :, t]
        Kₜ[:, :, t] .= (Pₜₜ₋₁[:, :, t] * B̃') / Mₜₜ₋₁[:, :, t]

    end

    #=
    function compute_Kₜ(Pₜₜ₋₁::AbstractMatrix{T},
        Mₜₜ₋₁::AbstractMatrix{T},
        params::QKParams{T,T2}
        t::Int)

        @unpack B̃, M, P = params

        return (Pₜₜ₋₁ * B̃') / Mₜₜ₋₁
    end
    =#

    function ensure_positive_definite(A::AbstractMatrix{T1}) where T1 <: Real
        ε = 1.5e-8
        return (A + A') / 2.0 + ε * I
    end
    function compute_Kₜ(Pₜₜ₋₁::AbstractMatrix{T}, Mₜₜ₋₁::AbstractMatrix{T}, params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}
        @unpack B̃, M, P = params
        S = B̃ * Pₜₜ₋₁ * B̃' + Mₜₜ₋₁
        return Pₜₜ₋₁ * B̃' / S
    end

    function predict_Mₜₜ₋₁!(Mₜₜ₋₁::AbstractArray{Real, 3}, Pₜₜ₋₁::AbstractArray{Real, 3},
            Zₜₜ₋₁::AbstractMatrix{T}, tmpB::AbstractMatrix{T}, params::QKParams{T,T2},
            t::Int) where {T <: Real, T2 <: Real}
                   

        @unpack B̃, V, M, P, wc, wu, wv, wuu, wuv, wvv = params
        zt = Zₜₜ₋₁[:, t]
        #Mₜₜ₋₁ = B̃Pₜₜ₋₁B̃' + V
        # tmpB = B̃Pₜₜ₋₁
       #=
        for i = 1:M
            for j = 1:P
                tmpB[i, j] = 0
                for k = 1:P
                    tmpB[i, j] += B̃[i, k] * Pₜₜ₋₁[k, j, t]
                end
            end
        end

        if typeof(V) <: Function
            V_tmp = V(zt)
            trHP = tr(HessR(zt) * Pₜₜ₋₁[:,:,t])
            for i = 1:M
                for j = 1:M
                    for k = 1:P
                        Mₜₜ₋₁[i, j, t] += tmpB[i, k] * B̃[j, k]
                    end
                    Mₜₜ₋₁[i, j, t] += V_tmp[i, j] 
                end
            end
            Mₜₜ₋₁[:, :, t] .+= 0.5 * tr(HessR(zt) * Pₜₜ₋₁[:,:,t])
        else
        #Mₜₜ₋₁ = tmpB * B̃' + V
        for i = 1:M
            for j = 1:M
                for k = 1:P
                    Mₜₜ₋₁[i, j, t] += tmpB[i, k] * B̃[j, k]
                end
                Mₜₜ₋₁[i, j, t] += V[i, j]
            end
        end
        =#

        V = [wc + wu * zt[2] + wv * zt[1] + wuu * zt[6] + wvv * zt[3] + 
            (wuv / 2.0) * (zt[4] + zt[5])]    
        Mₜₜ₋₁[:,:,t] = make_positive_definite(B̃ * Pₜₜ₋₁[:,:,t] * B̃' + V)
        #Mₜₜ₋₁[:,:,t] = B̃ * Pₜₜ₋₁[:,:,t] * B̃' + [V_tmp]


        #if any(Diagonal(Mₜₜ₋₁[:,:,t] .< 0.0))
        #    println("Negative Variance")
        #end
    end

    #=
    function predict_Mₜₜ₋₁(Pₜₜ₋₁::AbstractMatrix{T},
                       Zₜₜ₋₁::AbstractVector{T},
                       params::QKParams{T,T2}
                       t::Int)::AbstractMatrix{T}
        @unpack B̃, V, M, P = params
        zt = Zₜₜ₋₁
        V_tmp = (V isa Function) ? V(zt) : V
        Mₜₜ₋₁_t = B̃ * Pₜₜ₋₁ * B̃' + V_tmp
        return Matrix{<:Real}(Mₜₜ₋₁_t)  # Ensure we always return a concrete Matrix{<:Real}
    end
    =#
    function predict_Mₜₜ₋₁(Pₜₜ₋₁::AbstractMatrix{T}, Zₜₜ₋₁::AbstractVector{T}, params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}
        @unpack M, B̃, M, P, wc, wu, wv, wuu, wuv, wvv = params
        z = Zₜₜ₋₁
        V_tmp = [wc + wu * z[2] + wv * z[1] + wuu * z[6] + wvv * z[3] + 
            (wuv / 2.0) * (z[4] + z[5])]
        M_tmp = B̃ * Pₜₜ₋₁ * B̃' + V_tmp

        if M==1
            if M_tmp[1,1] < 0.0
                Mₜₜ₋₁ = reshape([1e-04], 1, 1)
            else
                Mₜₜ₋₁ = M_tmp
            end
        else
            if isposdef(M_tmp)
                Mₜₜ₋₁ = M_tmp
            else
                Mₜₜ₋₁ = make_positive_definite(M_tmp)
            end
        end
        return Mₜₜ₋₁
    end

    function predict_Yₜₜ₋₁!(Yₜₜ₋₁::AbstractMatrix{T}, Zₜₜ₋₁::AbstractMatrix{T},
        Y::AbstractMatrix{T}, params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}

        @unpack A, B̃, M, P, α = params

        Yₜₜ₋₁[:,t] = A + B̃ * Zₜₜ₋₁[:,t] + α * Y[:, t]
        #for i = 1:M
        #    Yₜₜ₋₁[i, t] = A[i]
        #    for j = 1:M
        #        Yₜₜ₋₁[i, t] += α[i, j] * Y[j, t]
        #    end
        #    for j = 1:P
        #        Yₜₜ₋₁[i, t] += B̃[i, j] * Zₜₜ₋₁[j, t]
        #    end
        #end

    end

    function predict_Yₜₜ₋₁!(Yₜₜ₋₁::AbstractVector{T}, Zₜₜ₋₁::AbstractMatrix{T},
        Y::AbstractVector{T1}, params::QKParams{T,T2}, t::Int) where {T1 <: Real, T <: Real, T2 <: Real}

        @unpack A, B̃, M, P, α = params

        Yₜₜ₋₁[t] = (A + B̃ * Zₜₜ₋₁[:,t] + α * Y[t])[1]
        #for i = 1:M
        #    Yₜₜ₋₁[i, t] = A[i]
        #    for j = 1:M
        #        Yₜₜ₋₁[i, t] += α[i, j] * Y[j, t]
        #    end
        #    for j = 1:P
        #        Yₜₜ₋₁[i, t] += B̃[i, j] * Zₜₜ₋₁[j, t]
        #    end
        #end

    end

    function predict_Yₜₜ₋₁!(Yₜₜ₋₁::AbstractMatrix{T}, Zₜₜ₋₁::AbstractMatrix{T},
        Y::AbstractVector{T1}, params::QKParams{T,T2}, t::Int) where {T1 <: Real, T <: Real, T2 <: Real}

        @unpack A, B̃, M, P, α = params

        Yₜₜ₋₁[:,t] = A + B̃ * Zₜₜ₋₁[:,t] + α * Y[t]
        #for i = 1:M
        #    Yₜₜ₋₁[i, t] = A[i]
        #    for j = 1:M
        #        Yₜₜ₋₁[i, t] += α[i, j] * Y[j, t]
        #    end
        #    for j = 1:P
        #        Yₜₜ₋₁[i, t] += B̃[i, j] * Zₜₜ₋₁[j, t]
        #    end
        #end

    end

    function predict_Yₜₜ₋₁(Zₜₜ₋₁::AbstractMatrix{T}, Y::AbstractMatrix{T1}, params::QKParams{T,T2},
        t::Int) where {T1 <: Real, T <: Real, T2 <: Real}
        @unpack A, B̃, M, P, α = params
        return A + B̃ * Zₜₜ₋₁[:,t] + α * Y[:, t]
    end

    function predict_Yₜₜ₋₁(Zₜₜ₋₁::AbstractVector{T},
        Y::AbstractVector{T1},
        params::QKParams{T,T2},
        t::Int) where {T1 <: Real, T <: Real, T2 <: Real}
        @unpack A, B̃, M, P, α = params
        return (A + B̃ * Zₜₜ₋₁ + α * Y[t])[1]
    end

    function predict_Pₜₜ₋₁!(Pₜₜ::AbstractArray{Real,3}, Pₜₜ₋₁::AbstractArray{Real, 3}, Σₜₜ₋₁::AbstractArray{Real, 3},
                           Zₜₜ::AbstractMatrix{T}, tmpP::AbstractMatrix{T}, params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}
        
        #vecΣₜ₋₁ₜ₋₁[:, t] .= ν .+ Ψ * Zₜₜ[:, t]
        compute_Σₜₜ₋₁!(Σₜₜ₋₁, Zₜₜ, params, t)

        #Pₜₜ₋₁ = Φ̃Pₜ₋₁ₜ₋₁Φ̃' + Σ̃(Zₜ₋₁ₜ₋₁)
        @unpack Φ̃, P = params
        #Φ̃Pₜ₋₁ₜ₋₁
        #for i = 1:P
        #    for j = 1:P
        #        tmpP[i, j] = Φ̃[i, j] * Pₜₜ[j, i, t]
        #    end
        #end
        #(Φ̃Pₜ₋₁ₜ₋₁)Φ̃' ...
        #for i = 1:P
        #    for j = 1:P
        #        Pₜₜ₋₁[i, j, t] = tmpP[i, j] * Φ̃[i, j]
        #    end
        #end
        #... + Σ̃(Zₜ₋₁ₜ₋₁)
        #for i = 1:P
        #    for j = 1:P
        #        Pₜₜ₋₁[i, j, t] += Σₜₜ₋₁[i, j, t]
        #    end
        #end
        Pₜₜ₋₁[:, :, t] = ensure_positive_definite(Φ̃ * Pₜₜ[:, :, t] * Φ̃' + Σₜₜ₋₁[:, :, t])
     
    end

    function predict_Pₜₜ₋₁(Pₜₜ::AbstractMatrix{T},
        Zₜₜ::AbstractVecOrMat{<:Real},
        params::QKParams{T,T2},
        t::Int) where {T <: Real, T2 <: Real}

        Σₜₜ₋₁ = compute_Σₜₜ₋₁(Zₜₜ, params, t)

        @unpack Φ̃ = params

        P_tmp = Φ̃ * Pₜₜ * Φ̃' + Σₜₜ₋₁

        if !isposdef(P_tmp)
            P = make_positive_definite(P_tmp)
        else
            P = P_tmp
        end

        return P
    end


    function compute_Σₜₜ₋₁!(Σₜₜ₋₁::AbstractArray{Real, 3}, Zₜₜ::AbstractMatrix{T},
                              params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}
        
        @unpack Λ, Σ, μ , Φ, N,L1, L2, L3, Λ, Σ, μ, Φ̃ = params
        #Σₜₜ₋₁[:, :, t] .= make_positive_definite(compute_Σ̃condZ(Zₜₜ[:,t], L1, L2, L3, Λ, Σ, μ, Φ̃))
        Σₜₜ₋₁[:, :, t] .= compute_Σ̃condZ(Zₜₜ[:,t], L1, L2, L3, Λ, Σ, μ, Φ̃)
        #=
        ndx1(n::Int) = rem(n - 1, P) + 1
        ndx2(n::Int) = div(n - 1, P) + 1
        #vecΣₜ₋₁ₜ₋₁[:, t] .= ν .+ Ψ * Zₜₜ[:, t]
        for i = 1:P^2
            vecΣₜₜ₋₁[i, t] = ν[i]
            for j = 1:P
                vecΣₜₜ₋₁[i, t] += Ψ[i, j] * Zₜₜ[j, t]
            end
            Σₜₜ₋₁[ndx1(i), ndx2(i), t] = vecΣₜₜ₋₁[i, t]
        end
        =#
    end

    

    function compute_Σₜₜ₋₁(Zₜₜ::AbstractVector{T},
        params::QKParams{T,T2},
        t::Int) where {T <: Real, T2 <: Real}

        @unpack Λ, Σ, μ, Φ, N, L1, L2, L3, Λ, Σ, μ, Φ̃ = params

        return compute_Σ̃condZ(Zₜₜ, L1, L2, L3, Λ, Σ, μ, Φ̃)
    end

    function predict_Zₜₜ₋₁!(Ztt::AbstractMatrix{T}, 
        Zttm1::AbstractMatrix{T}, params::QKParams{T,T2}, t::Int) where {T <: Real, T2 <: Real}

        @unpack μ̃, Φ̃, P = params
        #for i in 1:P
        #    Zttm1[i, t] = μ̃[i]
        #    for j = 1:P
        #        Zttm1[i, t] += Φ̃[i, j] * Ztt[j, t]
        #    end
        #end

        Zttm1[:, t] = μ̃ + Φ̃ * Ztt[:, t]

    end

    function predict_Zₜₜ₋₁(Ztt::AbstractVector{T}, params::QKParams{T,T2}) where {T <: Real, T2 <: Real}

        @unpack μ̃, Φ̃, P = params
        #Zttm1 = copy(μ̃)
        #for i in 1:P
        #    Zttm1[i] = μ̃[i]
        #    for j = 1:P
        #        Zttm1[i] += Φ̃[i, j] * Ztt[j]
        #    end
        #end

        return μ̃ + Φ̃ * Ztt

    end

    #function qrr(A::AbstractMatrix{T}, B::AbstractMatrix{T})
    #    ~, R = qr(vcat(A,B))
    #    return R
    #end

    function smooth_max(x, threshold=1e-8)
        return ((x + threshold) + sqrt((x - threshold)^2 + sqrt(eps()))) / 2.0
    end

    function d_eigen(A::AbstractMatrix{T}) where T <: Real
        vals, vecs = DifferentiableEigen.eigen(A)
        N = Int(length(vals) / 2)
        tmp = vecs[1:2:end]
        out_vecs = reshape(tmp, N, N)
        return vals[1:2:end], out_vecs
    end

    function make_positive_definite(A::AbstractMatrix{T}) where T <: Real
        # Compute the eigenvalues and eigenvectors
        
        A_tmp = (A + A') / 2.0
        eig_vals, eig_vecs = d_eigen(A_tmp)

        corrected_eigvals = smooth_max.(eig_vals)

        # Reconstruct the matrix using the modified eigenvalues
        corrected_A_tmp = eig_vecs * Diagonal(corrected_eigvals) * eig_vecs'
    
        # Ensure perfect symmetry
        corrected_A = (corrected_A_tmp + corrected_A_tmp') / 2.0

        return corrected_A

    end

    # Define a custom eigen function for ForwardDiff.Dual types
    #=
    function make_positive_definite(A::AbstractMatrix{T})
        # Compute the eigenvalues and eigenvectors
        eigvals, eigvecs = eigen(Hermitian(A))
        
        # Smooth approximation of max(x, ε)
        ε = sqrt(eps(T))
        smooth_max(x) = (x + √(x^2 + ε^2)) / 2
        
        # Apply smooth max to eigenvalues
        corrected_eigvals = smooth_max.(eigvals)
        
        # Reconstruct the matrix using the modified eigenvalues
        corrected_A = eigvecs * Diagonal(corrected_eigvals) * eigvecs'
        
        return corrected_A
    end
    =#

    
    #Quadratic Kalman Filter
    #=
    function srqkf_filter(data::QKData{<:Real}, params::QKParams{T,T2})

        @unpack T̄, Y = data
        @unpack N, M, μ̃ᵘ, Σ̃ᵘ, P = params

        # Predfine Matrix
        Zₜₜ =  zeros(T, P, T̄ + 1)
        Zₜₜ₋₁ = zeros(T, P, T̄)
        Fₜₜ = zeros(T, P, P, T̄ + 1)
        Fₜₜ₋₁ = zeros(T, P, P, T̄)
        Σₜₜ₋₁ = zeros(T, P, P, T̄)
        #vecΣₜₜ₋₁ = zeros(T, P^2, T̄)
        Kₜ = zeros(T, P, M, T̄)
        tmpP = zeros(T, P, P)
        tmpB = zeros(T, M, P)
        Yₜₜ₋₁ = zeros(T, M, T̄)
        Gₜₜ₋₁ = zeros(T, M, M, T̄)
        tmpPB = zeros(T, P, M)
        Yₜ = Y[:, 2:end]
        tmpϵ = zeros(T, M)
        tmpKM = zeros(T, P, M)
        tmpKMK = zeros(T, P, P)

        #Initalize: Z₀₀ = μ̃ᵘ, P₀₀ = Σ̃ᵘ
        Zₜₜ[:, 1] .= μ̃ᵘ
        ~, tmp = qr(Σ̃ᵘ)
        Fₜₜ[:, :, 1] .= Σ̃ᵘ
        
        # Loop over time
        for t in 1:T̄

            # State Prediction: Zₜₜ₋₁ = μ̃ + Φ̃Zₜ₋₁ₜ₋₁, Pₜₜ₋₁ = Φ̃Pₜ₋₁ₜ₋₁Φ̃' + Σ̃(Zₜ₋₁ₜ₋₁)
            predict_Zₜₜ₋₁!(Zₜₜ, Zₜₜ₋₁, params, t)
            predict_Pₜₜ₋₁!(Pₜₜ, Pₜₜ₋₁, Σₜₜ₋₁, Zₜₜ, tmpP, params, t)

            # Observation Prediction: Yₜₜ₋₁ = A + B̃Zₜₜ₋₁, Mₜₜ₋₁ = B̃Pₜₜ₋₁B̃' + V
            predict_Yₜₜ₋₁!(Yₜₜ₋₁, Zₜₜ₋₁, Y, params, t)
            predict_Mₜₜ₋₁!(Mₜₜ₋₁, Pₜₜ₋₁, tmpB, params, t)

            # Kalman Gain: Kₜ = Pₜₜ₋₁B̃′/Mₜₜ₋₁
            compute_Kₜ!(Kₜ, Pₜₜ₋₁, Mₜₜ₋₁, tmpPB, params, t)

            # Update States: Zₜₜ = Zₜₜ₋₁ + Kₜ(Yₜ - Yₜₜ₋₁); Pₜₜ = Pₜₜ₋₁ - KₜMₜₜ₋₁Kₜ'
            update_Zₜₜ!(Zₜₜ, Kₜ, Yₜ, Yₜₜ₋₁, Zₜₜ₋₁, tmpϵ, params, t)
            update_Pₜₜ!(Pₜₜ, Kₜ, Mₜₜ₋₁, Pₜₜ₋₁, tmpKM, tmpKMK, params, t)

        end

        return (Zₜₜ = Zₜₜ, Pₜₜ = Pₜₜ,  Yₜₜ₋₁ = Yₜₜ₋₁, Mₜₜ₋₁ = Mₜₜ₋₁, Kₜ = Kₜ, Zₜₜ₋₁ = Zₜₜ₋₁,
                Pₜₜ₋₁ = Pₜₜ₋₁, Σₜₜ₋₁ = Σₜₜ₋₁)

    end
    =#

    function make_symmetric(A::Matrix{T}) where T <: Real
        return (A + A') / 2
    end

    function compute_wc_c1(α::Real, Δt::Real, θy::Real, ξ1::Real, σz::Real, θz::Real)
        return α - α / exp(Δt * θy) + 
               (ξ1 * σz^2 * (θy + 2. * θz * coth(Δt * θz) + 
               exp(Δt * θy) * (θy - 2. * θz * coth(Δt * θz)))) / 
               (exp(Δt * θy) * (θy^3 - 4. * θy * θz^2))
    end

    function compute_wc_c2(ξ1::Real, σz::Real, Δt::Real, θz::Real, α::Real)
        numerator = 2. * ξ1 * σz^2 * (-1. + cosh(Δt * θz)) + 
                    6. * α * θz^2 * sinh(Δt * θz)
        
        denominator = 3. * (1. + exp(Δt * θz)) * θz^2
        
        return numerator / denominator
    end

    function compute_wc_c3(Δt::Real, θz::Real, α::Real, ξ1::Real, σz::Real)
        numerator = (-1. + coth(Δt * θz)) * (
            16. * α * θz^2 * sinh(Δt * θz)^2 + 
            ξ1 * σz^2 * (-2. * Δt * θz + sinh(2. * Δt * θz))
        )
        
        denominator = 8. * θz^2
        
        return numerator / denominator
    end
    
    function compute_wc(α::Real, ξ1::Real, σz::Real, Δt::Real, θz::Real, θy::Real)
        
        if θy ≈ θz
            return compute_wc_c2(ξ1, σz, Δt, θz, α)
        elseif θy ≈ 2.0 * θz
            return compute_wc_c3(Δt, θz, α, ξ1, σz)
        else
            return compute_wc_c1(α, Δt, θy, ξ1, σz, θz)
        end
    end
    

    function compute_wy0(Δt::Real,θy::Real)
        return exp(-θy * Δt)
    end

    function compute_wv_c1(ξ0::Real, θy::Real, θz::Real, Δt::Real)
        numerator = ξ0 * (
            θy - θz * coth(Δt * θz) + 
            exp(-Δt * θy) * θz * csch(Δt * θz)
        )
        
        denominator = θy^2 - θz^2
        
        return numerator / denominator
    end
    
    function compute_wv_c2(ξ0::Real, Δt::Real, θz::Real)
        return 1.0/2.0 * ξ0 * (Δt + 
               1.0/θz - 
               Δt * coth(Δt * θz))
    end
    
    function compute_wv_c3(Δt::Real, θz::Real, ξ0::Real)
        numerator = 2. * (2. + exp(Δt * θz)) * ξ0 * sinh(Δt * θz)
        denominator = 3. * (1. + exp(Δt * θz))^2 * θz
        
        return numerator / denominator
    end
    
    function compute_wv(ξ0::Real, θy::Real, θz::Real, Δt::Real)
        if θy ≈ θz
            return compute_wv_c2(ξ0, Δt, θz)
        elseif θy ≈ 2.0 * θz
            return compute_wv_c3(Δt, θz, ξ0)
        else
            return compute_wv_c1(ξ0, θy, θz, Δt)
        end
    end

    function compute_wvv_c1(Δt::Real, θy::Real, θz::Real, ξ1::Real)
        numerator = exp(-Δt * θy) * ξ1 * (
            exp(Δt * θy) * θy^2 + 
            θz * csch(Δt * θz)^2 * (
                -2. * θz + 
                exp(Δt * θy) * (2. * θz - θy * sinh(2. * Δt * θz))
            )
        )
        
        denominator = θy^3 - 4. * θy * θz^2
        
        return numerator / denominator
    end
    
    function compute_wvv_c2(Δt::Real, θz::Real, ξ1::Real)
        return ((1. - 4. / (1. + exp(Δt * θz))^2) * ξ1) / (3. * θz)
    end
    
    function compute_wvv_c3(Δt::Real, θz::Real, ξ1::Real)
        numerator = (3. - 4. * exp(2. * Δt * θz) + exp(4. * Δt * θz) + 
                     4. * Δt * θz) * ξ1
        
        denominator = 4. * (1. - exp(2. * Δt * θz))^2 * θz
        
        return numerator / denominator
    end
    
    function compute_wvv(Δt::Real, θy::Real, θz::Real, ξ1::Real)
        if θy ≈ θz
            return compute_wvv_c2(Δt, θz, ξ1)
        elseif θy ≈ 2.0 * θz
            return compute_wvv_c3(Δt, θz, ξ1)
        else
            return compute_wvv_c1(Δt, θy, θz, ξ1)
        end
    end 

    function compute_wu_c1(Δt::Real, θy::Real, θz::Real, ξ0::Real)
        numerator = -exp(-Δt * θy) * ξ0 * (θy + θz * coth(Δt * θz) - 
                     exp(Δt * θy) * θz * csch(Δt * θz))
        
        denominator = θy^2 - θz^2
        
        return numerator / denominator
    end
    
    function compute_wu_c2(Δt::Real, θz::Real, ξ0::Real)
        numerator = ξ0 * (-cosh(Δt * θz) + Δt * θz * csch(Δt * θz) + 
                     sinh(Δt * θz))
        
        denominator = 2. * θz
        
        return numerator / denominator
    end
    
    function compute_wu_c3(Δt::Real, θz::Real, ξ0::Real)
        numerator = (-exp(-2. * Δt * θz) + 2. / (1. + exp(Δt * θz))) * ξ0
        
        denominator = 3. * θz
        
        return numerator / denominator
    end
    
    function compute_wu(Δt::Real, θy::Real, θz::Real, ξ0::Real)
        if θy ≈ θz
            return compute_wu_c2(Δt, θz, ξ0)
        elseif θy ≈ 2.0 * θz
            return compute_wu_c3(Δt, θz, ξ0)
        else
            return compute_wu_c1(Δt, θy, θz, ξ0)
        end
    end

    function compute_wuu_c1(Δt::Real, θy::Real, θz::Real, ξ1::Real)
        numerator = -exp(-Δt * θy) * ξ1 * (
            θy^2 + 
            2. * θy * θz * coth(Δt * θz) - 
            2. * (-1. + exp(Δt * θy)) * θz^2 * csch(Δt * θz)^2
        )
        
        denominator = θy^3 - 4. * θy * θz^2
        
        return numerator / denominator
    end
    
    function compute_wuu_c2(Δt::Real, θz::Real, ξ1::Real)
        numerator = exp(-Δt * θz) * (-1. + exp(Δt * θz)) * (1. + 
                     3. * exp(Δt * θz)) * ξ1
        
        denominator = 3. * (1. + exp(Δt * θz))^2 * θz
        
        return numerator / denominator
    end
    
    function compute_wuu_c3(Δt::Real, θz::Real, ξ1::Real)
        numerator = exp(-Δt * θz) * ξ1 * (
            2. * exp(3. * Δt * θz) * Δt * θz + 
            (1. - 3. * exp(2. * Δt * θz)) * sinh(Δt * θz)
        )
        
        denominator = 2. * (-1. + exp(2. * Δt * θz))^2 * θz
        
        return numerator / denominator
    end
    
    function compute_wuu(Δt::Real, θy::Real, θz::Real, ξ1::Real)
        if θy ≈ θz
            return compute_wuu_c2(Δt, θz, ξ1)
        elseif θy ≈ 2.0 * θz
            return compute_wuu_c3(Δt, θz, ξ1)
        else
            return compute_wuu_c1(Δt, θy, θz, ξ1)
        end
    end

    function compute_wuv_c1(Δt::Real, θy::Real, θz::Real, ξ1::Real)
        numerator = 2. * exp(-Δt * θy) * θz * ξ1 * (
            θy + 
            2. * θz * coth(Δt * θz) + 
            exp(Δt * θy) * (θy - 2. * θz * coth(Δt * θz))
        ) * csch(Δt * θz)
        
        denominator = θy^3 - 4. * θy * θz^2
        
        return numerator / denominator
    end
    
    function compute_wuv_c2(Δt::Real, θz::Real, ξ1::Real)
        numerator = 4. * (-1. + exp(Δt * θz)) * ξ1
        
        denominator = 3. * (1. + exp(Δt * θz))^2 * θz
        
        return numerator / denominator
    end
    
    function compute_wuv_c3(Δt::Real, θz::Real, ξ1::Real)
        numerator = exp(Δt * θz) * ξ1 * (-2. * Δt * θz + 
                     sinh(2. * Δt * θz))
        
        denominator = (-1. + exp(2. * Δt * θz))^2 * θz
        
        return numerator / denominator
    end
    
    function compute_wuv(Δt::Real, θy::Real, θz::Real, ξ1::Real)
        if θy ≈ θz
            return compute_wuv_c2(Δt, θz, ξ1)
        elseif θy ≈ 2.0 * θz
            return compute_wuv_c3(Δt, θz, ξ1)
        else
            return compute_wuv_c1(Δt, θy, θz, ξ1)
        end
    end

    function compute_mean_y(u::Real, v::Real, y0::Real, Δt::Real, α::Real, θy::Real, θz::Real, σz::Real,
        ξ0::Real, ξ1::Real)

        return compute_wc(α, ξ1, σz, Δt, θz, θy) + compute_wy0(Δt,θy) * y0 + 
            compute_wv(ξ0, θy, θz, Δt) * v + compute_wu(Δt, θy, θz, ξ0) * u +
            compute_wvv(Δt, θy, θz, ξ1) * v^2 + compute_wuu(Δt, θy, θz, ξ1) * u^2 +
            compute_wuv(Δt, θy, θz, ξ1) * u * v

    end

    function compute_mean_y_aug(z::AbstractVector{T}, y0::Real, Δt::Real, α::Real, θy::Real, θz::Real, σz::Real,
        ξ0::Real, ξ1::Real) where T <: Real

        return compute_wc(α, ξ1, σz, Δt, θz, θy) + compute_wy0(Δt,θy) * y0 + 
            compute_wv(ξ0, θy, θz, Δt) * z[1] + compute_wu(Δt, θy, θz, ξ0) * z[2] +
            compute_wvv(Δt, θy, θz, ξ1) * z[3] + compute_wuu(Δt, θy, θz, ξ1) * z[6] +
            (compute_wuv(Δt, θy, θz, ξ1) / 2.0) * (z[4] + z[5])

    end

    function compute_qc_c1(Δt::Real, θy::Real, θz::Real, σy::Real, σz::Real, ξ0::Real, ξ1::Real)
        term1 = σy^2 / θy - (exp(-2. * Δt * θy) * σy^2) / θy
        
        term2 = (exp(-2. * Δt * θy) * ξ0^2 * σz^2 * (
            (-1. + exp(2. * Δt * θy)) * (θy^2 + θz^2) - 
            4. * exp(Δt * θy) * θy * θz * (-1. + 
                cosh(Δt * θy) * cosh(Δt * θz)) * csch(Δt * θz)
        )) / (θy * (θy^2 - θz^2)^2)
        
        term3 = (2. * exp(-2. * Δt * θy) * ξ1^2 * σz^4 * (
            (1. + exp(2. * Δt * θy)) * θy^2 * (θy^2 + 8. * θz^2) - 
            (-1. + exp(2. * Δt * θy)) * θy * θz * (5. * θy^2 + 4. * θz^2) * coth(Δt * θz) + 
            8. * (-1. + exp(Δt * θy))^2 * θz^2 * (θy^2 - θz^2) * csch(Δt * θz)^2
        )) / ((θy^2 - θz^2) * (θy^3 - 4. * θy * θz^2)^2)
        
        return 0.5 * (term1 + term2 + term3)
    end
    
    function compute_qc_c2(Δt::Real, θz::Real, σy::Real, σz::Real, ξ0::Real, ξ1::Real)
        term1 = 36. * (-1. + exp(2. * Δt * θz)) * θz^3 * σy^2
        
        term2 = 9. * θz * ξ0^2 * σz^2 * (-1. + exp(2. * Δt * θz))
        
        term3 = -18. * Δt^2 * θz^3 * ξ0^2 * σz^2 * (1. + coth(Δt * θz))
        
        term4 = -72. * Δt * θz * ξ1^2 * σz^4 * (1. + coth(Δt * θz))
        
        term5 = 2. * exp(Δt * θz) * ξ1^2 * σz^4 * (33. + 
                2. * cosh(Δt * θz) + 
                cosh(2. * Δt * θz)) * sech(Δt * θz / 2.)^2
    
        numerator = exp(-2. * Δt * θz) * (term1 + term2 + term3 + term4 + term5)
        
        denominator = 72. * θz^4
        
        return numerator / denominator
    end
    
    function compute_qc_c3(Δt::Real, θz::Real, σy::Real, σz::Real, ξ0::Real, ξ1::Real)
        term1 = 36. * θz^3 * σy^2 * (1. - exp(-4. * Δt * θz))
        
        term2 = (3. * ξ1^2 * σz^4 * (15. + 24. * Δt^2 * θz^2 - 
                 16. * cosh(2. * Δt * θz) + 
                 cosh(4. * Δt * θz))) / ((-1. + exp(2. * Δt * θz))^2)
        
        term3 = 32. * exp(-2. * Δt * θz) * θz * ξ0^2 * σz^2 * 
                (2. + cosh(Δt * θz)) * sinh(Δt * θz / 2.)^2 * 
                tanh(Δt * θz / 2.)
        
        numerator = term1 + term2 + term3
        
        denominator = 144. * θz^4
        
        return numerator / denominator
    end
    
    function compute_qc_c4(Δt::Real, θz::Real, σy::Real, σz::Real, ξ0::Real, ξ1::Real)
        term1 = -400. * (1. + exp(2. * Δt * θz)) * θz * ξ0^2 * σz^2 / 
            (-1. + exp(Δt * θz))
    
        term2 = 25. * (-1. + exp(Δt * θz)) * θz * 
                (9. * θz^2 * σy^2 + 20. * ξ0^2 * σz^2)
        
        term3 = 32. * exp(0.5 * Δt * θz) * σz^2 * csch(Δt * θz) * (
            25. * θz * ξ0^2 + 
            16. * ξ1^2 * σz^2 * (
                19. + 18. * cosh(0.5 * Δt * θz) + 3. * cosh(Δt * θz)
            ) * csch(Δt * θz) * sinh(0.25 * Δt * θz)^6
        )
        
        numerator = term1 + term2 + term3
        
        denominator = exp(Δt * θz) * (225. * θz^4)
        
        return numerator / denominator
    end
    
    function compute_qc(Δt::Real, θy::Real, θz::Real, σy::Real, σz::Real, ξ0::Real, ξ1::Real)
        if θy ≈ θz
            return compute_qc_c2(Δt, θz, σy, σz, ξ0, ξ1)
        elseif θy ≈ 2.0 * θz
            return compute_qc_c3(Δt, θz, σy, σz, ξ0, ξ1)
        elseif θy ≈ 0.5 * θz
            return compute_qc_c4(Δt, θz, σy, σz, ξ0, ξ1)
        else
            return compute_qc_c1(Δt, θy, θz, σy, σz, ξ0, ξ1)
        end
    end

    
    function compute_qv_c1(ξ0::Real, ξ1::Real, σz::Real, Δt::Real, θy::Real, θz::Real)
        common_factor = 2. * exp(-2. * Δt * θy) * ξ0 * ξ1 * σz^2 * csch(Δt * θz)
    
        term1 = 6. * θy * θz * (θy^2 + θz^2)
        
        term2 = 2. * exp(Δt * θy) * θz * (-4. * θy^2 + θz^2) * 
                (-θy + 2. * θz * coth(0.5 * Δt * θz))
        
        term3 = -4. * θz^2 * (-4. * θy^2 + θz^2) * coth(Δt * θz)
        
        term4 = exp(2. * Δt * θy) * csch(Δt * θz) * (
            -2. * θy^4 + 9. * θy^2 * θz^2 - 4. * θz^4 + 
            θy^2 * (2. * θy^2 + 7. * θz^2) * cosh(2. * Δt * θz) - 
            θy * θz * (7. * θy^2 + 2. * θz^2) * sinh(2. * Δt * θz)
        )
        
        numerator = common_factor * (term1 + term2 + term3 + term4)
        
        denominator = θy * (4. * θy^2 - θz^2) * (θy^4 - 5. * θy^2 * θz^2 + 4. * θz^4)
        
        return numerator / denominator
    end

    function compute_qv_c2(ξ0::Real, ξ1::Real, σz::Real, Δt::Real, θz::Real)
        numerator = 2. * ξ0 * ξ1 * σz^2 * (
            18. * Δt * θz + 
            exp(Δt * θz) * (
                -9. - 6. * Δt * θz + 
                9. * cosh(2. * Δt * θz) + 
                2. * sinh(Δt * θz) - 
                7. * sinh(2. * Δt * θz)
            )
        )
        
        denominator = 9. * (-1. + exp(Δt * θz)) * (1. + exp(Δt * θz))^2 * θz^3
        
        return numerator / denominator
    end

    function compute_qv_c3(ξ0::Real, ξ1::Real, σz::Real, Δt::Real, θz::Real)
        numerator = exp(-3. * Δt * θz) * (
            7. + 
            exp(Δt * θz) * (
                7. + 
                exp(Δt * θz) * (
                    15. + 
                    exp(Δt * θz) * (
                        -10. + 
                        exp(Δt * θz) * (
                            -25. + 
                            3. * exp(Δt * θz) * (1. + exp(Δt * θz))
                        )
                    ) + 
                    60. * Δt * θz
                )
            )
        ) * ξ0 * ξ1 * σz^2
        
        denominator = 45. * (-1. + exp(Δt * θz)) * (1. + exp(Δt * θz))^2 * θz^3
        
        return numerator / denominator
    end

    function compute_qv_c4(ξ0::Real, ξ1::Real, σz::Real, Δt::Real, θz::Real)
        numerator = 2. * ξ0 * ξ1 * σz^2 * (
            86. + 
            (83. + 89. * exp(Δt * θz) - 36. * exp(1.5 * Δt * θz) + 
            9. * exp(2. * Δt * θz) + 6. * exp(2.5 * Δt * θz) + 
            3. * exp(3. * Δt * θz)) / exp(0.5 * Δt * θz) - 
            60. * Δt * θz * (coth(0.25 * Δt * θz) + sinh(0.5 * Δt * θz))
        )
        
        denominator = exp(Δt * θz) * (
            45. * θz^3 * (cosh(0.25 * Δt * θz) + cosh(0.75 * Δt * θz))^2
        )
        
        return numerator / denominator
    end

    function compute_qv(ξ0::Real, ξ1::Real, σz::Real, Δt::Real, θy::Real, θz::Real)
        if θy ≈ θz
            return compute_qv_c2(ξ0, ξ1, σz, Δt, θz)
        elseif θy ≈ 2.0 * θz
            return compute_qv_c3(ξ0, ξ1, σz, Δt, θz)
        elseif θy ≈ 0.5 * θz
            return compute_qv_c4(ξ0, ξ1, σz, Δt, θz)
        else
            return compute_qv_c1(ξ0, ξ1, σz, Δt, θy, θz)
        end
    end

    function compute_qvv_c1(ξ1::Real, σz::Real, Δt::Real, θy::Real, θz::Real)
        common_factor = ξ1^2 * σz^2 * csch(Δt * θz)^2 / exp(2. * Δt * θy)
    
    term1 = exp(2. * Δt * θy) * θy^3 * (θy^2 + 8. * θz^2) * cosh(2. * Δt * θz)
    
    term2 = 16. * (1. - exp(Δt * θy))^2 * θz^3 * (-θy^2 + θz^2) * coth(Δt * θz)
    
    term3 = θy * (
        -5. * θy^2 * θz^2 - 4. * θz^4 + 
        16. * exp(Δt * θy) * θz^2 * (-θy + θz) * (θy + θz) - 
        exp(2. * Δt * θy) * (
            θy^4 - 13. * θy^2 * θz^2 + 12. * θz^4 + 
            θy * θz * (5. * θy^2 + 4. * θz^2) * sinh(2. * Δt * θz)
        )
    )
    
    numerator = common_factor * (term1 + term2 + term3)
    
    denominator = (θy - θz) * (θy + θz) * (θy^3 - 4. * θy * θz^2)^2
    
    return numerator / denominator
    end

    function compute_qvv_c2(ξ1::Real, σz::Real, Δt::Real, θz::Real)
        numerator = (83. + 36. * Δt * θz + 
                    exp(Δt * θz) * (-109. + 44. * exp(Δt * θz) - 
                                    20. * exp(2. * Δt * θz) + 
                                    exp(3. * Δt * θz) + 
                                    exp(4. * Δt * θz) + 
                                    36. * Δt * θz)) * ξ1^2 * σz^2
        
        denominator = 9. * (-1. + exp(Δt * θz))^2 * (1. + exp(Δt * θz))^3 * θz^3
        
        return numerator / denominator
    end

    function compute_qvv_c3(ξ1::Real, σz::Real, Δt::Real, θz::Real)
        numerator = (7. - 10. * exp(6. * Δt * θz) + exp(8. * Δt * θz) + 
                    24. * exp(4. * Δt * θz) * (1. + Δt * θz) - 
                    2. * exp(2. * Δt * θz) * (11. + 12. * Δt * θz * (1. + 2. * Δt * θz))) * 
                    ξ1^2 * σz^2
        
        denominator = exp(2. * Δt * θz) * (24. * (θz - exp(2. * Δt * θz) * θz)^3)
        
        return -(numerator / denominator)
    end

    function compute_qvv_c4(ξ1::Real, σz::Real, Δt::Real, θz::Real)
        common_factor = 4. * ξ1^2 * σz^2 / (225. * θz^3)
    
        term1 = sech(0.5 * Δt * θz)^3
        
        term2 = 84. + 138. * cosh(0.5 * Δt * θz) + 
                84. * cosh(Δt * θz) + 
                14. * cosh(1.5 * Δt * θz)
        
        term3 = -75. * sinh(0.5 * Δt * θz) - 
                66. * sinh(Δt * θz) - 
                11. * sinh(1.5 * Δt * θz)
        
        term4 = tanh(0.25 * Δt * θz)^3
        
        return common_factor * term1 * (term2 + term3) * term4
    end

    function compute_qvv(ξ1::Real, σz::Real, Δt::Real, θy::Real, θz::Real)
        if θy ≈ θz
            return compute_qvv_c2(ξ1, σz, Δt, θz)
        elseif θy ≈ 2.0 * θz
            return compute_qvv_c3(ξ1, σz, Δt, θz)
        elseif θy ≈ 0.5 * θz
            return compute_qvv_c4(ξ1, σz, Δt, θz)
        else
            return compute_qvv_c1(ξ1, σz, Δt, θy, θz)
        end
    end

    function compute_qu_c1(ξ0::Real, ξ1::Real, σz::Real, Δt::Real, θy::Real, θz::Real)
        common_factor = 4. * exp(-2. * Δt * θy) * ξ0 * ξ1 * σz^2
    
        term1 = -2. * θy^4 - 7. * θy^2 * θz^2
        
        term2 = θz * csch(Δt * θz) * (
            3. * exp(2. * Δt * θy) * θy * (θy^2 + θz^2) +
            exp(Δt * θy) * (4. * θy^2 - θz^2) * (θy + 2. * θz * coth(0.5 * Δt * θz)) +
            2. * θz * (-4. * θy^2 + θz^2) * csch(Δt * θz)
        )
        
        term3 = θz * coth(Δt * θz) * (
            -7. * θy^3 - 2. * θy * θz^2 +
            2. * exp(2. * Δt * θy) * θz * (-4. * θy^2 + θz^2) * csch(Δt * θz)
        )
        
        numerator = common_factor * (term1 + term2 + term3)
        
        denominator = θy * (4. * θy^2 - θz^2) * (θy^4 - 5. * θy^2 * θz^2 + 4. * θz^4)
        
        return numerator / denominator
    end

    function compute_qu_c2(ξ0::Real, ξ1::Real, σz::Real, Δt::Real, θz::Real)
        numerator = 2. * ξ0 * ξ1 * σz^2 * (
            9. + 6. * (-1. + 3. * exp(Δt * θz)) * Δt * θz - 
            9. * cosh(2. * Δt * θz) + 
            2. * sinh(Δt * θz) - 
            7. * sinh(2. * Δt * θz)
        )
        
        denominator = 9. * (-1. + exp(Δt * θz)) * (1. + exp(Δt * θz))^2 * θz^3
        
        return -(numerator / denominator)
    end

    function compute_qu_c3(ξ0::Real, ξ1::Real, σz::Real, Δt::Real, θz::Real)
        numerator = (3. + exp(Δt * θz) * (
            3. + exp(Δt * θz) * (
                -25. + exp(Δt * θz) * (
                    -10. + exp(Δt * θz) * (
                        15. + 7. * exp(Δt * θz) * (
                            1. + exp(Δt * θz)
                        ) - 60. * Δt * θz
                    )
                )
            )
        )) * ξ0 * ξ1 * σz^2
        
        denominator = exp(4. * Δt * θz) * (
            45. * (-1. + exp(Δt * θz)) * (1. + exp(Δt * θz))^2 * θz^3
        )
        
        return numerator / denominator
    end

    function compute_qu_c4(ξ0::Real, ξ1::Real, σz::Real, Δt::Real, θz::Real)
        numerator = 2. * (
            -3. + 
            48. * exp(1.5 * Δt * θz) + 
            128. * exp(2.5 * Δt * θz) + 
            80. * exp(3.5 * Δt * θz) - 
            10. * exp(2. * Δt * θz) * (17. + 3. * Δt * θz) + 
            exp(4. * Δt * θz) * (-83. + 30. * Δt * θz)
        ) * ξ0 * ξ1 * σz^2 * csch(Δt * θz)^2
        
        denominator = exp(3. * Δt * θz) * (45. * θz^3)
        
        return numerator / denominator
    end

    function compute_qu(ξ0::Real, ξ1::Real, σz::Real, Δt::Real, θy::Real, θz::Real)
        if θy ≈ θz
            return compute_qu_c2(ξ0, ξ1, σz, Δt, θz)
        elseif θy ≈ 2.0 * θz
            return compute_qu_c3(ξ0, ξ1, σz, Δt, θz)
        elseif θy ≈ 0.5 * θz
            return compute_qu_c4(ξ0, ξ1, σz, Δt, θz)
        else
            return compute_qu_c1(ξ0, ξ1, σz, Δt, θy, θz)
        end
    end

    function compute_quu_c1(ξ1::Real, σz::Real, θy::Real, θz::Real, Δt::Real)
        numerator = ξ1^2 * σz^2 * (
            -2. * (θy^5 + 8. * θy^3 * θz^2) + 
            θz * (
                -2. * θy^2 * (5. * θy^2 + 4. * θz^2) * coth(Δt * θz) + 
                (-1. + exp(Δt * θy)) * θz * (
                    (21. + 5. * exp(Δt * θy)) * θy^3 + 
                    4. * (-3. + exp(Δt * θy)) * θy * θz^2 + 
                    16. * (-1. + exp(Δt * θy)) * θz * (-θy^2 + θz^2) * coth(Δt * θz)
                ) * csch(Δt * θz)^2
            )
        )
        
        denominator = exp(2. * Δt * θy) * (
            (θy - θz) * (θy + θz) * (θy^3 - 4. * θy * θz^2)^2
        )
        
        return numerator / denominator
    end

    function compute_quu_c2(ξ1::Real, σz::Real, Δt::Real, θz::Real)
        numerator = (
            -1. + exp(Δt * θz) * (
                -1. + exp(Δt * θz) * (
                    20. + exp(Δt * θz) * (
                        -44. + exp(Δt * θz) * (
                            109. + 36. * Δt * θz + exp(Δt * θz) * (
                                -83. + 36. * Δt * θz
                            )
                        )
                    )
                )
            )
        ) * ξ1^2 * σz^2
        
        denominator = exp(2. * Δt * θz) * (
            9. * (-1. + exp(Δt * θz))^2 * (1. + exp(Δt * θz))^3 * θz^3
        )
        
        return numerator / denominator
    end

    function compute_quu_c3(ξ1::Real, σz::Real, Δt::Real, θz::Real)
        numerator = (
            1. - 10. * exp(2. * Δt * θz) + 
            7. * exp(8. * Δt * θz) - 
            24. * exp(4. * Δt * θz) * (-1. + Δt * θz) + 
            exp(6. * Δt * θz) * (-22. + 24. * Δt * θz * (1. - 2. * Δt * θz))
        ) * ξ1^2 * σz^2
        
        denominator = exp(4. * Δt * θz) * (24. * (θz - exp(2. * Δt * θz) * θz)^3)
        
        return -(numerator / denominator)
    end

    function compute_quu_c4(ξ1::Real, σz::Real, Δt::Real, θz::Real)
        common_factor = 16. * ξ1^2 * σz^2 / (exp(Δt * θz) * 225. * θz^3)
    
        term1 = 11. + 14. * coth(Δt * θz)
        
        term2 = -8. * exp(0.5 * Δt * θz) * csch(Δt * θz)^3 * sinh(0.25 * Δt * θz)^2 * (
            -1. - 2. * cosh(0.5 * Δt * θz) + 31. * cosh(Δt * θz) + 8. * sinh(Δt * θz)
        )
        
        return common_factor * (term1 + term2)
    end

    function compute_quu(ξ1::Real, σz::Real, Δt::Real, θy::Real, θz::Real)
        if θy ≈ θz
            return compute_quu_c2(ξ1, σz, Δt, θz)
        elseif θy ≈ 2.0 * θz
            return compute_quu_c3(ξ1, σz, Δt, θz)
        elseif θy ≈ 0.5 * θz
            return compute_quu_c4(ξ1, σz, Δt, θz)
        else
            return compute_quu_c1(ξ1, σz, θy, θz, Δt)
        end
    end

    function compute_quv_c1(Δt::Real, θy::Real, θz::Real, ξ1::Real, σz::Real)
        common_factor = 4. * θz * ξ1^2 * σz^2 * csch(Δt * θz)
    
    term1 = 2. * (θy - θz) * (θy + θz) * (θy^2 - 4. * θz^2 - 8. * θz^2 * csch(Δt * θz)^2)
    
    term2 = cosh(Δt * θy) * (
        3. * θy^4 + 14. * θy^2 * θz^2 - 8. * θz^4 + 
        16. * (θy - θz) * θz^2 * (θy + θz) * csch(Δt * θz)^2
    )
    
    term3 = θy * θz * (-13. * θy^2 + 4. * θz^2) * coth(Δt * θz) * sinh(Δt * θy)
    
    numerator = common_factor * (term1 + term2 + term3)
    
    denominator = exp(Δt * θy) * ((θy - θz) * (θy + θz) * (θy^3 - 4. * θy * θz^2)^2)
    
    return numerator / denominator
    end
    
    function compute_quv_c2(Δt::Real, θz::Real, ξ1::Real, σz::Real)
        common_factor = 2. * exp(-2. * Δt * θz) * ξ1^2 * σz^2 * (1. + coth(Δt * θz))
        
        term1 = -12.
        
        term2 = 5. * cosh(Δt * θz)
        
        term3 = -9. * Δt * θz * csch(Δt * θz)
        
        term4 = 16. * sech(0.5 * Δt * θz)^2
        
        numerator = common_factor * (term1 + term2 + term3 + term4)
        
        denominator = 9. * θz^3
        
        return numerator / denominator
    end
    
    function compute_quv_c3(Δt::Real, θz::Real, ξ1::Real, σz::Real)
        numerator = exp(Δt * θz) * ξ1^2 * σz^2 * (
            3. + 
            12. * Δt^2 * θz^2 - 
            4. * cosh(2. * Δt * θz) + 
            cosh(4. * Δt * θz) - 
            6. * Δt * θz * sinh(2. * Δt * θz)
        )
        
        denominator = 3. * (-1. + exp(2. * Δt * θz))^3 * θz^3
        
        return numerator / denominator
    end
    
    function compute_quv_c4(Δt::Real, θz::Real, ξ1::Real, σz::Real)
        numerator = 128. * ξ1^2 * σz^2 * (
            53. + 
            66. * cosh(0.5 * Δt * θz) + 
            21. * cosh(Δt * θz)
        ) * sinh(0.25 * Δt * θz)^3

    denominator = exp(0.5 * Δt * θz) * (
        225. * θz^3 * 
        (cosh(0.25 * Δt * θz) + cosh(0.75 * Δt * θz))^3
    )

    return numerator / denominator

    end
    
    function compute_quv(Δt::Real, θy::Real, θz::Real, ξ1::Real, σz::Real)
        if θy ≈ θz
            return compute_quv_c2(Δt, θz, ξ1, σz)
        elseif θy ≈ 2.0 * θz
            return compute_quv_c3(Δt, θz, ξ1, σz)
        elseif θy ≈ 0.5 * θz
            return compute_quv_c4(Δt, θz, ξ1, σz)
        else
            return compute_quv_c1(Δt, θy, θz, ξ1, σz)
        end
    end

    function compute_var_y(u::Real, v::Real, Δt::Real, θy::Real, θz::Real, σz::Real,σy::Real,
        ξ0::Real, ξ1::Real)
    
        return compute_qc(Δt, θy, θz, σy, σz, ξ0, ξ1) + 
            compute_qv(ξ0, ξ1, σz, Δt, θy, θz) * v + 
            compute_qu(ξ0, ξ1, σz, Δt, θy, θz) * u +
            compute_qvv(ξ1, σz, Δt, θy, θz) * v^2 + 
            compute_quu(ξ1, σz, Δt, θy, θz) * u^2 +
            compute_quv(Δt, θy, θz, ξ1, σz) * u * v
    
    
    end

    function compute_var_y_aug(z::AbstractVector{T}, Δt::Real, θy::Real, θz::Real, σz::Real, σy::Real,
        ξ0::Real, ξ1::Real, t0::Real) where T<:Real
    
        return compute_qc(Δt, θy, θz, σy, σz, ξ0, ξ1) + 
            compute_qv(ξ0, ξ1, σz, Δt, θy, θz) * z[1] + 
            compute_qu(ξ0, ξ1, σz, Δt, θy, θz) * z[2] +
            compute_qvv(ξ1, σz, Δt, θy, θz) * z[3] + 
            compute_quu(ξ1, σz, Δt, θy, θz) * z[6] +
            (compute_quv(Δt, θy, θz, ξ1, σz) / 2.0) * (z[4] + z[5])
    
    
    end

    @with_kw struct UnivariateLatentModel{T<:Real, T2 <: Real}
        #Deep parameters
        θz::T
        σz::T
        α::T
        θy::T
        σy::T
        ξ0::T
        ξ1::T
        Δt::T2
        #Precomputed Parameters
        wc::T
        wy0::T
        wu::T
        wv::T
        wuu::T
        wvv::T
        wuv::T
        qc::T
        qu::T
        qv::T
        quu::T
        qvv::T
        quv::T
    end
    function UnivariateLatentModel(θz::T, σz::T, α::T, θy::T,
        σy::T, ξ0::T, ξ1::T, Δt::T2) where {T <: Real, T2 <: Real}
        # Assert parameter restrictions
        @assert θz > zero(T)
        @assert σz > zero(T)
        @assert θy > zero(T)
        @assert σy > zero(T)
        # Precompute parameters
        wc = compute_wc(α, ξ1, σz, Δt, θz, θy)
        wy0 = compute_wy0(Δt, θy)
        wu = compute_wu(Δt, θy, θz, ξ0)
        wv = compute_wv(ξ0, θy, θz, Δt)
        wuu = compute_wuu(Δt, θy, θz, ξ1)
        wvv = compute_wvv(Δt, θy, θz, ξ1)
        wuv = compute_wuv(Δt, θy, θz, ξ1)
        qc = compute_qc(Δt, θy, θz, σy, σz, ξ0, ξ1)
        qu = compute_qu(ξ0, ξ1, σz, Δt, θy, θz)
        qv = compute_qv(ξ0, ξ1, σz, Δt, θy, θz)
        quu = compute_quu(ξ1, σz, Δt, θy, θz)
        qvv = compute_qvv(ξ1, σz, Δt, θy, θz)
        quv = compute_quv(Δt,θy, θz, ξ1, σz)
        return UnivariateLatentModel{T,T2}(
            θz = θz,
            σz = σz,
            α = α,
            θy = θy,
            σy = σy,
            ξ0 = ξ0,
            ξ1 = ξ1,
            Δt = Δt,
            wc = wc,
            wy0 = wy0,
            wu = wu,
            wv = wv,
            wuu = wuu,
            wvv = wvv,
            wuv = wuv,
            qc = qc,
            qu = qu,
            qv = qv,
            quu = quu,
            qvv = qvv,
            quv = quv
        )
    end

    function convert_to_qkparams(params_in::UnivariateLatentModel{T, T2}) where {T <: Real, T2 <: Real}
        @unpack θz, Δt, σz, wc, wy0, wu, wv, wuu, wvv, wuv, qc, qu, qv, quu, qvv, quv = params_in
        N = 2
        M = 1
        μ = zeros(T, N)
        Φ = T[exp(-θz * Δt) 0.0; 1.0 0.0]
        Ω = T[sqrt(((σz^2)/(2.0 * θz))*(1-exp(-2*θz*Δt))) 0.0; 0.0 0.0]
        A = T[wc]
        B = T[wv wu]
        α = reshape(T[wy0], 1, 1)
        C = [T[wvv (wuv / 2.0); (wuv / 2.0) wuu]]
        return QKParams(N, M, μ, Φ, Ω, A, B, C, qc, qu, qv, quu, quv, qvv, α, Δt)
    end

    export QKParams, get_e, compute_e, compute_Λ, compute_μ̃, compute_Φ̃, compute_L1,
            compute_L2, compute_L3, compute_ν, compute_Ψ, compute_μᵘ, compute_Σ̃ᵘ, vech,
            issymmetric, compute_B̃, selection_matrix, duplication_matrix, spectral_radius,
            compute_H̃, compute_G̃, QKData, qkf_filter, compute_Σᵘ, compute_μ̃ᵘ, compute_Σ̃condZ,
            make_positive_definite, make_symmetric, compute_Σ̃condX, predict_Zₜₜ₋₁,
            compute_moments_I1I2, compute_mean_y, compute_var_y, compute_mean_y_aug,
            compute_var_y_aug, compute_wc, compute_wv, compute_wu, compute_wuu, compute_wvv,
            compute_wuv, compute_wy0, compute_qc, compute_qv, compute_qu, compute_qvv,
            compute_quu, compute_quv, UnivariateLatentModel, convert_to_qkparams,
            compute_loglik!, log_pdf_normal, compute_μ̃_old, compute_Σ̃_old, compute_Σ̃condZ_old,
            compute_Σ̃condX_old, compute_μ̃ᵘ_old, compute_Σ̃ᵘ_old, compute_Σᵘ_old, compute_μᵘ_old,
            compute_H̃_old, compute_G̃_old, compute_Ψ_old, compute_Λ_old, compute_Φ̃_old,
            compute_L1_old, compute_L2_old, compute_L3_old, compute_ν_old, compute_B̃_old,
            selection_matrix_old, duplication_matrix_old, spectral_radius_old, qkf_filter!,
            QKParams_old

end