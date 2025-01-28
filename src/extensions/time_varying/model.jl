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