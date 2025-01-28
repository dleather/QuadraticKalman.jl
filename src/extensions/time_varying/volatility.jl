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