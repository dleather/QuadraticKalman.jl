t0 = 0.0
T = 0.25
u  = 0.0
v = 0.4
θz = 0.5
σz = 1.0

ξ

(1/(2 * θz)) * exp(2 * (-T + t0) * θz) * (-σy^2 + exp(2 * (T - t0) * θz) * σy^2 +
    (1 / ( 144 * θz^3)) * ((36 * θz * (exp(4 * T * θz) + exp(4 * t0 * θz) - 
    2 * exp(2 * (T + t0) * θz) * (1 + 2 * (T - t0)^2 * θz^2)) * ξ0^2 * σz^2) / 
    (-exp(4 * t0 * θz) + exp(2 * (T + t0) * θz)) - 2 * ξ1^2 * σz^4 * 
    (3 + exp(4 * (T - t0) * θz) + exp((T - t0) * θz) * (-4 - 6 * cosh((T - t0) * θz)) + 
    6 * cosh((T - t0) * θz))^2 * coth((T - t0) * θz)^2 + 2 * θz * ξ0 * ξ1 * 
    csch((T - t0) * θz) * (6 * exp(-2 * (T + 2 * t0) * θz) * 
    (exp(T * θz) - exp(t0 * θz))^3 * (exp(3 * t0 * θz) * u + exp(3 * T * θz) * v + 
    exp((2 * T + t0) * θz) * u * (-1 + 2 * T * θz - 2 * t0 * θz) - exp((T + 2 * t0) * θz) *
    v * (1 + 2 * T * θz - 2 * t0 * θz)) * σz^2 * coth((T - t0) * θz) + 4 * exp(-3 * T * θz - 
          5 * t0 * θz) * (8 * exp(2 * (3 * T + t0) * θz) * u - 
           exp((T + 7 * t0) * θz) * u + exp((7 T + t0) * θz) * v - 
           8 * exp(2 * (T + 3 * t0) * θz) * v + 
           8 * exp(4 * (T + t0) * θz) * (-u + v + 
              3 * (T - t0) * (u + v) * θz) - 
           exp((5 T + 3 * t0) * θz) * (9 u + 10 v + 
              6 * (T - t0) * (3 u + v) * θz) + 
           exp(3 * T * θz + 
             5 * t0 * θz) * (10 u + 9 v - 
              6 * (T - t0) * (u + 3 v) * θz)) * σz^2 csch((T - 
             t0) * θz) + 
        3 * exp(-3 * T * θz - 
          5 * t0 * θz) * (exp(T * θz) - exp(
           t0 * θz))^3 * (exp(2 * t0 * θz) * u^2 + 
           exp(2 * T * θz) * v^2 + 
           exp((T + t0) * θz) * (3 u^2 + 4 u v + 
              3 v^2)) * θz (exp(3 * t0 * θz) * u + 
           exp(3 * T * θz) * v + 
           exp((2 * T + t0) * θz)
             u * (-1 + 2 * T * θz - 2 * t0 * θz) - 
           exp((T + 2 * t0) * θz)
             v * (1 + 2 * T * θz - 2 * t0 * θz)) csch((T - 
             t0) * θz)^2) + 
     2 * ξ1^2 * (4 * exp(-2 * (T + 2 * t0) * θz) * (exp(T * θz) - exp(
           t0 * θz))^6 * σz^4 * coth((T - t0) * θz)^2 + 
        2 * exp(-2 * (T + 2 * t0) * θz) * (exp(6 T * θz) + exp(
           6 * t0 * θz) - 128 * exp(3 * (T + t0) * θz) - 
           9 * exp(2 * (2 * T + t0) * θz) * (-7 + 4 T * θz - 
              4 * t0 * θz) + 
           9 * exp(2 * (T + 2 * t0) * θz) * (7 + 4 T * θz - 
              4 * t0 * θz)) * σz^4 csch((T - 
             t0) * θz)^2 + 
        4 * exp(-3 * T * θz - 
          5 * t0 * θz) * (exp(T * θz) - exp(
           t0 * θz))^6 * (exp(2 * t0 * θz) * u^2 + 
           exp(2 * T * θz) * v^2 + 
           exp((T + t0) * θz) * (3 u^2 + 4 u v + 
              3 v^2)) * θz σz^2 * coth((T - 
             t0) * θz) csch((T - t0) * θz)^2 + 
        2 * exp(-4 * (2 * T + 3 * t0) * θz) * θz (exp(
            5 * (T + 3 * t0) * θz) * u^2 + 
           10 * exp(12 * T * θz + 8 * t0 * θz) * u v + 
           10 * exp(6 T * θz + 14 * t0 * θz) * u v + 
           exp((13 * T + 7 * t0) * θz) * v^2 + 
           exp((11 T + 9 * t0) * θz) * (-48 u v - 21 v^2 + 
              u^2 * (-83 + 36 T * θz - 36 * t0 * θz)) - 
           exp((7 T + 13 * t0) * θz) * (21 u^2 + 48 u v + 
              v^2 * (83 + 36 T * θz - 36 * t0 * θz)) + 
           2 * exp(8 T * θz + 
             12 * t0 * θz) * (32 u^2 + 96 v^2 + 
              3 u v * (41 + 12 * T * θz - 12 * t0 * θz)) - 
           exp((9 T + 11 * t0) * θz) * (416 u v + 
              9 u^2 * (17 + 4 T * θz - 4 * t0 * θz) + 
              9 v^2 * (17 - 4 T * θz + 4 * t0 * θz)) + 
           2 * exp(10 * (T + t0) * θz) * (96 u^2 + 32 v^2 + 
              3 u v * (41 - 12 * T * θz + 
                 12 * t0 * θz))) * σz^2 csch((T - 
             t0) * θz)^3 + 
        exp(-4 T * θz - 
          6 * t0 * θz) * (exp(T * θz) - exp(
           t0 * θz))^6 * (exp(2 * t0 * θz) * u^2 + 
           exp(2 * T * θz) * v^2 + 
           exp((T + t0) * θz) * (3 u^2 + 4 u v + 
              3 v^2))^2 * θz^2 csch((T - t0) * θz)^4)))

