using Test
import QuadraticKalman as QK
using LinearAlgebra, Random
@testset "QKData Tests" begin
    @testset "Constructor & Validation" begin
        # Valid 2D
        Y = rand(3, 5) # M=3, T̄=4
        data = QK.QKData(Y)
        @test data.M == 3
        @test data.T_bar == 4
        @test length(data.Y) == 15  # total elements

        # Invalid: single time step
        Y_bad = rand(3, 1)
        @test_throws ArgumentError QK.QKData(Y_bad)

        # 1D
        Y1d = rand(10)
        data1d = QK.QKData(Y1d)
        @test data1d.M == 1
        @test data1d.T_bar == 9
    end

    @testset "get_measurement" begin
        # 2D
        data2 = QK.QKData(rand(4, 10))
        @test QK.get_measurement(data2, 1) == data2.Y[:, 1]
        @test QK.get_measurement(data2, 10) == data2.Y[:, 10]
        @test_throws ArgumentError QK.get_measurement(data2, 11)

        # 1D
        data1d = QK.QKData(rand(10))
        @test QK.get_measurement(data1d, 1) == data1d.Y[1]
        @test QK.get_measurement(data1d, 10) == data1d.Y[10]
        @test_throws ArgumentError QK.get_measurement(data1d, 11)
    end

    @testset "validate_data" begin
        # Valid cases
        Y2d = rand(3, 5)
        data2d = QK.QKData(Y2d)
        @test QK.validate_data(data2d) == true

        Y1d = rand(10) 
        data1d = QK.QKData(Y1d)
        @test QK.validate_data(data1d) == true

        # Invalid cases
        # Empty data
        @test_throws ArgumentError QK.validate_data(QK.QKData(Array{Float64}(undef,0)))
        
        # Single observation
        @test_throws ArgumentError QK.validate_data(QK.QKData(rand(3,1)))
        @test_throws ArgumentError QK.validate_data(QK.QKData([1.0]))

        # Non-finite values
        Y_inf = rand(3,5)
        Y_inf[2,3] = Inf
        @test_throws ArgumentError QK.validate_data(QK.QKData(Y_inf))

        Y_large = rand(3,5)
        Y_large[1,2] = 1e11  # Larger than threshold
        @test_throws ArgumentError QK.validate_data(QK.QKData(Y_large))
    end
end
