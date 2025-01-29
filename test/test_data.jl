using Test
using QuadraticKalman

@testset "QKData Tests" begin
    @testset "Constructor & Validation" begin
        # Valid 2D
        Y = rand(3, 5) # M=3, T̄=4
        data = QKData(Y)
        @test data.M == 3
        @test data.T̄ == 4
        @test length(data.Y) == 15  # total elements

        # Invalid: single time step
        Y_bad = rand(3, 1)
        @test_throws ArgumentError QKData(Y_bad)

        # 1D
        Y1d = rand(10)
        data1d = QKData(Y1d)
        @test data1d.M == 1
        @test data1d.T̄ == 9
    end

    @testset "get_measurement" begin
        # 2D
        data2 = QKData(rand(4, 10))
        @test get_measurement(data2, 1) == data2.Y[:, 1]
        @test get_measurement(data2, 10) == data2.Y[:, 10]
        @test_throws ArgumentError get_measurement(data2, 11)

        # 1D
        data1d = QKData(rand(10))
        @test get_measurement(data1d, 1) == data1d.Y[1]
        @test get_measurement(data1d, 10) == data1d.Y[10]
        @test_throws ArgumentError get_measurement(data1d, 11)
    end

    @testset "validate_data" begin
        # Valid cases
        Y2d = rand(3, 5)
        data2d = QKData(Y2d)
        @test QuadraticKalman.validate_data(data2d) == true

        Y1d = rand(10) 
        data1d = QKData(Y1d)
        @test QuadraticKalman.validate_data(data1d) == true

        # Invalid cases
        # Empty data
        @test_throws ArgumentError QuadraticKalman.validate_data(QKData(Array{Float64}(undef,0)))
        
        # Single observation
        @test_throws ArgumentError QuadraticKalman.validate_data(QKData(rand(3,1)))
        @test_throws ArgumentError QuadraticKalman.validate_data(QKData([1.0]))

        # Non-finite values
        Y_inf = rand(3,5)
        Y_inf[2,3] = Inf
        @test_throws ArgumentError QuadraticKalman.validate_data(QKData(Y_inf))

        Y_large = rand(3,5)
        Y_large[1,2] = 1e11  # Larger than threshold
        @test_throws ArgumentError QuadraticKalman.validate_data(QKData(Y_large))
    end
end
