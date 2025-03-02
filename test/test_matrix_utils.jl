using Test
import QuadraticKalman as QK
using LinearAlgebra, Random
@testset "Matrix Utils Tests" begin
    @testset "selection_matrix" begin
        S = QK.selection_matrix(3)
        @test size(S) == (6, 9)
        # Test extracting lower-triangular elements from e.g. 3×3
        A = [1. 2. 3.; 4. 5. 6.; 7. 8. 9.]
        tA = tril(A)[:]
        @test S * A[:] == [tA[i] for i in 1:length(tA) if tA[i] != 0]
    end

    @testset "make_positive_definite" begin
        A = [1.0 0.2; 0.2 1.0]
        # already PD
        A_pd = QK.make_positive_definite(A)
        @test isposdef(A_pd)

        # Slightly negative eigenvalue
        B = [ 1.0 0.99
              0.99 0.98 ]
        @test !isposdef(B)
        B_pd = QK.make_positive_definite(B)
        @test isposdef(B_pd)
    end

    @testset "spectral_radius" begin
        M = [0.5 0.0; 0.0 0.9]
        @test QK.spectral_radius(M) ≈ 0.9
    end
    
    @testset "make_positive_definite!" begin
        Random.seed!(123)  # For reproducibility
        
        # Test case 1: Already positive definite matrix
        A1 = [2.0 0.5; 0.5 3.0]
        n = size(A1, 1)
        work_matrix = similar(A1)
        eig_vals = Vector{Float64}(undef, n)
        eig_vecs = Matrix{Float64}(undef, n, n)
        
        result1 = copy(A1)
        QK.make_positive_definite!(result1, work_matrix, eig_vals, eig_vecs)
        
        # Check that eigenvalues are positive
        @test all(eigvals(result1) .> 0)
        # Check that the matrix is still approximately the same
        @test isapprox(result1, A1, atol=1e-6)
        # Check that the matrix is symmetric
        @test isapprox(result1, result1', atol=1e-10)
        
        # Test case 2: Matrix with negative eigenvalues
        A2 = [1.0 2.0; 2.0 -1.0]  # Has one negative eigenvalue
        result2 = copy(A2)
        QK.make_positive_definite!(result2, work_matrix, eig_vals, eig_vecs)
        
        # Check that all eigenvalues are now positive
        @test all(eigvals(result2) .> 0)
        # Check that the matrix is symmetric
        @test isapprox(result2, result2', atol=1e-10)
        
        # Test case 3: Non-symmetric matrix
        A3 = [1.0 2.0; 1.5 3.0]
        result3 = copy(A3)
        QK.make_positive_definite!(result3, work_matrix, eig_vals, eig_vecs)
        
        # Check that all eigenvalues are positive
        @test all(eigvals(result3) .> 0)
        # Check that the matrix is symmetric
        @test isapprox(result3, result3', atol=1e-10)
        
        # Test case 4: Matrix with very small eigenvalues
        A4 = [1.0 0.0; 0.0 1e-10]
        result4 = copy(A4)
        QK.make_positive_definite!(result4, work_matrix, eig_vals, eig_vecs, clamp_threshold=1e-6)
        
        # Check that all eigenvalues are above the threshold
        @test all(eigvals(result4) .> 1e-6)
        # Check that the matrix is symmetric
        @test isapprox(result4, result4', atol=1e-10)
    end
    
    @testset "make_positive_definite_wrapper" begin
        # Test the wrapper function with various matrices
        
        # Already positive definite
        A1 = [2.0 0.5; 0.5 3.0]
        result1 = QK.make_positive_definite_wrapper(A1)
        @test all(eigvals(result1) .> 0)
        @test isapprox(result1, A1, atol=1e-6)
        
        # Matrix with negative eigenvalues
        A2 = [1.0 2.0; 2.0 -1.0]
        result2 = QK.make_positive_definite_wrapper(A2)
        @test all(eigvals(result2) .> 0)
        
        # Custom threshold
        A4 = [1.0 0.0; 0.0 1e-10]
        result4 = QK.make_positive_definite_wrapper(A4, clamp_threshold=1e-5)
        @test all(eigvals(result4) .> 1e-5)
    end
    
    @testset "mul_diag!" begin
        # Test multiplication with diagonal matrix
        A = zeros(3, 3)
        B = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        v = [0.5, 1.0, 2.0]
        
        # Expected result: B * Diagonal(v)
        expected = B * Diagonal(v)
        
        # Test our implementation
        QK.mul_diag!(A, B, v)
        
        @test isapprox(A, expected)
        
        # Test with random matrices
        n = 5
        A_rand = zeros(n, n)
        B_rand = randn(n, n)
        v_rand = randn(n)
        
        expected_rand = B_rand * Diagonal(v_rand)
        QK.mul_diag!(A_rand, B_rand, v_rand)
        
        @test isapprox(A_rand, expected_rand)
    end
    
    @testset "mul_transpose!" begin
        # Test multiplication with transposed matrix
        A = zeros(3, 3)
        B = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
        C = [0.5 1.0 1.5; 2.0 2.5 3.0; 3.5 4.0 4.5]
        
        # Expected result: B * C'
        expected = B * C'
        
        # Test our implementation
        QK.mul_transpose!(A, B, C)
        
        @test isapprox(A, expected)
        
        # Test with random matrices
        n = 5
        A_rand = zeros(n, n)
        B_rand = randn(n, n)
        C_rand = randn(n, n)
        
        expected_rand = B_rand * C_rand'
        QK.mul_transpose!(A_rand, B_rand, C_rand)
        
        @test isapprox(A_rand, expected_rand)
    end
    
    @testset "d_eigen!" begin
        # Test eigendecomposition
        A = [2.0 1.0; 1.0 3.0]  # Symmetric matrix
        n = size(A, 1)
        vals = Vector{Float64}(undef, n)
        vecs = Matrix{Float64}(undef, n, n)
        
        # Get eigendecomposition
        QK.d_eigen!(A, vals, vecs, assume_symmetric=true)
        
        # Check that we can reconstruct the original matrix
        reconstructed = vecs * Diagonal(vals) * vecs'
        @test isapprox(reconstructed, A, atol=1e-6)
        
        # Check against standard eigen
        std_vals, std_vecs = LinearAlgebra.eigen(Symmetric(A))
        # Note: Eigenvalues should match, but eigenvectors might differ in sign
        @test isapprox(sort(vals), sort(std_vals), atol=1e-6)
    end
    
    @testset "smooth_max" begin
        # Test smooth maximum function
        
        # Positive value should remain approximately the same
        @test isapprox(QK.smooth_max(5.0), 5.0, atol=1e-6)
        
        # Negative value should be clamped to approximately zero
        @test QK.smooth_max(-5.0) > 0
        @test isapprox(QK.smooth_max(-5.0), 0.0, atol=1e-6)
        
        # Value near zero should be handled smoothly
        @test QK.smooth_max(1e-9) > 0
        
        # Test with custom threshold
        @test QK.smooth_max(0.01, threshold=0.1) > 0.01
        @test isapprox(QK.smooth_max(0.2, threshold=0.1), 0.2, atol=1e-6)
    end

    @testset "kronSigmaSigmaIndex tests" begin

        # A helper that does the "naive" indexing approach for comparison
        function naive_kronSigmaSigmaIndex(Sigma, i, N)
            return vec(kron(Sigma, Sigma))[i]
        end
    
        # 1) Test with a simple 2×2 Sigma
        @testset "N=2 small manual Sigma" begin
            Sigma = [1.0 2.0; 
                     3.0 4.0]
            N = size(Sigma,1)
            @test N == 2
            # Just compare all elements
            bigvec = vec(kron(Sigma, Sigma))  # length 16
            for i in 1:length(bigvec)
                expected = bigvec[i]
                got = QK.kronSigmaSigmaIndex(Sigma, i, N)
                @test isapprox(got, expected; rtol=1e-12, atol=1e-12)
            end
        end
    
        # 2) Random tests for N=2 or N=3:
        for N in (2, 3)
            @testset "N=$N random Sigma" begin
                Sigma = rand(N, N)
                bigvec = vec(kron(Sigma, Sigma))
                @test length(bigvec) == N^4
                for i in 1:N^4
                    @test isapprox(QK.kronSigmaSigmaIndex(Sigma, i, N),
                                   bigvec[i];
                                   rtol=1e-12, atol=1e-12)
                end
            end
        end
    
        # 3) Possibly a bigger test if you want:
        @testset "N=4 (bigger random test)" begin
            N = 4
            Sigma = rand(N, N)
            bigvec = vec(kron(Sigma, Sigma))
            for i in 1:N^4
                # Compare function vs. full Kron
                @test isapprox(QK.kronSigmaSigmaIndex(Sigma, i, N),
                               bigvec[i];
                               rtol=1e-12)
            end
        end
    
    end
    
    @testset "ensure_positive_definite! tests" begin

        @testset "Dimension edge cases" begin
    
            @testset "Empty matrix" begin
                A_empty = Matrix{Float64}(undef, 0, 0)
                returned = QK.ensure_positive_definite!(A_empty)
                @test returned === A_empty   # same reference
                @test size(A_empty) == (0, 0)
                # Shouldn't error
            end
    
            @testset "1x1 matrix" begin
                A_1x1 = reshape([ -10.0 ], 1, 1)
                returned = QK.ensure_positive_definite!(A_1x1; shift=0.1)
                @test returned === A_1x1
                @test A_1x1[1,1] == -10.0 + 0.1  # diagonal increment
            end
        end
    
        @testset "Symmetry test" begin
            # Make a random 5x5 with some random entries
            A = randn(5, 5)
            # Store a copy so we can check how it changes
            A_orig = copy(A)
            # shift for test
            shiftval = 0.05
    
            returned = QK.ensure_positive_definite!(A; shift=shiftval)
            @test returned === A   # must return same reference
    
            # Check symmetry
            @test isapprox(A, A', atol=1e-14)
    
            # Check diagonal increments
            for i in 1:5
                @test isapprox(A[i,i], 0.5*(A_orig[i,i] + A_orig[i,i]) + shiftval; atol=1e-14)
            end
        end
        
        @testset "Already PD matrix" begin
            # A random positive-definite matrix:
            # We'll build it by A = Q * D * Q', with D diagonal positive
            Q = randn(5,5)
            Q = qr(Q).Q  # Or use some orthonormal factor, e.g. from SVD
            D = Diagonal(range(0.1, 1.0, length=5))  # strictly positive diagonal
            A_pd = Q * D * Q'
            A_pd_copy = copy(A_pd)
    
            QK.ensure_positive_definite!(A_pd; shift=1e-5)
    
            # The matrix is still PD and was changed only by the shift on the diagonal
            for i in 1:5
                @test A_pd[i,i] ≈ A_pd_copy[i,i] + 1e-5  # diagonal increment
            end
            @test isapprox(A_pd, A_pd', atol=1e-14)
        end
        
        @testset "Compare with non-mutating version" begin
            # Create a random matrix
            A = [-1e-10 0.1; 0.1 2.0]
            A_copy = copy(A)
            
            # Apply both versions with the same shift
            shift_val = 1e-6
            A_mutated = QK.ensure_positive_definite!(copy(A); shift=shift_val)
            A_nonmutated = QK.ensure_positive_definite(A_copy; shift=shift_val)
            
            # Results should be identical
            @test isapprox(A_mutated, A_nonmutated, atol=1e-14)
            
            # Original matrix should be unchanged
            @test A_copy == A
            
            # Both results should be symmetric
            @test isapprox(A_mutated, A_mutated', atol=1e-14)
            @test isapprox(A_nonmutated, A_nonmutated', atol=1e-14)
            
            # --- Check how the eigenvalues changed ---
            # If the function is just A + shift_val*I, then each eigenvalue
            # is old_eigenval + shift_val. So let's check that the
            # difference is >= shift_val (up to small numerical tolerance).
        
            original_evals      = eigvals(A)
            mutated_evals       = eigvals(A_mutated)
            nonmutated_evals    = eigvals(A_nonmutated)
        
            # Sort to match eigenvalues in ascending order
            sort_orig    = sort(original_evals)
            sort_mutated = sort(mutated_evals)
            sort_nonmut  = sort(nonmutated_evals)
        
            # Check each new eigenvalue >= old eigenvalue + (shift_val - tiny epsilon)
            @test all(sort_mutated  .- sort_orig .≥ shift_val - 1e-10)
            @test all(sort_nonmut   .- sort_orig .≥ shift_val - 1e-10)
        end
    
        @testset "No unwanted allocations" begin
            # We can roughly check that ensure_positive_definite! is in-place
            # by measuring allocated bytes. If your function is truly
            # non-allocating (besides minimal ephemeral stack usage in BLAS, etc.),
            # this should be near 0.
            A_big = randn(100, 100)
            alloc_before = Base.gc_bytes()  # or use @allocated macro
            QK.ensure_positive_definite!(A_big; shift=1e-8)
            alloc_after = Base.gc_bytes()
    
            # This check is somewhat heuristic; you could also do:
            allocated = @allocated QK.ensure_positive_definite!(A_big; shift=1e-8)
            @test allocated < 1000  # e.g. some small threshold
        end
    
    end

end

