# QuadraticKalman.jl Development Guide

## Build/Test Commands
- Run all tests: `julia --project=. test/runtests.jl`
- Run single test: `julia --project=. -e 'using Pkg; Pkg.test("QuadraticKalman", test_args=["test_core_filter"])'`
- Test with code coverage: `julia --project=. -e 'using Pkg; Pkg.test("QuadraticKalman"; coverage=true)'`
- Quick build check: `julia --project=. -e 'using Pkg; Pkg.build("QuadraticKalman")'`

## Code Style Guidelines
- Follow standard Julia style: 4-space indentation, lowercase snake_case for functions/variables
- Type annotations: Use for function parameters and return values
- Docstrings: Required for all public functions (triple quotes)
- Error handling: Use `@assert` for internal checks, throw typed exceptions with descriptive messages
- Imports: Group standard library, then external packages, then local modules
- Performance: Avoid global variables, use type stability, minimize allocations
- Tests: Every public function should have test coverage
- Mutating functions: Follow Julia convention of ending with `!` (e.g., `filter!`)

## Matrix Math Conventions
- Use standard linear algebra notation where possible
- Document dimensions for all matrix parameters
- Prefer broadcasting over explicit loops when appropriate