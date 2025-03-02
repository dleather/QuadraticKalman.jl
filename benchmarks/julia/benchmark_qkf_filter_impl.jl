# benchmarks/julia/benchmark_qkf_filter_impl.jl

using BenchmarkTools, Random, LinearAlgebra, DataFrames, CSV, Plots, UnPack
using QuadraticKalman

# Define a structure to hold benchmark parameters
struct QKFBenchmarkScenario
    id::Int       # unique scenario id
    T_bar::Int    # length of the observation sequence
    N::Int        # dimension of state
    M::Int        # dimension of measurement
end

# Function to create a model and data for benchmarking
function create_benchmark_data(scn::QKFBenchmarkScenario)
    id = scn.id
    T = scn.T_bar
    N = scn.N
    M = scn.M

    # Create a local RNG with deterministic seed
    rng = MersenneTwister(1234 + 7*T + 13*N + 17*M)

    # Create model parameters
    mu = fill(0.1, N)                                # constant state drift
    Phi = diagm(0 => fill(0.8, N))                   # state transition matrix
    Omega = diagm(0 => fill(0.1, N))                 # state noise scaling
    A = zeros(M)                                      # zero measurement drift
    B = fill(0.1, M, N)                              # measurement matrix
    C = [diagm(0 => fill(0.01, N)) for _ in 1:M]     # quadratic effects
    D = diagm(0 => fill(0.1, M))                     # measurement noise scaling 
    alpha = zeros(M, M)                              # no dependency on previous measurements

    # Create the model
    model = QuadraticKalman.QKModel(N, M, mu, Phi, Omega, A, B, C, D, alpha)

    # Generate synthetic data
    Y = zeros(M, T)
    X = zeros(N, T)
    X[:, 1] = zeros(N)  # initial state
    
    for t in 1:(T-1)
        shock = randn(rng, N)
        X[:, t+1] = mu + Phi * X[:, t] + Omega * shock
    end
    
    for t in 1:T
        noise = randn(rng, M)
        xt = X[:, t]
        Y[:, t] = A + B * xt
        if t > 1
            Y[:, t] += alpha * Y[:, t-1]
        end
        for i in 1:M
            Y[i, t] += xt' * C[i] * xt
        end
        Y[:, t] += D * noise
    end
    
    # Create QKData
    data = QuadraticKalman.QKData(Y)
    
    return model, data
end

# Function to benchmark the low-level _qkf_filter_impl! function
function benchmark_qkf_filter_impl(scn::QKFBenchmarkScenario)
    model, data = create_benchmark_data(scn)
    
    # Get dimensions
    @unpack Y = data
    T̄ = size(Y, 2)                # Number of time steps
    P = size(model.aug_state.Phi_aug, 1)  # Augmented state dimension
    M = size(Y, 1)                # Measurement dimension
    
    # Preallocate all arrays needed for _qkf_filter_impl!
    Z_tt = zeros(Float64, P, T̄)
    P_tt = zeros(Float64, P, P, T̄)
    Z_ttm1 = zeros(Float64, P, T̄ - 1)
    P_ttm1 = zeros(Float64, P, P, T̄ - 1)
    Y_ttm1 = zeros(Float64, M, T̄ - 1)
    M_ttm1 = zeros(Float64, M, M, T̄ - 1)
    K_t = zeros(Float64, P, M, T̄ - 1)
    ll_t = zeros(Float64, T̄ - 1)
    Sigma_ttm1 = zeros(Float64, P, P, T̄ - 1)
    tmpP = zeros(Float64, P, P)
    tmpKM = zeros(Float64, P, P)
    tmpKMK = zeros(Float64, P, P)
    tmpB = zeros(Float64, M, P)
    
    # Benchmark the low-level implementation
    benchmark_result = @benchmark QuadraticKalman.qkf_filter!(
        $data, $model
    ) samples=100 evals=1
    
    # Also benchmark the standard qkf_filter! for comparison
    standard_benchmark = @benchmark QuadraticKalman.qkf_filter($data, $model) samples=100 evals=1
    
    # Return the benchmark results
    return scn.id, benchmark_result, standard_benchmark
end

# Define the benchmark scenarios
nm_configs = [
    (N = 1, M = 1),
    (N = 2, M = 2),
    (N = 5, M = 5)
    #(N = 10, M = 5)  # Added a larger state dimension case
]
T_values = [10, 100, 1000]#, 5000]  # Added a larger time series case

# Create the scenarios
scenarios = QKFBenchmarkScenario[]
id = 1
for config in nm_configs
    for T_val in T_values
        push!(scenarios, QKFBenchmarkScenario(id, T_val, config.N, config.M))
        id += 1
    end
end

# Run the benchmarks and collect results
results = DataFrame(
    id = Int[], 
    T_bar = Int[], 
    N = Int[], 
    M = Int[],
    impl_median_time_ms = Float64[], 
    impl_min_time_ms = Float64[],
    impl_allocs = Int[],
    impl_memory_bytes = Int[],
    std_median_time_ms = Float64[], 
    std_min_time_ms = Float64[],
    std_allocs = Int[],
    std_memory_bytes = Int[]
)

for scn in scenarios
    id, impl_bench, std_bench = benchmark_qkf_filter_impl(scn)
    
    # Extract metrics for the low-level implementation
    impl_med_time = median(impl_bench).time / 1e6  # convert from ns to ms
    impl_min_time = minimum(impl_bench).time / 1e6
    impl_allocs = median(impl_bench).allocs
    impl_memory = median(impl_bench).memory
    
    # Extract metrics for the standard implementation
    std_med_time = median(std_bench).time / 1e6
    std_min_time = minimum(std_bench).time / 1e6
    std_allocs = median(std_bench).allocs
    std_memory = median(std_bench).memory
    
    push!(results, (
        id, 
        scn.T_bar, 
        scn.N, 
        scn.M, 
        impl_med_time, 
        impl_min_time,
        impl_allocs,
        impl_memory,
        std_med_time, 
        std_min_time,
        std_allocs,
        std_memory
    ))
    
    println("Completed scenario: T_bar=$(scn.T_bar), N=$(scn.N), M=$(scn.M)")
    println("  qkf_filter!: median time: $(impl_med_time) ms, allocs: $(impl_allocs)")
    println("  qkf_filter: median time: $(std_med_time) ms, allocs: $(std_allocs)")
end

# Save benchmark results to a CSV file
CSV.write("benchmarks/results/qkf_filter_impl_benchmark_results.csv", results)

# Create a new column 'scenario' in the DataFrame for labeling
results.scenario = map((n, m) ->
    n == 1 && m == 1 ? "(N = 1, M = 1)" :
    n == 2 && m == 2 ? "(N = 2, M = 2)" :
    n == 5 && m == 5 ? "(N = 5, M = 5)" :
    n == 10 && m == 5 ? "(N = 10, M = 5)" : "Other", 
    results.N, results.M)

# Set up plotting environment
ENV["GKSwstype"] = "nul"  # For headless environments
gr(dpi=300)

# Plot execution time comparison
plt_time = plot(
    layout = (4,1),
    size = (800, 1200),
    legend = :topleft,
    leftmargin = 10,
)

# Plot allocation comparison
plt_alloc = plot(
    layout = (4,1),
    size = (800, 1200),
    legend = :topleft,
    leftmargin = 10,
)

# Colors and markers for consistent styling
colors = Dict("_qkf_filter_impl!" => :blue, "qkf_filter!" => :red)
markers = Dict("_qkf_filter_impl!" => :circle, "qkf_filter!" => :square)

# Create separate subplot for each N,M configuration
for (i, config) in enumerate(["(N = 1, M = 1)", "(N = 2, M = 2)", "(N = 5, M = 5)"])
    config_data = filter(row -> row.scenario == config, results)
    
    # Extract N and M values for the subplot title
    N = config_data[1, :N]
    M = config_data[1, :M]
    
    # Create time subplot
    plot!(plt_time, subplot = i,
        xlabel = "Sequence Length (T)",
        ylabel = "Median Time (ms)", 
        title = "State Dim: $N, Measurement Dim: $M",
        xscale = :log10,
        yscale = :log10,
        xticks = 10.0 .^ (1:4),
        grid = true,
        minorgrid = true,
    )
    
    # Plot execution time for each implementation
    plot!(plt_time, subplot = i,
        config_data.T_bar,
        config_data.impl_median_time_ms,
        label = "_qkf_filter_impl!",
        color = colors["_qkf_filter_impl!"],
        marker = markers["_qkf_filter_impl!"],
        markersize = 6,
        lw = 2
    )
    
    plot!(plt_time, subplot = i,
        config_data.T_bar,
        config_data.std_median_time_ms,
        label = "qkf_filter!",
        color = colors["qkf_filter!"],
        marker = markers["qkf_filter!"],
        markersize = 6,
        lw = 2
    )
    
    # Create allocation subplot
    plot!(plt_alloc, subplot = i,
        xlabel = "Sequence Length (T)",
        ylabel = "Allocations", 
        title = "State Dim: $N, Measurement Dim: $M",
        xscale = :log10,
        yscale = :log10,
        xticks = 10.0 .^ (1:4),
        grid = true,
        minorgrid = true,
    )
    
    # Plot allocations for each implementation
    plot!(plt_alloc, subplot = i,
        config_data.T_bar,
        config_data.impl_allocs,
        label = "_qkf_filter_impl!",
        color = colors["_qkf_filter_impl!"],
        marker = markers["_qkf_filter_impl!"],
        markersize = 6,
        lw = 2
    )
    
    plot!(plt_alloc, subplot = i,
        config_data.T_bar,
        config_data.std_allocs,
        label = "qkf_filter!",
        color = colors["qkf_filter!"],
        marker = markers["qkf_filter!"],
        markersize = 6,
        lw = 2
    )
end

# Save the plots
savefig(plt_time, "results/qkf_filter_impl_time_comparison.png")
savefig(plt_alloc, "results/qkf_filter_impl_alloc_comparison.png")

# Print summary statistics
println("\nSummary Statistics:")
println("==================")

# For each config compute the ratio of qkf_filter! to _qkf_filter_impl!
for config in ["(N = 1, M = 1)", "(N = 2, M = 2)", "(N = 5, M = 5)", "(N = 10, M = 5)"]
    config_data = filter(row -> row.scenario == config, results)
    time_ratio = mean(config_data.std_median_time_ms ./ config_data.impl_median_time_ms)
    alloc_ratio = mean(config_data.std_allocs ./ config_data.impl_allocs)
    
    println("$config:")
    println("  Time ratio (qkf_filter! / _qkf_filter_impl!): $(round(time_ratio, digits=2))")
    println("  Allocation ratio (qkf_filter! / _qkf_filter_impl!): $(round(alloc_ratio, digits=2))")
end


scn = QKFBenchmarkScenario(1, 1000, 1, 1)
model, data = create_benchmark_data(scn)
    
# Get dimensions
@unpack Y = data
T̄ = size(Y, 2)                # Number of time steps
P = size(model.aug_state.Phi_aug, 1)  # Augmented state dimension
M = size(Y, 1)                # Measurement dimension

# Preallocate all arrays needed for _qkf_filter_impl!
Z_tt = zeros(Float64, P, T̄)
P_tt = zeros(Float64, P, P, T̄)
Z_ttm1 = zeros(Float64, P, T̄ - 1)
P_ttm1 = zeros(Float64, P, P, T̄ - 1)
Y_ttm1 = zeros(Float64, M, T̄ - 1)
M_ttm1 = zeros(Float64, M, M, T̄ - 1)
K_t = zeros(Float64, P, M, T̄ - 1)
ll_t = zeros(Float64, T̄ - 1)
Sigma_ttm1 = zeros(Float64, P, P, T̄ - 1)
tmpP = zeros(Float64, P, P)
tmpKM = zeros(Float64, P, P)
tmpKMK = zeros(Float64, P, P)
tmpKV = zeros(Float64, P, M)
tmpB = zeros(Float64, M, P)

# Benchmark the low-level implementation
benchmark_result = @benchmark _qkf_filter_impl!(
    $Z_tt, $P_tt, $Z_ttm1, $P_ttm1, 
    $Y_ttm1, $M_ttm1, $K_t, $ll_t,
    $Sigma_ttm1, $tmpP, $tmpKM, $tmpKMK, $tmpKV, $tmpB,
    $data, $model
) samples=100 evals=1


@profview_allocs QuadraticKalman._qkf_filter_impl!(
    Z_tt, P_tt, Z_ttm1, P_ttm1, 
    Y_ttm1, M_ttm1, K_t, ll_t,
    Sigma_ttm1, tmpP, tmpKM, tmpKMK, tmpKV, tmpB,
    data, model
)

@benchmark result = QuadraticKalman.qkf_filter!(
    data, model
)

@benchmark result = QuadraticKalman.qkf_filter(
    data, model
)


# Also benchmark the standard qkf_filter! for comparison
standard_benchmark = @benchmark qkf_filter!($data, $model) samples=100 evals=1

# Return the benchmark results
return scn.id, benchmark_result, standard_benchmark