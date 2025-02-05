using BenchmarkTools, DataFrames, CSV, Plots, LinearAlgebra, RCall, JSON, Random
using StatsPlots, Measures
using QuadraticKalman

# Define a structure to hold a benchmark scenario's parameters, including a unique scenario id
struct BenchmarkScenario
    id::Int       # unique scenario id
    T_bar::Int    # length of the observation sequence
    N::Int        # dimension of state
    M::Int        # dimension of measurement
end

# Function to simulate benchmark data for a given scenario with deterministic parameters.
function simulate_data(scn::BenchmarkScenario)
    id = scn.id
    T = scn.T_bar
    N = scn.N
    M = scn.M

    # Create a local RNG using a seed that depends deterministically on T, N, and M.
    # This ensures that for a given scenario, the simulation is reproducible.
    rng = MersenneTwister(1234 + 7*T + 13*N + 17*M)

    # Deterministic and stable model parameters:
    mu    = fill(0.1, N)                                # constant state drift
    Phi   = diagm(0 => fill(0.8, N))                    # state transition matrix with eigenvalues 0.8 (stable)
    Omega = diagm(0 => fill(0.1, N))                    # state noise scaling, small noise
    A     = zeros(M)                                    # zero measurement drift
    B     = fill(0.1, M, N)                             # measurement matrix with uniform small weights
    C     = [diagm(0 => fill(0.01, N)) for _ in 1:M]    # quadratic effects are mild; each channel gets a diagonal matrix
    D     = diagm(0 => fill(0.1, M))                    # measurement noise scaling 
    alpha = zeros(M, M)                                 # no dependency on previous measurements

    # Simulate state sequence X (size: N x T)
    X = zeros(N, T)
    X[:, 1] = zeros(N)                                # set initial state to zero (deterministic)
    for t in 1:(T - 1)
        shock = randn(rng, N)
        X[:, t+1] = mu + Phi * X[:, t] + Omega * shock
    end

    # Simulate observations Y (size: M x T)
    Y = zeros(M, T)
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

    params = (mu = mu, Phi = Phi, Omega = Omega, A = A, B = B, C = C, D = D, alpha = alpha,
             N = N, M = M)

    # Save simulated data in CSV
    CSV.write("data/scenario_$(id)_data.csv", DataFrame(Y, :auto))

    # Save model parameters in JSON format.
    open("data/scenario_$(id)_params.json", "w") do io
        write(io, JSON.json(params))
    end

    r_data = (X, Y, params, id)
    return r_data

end

# Function to benchmark the filter (qkf) for a given scenario.
function benchmark_filter(scn::BenchmarkScenario)
    X, Y, params, id = simulate_data(scn)
    # Construct the model using the parameters.
    # QKModel expects: (state dimension, measurement dimension, mu, Phi, Omega, a, B, C, D, alpha)

    model = QKModel(scn.N, scn.M, params.mu, params.Phi, params.Omega,
                    params.A, params.B, params.C, params.D, params.alpha)

    # QKData expects the measurements in rows, so we transpose Y (Y' has each row as an observation)
    data = QKData(Y)
    # Benchmark the qkf function.
    benchmark_result = @benchmark qkf_filter($data, $model) samples=100
    return id, benchmark_result
end


# Define the three (N, M) configurations and three data-lengths (T_bar)
nm_configs = [
    (N = 1, M = 1),
    (N = 2, M = 2),
    (N = 5, M = 5)
]
T_values = [10, 100, 1000]

# Create the Cartesian product of specifications before running benchmarks.
scenarios = BenchmarkScenario[]
id = 1
for config in nm_configs
    for T_val in T_values
        push!(scenarios, BenchmarkScenario(id, T_val, config.N, config.M))
        id += 1

    end
end

# Run the benchmarks and collect results in a single dataframe
results = DataFrame(id = Int[], T_bar = Int[], N = Int[], M = Int[],
                    median_time = Float64[], min_time = Float64[],
                    language = String[])
# Julia benchmarks
for scn in scenarios
    id, b = benchmark_filter(scn)
    med_time = median(b).time / 1e6  # convert from nanoseconds to milliseconds
    min_time = minimum(b).time / 1e6
    push!(results, (id, scn.T_bar, scn.N, scn.M, med_time, min_time, "Julia"))
    println("Completed Julia scenario: T_bar=$(scn.T_bar), N=$(scn.N), M=$(scn.M) => median time: $(med_time) ms")
end

# R benchmarks
for scenario_id in 1:length(scenarios)
    # If pwd child is "r", navigate to ".."
    if basename(pwd()) == "r"
        cd("..")
    end
    scn = scenarios[scenario_id]
    @rput scenario_id
    R"source('r/run_benchmarks.R')"
    @rget median_time
    @rget minimum_time

    push!(results, (scenario_id, scn.T_bar, scn.N, scn.M, median_time, minimum_time, "R"))
    println("Completed R scenario: T_bar=$(scn.T_bar), N=$(scn.N), M=$(scn.M) => median time: $median_time ms")
end


# Save benchmark results to a CSV file.
CSV.write("results/benchmark_results.csv", results)

# Load previously saved benchmark results
if isfile("results/benchmark_results.csv")
    results = CSV.read("results/benchmark_results.csv", DataFrame)
    println("Loaded existing benchmark results from CSV file")
else
    println("No existing benchmark results found")
end


# Create a new column 'scenario' in the DataFrame for labeling.
results.scenario = map((n, m) ->
    n == 1 && m == 1 ? "(N = 1, M = 1)" :
    n == 2 && m == 2 ? "(N = 2, M = 2)" :
    n == 5 && m == 5 ? "(N = 5, M = 5)" : "Other", results.N, results.M)

# Group the results for better visualization
grouped_results = groupby(results, [:scenario, :language])

ENV["GKSwstype"] = "nul"  # Needed for headless environments
gr(dpi=300)
# Increase left margin to prevent y-axis label cutoff
plt = plot(
    layout = (3,1),                # 3 rows, 1 column
    size = (800, 1200),            # taller figure size for more bottom margin
    legend = :topleft,
    leftmargin = 10mm,             # Added left margin to ensure y-labels are fully visible
)

# Colors and markers for consistent styling
colors = Dict("Julia" => :blue, "R" => :red)
markers = Dict("Julia" => :circle, "R" => :square)

# Create separate subplot for each N,M configuration
for (i, config) in enumerate(["(N = 1, M = 1)", "(N = 2, M = 2)", "(N = 5, M = 5)"])
    config_data = filter(row -> row.scenario == config, results)
    
    # Extract N and M values for the subplot title
    N = config_data[1, :N]
    M = config_data[1, :M]
    
    # Create subplot using plot! with explicit subplot index
    plot!(plt, subplot = i,
    xlabel = "Sequence Length (T)",
    ylabel = "Median Time (ms)", 
    title = "State Dimensionality: $N, Measurement Dimensionality: $M",
    xscale = :log10,
    yscale = :log10,
    xticks = 10.0 .^ (1:3),
    ylim = i == 1 ? (10.0^(-1), 10.0^4) :
           i == 2 ? (10.0^(-1), 10.0^4) :
           (10.0^(1), 10.0^6),
    minorticks = 5,
    grid = true,
    minorgrid = true,


)
    
    # Plot each implementation - add subplot=i to each plot! call
    for lang in ["Julia", "R"]
        lang_data = filter(row -> row.language == lang, config_data)
        plot!(plt, subplot = i,
            lang_data.T_bar,
            lang_data.median_time,
            label = lang,
            color = colors[lang],
            marker = markers[lang],
            markersize = 6,
            lw = 2
        )
    end
    
end

# Save without dpi parameter
savefig(plt, "results/scaling_comparison.png")

# For each config compute the ratio of the median time of R to Julia
for config in ["(N = 1, M = 1)", "(N = 2, M = 2)", "(N = 5, M = 5)"]
    config_data = filter(row -> row.scenario == config, results)
    R_median = mean(filter(row -> row.language == "R", config_data).median_time)
    Julia_median = mean(filter(row -> row.language == "Julia", config_data).median_time)
    println("Ratio of R to Julia median time for $config: $(R_median / Julia_median)")
end

