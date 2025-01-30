using Documenter
using QuadraticKalman 

makedocs(
    sitename = "QuadraticKalman.jl",
    format = Documenter.HTML(),
    modules = [QuadraticKalman],
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md"
    ],
    # Add this to only document specific exports
    public = [
        # Types
        :QKData,
        :QKModel,
        :QKFOutput,
        :FilterOutput,
        :SmootherOutput,
        # Functions
        :qkf_filter,
        :qkf_filter!,
        :qkf_smoother,
        :qkf_smoother!,
        :qkf,
        :get_measurement
    ]
)

deploydocs(
    repo = "github.com/dleather/QuadraticKalman.jl.git",
    devbranch = "main"
)