# make.jl
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
    ]
    # Removed invalid `public` parameter
)

deploydocs(
    repo = "https://github.com/dleather/QuadraticKalman.jl.git", # Added https://
    devbranch = "main"
)