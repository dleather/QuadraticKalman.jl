using Documenter
using QuadraticKalman

# Define output directory that won't conflict with Quarto
const BUILD_DIR = joinpath(@__DIR__, "build")

makedocs(
    sitename = "QuadraticKalman.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://dleather.github.io/QuadraticKalman.jl/stable/",
        assets = String[],
        analytics = "UA-XXXXXXXXX-X"
    ),
    modules = [QuadraticKalman],
    checkdocs = :exports,  # Only check for docstrings in exported functions
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
    build = BUILD_DIR
)

# Only deploy if we're on CI
if get(ENV, "CI", nothing) == "true"
    deploydocs(
        repo = "github.com/dleather/QuadraticKalman.jl.git",
        target = BUILD_DIR,
        push_preview = true
    )
end