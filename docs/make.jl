using Documenter
using QuadraticKalman

makedocs(
    sitename = "QuadraticKalman.jl",
    format = Documenter.HTML(),
    modules = [QuadraticKalman],
    pages = [
        "Home" => "index.md",
        "Plotting Guide" => "plots.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md"
    ],
    checkdocs = :none,
)

deploydocs(
    repo  = "https://github.com/dleather/QuadraticKalman.jl.git",
    devbranch = "main",
    deploy_config = Documenter.GitHubActions()
)