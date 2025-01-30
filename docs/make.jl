using Documenter
using QuadraticKalman

doc = makedocs(
    sitename = "QuadraticKalman.jl",
    format = Documenter.HTML(),
    modules = [QuadraticKalman],
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
        "Examples" => "examples.md"
    ],
    checkdocs = :none,
)

deploydocs(
    doc   = doc,
    repo  = "https://github.com/dleather/QuadraticKalman.jl.git",
    devbranch = "main",
    deploy_config = Documenter.GitHubActions()
)