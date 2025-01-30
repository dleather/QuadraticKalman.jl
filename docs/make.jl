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
    checkdocs = :none,
)

deploydocs(
    doc   = doc,
    repo  = "https://github.com/dleather/QuadraticKalman.jl.git",
    devbranch = "main",
    # Possibly: deploy_config = Documenter.GitHubActions()
)