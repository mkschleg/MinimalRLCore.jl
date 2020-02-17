push!(LOAD_PATH,"../src/")

using Documenter, MinimalRLCore

makedocs(
    sitename="MinimalRLCore",
    modules = [MinimalRLCore],
    format = Documenter.HTML(prettyurls = false),
    pages = [
        "Home"=>"index.md",
        "Manual" => Any[
             "Environments" => "manual/environment.md"
             # "Agents" => "docs/agents.md"
             # "Learning" => "docs/learning.md"
             # "Feature Creators" => "docs/feature_creators.md"
             ],
         "Documentation" => Any[
             "Environments" => "docs/environments.md"
             "Agents" => "docs/agents.md"
             "GVF" => "docs/gvf.md"
             # "Learning" => "docs/learning.md"
             "Feature Constructor" => "docs/feature_creators.md"
             ]
    ]
)

deploydocs(
    repo = "github.com/mkschleg/MinimalRLCore.jl.git",
    devbranch = "master"
)
