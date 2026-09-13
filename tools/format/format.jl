using JuliaFormatter

root = normpath(joinpath(@__DIR__, "..", ".."))
for directory in ("src", "test", "ext", "examples/scientific", "bin", "tools/format")
    format(joinpath(root, directory))
end
