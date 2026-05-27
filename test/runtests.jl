using REOB
using Test
using Random
using Statistics

@testset "REOB" begin
    include("test_statistics.jl")
    include("test_filters.jl")
    include("test_evaluation.jl")
    include("test_reo.jl")
    include("test_tsp.jl")
    include("test_vote.jl")
end
