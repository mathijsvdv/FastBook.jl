include("../imports.jl")

@testset ExtendedTestSet "`ParamGroups`" begin
    model = Chain(Dense(3, 5), Dense(5, 3))
    paramgroups = ParamGroups(IndexGrouper([1, 2]), model)

    @test getgroup(paramgroups, model[1].weight) == 1
    @test getgroup(paramgroups, model[2].weight) == 2
    @test getgroup(paramgroups, rand(10)) === nothing
end
