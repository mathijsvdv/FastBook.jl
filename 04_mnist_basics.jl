### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ 80a3cfdc-89fd-485b-a6a0-1bd36d4404c1
let
	import Pkg
	Pkg.activate(".")
end

# ╔═╡ 2701ea9e-195e-11ec-3e0a-e9fe959701a0
begin
	using FastAI
	using FastAI.Datasets
	using MLDatasets
	using DataLoaders
end

# ╔═╡ 458ef9cf-bc7e-4eb7-9fa8-ec10e2a1e0c2


# ╔═╡ 9d936ab3-f730-44e8-8d35-841356016e75
begin
	data, blocks = loaddataset("imagenette2-160", (Image, Label))
	method = ImageClassificationSingle(blocks)
	learner = methodlearner(method, data, callbacks=[ToGPU()])
	fitonecycle!(learner, 10)
	plotpredictions(method, learner)
end

# ╔═╡ Cell order:
# ╠═80a3cfdc-89fd-485b-a6a0-1bd36d4404c1
# ╠═2701ea9e-195e-11ec-3e0a-e9fe959701a0
# ╠═458ef9cf-bc7e-4eb7-9fa8-ec10e2a1e0c2
# ╠═9d936ab3-f730-44e8-8d35-841356016e75
