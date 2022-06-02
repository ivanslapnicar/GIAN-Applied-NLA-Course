### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 41e9ad5f-0777-48aa-a330-603622933518
# Pkg.add("MatrixMarket")
using MatrixMarket, GZip

# ╔═╡ ff0d8b46-8f27-4cb1-a76f-3fb71b3635d3
using LinearAlgebra

# ╔═╡ a5982505-2f86-4549-bc93-c8b593fef6e2
begin
	using Plots
	using SparseArrays
	plotly()
end

# ╔═╡ 4343f2c3-d1a4-4ed6-8d16-141fc8756497
using MatrixDepot

# ╔═╡ acf50b8c-9c00-4ba0-9cc8-bfeba8ede017
include("ModuleB.jl")

# ╔═╡ 551b8fba-1536-4a45-bcb3-560c6abdfd43
md"""
# Solutions 2 - Examples in Eigenvalue Decomposition
"""

# ╔═╡ a5b94721-1cb6-44f9-a91f-d1bac376cf6b
md"""
## Assignment 1
"""

# ╔═╡ 5bb3aa4d-470b-4b46-bd01-1a91e64b8cf3
# using Pkg

# ╔═╡ 17805b1d-411d-4501-ac32-2e2ea570ff21
# Pkg.add("MatrixMarket")

# ╔═╡ 7caac058-2bb4-428a-a688-af69d4151cb6
# Pkg.add("GZip")

# ╔═╡ 14dcd968-879c-4a22-9a48-178f14f6c705
# Pkg.add("MatrixDepot")

# ╔═╡ faf36674-f06a-4cb1-abff-848876feb126
varinfo(MatrixMarket)

# ╔═╡ 184958f1-c84a-4ce9-8e96-e4bb6604e554
function gunzip(fname)
    destname, ext = splitext(fname)
    if ext != ".gz"
        error("gunzip: $fname: unknown suffix -- ignored")
    end
    open(destname, "w") do f
        GZip.open(fname) do g
            write(f, read(g, String))
        end
    end
    destname
end

# ╔═╡ 98e3d891-f012-49f9-b30e-a0897c51eec7
#Download and parse master list of matrices
if !isfile("matrices.html")
    download("math.nist.gov/MatrixMarket/matrices.html", "matrices.html")
end

# ╔═╡ 4e58fa54-dfe1-473a-b7e8-2cfc8290c486
begin
	matrixmarketdata = Any[]
	open("matrices.html") do f
	   for line in readlines(f)
	       if occursin("""<A HREF="/MatrixMarket/data/""", line)
	           collectionname, setname, matrixname = split(split(line, '"')[2], '/')[4:6]
	           matrixname = split(matrixname, '.')[1]
	           push!(matrixmarketdata, (collectionname, setname, matrixname) )
	       end
	   end
	end
end

# ╔═╡ 7f5508a7-176d-4baa-8d3b-2f99aa4dcde3
begin
	#Download one matrix at random plus some specifically chosen ones.
	n = rand(1:length(matrixmarketdata))
	testmatrices = [ ("NEP", "mhd", "mhd1280b")
	               , ("Harwell-Boeing", "acoust", "young4c")
	               , ("Harwell-Boeing", "platz", "plsk1919")
	               , matrixmarketdata[n]
	               ]
	for (collectionname, setname, matrixname) in testmatrices
	    fn = string(collectionname, '_', setname, '_', matrixname)
	    mtxfname = string(fn, ".mtx")
	    if !isfile(mtxfname)
	        url = "ftp://math.nist.gov/pub/MatrixMarket2/$collectionname/$setname/$matrixname.mtx.gz"
	        gzfname = string(fn, ".mtx.gz")
	        try
	            download(url, gzfname)
	        catch
	            continue
	        end
	        gunzip(gzfname)
	        rm(gzfname)
	    end
	end
end

# ╔═╡ 5cdd50ad-b60b-4d17-b212-d2f820077deb
matrixmarketdata

# ╔═╡ 367f59f4-9001-466c-9c60-c14367bd41a4
readdir()

# ╔═╡ edcdd53d-2421-4266-9925-381fb699e3d2
A=mmread("Harwell-Boeing_acoust_young4c.mtx")

# ╔═╡ 87f08a34-c47b-41ff-8379-b3b9381e5452
size(A)

# ╔═╡ a52467d5-6603-4e84-94ec-88eabcf88604
issymmetric(A)

# ╔═╡ 4c7cd910-1333-4845-8b5a-67d07ddb95ac
cond(Matrix(A))

# ╔═╡ a8ef89d9-74d6-463b-b116-1ed28bd7c723
begin
	# Make a nicer spy function for regular matrices
	import Plots.spy
	spy(A::Matrix)=heatmap(A,yflip=true,color=:RdBu,aspectratio=1,clim=(-1,1.0)) 
end

# ╔═╡ d23ce0b0-ad07-4c52-b534-07db5031724b
begin
	# Plot of a small random matrix, for illustration
	B=rand(10,10)
	spy(sparse(B))
end

# ╔═╡ 84c2abd2-7dd8-4846-b63d-2b59b8881f9c
spy(B)

# ╔═╡ 3cd03889-a274-4394-8062-7357f232ebbf
spy(abs.(A))

# ╔═╡ 87f8f693-36e2-43c4-bf5b-293a371e46c4
A[1,1]

# ╔═╡ 2e953e14-b2fc-4501-b6f1-de248071dbaf
md"""
## Assignment 2
"""

# ╔═╡ af344540-75bd-45b6-b0ab-65f90b5c0d4f
varinfo(MatrixDepot)

# ╔═╡ 73bf9275-3926-412f-aa03-0d9c4d41d6f0
listnames(:eigen)

# ╔═╡ c5772572-adfe-4fa6-bbd1-ad047c02d5f5
mdlist(:sparse & :symmetric)

# ╔═╡ c7ec0ac0-30a8-46a0-83d0-d43aa99ec236
matrixdepot("fiedler")

# ╔═╡ d536e67f-9eda-4b06-a6f0-a9294ccc9495
MatrixDepot.download()

# ╔═╡ 1f0402e9-1ec7-4cb8-ad2d-8178f55ce6de
A₁=MatrixDepot.islocal("fiedler")

# ╔═╡ ff4f60b4-705c-4811-a61a-092a74863624
mdopen("Gset/G24")

# ╔═╡ 250ec8cc-9f12-4bfa-b636-9c04a0a20f34
cond(A₁)

# ╔═╡ b40cb06e-36da-4587-8dfb-da926e839aff
ModuleB.myPowerMethod(A₁,1e-10)

# ╔═╡ b6b6efc1-3d4d-4c23-8ec7-dfb9cf874780
eigvals(A₁)

# ╔═╡ 69548701-43ac-4c78-b3fb-2079e98cd924
MatrixDepot.mdlist(:illcond & :symmetric)

# ╔═╡ Cell order:
# ╟─551b8fba-1536-4a45-bcb3-560c6abdfd43
# ╟─a5b94721-1cb6-44f9-a91f-d1bac376cf6b
# ╠═5bb3aa4d-470b-4b46-bd01-1a91e64b8cf3
# ╠═17805b1d-411d-4501-ac32-2e2ea570ff21
# ╠═7caac058-2bb4-428a-a688-af69d4151cb6
# ╠═14dcd968-879c-4a22-9a48-178f14f6c705
# ╠═41e9ad5f-0777-48aa-a330-603622933518
# ╠═faf36674-f06a-4cb1-abff-848876feb126
# ╠═184958f1-c84a-4ce9-8e96-e4bb6604e554
# ╠═98e3d891-f012-49f9-b30e-a0897c51eec7
# ╠═4e58fa54-dfe1-473a-b7e8-2cfc8290c486
# ╠═7f5508a7-176d-4baa-8d3b-2f99aa4dcde3
# ╠═5cdd50ad-b60b-4d17-b212-d2f820077deb
# ╠═367f59f4-9001-466c-9c60-c14367bd41a4
# ╠═edcdd53d-2421-4266-9925-381fb699e3d2
# ╠═87f08a34-c47b-41ff-8379-b3b9381e5452
# ╠═ff0d8b46-8f27-4cb1-a76f-3fb71b3635d3
# ╠═a52467d5-6603-4e84-94ec-88eabcf88604
# ╠═4c7cd910-1333-4845-8b5a-67d07ddb95ac
# ╠═a5982505-2f86-4549-bc93-c8b593fef6e2
# ╠═a8ef89d9-74d6-463b-b116-1ed28bd7c723
# ╠═d23ce0b0-ad07-4c52-b534-07db5031724b
# ╠═84c2abd2-7dd8-4846-b63d-2b59b8881f9c
# ╠═3cd03889-a274-4394-8062-7357f232ebbf
# ╠═87f8f693-36e2-43c4-bf5b-293a371e46c4
# ╟─2e953e14-b2fc-4501-b6f1-de248071dbaf
# ╠═4343f2c3-d1a4-4ed6-8d16-141fc8756497
# ╠═af344540-75bd-45b6-b0ab-65f90b5c0d4f
# ╠═73bf9275-3926-412f-aa03-0d9c4d41d6f0
# ╠═c5772572-adfe-4fa6-bbd1-ad047c02d5f5
# ╠═c7ec0ac0-30a8-46a0-83d0-d43aa99ec236
# ╠═d536e67f-9eda-4b06-a6f0-a9294ccc9495
# ╠═1f0402e9-1ec7-4cb8-ad2d-8178f55ce6de
# ╠═ff4f60b4-705c-4811-a61a-092a74863624
# ╠═250ec8cc-9f12-4bfa-b636-9c04a0a20f34
# ╠═acf50b8c-9c00-4ba0-9cc8-bfeba8ede017
# ╠═b40cb06e-36da-4587-8dfb-da926e839aff
# ╠═b6b6efc1-3d4d-4c23-8ec7-dfb9cf874780
# ╠═69548701-43ac-4c78-b3fb-2079e98cd924
