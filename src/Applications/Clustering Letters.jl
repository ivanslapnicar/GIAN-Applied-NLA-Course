### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# ╔═╡ 5d365f5e-f1da-486e-ac74-07d9dfd4906c
begin
    import Pkg
    # activate a temporary environment
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(name="Arpack", version="0.5.3")
		Pkg.PackageSpec(name="Plots")
		Pkg.PackageSpec(name="Images")
		Pkg.PackageSpec(name="Clustering")
		Pkg.PackageSpec(name="Distances")
    ])
    using Images, Clustering, Plots, LinearAlgebra, SparseArrays, Distances, Arpack, Statistics
end

# ╔═╡ d49254d5-17bd-4213-ba92-f52855e4559f
md"""
# Clustering Letters

The letters are written in the file `files/Indore.jpg`. Some preprocessing is needed.
"""

# ╔═╡ ff88c198-2181-4aeb-a40a-21c1cd3f5c75
img₀=load("files/Indore.jpg")

# ╔═╡ 329d8fa3-da45-474e-9339-948b3ccb65a0
img=map(Gray,img₀)

# ╔═╡ fa13ec79-6745-483f-b34a-644fc929276d
# Extract the matrix
A=map(Float32,img)

# ╔═╡ d40532c8-81bd-4344-bc12-54a68897cdc5
sort(A[:])

# ╔═╡ b29e4044-6a57-44d0-bbaa-b36a711cb019
# Truncate small elements (0 is black, 1 is white)
sum(A.>0.2)

# ╔═╡ 640836a4-c8c3-41ef-b5d6-cd325004686d
A.*=(A.<=0.2)

# ╔═╡ 2ef827dc-acb6-40d2-a063-1e8c6d7c94e5
sum(A.>0), prod(size(A))

# ╔═╡ c1f80a30-0431-424e-88ef-3df933faabee
colorview(Gray,A)

# ╔═╡ 56f7d206-2ede-4259-9f30-0003979215f3
# Increase the contrast for clustering - turn into B/W
A₁=ones(size(A)).*map(Float64,A.>0)

# ╔═╡ dde9e719-3b23-44b5-87dd-cc3184c44b1d
colorview(Gray,rotl90(A₁))

# ╔═╡ cd237619-9597-4813-97d6-63edef5d8731
ind=findall(A₁.==1)

# ╔═╡ 663f1862-0d1a-4da7-a166-154faf30de1c
# Create array
X=transpose(getindex.(ind,[1 2]))

# ╔═╡ 8f6c1be8-890e-4586-afda-3fb830be11c0
# We are looking for 6 letters
out=kmeans(X,6)

# ╔═╡ feb2c116-1bf9-406f-a9ae-15117a7076ac
md"""
For plotting we use the function similar to the one from the notebook
[K-means Algorithm](L09+K-means+Algorithm.ipynb).
"""

# ╔═╡ 6f01a977-7a31-4e93-b472-83c644bf54d6
function plotKmeansresult(out::KmeansResult,X::AbstractArray)
    k=size(out.centers,2)
	scatter()
    # Clusters
    for j=1:k
        scatter!(X[1,findall(out.assignments.==j)],
            X[2,findall(out.assignments.==j)],label="Cluster $j",ms=1,markerstrokewidth = 0.1)
    end
    # Centers
    scatter!(out.centers[1,:],out.centers[2,:],markercolor=:red,label="Centers")
end

# ╔═╡ bc67be31-af26-43e8-90bf-a0f3f7afbf2d
plotKmeansresult(out,X)

# ╔═╡ 17b201ec-3390-4d3a-ac3a-616281e62090
md"""
Now we try the spectral $k$-partitioning:

* form the (normalized) Laplacian matrix, 
* compute eigenvectors corresponding to $k$ smallest eigenvalues, and
* cluster those vectors with $k$-means algorithm.

We need some functions from previous notebooks
"""

# ╔═╡ 93d04deb-2a5e-476e-8e4a-af8f7784a9db
begin
	function WeightMatrix(src::Array,dst::Array,weights::Array)
		n=nv(G)
		sparse([src;dst],[dst;src],[weights;weights],n,n)
	end
	
	Laplacian(W::AbstractMatrix)=Diagonal(vec(sum(W,dims=2)))-W
	
	function NormalizedLaplacian(L::AbstractMatrix)
		D=inv(√Diagonal(L))
		Symmetric(D*L*D)
	end
	
	function plotKpartresult(C::Vector,X::AbstractArray)
	    k=maximum(C)
		scatter()
	    # Clusters
	    for j=1:k
	        scatter!(X[1,findall(C.==j)],X[2,findall(C.==j)],label="Cluster $j",ms=1,markerstrokewidth = 0.1)
	    end
		scatter!()
	end
end

# ╔═╡ ed4e90c1-777f-48e6-9c8c-6f7ee70fc0aa
S=pairwise(SqEuclidean(),X,dims=2)

# ╔═╡ 6a772863-8cd9-4311-90be-8882f35831e2
β=1

# ╔═╡ f00ec83d-9062-4d41-857b-87683197a0e3
# Weight matrix
W=exp.(-β*S)-I

# ╔═╡ eae62b55-8154-4d54-9873-3acf41c2e79a
# Laplace matrix
L=Laplacian(W)

# ╔═╡ 6e0b884e-32f1-4350-bcce-1ef281649e90
begin
	# Cluster Laplacian
	k=6
	m=size(L,1)
	λₗ,Xₗ=eigs(L,nev=k,which=:SM, v0=ones(m))
end

# ╔═╡ d94a07cb-6f17-4769-a2c5-4882cd157ea5


# ╔═╡ 11964322-ce88-4269-9dfc-7bb5d9c14cd6
outₗ=kmeans(transpose(Xₗ),k)

# ╔═╡ 451686b5-463d-4303-9616-044b94f4f956
plotKpartresult(outₗ.assignments,X)

# ╔═╡ 2f728aaa-c7ea-4c9a-bd80-81907fa91502
# Normalized Laplacian matrix
Lₙ=NormalizedLaplacian(L)

# ╔═╡ c61d9088-cfde-4653-a255-4d8bbc0fad75
# Cluster normalized Laplacian
λₙ,Xₙ=eigs(Lₙ,nev=k,which=:SM, v0=ones(m))

# ╔═╡ ac916193-521f-4fbc-9a24-a646af3ed29a
Yₙ=inv(√Diagonal(L))*Xₙ

# ╔═╡ 85589cdb-919b-4a0c-92b0-ef5d9a20a84a
outₙ=kmeans(transpose(Yₙ),k)

# ╔═╡ 51b6917c-fcf5-401e-b6fe-c02a43a19851
plotKpartresult(outₙ.assignments,X)

# ╔═╡ 286738c6-8b38-41ce-8121-ca163c74a031
md"""
We now visualize rows of eigenvector matrices using Principal Component Analysis.
For details see the [PCA notebook](https://ivanslapnicar.github.io/GIAN-Applied-NLA-Course/L16%20Principal%20Component%20Analysis.jl.html).
"""

# ╔═╡ 806a63ed-4e5b-4208-bb13-58865b75c3be
# Fact 4 - PCA using SVD
function PCA(X::Array{T}, k::Int) where T
    μ=mean(X,dims=1)
    U=svd(X.-μ)
    U.U[:,1:k]*Diagonal(U.S[1:k])
end

# ╔═╡ 190729b5-b1cf-4435-9c2b-6724d2592c1f
function Plot(C::Vector, T::Array, k::Int)
    P=Array{Any}(undef,k)
	scatter(legend=false)
    for j=1:k
        P[j]=T[C.==j,1:3]
        scatter!(P[j][:,1],P[j][:,2],P[j][:,3],ms=2)
    end
	scatter!()
end

# ╔═╡ 720a23fc-a52d-4284-824f-89c9d66c3443
Tₗ=PCA(Xₗ,3)

# ╔═╡ 0a943910-0295-40c7-aad6-08d30f564b0d
scatter3d(Tₗ[:,1],Tₗ[:,2],Tₗ[:,3])

# ╔═╡ 327f17dd-88ec-4cc2-b680-20db76872b50
Tₙ=PCA(Yₙ,3)

# ╔═╡ ba06fe73-f259-4767-89d1-d66203550c50
scatter3d(Tₙ[:,1],Tₙ[:,2],Tₙ[:,3])

# ╔═╡ Cell order:
# ╟─d49254d5-17bd-4213-ba92-f52855e4559f
# ╠═5d365f5e-f1da-486e-ac74-07d9dfd4906c
# ╠═ff88c198-2181-4aeb-a40a-21c1cd3f5c75
# ╠═329d8fa3-da45-474e-9339-948b3ccb65a0
# ╠═fa13ec79-6745-483f-b34a-644fc929276d
# ╠═d40532c8-81bd-4344-bc12-54a68897cdc5
# ╠═b29e4044-6a57-44d0-bbaa-b36a711cb019
# ╠═640836a4-c8c3-41ef-b5d6-cd325004686d
# ╠═2ef827dc-acb6-40d2-a063-1e8c6d7c94e5
# ╠═c1f80a30-0431-424e-88ef-3df933faabee
# ╠═56f7d206-2ede-4259-9f30-0003979215f3
# ╠═dde9e719-3b23-44b5-87dd-cc3184c44b1d
# ╠═cd237619-9597-4813-97d6-63edef5d8731
# ╠═663f1862-0d1a-4da7-a166-154faf30de1c
# ╠═8f6c1be8-890e-4586-afda-3fb830be11c0
# ╟─feb2c116-1bf9-406f-a9ae-15117a7076ac
# ╠═6f01a977-7a31-4e93-b472-83c644bf54d6
# ╠═bc67be31-af26-43e8-90bf-a0f3f7afbf2d
# ╟─17b201ec-3390-4d3a-ac3a-616281e62090
# ╠═93d04deb-2a5e-476e-8e4a-af8f7784a9db
# ╠═ed4e90c1-777f-48e6-9c8c-6f7ee70fc0aa
# ╠═6a772863-8cd9-4311-90be-8882f35831e2
# ╠═f00ec83d-9062-4d41-857b-87683197a0e3
# ╠═eae62b55-8154-4d54-9873-3acf41c2e79a
# ╠═6e0b884e-32f1-4350-bcce-1ef281649e90
# ╠═d94a07cb-6f17-4769-a2c5-4882cd157ea5
# ╠═11964322-ce88-4269-9dfc-7bb5d9c14cd6
# ╠═451686b5-463d-4303-9616-044b94f4f956
# ╠═2f728aaa-c7ea-4c9a-bd80-81907fa91502
# ╠═c61d9088-cfde-4653-a255-4d8bbc0fad75
# ╠═ac916193-521f-4fbc-9a24-a646af3ed29a
# ╠═85589cdb-919b-4a0c-92b0-ef5d9a20a84a
# ╠═51b6917c-fcf5-401e-b6fe-c02a43a19851
# ╟─286738c6-8b38-41ce-8121-ca163c74a031
# ╠═806a63ed-4e5b-4208-bb13-58865b75c3be
# ╠═190729b5-b1cf-4435-9c2b-6724d2592c1f
# ╠═720a23fc-a52d-4284-824f-89c9d66c3443
# ╠═0a943910-0295-40c7-aad6-08d30f564b0d
# ╠═327f17dd-88ec-4cc2-b680-20db76872b50
# ╠═ba06fe73-f259-4767-89d1-d66203550c50
