### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ c7e41465-41ad-4f11-baec-bfaae9edbf59
begin
	using PlutoUI
	PlutoUI.TableOfContents(aside=true)
end

# ╔═╡ d0233ece-1f69-11eb-3da1-43b8ddc9fbe1
begin
	# Necessary packages
	using LinearAlgebra
	using Random
	using Statistics
	using Clustering
	using Plots
end

# ╔═╡ 00d72ad7-ea80-4954-8c75-590d4e9ccdc4
md"""
# K-means Algorithm


Data clustering is one of the main mathematical applications variety of algorithms have been developed to tackle the problem. K-means is one of the basic algorithms for data clustering.

__Prerequisites__

The reader should be familiar with basic linear algebra. 
 
__Competences__

The reader should be able to recognise applications where K-means algorithm can be efficiently used and use it.

__Credits.__ The notebook was initially derived from M.Sc. Thesis of Ivančica Mirošević. 
"""

# ╔═╡ 235e1901-eed3-4934-9214-ffdd82cf21d6
md"""
## Definitions

__Data clustering problem__ is the following: partition the given set of $m$ objects of the same type into $k$ subsets according to some criterion. Additional request may be to find the optimal $k$.

__K-means clustering problem__ is the following: partition the set  $X=\{x_{1},x_{2},\cdots ,x_{m}\}$ , where $x_{i}\in\mathbb{R}^{n}$, into $k$ _clusters_ $\pi=\{C_{1},C_{2},...,C_{k}\}$ such that

$$
J(\pi)=\sum_{i=1}^{k}\sum_{x\in
C_{i}}\| x-c_{i}\|_{2}^{2} \to \min$$

over all possible partitions. Here $c_{i}=\displaystyle\frac{1}{|C_{i}|}\sum_{x\in C_{i}} x$ is the mean of points in $C_i$ and $|C_i|$ is the cardinality of $C_i$.
"""

# ╔═╡ 26640b90-1f69-11eb-029e-4dc3a825f639
md"
## K-means clustering algorithm

1. __Initialization__: Choose initial set of $k$ means $\{c_1,\ldots,c_k\}$ (for example, by choosing randomly $k$ points from $X$).
2. __Assignment step__: Assign each point $x$ to one nearest mean $c_i$.
3. __Update step__: Compute the new means.
4. __Iteration__: Repeat Steps 2 and 3 until the assignment no longer changes.
"

# ╔═╡ 48914840-1f69-11eb-30a9-3943767bc261
md"
## First Variation clustering algorithm

A __first variation__ of a partition $\pi=\{C_1,\ldots,C_k\}$ is 
a partition $\pi^{\prime}=\{C_{1}^{\prime},\cdots ,C_{k}^{\prime }\}$ 
obtained by moving a single point $x$ from a cluster  $C_{i}$ to a cluster $C_{j}$. Notice that $\pi$ is a first variation of itself.

A __next partition__ of the partition $\pi$ is a partition 
$\mathop{\mathrm{next}}(\pi)=\mathop{\mathrm{arg min}}\limits_{\pi^{\prime}} J(\pi^{\prime})$.

We have the following algorithm:

1. Choose initial partition $\pi$.
2. Compute $\mathop{\mathrm{next}}(\pi)$
3. If $J(\mathop{\mathrm{next}}(\pi))<J(\pi)$, set $\pi=\mathop{\mathrm{next}}(\pi)$ and go to Step 2
4. Stop.
"

# ╔═╡ 2b76953c-edf2-41cf-9d75-7efb2d6a58f1
md"""
## Facts

1. The k-means clustering problem is NP-hard.

2. In the k-means algorithm, $J(\pi)$ decreases in every iteration.

3. K-means algorithm can converge to a local minimum.

4. Each iteration of the k-means algorithm requires $O(mnk)$ operations.

4. K-means algorithm is implemented in the function `kmeans()` in the package [Clustering.jl](https://github.com/JuliaStats/Clustering.jl).

5.  $J(\pi)=\mathop{\mathrm{trace}}(S_W)$, where

$$
S_{W}=\sum\limits_{i=1}^k\sum\limits_{x\in C_{i}}
(x-c_i)(x-c_{i})^{T}
=\sum_{i=1}^k\frac{1}{2|C_{i}|}\sum_{x\in C_{i}}\sum_{y \in C_{i}}
(x-y)(x-y)^{T}.$$

Let $c$ denote the mean of $X$. Then $S_W=S_T-S_B$, where

$$
\begin{aligned}
S_{T}&=\sum_{x\in X}(x-c)(x-c)^{T} = 
\frac{1}{2m}\sum_{i=1}^m\sum_{j=1}^m
(x_{i}-x_{j})(x_{i}-x_{j})^{T}, \\
S_{B}&=\sum_{i=1}^k|C_{i}|(c_{i}-c)(c_{i}-c)^{T} =
\frac{1}{2m}\sum_{i=1}^k\sum_{j=1}^k|C_{i}||C_{j}|
(c_{i}-c_{j})(c_{i}-c_{j})^{T}.
\end{aligned}$$

6. In order to try to avoid convergence to local minima, the k-means algorithm can be enhanced with first variation by adding the following steps:
    1. Compute $\mathop{\mathrm{next}}(\pi)$. 
    2. If $J(\mathop{\mathrm{next}}(\pi))<J(\pi)$, set $\pi=\mathop{\mathrm{next}}(\pi)$ and go to Step 2.
     
"""

# ╔═╡ 86d55833-ee5a-4f73-b0bf-e8b6e6d65617
function myKmeans(X::Vector{T}, k::Int) where T
    # X is Array of Arrays
    m,n=length(X),length(X[1])
    C=Vector{Int}(undef,m)
    # Choose random k means among X
    c=X[randperm(m)[1:k]]
    # This is just to start the while loop
    cnew=copy(c)
    cnew[1]=cnew[1].+(1.0,1.0)
    # Loop
    iterations=0
    while cnew!=c
        iterations+=1
        cnew=copy(c)
        # Assignment
        for i=1:m
            C[i]=findmin([norm(X[i].-c[j]) for j=1:k])[2]
        end
        # Update
        for j=1:k
          c[j]=(mean([x[1] for x in X[C.==j]]),mean([x[2] for x in X[C.==j]]))
        end
    end
    C,c,iterations
end

# ╔═╡ e15fa4cf-930a-4065-b839-c2ec944afcfa
md"""

## Examples

### Random clusters

We generate $k$ random clusters around points with integer coordinates.
"""

# ╔═╡ bacdc83d-1acd-46df-8d06-3e0279978a33
begin
	# Generate points as Tuple()
	k=5
	Random.seed!(1235)
	centers= [Tuple(rand(-5:5,2)) for i=1:k]
	# Number of points in cluster
	sizes=rand(10:50,k)
	csizes=cumsum(sizes)
	# X is array of arrays
	X=Vector{Tuple{Float64,Float64}}(undef,sum(sizes))
	X[1:csizes[1]]=[centers[1].+Tuple((rand(2).-0.5)/2) for i=1:sizes[1]]
	for j=2:k
		X[csizes[j-1]+1:csizes[j]]=[centers[j].+Tuple((rand(2).-0.5)/2) for i=1:sizes[j]]
	end
	centers, sizes
end

# ╔═╡ 99aba610-1f75-11eb-1b5e-2f2ee211ed7f
X

# ╔═╡ fceab134-0921-459b-8d91-952a44abb07b
begin
	# Plot
	scatter(X,label="Points",title="Points")
	scatter!(centers,markershape = :hexagon, ms = 6,label="Centers")
end

# ╔═╡ f559b870-608a-426e-a23b-b66ef6d4b47f
# Plot the solution
function plotKmeansresult(C::Vector,c::Vector,X::Vector)
    scatter()
    # Clusters
    for j=1:k
        scatter!(X[findall(C.==j)],label="Cluster $j")
    end
    # Means
    scatter!(c,markershape=:hexagon,ms=6,color=:red,label="Centers")
	plot!(title="Computed clusters")
end

# ╔═╡ ba84fe97-4dd0-47d3-b980-f5e5a8de6fa3
X

# ╔═╡ 310a4957-cc58-424b-845c-3ebaf7db77f3
md"""
__What happens?__

We see that the algorithm, although simple, for this example 
converges to a local minimum.

Let us try the function `kmeans()` from the package `Clustering.jl`.
The inputs to `kmeans()` are:

* a matrix whose columns are points, 
* number of clusters we are looking for, and, 
* optionally, the method to compute initial means. 

If we choose `init=:rand`, the results are similar. If we choose
`init=:kmpp`, which is the default, the results are better, but convergence to a local minimum is still possible.

__Run the clustering several times!__

```
seeding_algorithm(s::Symbol) = 
    s == :rand ? RandSeedAlg() :
    s == :kmpp ? KmppAlg() :
    s == :kmcen ? KmCentralityAlg() :
    error("Unknown seeding algorithm $s")
```
"""

# ╔═╡ f9af8a75-48fd-4e82-b43e-61867bead6f9
methods(kmeans)

# ╔═╡ 7e869699-8eb1-4e38-905d-a9448d037841
begin
	Xₘ=transpose([[x[1] for x in X] [x[2] for x in X]])
	output=kmeans(Matrix(Xₘ),k,init=:kmpp)
end

# ╔═╡ 4f8152ee-ce86-4df6-81d2-be2df33df980
fieldnames(KmeansResult)

# ╔═╡ 95fb2e66-e811-47f4-a772-8270d0c5d83b
output.centers

# ╔═╡ 26b7bac0-1f72-11eb-16fc-c1d74920c462
output.assignments

# ╔═╡ 7bb0e9ce-72dd-4623-ab3d-a6d18fb9dd15
# We need to modify the plotting function
function plotKmeansresult(out::KmeansResult,X::AbstractArray)
    k=size(out.centers,2)
    scatter(aspect_ratio=1)
    # Clusters
    for j=1:k
        scatter!(X[1,findall(out.assignments.==j)], X[2,findall(out.assignments.==j)], label="Cluster $j")
    end
    # Means
    scatter!(out.centers[1,:], out.centers[2,:], markershape=:hexagon,ms=6,color=:red,label="Centers") 
end

# ╔═╡ 8e5b89ce-e6b5-4473-bd98-a2285331418c
begin
	C,c,iterations=myKmeans(X,k)
	plotKmeansresult(C,c,X)
end

# ╔═╡ 406bb1b2-a8bb-4cd2-95c3-3975f99e3f32
begin
	out=kmeans(Xₘ,k,init=:kmpp)
	plotKmeansresult(out,Xₘ)
end

# ╔═╡ 2445e75a-77f9-4c27-b400-44d8b49fc03b
md"""
### Concentric rings

The k-means algorithm works well if clusters can be separated by hyperplanes. In this example it is not the case.
"""

# ╔═╡ 21bdc615-76e8-46cc-ac86-310d28f6cd25
begin
	# Number of rings, try also k=3
	k₁=2
	# Center
	Random.seed!(5361)
	center=[rand(-5:5);rand(-5:5)]
	# Radii
	radii=randperm(10)[1:k₁]
	# Number of points in circles
	sizes₁=rand(1000:2000,k₁)
	center,radii,sizes₁
end

# ╔═╡ 9a4060b7-ba5a-401f-90f2-b9780ed93ddb
begin
	# Generate points
	X₁=Array{Float64}(undef,2,sum(sizes₁))
	csizes₁=cumsum(sizes₁)
	# Random angles
	ϕ=2*π*rand(sum(sizes₁))
	for i=1:csizes₁[1]
		X₁[:,i]=center+radii[1]*[cos(ϕ[i]);sin(ϕ[i])] + (rand(2).-0.5)/50
	end
	for j=2:k₁
		for i=csizes₁[j-1]+1:csizes₁[j]
			X₁[:,i]=center+radii[j]*[cos(ϕ[i]);sin(ϕ[i])] + (rand(2).-0.5)/50
		end
	end
	scatter(X₁[1,:],X₁[2,:],title="Concentric Rings", aspect_ratio=1,label="Points")
end

# ╔═╡ 0be24dd5-f3b0-4daa-aad8-5dbcdfca9313
begin
	out₁=kmeans(X₁,k₁,init=:rand)
	plotKmeansresult(out₁,X₁)
end

# ╔═╡ 1637ce15-f55a-4d66-8a00-45a555bf5d61


# ╔═╡ Cell order:
# ╟─c7e41465-41ad-4f11-baec-bfaae9edbf59
# ╟─00d72ad7-ea80-4954-8c75-590d4e9ccdc4
# ╟─235e1901-eed3-4934-9214-ffdd82cf21d6
# ╟─26640b90-1f69-11eb-029e-4dc3a825f639
# ╟─48914840-1f69-11eb-30a9-3943767bc261
# ╟─2b76953c-edf2-41cf-9d75-7efb2d6a58f1
# ╠═d0233ece-1f69-11eb-3da1-43b8ddc9fbe1
# ╠═86d55833-ee5a-4f73-b0bf-e8b6e6d65617
# ╟─e15fa4cf-930a-4065-b839-c2ec944afcfa
# ╠═bacdc83d-1acd-46df-8d06-3e0279978a33
# ╠═99aba610-1f75-11eb-1b5e-2f2ee211ed7f
# ╠═fceab134-0921-459b-8d91-952a44abb07b
# ╠═f559b870-608a-426e-a23b-b66ef6d4b47f
# ╠═8e5b89ce-e6b5-4473-bd98-a2285331418c
# ╠═ba84fe97-4dd0-47d3-b980-f5e5a8de6fa3
# ╟─310a4957-cc58-424b-845c-3ebaf7db77f3
# ╠═f9af8a75-48fd-4e82-b43e-61867bead6f9
# ╠═7e869699-8eb1-4e38-905d-a9448d037841
# ╠═4f8152ee-ce86-4df6-81d2-be2df33df980
# ╠═95fb2e66-e811-47f4-a772-8270d0c5d83b
# ╠═26b7bac0-1f72-11eb-16fc-c1d74920c462
# ╠═7bb0e9ce-72dd-4623-ab3d-a6d18fb9dd15
# ╠═406bb1b2-a8bb-4cd2-95c3-3975f99e3f32
# ╟─2445e75a-77f9-4c27-b400-44d8b49fc03b
# ╠═21bdc615-76e8-46cc-ac86-310d28f6cd25
# ╠═9a4060b7-ba5a-401f-90f2-b9780ed93ddb
# ╠═0be24dd5-f3b0-4daa-aad8-5dbcdfca9313
# ╠═1637ce15-f55a-4d66-8a00-45a555bf5d61
