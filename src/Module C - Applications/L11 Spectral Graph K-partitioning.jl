### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ c9cf879e-0b01-4e0a-b31c-2ca5c6f03e63
begin
	using PlutoUI
	PlutoUI.TableOfContents(aside=true)
end

# ╔═╡ 94dde840-54f6-4fe4-846b-bc15f8fca9fa
begin
	# Packages
	using LightGraphs
	using GraphPlot
	using Clustering
	using SparseArrays
	using LinearAlgebra
	using Plots
	using Distances
	using Random
end

# ╔═╡ 81efb137-7e14-4091-a7f0-c6b72e2d77f7
md"""
# Spectral Graph K-partitioning

Instead of using recursive spectral bipartitioning, the graph $k$-partitioning problem can be solved using $k$ eigenvectors which correspond to $k$ smallest eigenvalues of Laplace matrix (Laplacian) or normalized Laplace matrix (normalized Laplacian), respectively.

Suggested reading is [U. von Luxburg, A Tutorial on Spectral Clustering](http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/luxburg06_TR_v2_4139%5b1%5d.pdf), which includes the quote 

_"Spectral clustering cannot serve as a “black box algorithm” which automatically detects the correct clusters in any given data set. But it can be considered as a powerful tool which can produce good results if applied with care."_


__Prerequisites__

The reader should be familiar with k-means algorithm, spectral graph bipartitioning and recursive bipartitioning.
 
__Competences__

The reader should be able to apply graph spectral k-partitioning to data clustering problems.

__Credits.__ The notebook was initially derived from M.Sc. Thesis of Ivančica Mirošević.
"""

# ╔═╡ 75cd6187-4c51-426c-8a3c-d7b6c922ae0b
md"""

## Definitions

Let $G=(V,E)$ be a weighted graph with weights $\omega$, with weights matrix $W$, Laplacian matrix $L=D-W$, and normalized Laplacian matrix $L_n=D^{-1/2}(D-W)D^{-1/2}$. 

Let the $k$-partition $\pi_k =\{V_{1},V_{2},...,V_{k}\}$, the cut $\mathop{\mathrm{cut}}(\pi_k)$, the proportional cut $\mathop{\mathrm{pcut}}(\pi_k)$ and the normalized cut $\mathop{\mathrm{ncut}}(\pi_k)$ be defined as in the
[Spectral Graph Bipartitioning](https://ivanslapnicar.github.io/NumericalMathematics/C2%20Spectral%20Graph%20Bipartitioning.jl.html) notebook.


__Partition vectors__ of a $k$-partition $\pi_k$ are

$$
\begin{aligned}
h_{1} &=[\overset{\displaystyle |V_{1}|}{\overbrace{1,\cdots
,1}},0,\cdots
,0,\cdots ,0,\cdots ,0]^{T}  \\
h_{2} &=[0,\cdots
,0,\overset{\displaystyle |V_{2}|}{\overbrace{1,\cdots ,1}} ,\cdots ,0,\cdots
,0]^{T} \\
&\vdots \\
h_{k} &=[0,\cdots ,0,0,\cdots ,0,\cdots
,\overset{\displaystyle |V_{k}|}{ \overbrace{1,\cdots ,1}}]^{T}.
\end{aligned}$$

## Facts

1. Set

$$
\begin{aligned}
X&=\begin{bmatrix} x_1 & x_2 &\cdots & x_k \end{bmatrix}, \quad 
x_i=\displaystyle \frac{h_i}{\|h_i\|_2}, \\
Y&=\begin{bmatrix} y_1 & y_2 &\cdots & y_k \end{bmatrix}, \quad
y_i=\displaystyle \frac{D^{1/2}h_i}{\|D^{1/2}h_i\|_2}.
\end{aligned}$$

It holds

$$
\begin{aligned}
& \mathop{\mathrm{cut}}(V_{i},V\backslash V_{i})=h_{i}^{T}(D-W)h_{i}=h_{i}^{T}L h_{i},\quad 
\omega( C_{i})=h_{i}^{T}D h_{i},\quad |C_{i}| =h_{i}^{T}h_{i},\\
& \mathop{\mathrm{pcut}}(\pi_k) =\frac{h_{1}^{T}L h_{1}}{h_{1}^{T} h_{1}}+
\cdots + \frac{h_{k}^{T}L h_{k}}{h_{k}^{T}h_{k}}
=x_{1}^{T}L x_{1}+\cdots +x_{k}^{T}Lx_{k}=\mathop{\mathrm{trace}}(X^{T}LX),\\
& \mathop{\mathrm{ncut}}(\pi_k)=\frac{h_{1}^{T}L h_{1}}{h_{1}^{T}D h_{1}}+\cdots
+\frac{h_{k}^{T}L h_{k}}{h_{k}^{T}D h_{k}}
=\mathop{\mathrm{trace}}(Y^{T}L_{n}Y).
\end{aligned}$$

2. The __relaxed__ $k$-partitioning problems are trace-minimization problems,

$$
\begin{aligned}
\min_{\displaystyle \pi_k} \mathop{\mathrm{pcut}}(\pi_k) &\geq
\underset{\displaystyle X\in\mathbb{R}^{n\times k}}{\min_{\displaystyle X^{T}X=I}}
\mathop{\mathrm{trace}}(X^{T}LX),\\
\min_{\displaystyle \pi_k} \mathop{\mathrm{ncut}}(\pi_k) &\geq
\underset{\displaystyle Y\in\mathbb{R}^{n\times
k}}{\min_{\displaystyle Y^{T}Y=I}}\mathop{\mathrm{trace}}(Y^{T}L_{n}Y).
\end{aligned}$$

3. __Ky-Fan  Theorem.__ Let $A\in \mathbb{R}^{n\times n}$ be a symmetric matrix with eigenvalues $\lambda _1\leq \cdots \leq \lambda_n$. Then

$$
\underset{\displaystyle Z^TZ=I}{\min_{\displaystyle Z\in \mathbb{R}^{n\times
k}}}\mathop{\mathrm{trace}}\left(Z^{T}AZ\right)
=\sum_{i=1}^{k}\lambda_{i}.$$

4. Let $\lambda_1\leq \cdots \leq \lambda_n$ be the eigenvalues of $L$ with eigenvectors $v^{[1]},\cdots ,v^{[k]}$. The solution of the relaxed proportional cut problem is the matrix $X=\begin{bmatrix}v^{[1]} & \cdots & v^{[k]}\end{bmatrix}$, and it holds 

$\min\limits_{\displaystyle \pi_k} \mathop{\mathrm{pcut}}(\pi_k)\geq \sum\limits_{i=1}^k \lambda_i.$

5. Let $\mu_1\leq \cdots \leq \mu_n$ be the eigenvalues of $L_n$ with eigenvectors $w^{[1]},\cdots ,w^{[k]}$. The solution of the relaxed normalized cut problem is the matrix $Y=\begin{bmatrix}w^{[1]} & \cdots & w^{[k]}\end{bmatrix}$, and it holds

$\min\limits_{\displaystyle \pi_k} \mathop{\mathrm{ncut}}(\pi_k)\geq \sum\limits_{i=1}^k \mu_i$.

6. It remains to recover the $k$-partition. The k-means algorithm applied to rows of the matrices $X$ or $D^{-1/2}Y$, will compute the $k$ centers and the assignment vector whose $i$-th component denotes the subset $V_j$ to which the vertex $i$ belongs.
"""

# ╔═╡ 639758a5-413d-408a-ace9-3741322c3dc2
md"""

## Examples

### Graph with three clusters
"""

# ╔═╡ d367f447-16e1-4cbc-b05f-b807d4ef955b
begin
	# Sources, targets, and weights
	n=9
	sn=[1,1,1,2,2,2,3,3,5,6,7,7,8]
	tn=[2,3,4,3,4,7,4,5,6,9,8,9,9]
	wn=[2,3,4,4,5,1,6,1,7,1,4,3,2]
	[sn tn wn]
end

# ╔═╡ 9438ace8-4290-4caa-bb34-24c1c8f4f66a
begin
	# What is the optimal tripartition?
	G=Graph(n)
	for i=1:length(sn)
	    add_edge!(G,sn[i],tn[i])
	end
	gplot(G, nodelabel=1:n, edgelabel=wn)
end

# ╔═╡ f05bae5c-8d1c-4f68-a299-fab3af2432ae
begin
	# We define some functions
	function WeightMatrix(src::Array,dst::Array,weights::Array)
	    n=nv(G)
	    sparse([src;dst],[dst;src],[weights;weights],n,n)
	end
	
	Laplacian(W::AbstractMatrix)=spdiagm(0=>vec(sum(W,dims=2)))-W
	
	function NormalizedLaplacian(L::AbstractMatrix)
	    D=1.0./sqrt.(diag(L))
	    n=length(D)
	    [L[i,j]*(D[i]*D[j]) for i=1:n, j=1:n]
	end
end

# ╔═╡ 357b8af9-700a-4a2d-9703-575af5b9ab13
begin
	W=WeightMatrix(sn,tn,wn)
	L=Laplacian(W)
	Lₙ=NormalizedLaplacian(L)
end

# ╔═╡ 085b49b4-08a2-4e6e-8cbe-6a011a93b551
Matrix(L)

# ╔═╡ 280049be-261e-49d7-a6fc-2211afe94e06
# Proportional cut. The clustering is visible in 
# the components of v₂ and v₃
# E=eigs(L,nev=3,which=:SM)
E=eigen(Matrix(L))

# ╔═╡ 64730ce5-61b4-405d-9a41-81d0253a4713
# Check the assignments
out=kmeans(Matrix(transpose(E.vectors[:,1:3])),3)

# ╔═╡ 692cbb97-2a5f-42c9-b829-e08d5e7479a8
begin
	# Normalized cut
	# Lanczos cannot be used for the "smallest in magnitude"
	# eienvalues of a singular matrix
	# λ,Y=eigs(Ln,nev=3,which=:SM) does not work
	Eₙ=eigen(Lₙ)
	D=sqrt.(diag(L))
	Y=inv(Diagonal(D))*Eₙ.vectors[:,1:3]
	outₙ=kmeans(Matrix(transpose(Y)),3)
end

# ╔═╡ 662b7b01-883f-4d51-b696-26f7559e0211
md"""
### Concentric rings
"""

# ╔═╡ 55d217d5-c31a-431d-bd46-f9f09280cf5b
function plotKpartresult(C::Vector,X::Array)
    k=maximum(C)
    scatter(aspect_ratio=1)
    for j=1:k
        scatter!(X[1,findall(C.==j)],X[2,findall(C.==j)],label="Cluster $j")
    end
	scatter!()
end

# ╔═╡ 3f815c3e-850d-4ca7-af1c-a2ff14e429db
begin
	# Generate concentric rings
	k=4
	Random.seed!(541)
	# Center
	center=[0,0]
	# Radii
	radii=Random.randperm(10)[1:k]
	# Number of points in rings
	sizes=rand(300:500,k)
	center,radii,sizes
end

# ╔═╡ 967b8e40-1f95-11eb-0aa1-8f4592054357
begin
	# Generate points
	X=Array{Float64}(undef,2,sum(sizes))
	csizes=cumsum(sizes)
	# Random angles
	ϕ=2*π*rand(sum(sizes))
	for i=1:csizes[1]
		X[:,i]=center+radii[1]*[cos(ϕ[i]);sin(ϕ[i])] + (rand(2).-0.5)/50
	end
	for j=2:k
		for i=csizes[j-1]+1:csizes[j]
			X[:,i]=center+radii[j]*[cos(ϕ[i]);sin(ϕ[i])] + (rand(2).-0.5)/50
		end
	end
	scatter(X[1,:],X[2,:],title="Concentric rings", aspect_ratio=1,label="Points")
end

# ╔═╡ 708a16c5-f755-490b-abcb-89788813b7d0
begin
	S=pairwise(SqEuclidean(),X,dims=2)
	# S=pairwise(Cityblock(),X)
	β=60
	W₁=exp.(-β*S);
end

# ╔═╡ c298ccfe-f15e-4652-baa7-c122cb22f22f
begin
	L₁=Laplacian(W₁)
	Ln₁=NormalizedLaplacian(L₁);
end

# ╔═╡ 9d8aef37-9640-4d67-9c33-bb6513938ae1
begin
	# Laplacian
	E₁=eigen(L₁)
	sp=sortperm(abs.(E₁.values))[1:k]
	λ₁=E₁.values[sp]
	Y₁=E₁.vectors[:,sp]
	out₁=kmeans(Matrix(transpose(Y₁)),k)
	plotKpartresult(out₁.assignments,X)
end

# ╔═╡ 7c72773d-0915-4ef0-aa7e-7f0474426bfa
begin
	# Normalized Laplacian
	En₁=eigen(Ln₁)
	spn=sortperm(abs.(En₁.values))[1:k]
	λ=En₁.values[spn]
	Yn₁=En₁.vectors[:,spn]
	Yn₁=Diagonal(1.0./sqrt.(diag(L₁)))*Y₁
	outn₁=kmeans(Matrix(transpose(Y₁)),k)
	plotKpartresult(outn₁.assignments,X)
end

# ╔═╡ 68d822d0-1f97-11eb-1a79-8f58785ae13a
L₁[3,1]

# ╔═╡ Cell order:
# ╟─c9cf879e-0b01-4e0a-b31c-2ca5c6f03e63
# ╟─81efb137-7e14-4091-a7f0-c6b72e2d77f7
# ╟─75cd6187-4c51-426c-8a3c-d7b6c922ae0b
# ╟─639758a5-413d-408a-ace9-3741322c3dc2
# ╠═94dde840-54f6-4fe4-846b-bc15f8fca9fa
# ╠═f05bae5c-8d1c-4f68-a299-fab3af2432ae
# ╠═d367f447-16e1-4cbc-b05f-b807d4ef955b
# ╠═9438ace8-4290-4caa-bb34-24c1c8f4f66a
# ╠═357b8af9-700a-4a2d-9703-575af5b9ab13
# ╠═085b49b4-08a2-4e6e-8cbe-6a011a93b551
# ╠═280049be-261e-49d7-a6fc-2211afe94e06
# ╠═64730ce5-61b4-405d-9a41-81d0253a4713
# ╠═692cbb97-2a5f-42c9-b829-e08d5e7479a8
# ╟─662b7b01-883f-4d51-b696-26f7559e0211
# ╠═55d217d5-c31a-431d-bd46-f9f09280cf5b
# ╠═3f815c3e-850d-4ca7-af1c-a2ff14e429db
# ╠═967b8e40-1f95-11eb-0aa1-8f4592054357
# ╠═708a16c5-f755-490b-abcb-89788813b7d0
# ╠═c298ccfe-f15e-4652-baa7-c122cb22f22f
# ╠═7c72773d-0915-4ef0-aa7e-7f0474426bfa
# ╠═9d8aef37-9640-4d67-9c33-bb6513938ae1
# ╠═68d822d0-1f97-11eb-1a79-8f58785ae13a
