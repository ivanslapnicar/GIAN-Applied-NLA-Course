### A Pluto.jl notebook ###
# v0.14.6

using Markdown
using InteractiveUtils

# ╔═╡ 0f6cd296-6cb8-4a9d-a21f-02b689dfe4e9
begin
	using PlutoUI
	PlutoUI.TableOfContents(aside=true)
end

# ╔═╡ 646a9fc2-3a73-4f8d-a952-2cdcf0e6287a
begin
	# Packages
	using LightGraphs
	using GraphPlot
	using Clustering
	using SparseArrays
	using LinearAlgebra
	using Plots
end

# ╔═╡ 7cd3af86-f459-4e4b-8e35-c29bc6966eed
begin
	# Define sizes
	using Random
	m=[200,100,100]
	n=[100,200,100]
	density=[0.5,0.7,0.2]
	Pts=Array{Any}(undef,3)
	Random.seed!(421)
	for i=1:3
	    # Generate sparse random
	    Pts[i]=sprand(m[i],n[i],density[i])
	end
	B=blockdiag(Pts[1],Pts[2],Pts[3])
end

# ╔═╡ 9ab98160-3e37-44c3-b219-a3e1bf95fd68
using Arpack

# ╔═╡ 89299cb7-a4eb-48f3-b3f1-ab2b5c6cfa2e
md"""
# Spectral Partitioning of Bipartite Graphs

Typical example of bipartite graph is a graph obtained from a collection of documents presented as a _term $\times$ document_  matrix.

__Prerequisites__

The reader should be familiar with k-means algorithm and spectral graph partitioning theory and algorithms.
 
__Competences__

The reader should be able to apply spectral partitioning of bipartite graphs to data clustering problems.

__Credits.__ The notebook was initially derived from M.Sc. Thesis of Ivančica Mirošević.
"""

# ╔═╡ a92d0e70-479d-444e-b06d-e47d2e385b10
md"""
## Definitions

__Undirected bipartite graph__ $G$ is a triplet $G=(T,D,E)$, where $T=\{t_{1},\cdots ,t_{m}\}$ and $D=\{d_{1},...,d_{n}\}$ are two sets of vertices and $E=\{(t_{i},d_{j}):t_{i}\in R,d_{j}\in D\}$, is a set of edges.

 $G$ is __weighted__ if there is weight $\omega(e)$ associated with each edge $e\in E$.

For example, $D$ is a set of documents, $T$ is a set of terms (words) and  edge $e=(t_{i},d_{j})$ exists if document $d_{j}$ contains term $t_{i}$. Weight $\omega(e)$ can be number of appearances of the term $t_i$ in the document $d_j$.

A __term-by-document-matrix__ is a matrix $A\in\mathbb{R}^{m\times n}$ with $A_{ij}=\omega((t_i,d_j))$.
"""

# ╔═╡ fddcc2a7-f718-4900-a135-764eb36496a7
md"""
## Facts

1. The weight matrix of $G$ is $W=\begin{bmatrix}0 & A \\ A^{T} & 0 \end{bmatrix}$.

2. The Laplacian matrix of $G$ is

$$L=\begin{bmatrix} \Delta_{1} & -A \\ -A^{T} & \Delta_{2}\end{bmatrix},$$

where $\Delta_1$ and $\Delta_2$ are diagonal matrices with elements 

$\Delta_{1,ii}=\sum\limits_{j=1}^n A_{ij},\quad i=1,\ldots,m,$

$\Delta_{2,jj}=\sum\limits_{i=1}^m A_{ij},\quad j=1,\ldots,n.$

3. The normalized Laplacian matrix of $G$ is 

$$L_n=\begin{bmatrix}
I & -\Delta_{1}^{-\frac{1}{2}}A\Delta_{2}^{-\frac{1}{2}} \\
-\Delta_{2}^{-\frac{1}{2}}A^T\Delta_{1}^{-\frac{1}{2}} & I
\end{bmatrix} \equiv 
\begin{bmatrix} I & -A_n \\ -A_n^T & I \end{bmatrix}.$$
"""

# ╔═╡ b3b44560-203a-11eb-2782-170e4c12627a
md"""
4. Let $\lambda$ be an eigenvalue of $L_n$ with an eigenvector $w=\begin{bmatrix} u \\ v\end{bmatrix}$, where $u\in \mathbb{R}^{m}$ $v\in\mathbb{R}^{n}$. Then $L_n w=\lambda w$ implies $A_n v =(1-\lambda)u$ and $A_n^T u=(1-\lambda)v$. Vice versa, if $(u,\sigma,v)$ is a singular triplet of $A_n$, then $1-\sigma$ is an eigenvalue of $L_n$ with (non-unit) eigenvector $w=\begin{bmatrix} u \\ v\end{bmatrix}$. 

5. The second largest singular value of $A_n$ corresponds to the second smallest eigenvalue of $L_n$, and computing the former is numerically more stable. 
"""

# ╔═╡ 1be90f30-203b-11eb-34da-c3116523ec3a
md"

## Algorithms

### Bipartitioning algorithm

1. For given $A$ compute $A_{n}$.
2. Compute singular vectors of $A_{n}$, $u^{[2]}$ and $v^{[2]}$, which correspond to the second largest singular value, $\sigma_2(A_n)$.
3. Assign the partitions $T=\{T_1,T_2\}$ and $D=\{D_1,D_2\}$ according to the signs of $u^{[2]}$ and $v^{[2]}$. The pair $(T,D)$ is now partitioned as $\{(T_1,D_1),(T_2,D_2)\}$.
"

# ╔═╡ 5211ce2e-203b-11eb-0b6f-55e809177fb9
md"
### Recursive bipartitioning algorithm

1. Compute the bipartition $\pi=\{(T_1,D_1),(T_2,D_2)\}$ of $(T,D)$. Set the counter $c=2$.

2. While $c<k$ repeat

   - compute bipartitions of each of the subpartitions of $(T,D)$,
   - among all $(c+1)$-subpartitions, choose the one with the smallest $\mathop{\mathrm{pcut}}(\pi_{c+1})$ or $\mathop{\mathrm{ncut}}(\pi_{c+1})$, respectively.
   - Set $c=c+1$

3. Stop
"

# ╔═╡ 38333a20-203c-11eb-2b28-59e07b3e50ee
md"
### Multipartitioning algorithm
1. For given $A$ compute $A_{n}$.
2. Compute $k$ left and right singular vectors, $u^{[1]},\ldots,u^{[k]}$ and $v^{[1]},\ldots,v^{[k]}$, which correspond to $k$ largest singular values $\sigma_1\geq \cdots \geq \sigma_k$ of $A_n$.
3. Partition the rows of matrices $\Delta_{1}^{-\frac{1}{2}}\begin{bmatrix} u^{[1]} & \ldots & u^{[k]}\end{bmatrix}$ and $\Delta_{2}^{-\frac{1}{2}}\begin{bmatrix} v^{[1]} & \ldots & v^{[k]}\end{bmatrix}$ with the k-means algorithm.
"

# ╔═╡ 1bc20329-3a6e-435d-a25c-8cce79fb7d9d
md"
## Examples

### Small term-by- document matrix
"

# ╔═╡ b9d0830a-0a33-40c1-a08a-49538cf7a879
begin
	# Make a nicer spy function
	import Plots.spy
	spy(A)=heatmap(A,yflip=true,color=:RdBu,aspectratio=1,clim=(-1,1.0)) 
end

# ╔═╡ f6492dfe-faac-407f-9a0e-5dcf321a4f2d
begin
	# Sources, targets, and weights
	dn=[6,6,7,6,7,7]
	tn=[1,2,2,3,4,5]
	wn=[3,1,3,2,2,3]
	[dn tn wn]
end

# ╔═╡ 1a8b6f7b-4098-4cd2-adad-8ee788b47333
mynames=["Term 1";"Term 2";"Term 3";"Term 4";"Term 5";"Doc 1";"Doc 2"]

# ╔═╡ 4079aa8e-2bb5-4d51-bca8-04d0997db2bf
begin
	G=Graph(7)
	for i=1:length(dn)
	    add_edge!(G,tn[i],dn[i])
	end
	gplot(G, nodelabel=mynames, edgelabel=wn)
end

# ╔═╡ b6d41ab1-24f2-417d-b604-3e12384e65f8
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

# ╔═╡ d5d8ee5d-9cce-45fc-a31b-b2ed8237cc86
W=WeightMatrix(tn,dn,wn)

# ╔═╡ 86fd4f5a-af57-490d-a17f-2339a9414057
Matrix(W)

# ╔═╡ 1f53b5cf-ab6a-4107-b53b-9ce96afcdb7f
begin
	L=Laplacian(W)
	Matrix(L)
end

# ╔═╡ 63403411-cca4-4d71-8a76-8dea8638f69f
Lₙ=NormalizedLaplacian(L)

# ╔═╡ 3cf471bc-d316-4de9-af63-71db9b035c40
begin
	A=W[1:5,6:7]
	Δ₁=sqrt.(sum(A,dims=2))
	Δ₂=sqrt.(sum(A,dims=1))
	Aₙ=[A[i,j]/(Δ₁[i]*Δ₂[j]) for i=1:size(A,1), j=1:size(A,2)]
end

# ╔═╡ ac0b5eda-299c-4ae2-ae78-bda92b378f19
# The partitioning - explain the results!
U,σ,V=svd(Aₙ)

# ╔═╡ 9bf6bbca-f96c-4b38-a4fd-5188ec29a2a2
U[:,2]

# ╔═╡ aa51688c-21a0-40f4-a6d6-01d441bfd4b7
V[:,2]

# ╔═╡ 5f27f1a2-80de-41d1-b1e2-da3a99493217
md"
### Sets of points
"

# ╔═╡ e1149b05-d594-4d75-ae56-1ce18bd0f60d
spy(Matrix(B))

# ╔═╡ b95d4070-c127-4db0-987b-5fc17421b667
# The structure of singular vectors reflects the block
S,rest₀=svds(B,nsv=3);

# ╔═╡ 57d472c1-548d-4678-b884-85f863771aa7
# S is a structure
S.S

# ╔═╡ 4806378b-63cb-445a-8c71-583458fce846
begin
	# Plot the first three left singular vectors
	k=size(B,1)
	x=collect(1:k)
	scatter(x,S.U[:,1],title="Left Singular Vectors",label="U[:,1]")
	scatter!(x,S.U[:,2],label="U[:,2]",legend=:topleft)
	scatter!(x,S.U[:,3],label="U[:,3]")
end

# ╔═╡ a08c99a3-9c10-47be-9dc6-987d92a85087
begin
	# Plot the first three right singular vectors
	scatter(x,S.Vt[1,:],title="Right Singular Vectors",label="V[:,1]")
	scatter!(x,S.Vt[2,:],label="V[:,2]")
	scatter!(x,S.Vt[3,:],label="V[:,3]")
end

# ╔═╡ 6888e8a9-f306-4490-9147-2460fbda4889
begin
	# Add random noise
	noise=sprand(k,k,0.3)
	C=B+noise
	spy(Matrix(C))
end

# ╔═╡ 3bd5ad15-31cd-4efd-b4e3-7cf7ede3edae
begin
	# Apply random permutation to rows and columns of C
	D=C[randperm(k),randperm(k)]
	spy(Matrix(D))
end

# ╔═╡ e4487f35-5de0-492e-96f1-69318d7d8e8b
md"""
__Question.__ Given D, can we recover C?

__Answer.__ Yes (with spectral partitioning)!
"""

# ╔═╡ 81d99f24-82b9-4216-8d17-6a18317beb77
Sₙ,rest=svds(D,nsv=3);

# ╔═╡ 8f925ff7-90b4-4fe4-a869-86dec693c0f4
# K-means on rows of U 
outU=kmeans(Matrix(transpose(Sₙ.U)),3)

# ╔═╡ d446724a-97d1-403f-baaf-d0619203f351
# K-means on Vt
outV=kmeans(Sₙ.Vt,3)

# ╔═╡ 7572c241-dbcc-4094-97e3-ee3172255294
sortperm(outU.assignments)

# ╔═╡ 22e90c11-ba4e-4073-9f78-f3b7beb24e69
begin
	# RECOVERY of B
	E=D[sortperm(outU.assignments),sortperm(outV.assignments)]
	spy(Matrix(E))
end

# ╔═╡ Cell order:
# ╟─0f6cd296-6cb8-4a9d-a21f-02b689dfe4e9
# ╟─89299cb7-a4eb-48f3-b3f1-ab2b5c6cfa2e
# ╟─a92d0e70-479d-444e-b06d-e47d2e385b10
# ╟─fddcc2a7-f718-4900-a135-764eb36496a7
# ╟─b3b44560-203a-11eb-2782-170e4c12627a
# ╟─1be90f30-203b-11eb-34da-c3116523ec3a
# ╟─5211ce2e-203b-11eb-0b6f-55e809177fb9
# ╟─38333a20-203c-11eb-2b28-59e07b3e50ee
# ╟─1bc20329-3a6e-435d-a25c-8cce79fb7d9d
# ╠═646a9fc2-3a73-4f8d-a952-2cdcf0e6287a
# ╠═b9d0830a-0a33-40c1-a08a-49538cf7a879
# ╠═b6d41ab1-24f2-417d-b604-3e12384e65f8
# ╠═f6492dfe-faac-407f-9a0e-5dcf321a4f2d
# ╠═1a8b6f7b-4098-4cd2-adad-8ee788b47333
# ╠═4079aa8e-2bb5-4d51-bca8-04d0997db2bf
# ╠═d5d8ee5d-9cce-45fc-a31b-b2ed8237cc86
# ╠═86fd4f5a-af57-490d-a17f-2339a9414057
# ╠═1f53b5cf-ab6a-4107-b53b-9ce96afcdb7f
# ╠═63403411-cca4-4d71-8a76-8dea8638f69f
# ╠═3cf471bc-d316-4de9-af63-71db9b035c40
# ╠═ac0b5eda-299c-4ae2-ae78-bda92b378f19
# ╠═9bf6bbca-f96c-4b38-a4fd-5188ec29a2a2
# ╠═aa51688c-21a0-40f4-a6d6-01d441bfd4b7
# ╟─5f27f1a2-80de-41d1-b1e2-da3a99493217
# ╠═7cd3af86-f459-4e4b-8e35-c29bc6966eed
# ╠═e1149b05-d594-4d75-ae56-1ce18bd0f60d
# ╠═9ab98160-3e37-44c3-b219-a3e1bf95fd68
# ╠═b95d4070-c127-4db0-987b-5fc17421b667
# ╠═57d472c1-548d-4678-b884-85f863771aa7
# ╠═4806378b-63cb-445a-8c71-583458fce846
# ╠═a08c99a3-9c10-47be-9dc6-987d92a85087
# ╠═6888e8a9-f306-4490-9147-2460fbda4889
# ╠═3bd5ad15-31cd-4efd-b4e3-7cf7ede3edae
# ╟─e4487f35-5de0-492e-96f1-69318d7d8e8b
# ╠═81d99f24-82b9-4216-8d17-6a18317beb77
# ╠═8f925ff7-90b4-4fe4-a869-86dec693c0f4
# ╠═d446724a-97d1-403f-baaf-d0619203f351
# ╠═7572c241-dbcc-4094-97e3-ee3172255294
# ╠═22e90c11-ba4e-4073-9f78-f3b7beb24e69
