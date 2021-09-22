### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ e166839c-7451-471f-b10f-3cc8d3d751f3
begin
	using PlutoUI, Plots, LinearAlgebra, Statistics, DelimitedFiles, Clustering, DataFrames
	import Random, CSV
	plotly()
end

# ╔═╡ d6b500f1-5d54-4cdb-8d74-56e8a9ab367e
# For binder, uncomment this cell ...
#=
begin
	import Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
		Pkg.PackageSpec(name="PlutoUI"),
        Pkg.PackageSpec(name="Plots"),
		Pkg.PackageSpec(name="Statistics"),
		Pkg.PackageSpec(name="DelimitedFiles"),
		Pkg.PackageSpec(name="Distributions"),
		Pkg.PackageSpec(name="Clustering"),
		Pkg.PackageSpec(name="CSV"),
		Pkg.PackageSpec(name="DataFrames")
    ])
end
=#

# ╔═╡ c810e2a4-03a5-4d19-8717-3de25f013269
TableOfContents(title="📚 Table of Contents", aside=true)

# ╔═╡ d7ef9ec9-1acf-48e6-bfbc-55ec01b36b0b
md"""
# Principal Component Analysis


PCA is an orthogonal linear transformation that transforms the data to a new coordinate system such that the greatest variance by some projection of the data comes to lie on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on. PCA analysis is perfromed either using EVD of the covariance matrix or the SVD of the mean-centered data.


__Prerequisites__

The reader should be familiar with linear algebra and statistics concepts.

__Competences__

The reader should be able to perform PCA on a given data set.

__References__

For more details see

* [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis),
* [L. I. Smith, A tutorial on Principal Components Analysis](http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf),
* [J. Shlens, A Tutorial on Principal Components Analysis](http://arxiv.org/abs/1404.1100).

"""

# ╔═╡ 728f10ec-2843-41ae-ae8e-c913155fe303
md"""
## Definitions

A __data matrix__ is a matrix $X\in\mathbb{R}^{m\times n}$, where each column corresponds to a feature (say, certain gene), and each row correspond to an observation (say, individual).

A __mean__ of a vector $x\in\mathbb{R}^{n}$ is $\mu(x)=\displaystyle \frac{x_1+x_2+\cdots x_n}{n}$.

A __standard deviation__  of a vector $x$ is $\sigma(x)=\displaystyle \sqrt{\frac{\sum_{i=1}^n (x_i-\mu(x))^2}{n-1}}$. A __variance__ of a vector $x$ is $\mathop{\mathrm{var}}(x)=\sigma^2(x)$.

A __vector of means__ of a data matrix $X$ is a row-vector of means of the columns of $X$, $\mu(X)=\begin{bmatrix}\mu(X_{:,1}) & \mu(X_{:,2}) & \ldots & \mu(X_{:,n})\end{bmatrix}$. 

A __zero-mean centered data matrix__ is a matrix $\bar X$ obtained from a data matrix $X$ by subtracting from each column the mean of this column,

$$
\bar X= \begin{bmatrix} X_{:,1}-\mu(X_{:,1}) & X_{:,2}-\mu(X_{:,2}) & \cdots & 
X_{:,n}-\mu(X_{:,n}) \end{bmatrix}\equiv
X-\mathbf{1}\mu(X),$$

where $\mathbf{1}=[1,1,\ldots,1]^T$.

A __covariance matrix__ of a data matrix $X$ is a matrix

$$
\mathop{\mathrm{cov}}(X)=\displaystyle \frac{1}{n-1}[X-\mathbf{1}\mu(X)]^T[X-\mathbf{1}\mu(X)]
\equiv \frac{\bar X^T \bar X}{n-1}.$$
"""

# ╔═╡ 1eec2efb-62b7-493b-8a9c-d3784d2e7825
md"""
## Facts

Given a data matrix $X$, let $\mathop{\mathrm{cov}}(X)=U\Lambda U^T$ be the EVD with non-increasingly ordered eigenvalues, $\lambda_1\geq \lambda_2\geq \cdots \geq \lambda_n$. 

1.  $\mathop{\mathrm{cov}}(X)$  is a symmetric PSD matrix.

2.  $\mathop{\mathrm{cov}}(X)=\mathop{\mathrm{cov}}(\bar X)$.

3. Let $T=\bar X U$. The columns of $T$ are the __principal components__ of $\bar X$. In particular:
    1. The first principal component of $\bar X$ (or $X$) is the first column, $T_{:,1}$. It is a projection of the zero-mean centered data set $\bar X$ on the line defined by $U_{:,1}$. This is the direction along which the data have the largest variance.  
    2. The second column (the second principal component), $T_{:,1}$, is a projection of $\bar X$ on the line defined by $U_{:,2}$, which is orthogonal to the first projection. This is direction with the largest variance _after_ subtracting the first principal component from $\bar X$. 
    3. The $k$-th principal component is the direction with the largest variance _after_ subtracting the first $k-1$ principal components from $\bar X$, that is, the first principal component of the matrix

$$
\hat X=\bar X-\sum_{i=1}^{k-1} \bar X U_{:,i} U_{:,i}^T.$$

4. Let $\bar X=\bar U \Sigma V^T$ be the SVD of $\bar X$. Then $V=U$ and $T=\bar U\Sigma V^T V\equiv \bar U \Sigma$.

5. Reconstruction of the principal components is the following:
    1. Full reconstruction is $X=T U^T +\mathbf{1} \mu(X)$.
    2. Reconstruction from the first $k$ principal components is
    
$$\tilde X =T U_{:,1:k}^T +\mathbf{1} \mu(X).$$
    
6. Partial reconstructions can help obtaining various insights about the data. For example, the rows of the matrix $T_{:,1:k}$ can be clustered by the $k$-means algorithm, and the points defined by first three columns of $T$ can be plotted to visualize projections of clusters. Afterwards, the computed clusters can be mapped back to original data.

7. Heuristical guess for number of important clusters is given be the location of the "knee" in the plot of the singular values of $\bar X$.
"""

# ╔═╡ 63929a68-2407-4b46-80bc-91c5da382931
md"""

## Examples

### Elliptical data set

We generate a "quasi" elliptical set of points and compute its principal components. 
"""

# ╔═╡ 1c25240c-d396-4661-b3f3-fd2b30876a00
begin
	# Generate data points
	Random.seed!(456)
	n=3
	m=500
	ax=[8,3,1]
	X=Array{Float64}(undef,m,n)
	for i=1:n
	    X[:,i]=Random.rand!(X[:,i])*ax[i]
	end
	# Parameters
	u=(rand(m).-0.5)*π
	v=(rand(m).-0.5)*2*π
	for i=1:m
	    X[i,1]=X[i,1]*cos(u[i])*cos(v[i])
	    X[i,2]=X[i,2]*cos(u[i])*sin(v[i])
	    X[i,3]=X[i,3]*sin(u[i])
	end
end

# ╔═╡ 61435c77-989b-4490-ba29-acc23420d021
X₀=copy(X)

# ╔═╡ 4bb4a5d4-4cfd-4ddc-b3a8-6b59f1e6f562
sum(abs.(X₀),dims=1)

# ╔═╡ 9ce6a3e8-b706-4014-92e9-4c03b5eb381b
# Plot the set
scatter(X₀[:,1],X₀[:,2],X₀[:,3],legend=false,ms=2,aspect_ratio=:equal,xlabel="x")

# ╔═╡ 7f033a55-ebf3-4265-85c1-f3358d165ab9
# Compute the means. How good is the RNG?
μ₀=mean(X₀,dims=1)

# ╔═╡ f427adaa-c41f-4017-ada6-cd018bbc642c
begin
	# Subtract the means, rotate by a random orthogonal matrix Q and 
	# translate by S
	Q,r=qr(rand(3,3))
	S=[3,-2,4]
	X₁=(X₀.-μ₀)*Q .+S'
	scatter(X₁[:,1],X₁[:,2],X₁[:,3],legend=false,ms=2,aspect_ratio=:equal,xlabel="x")
end

# ╔═╡ 66a8e281-4709-4548-b13b-5dd5c23805e6
C₁=cov(X₁)

# ╔═╡ 685bdf11-79bd-4070-9b30-3dfa0341e779
μ₁=mean(X₁,dims=1)

# ╔═╡ 5533faf5-7847-468b-bd62-244b2581cad6
# Fact 2
cov(X₁.-μ₁), (X₁.-μ₁)'*(X₁.-μ₁)/(m-1)

# ╔═╡ db7e0f78-0a05-4719-81f1-cd007ff1163b
# Principal components, evals are non-decreasing
λ₁,U₁=eigen(C₁)

# ╔═╡ 0e179343-1f3e-4b5f-8560-c5acbfa8f6e6
begin
	# Largest principal component
	T₁=(X₁.-μ₁)*U₁[:,3]
	p₁=scatter(T₁,zero(T₁),marker=:plus,ms=2)
	p₂=scatter(T₁)
	plot(p₁,p₂,layout=(1,2),legend=false,m=:plus,ms=2)
end

# ╔═╡ 5b208afb-13cc-4937-82f8-df67830a4914
begin
	# Two largest principal components
	T₂=(X₁.-μ₁)*U₁[:,[3,2]]
	scatter(T₂[:,1],T₂[:,2],legend=false,m=:plus,ms=2,ratio=:equal)
end

# ╔═╡ d2dacf61-8728-4046-9dae-0bafe842cebf
begin
	# All  three principal components
	T₃=(X₁.-μ₁)*U₁[:,[3,2,1]]
	scatter(T₃[:,1],T₃[:,2],T₃[:,3],legend=false,m=:plus,ms=2,ratio=:equal)
end

# ╔═╡ e1b16942-6e32-45f6-a3c8-eb259ccd9e5b
begin
	# Fact 5 - Recovery of the largest component
	Y₁=T₁*U₁[:,3]'.+μ₁
	scatter(Y₁[:,1],Y₁[:,2],Y₁[:,3],legend=false,m=:plus,ms=2,ratio=:equal)
end

# ╔═╡ 56ddc45c-9b40-4c76-8ea1-8235f9690d73
begin
	# Recovery of the two largest components
	Y₂=T₂*U₁[:,[3,2]]'.+μ₁
	scatter(Y₂[:,1],Y₂[:,2],Y₂[:,3],legend=false,m=:plus,ms=2,ratio=:equal)
end

# ╔═╡ d06b73ae-1556-4c04-8f39-b7b12ae5bc72
begin
	# Recovery of all three components (exact)
	Y₃=T₃*U₁[:,[3,2,1]]'.+μ₁
	scatter(Y₃[:,1],Y₃[:,2],Y₃[:,3],legend=false,m=:plus,ms=2,ratio=:equal)
	scatter!(X₁[:,1],X₁[:,2],X₁[:,3],legend=false,m=:circle,ms=2,mc=:red,ratio=:equal)
end

# ╔═╡ e50ce490-bf7e-40b0-9dd5-7e2ee5a057cb
# Fact 4 - PCA using SVD
function PCA(X::Array{T}, k::Int) where T
    μ=mean(X,dims=1)
    U=svd(X.-μ)
    U.U[:,1:k]*Diagonal(U.S[1:k])
end

# ╔═╡ 12580ca2-5703-4933-ae1a-eb69c3e22109
begin
	T₁s=PCA(X,1)
	[T₁s T₁]
end

# ╔═╡ 38318bfc-d72a-4da0-9862-235fb43f7c61
begin
	# The two largest components using SVD
	T₂s=PCA(X,2)
	[T₂s T₂]
end

# ╔═╡ ff6f6c9f-0e14-47a5-8bcf-29337a45d042
md"""
### Real data 1 

We will cluster three datasets from [Workshop "Principal Manifolds-2006"](http://www.ihes.fr/~zinovyev/princmanif2006/):

Data set | # of genes (m) | # of samples (n)
:---:|---:|---:
D1 | 17816 | 286
D2 | 3036 | 40
D3 | 10383| 103
"""

# ╔═╡ 8c5c604c-5043-4dd0-a604-99c958860c14
# Data set D₁
# readdlm("files/d1.txt")

# ╔═╡ 73c0571b-60c9-4a85-bbfa-39a6f42192a2
D₁=map(Float64,readdlm("files/d1.txt")[2:end,2:end])

# ╔═╡ d70d8dd7-a9a5-48bf-8a1b-789d02c35d02
sizeof(D₁)

# ╔═╡ 21bc5eee-ebd9-4192-a240-59cd3f4a33b2
# Fact 7 - Plot σ's and observe the knee
scatter(svdvals(D₁.-mean(D₁,dims=1)),leg=false,ms=2)

# ╔═╡ 83660bc1-8116-476e-9c88-0ea176120875
begin
	# PCA on D₁, keep 20 singular values
	k₁=20
	T=PCA(D₁,k₁)
end

# ╔═╡ 500d16ac-0dbe-4815-8fbb-922adf751d9f
# Find k clusters
out=kmeans(T',k₁);

# ╔═╡ 8c6255cb-60f4-4c63-b4b2-e403d74360e1
# Plot points defined by the first three principal components
function Plot(C::Vector, T::Array, k::Int)
    P=Array{Any}(undef,k)
	scatter(legend=false)
    for j=1:k
        P[j]=T[C.==j,1:3]
        scatter!(P[j][:,1],P[j][:,2],P[j][:,3],ms=2)
    end
	scatter!()
end

# ╔═╡ dff6d995-3753-4c70-b286-dd5aaf7f6461
Plot(out.assignments,T,k₁)

# ╔═╡ 54d17292-195e-4ad4-8848-c725d1cbf8bf
md"
### Real data 2
"

# ╔═╡ e133dab4-f9fe-47ed-a9a5-2644c1a954c9
# Data set D₂
D₂=readdlm("files/d2fn.txt")

# ╔═╡ 69016fc2-4249-4ce8-9d02-4371ba4ac6d7
# Plot σ's and observe knee
scatter(svdvals(D₂.-mean(D₂,dims=1)),leg=false,ms=2)

# ╔═╡ d5713d68-6413-4873-9d40-d8a5134ceb3d
begin
	k₂=5
	T²=PCA(D₂,k₂)
	out₂=kmeans(T²',k₂)
	Plot(out₂.assignments,T²,k₂)
end

# ╔═╡ 60d2fab0-9951-48b1-a9e4-81ff10b1bf40
md"""

### Real data 3

We use the package [CSV.jl](https://github.com/JuliaData/CSV.jl) and replace the "NULL" string with `missing`.
"""

# ╔═╡ dcef3495-9a8b-4e7f-bbda-865cac3e894e
sf = CSV.read("files/d3efn.txt",DataFrame,header=0,missingstring="NULL")

# ╔═╡ 5cabfc3e-54a1-40aa-8979-bc5cc10d2ffd
# sf[:,103]

# ╔═╡ 53191664-ec06-441c-b1ee-5ee8034c4984
typeof(sf)

# ╔═╡ 9ab1ba2b-30fd-4cca-be25-3090608d18be
# Set missing values to zero
D₃=coalesce.(Matrix(sf[:,1:end-1]),0.0)

# ╔═╡ a4ea64d4-57f3-4d43-9083-3e52b8e213d7
# Plot σ's and observe knee
scatter(svdvals(D₃.-mean(D₃,dims=1)),leg=false,ms=2)

# ╔═╡ d6143ca0-ba0e-406a-8b3f-fbcd348777bb
begin
	k₃=10
	T³=PCA(D₃,k₃)
	out₃=kmeans(T³',k₃)
	Plot(out₃.assignments,T³,k₃)
end

# ╔═╡ 2a85afb0-a3e8-477b-bc9b-e97701c40eae
begin
	# Clustering the transpose
	k₄=20
	T⁴=PCA(Matrix(D₃'),k₄)
	out₄=kmeans(T⁴',k₄)
	Plot(out₄.assignments,T⁴,k₄)
end

# ╔═╡ Cell order:
# ╠═d6b500f1-5d54-4cdb-8d74-56e8a9ab367e
# ╠═e166839c-7451-471f-b10f-3cc8d3d751f3
# ╠═c810e2a4-03a5-4d19-8717-3de25f013269
# ╟─d7ef9ec9-1acf-48e6-bfbc-55ec01b36b0b
# ╟─728f10ec-2843-41ae-ae8e-c913155fe303
# ╟─1eec2efb-62b7-493b-8a9c-d3784d2e7825
# ╟─63929a68-2407-4b46-80bc-91c5da382931
# ╠═1c25240c-d396-4661-b3f3-fd2b30876a00
# ╠═61435c77-989b-4490-ba29-acc23420d021
# ╠═4bb4a5d4-4cfd-4ddc-b3a8-6b59f1e6f562
# ╠═9ce6a3e8-b706-4014-92e9-4c03b5eb381b
# ╠═7f033a55-ebf3-4265-85c1-f3358d165ab9
# ╠═f427adaa-c41f-4017-ada6-cd018bbc642c
# ╠═66a8e281-4709-4548-b13b-5dd5c23805e6
# ╠═685bdf11-79bd-4070-9b30-3dfa0341e779
# ╠═5533faf5-7847-468b-bd62-244b2581cad6
# ╠═db7e0f78-0a05-4719-81f1-cd007ff1163b
# ╠═0e179343-1f3e-4b5f-8560-c5acbfa8f6e6
# ╠═5b208afb-13cc-4937-82f8-df67830a4914
# ╠═d2dacf61-8728-4046-9dae-0bafe842cebf
# ╠═e1b16942-6e32-45f6-a3c8-eb259ccd9e5b
# ╠═56ddc45c-9b40-4c76-8ea1-8235f9690d73
# ╠═d06b73ae-1556-4c04-8f39-b7b12ae5bc72
# ╠═e50ce490-bf7e-40b0-9dd5-7e2ee5a057cb
# ╠═12580ca2-5703-4933-ae1a-eb69c3e22109
# ╠═38318bfc-d72a-4da0-9862-235fb43f7c61
# ╟─ff6f6c9f-0e14-47a5-8bcf-29337a45d042
# ╠═8c5c604c-5043-4dd0-a604-99c958860c14
# ╠═73c0571b-60c9-4a85-bbfa-39a6f42192a2
# ╠═d70d8dd7-a9a5-48bf-8a1b-789d02c35d02
# ╠═21bc5eee-ebd9-4192-a240-59cd3f4a33b2
# ╠═83660bc1-8116-476e-9c88-0ea176120875
# ╠═500d16ac-0dbe-4815-8fbb-922adf751d9f
# ╠═8c6255cb-60f4-4c63-b4b2-e403d74360e1
# ╠═dff6d995-3753-4c70-b286-dd5aaf7f6461
# ╟─54d17292-195e-4ad4-8848-c725d1cbf8bf
# ╠═e133dab4-f9fe-47ed-a9a5-2644c1a954c9
# ╠═69016fc2-4249-4ce8-9d02-4371ba4ac6d7
# ╠═d5713d68-6413-4873-9d40-d8a5134ceb3d
# ╟─60d2fab0-9951-48b1-a9e4-81ff10b1bf40
# ╠═dcef3495-9a8b-4e7f-bbda-865cac3e894e
# ╠═5cabfc3e-54a1-40aa-8979-bc5cc10d2ffd
# ╠═53191664-ec06-441c-b1ee-5ee8034c4984
# ╠═9ab1ba2b-30fd-4cca-be25-3090608d18be
# ╠═a4ea64d4-57f3-4d43-9083-3e52b8e213d7
# ╠═d6143ca0-ba0e-406a-8b3f-fbcd348777bb
# ╠═2a85afb0-a3e8-477b-bc9b-e97701c40eae
