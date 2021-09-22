### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 53107773-07e7-4da5-9900-8ba5cefd7b68
begin
	using Plots, Clp, JuMP, Distributions, LinearAlgebra, TestImages
	using SparseArrays, FFTW, Random, PlutoUI, Wavelets, Images
	plotly()
end

# ╔═╡ c806af6e-fdda-4e4c-8768-9e9b07e286c6
# On binder, remove the comments
#=
begin
	import Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
		Pkg.PackageSpec(name="Plots"),
        Pkg.PackageSpec(name="PlutoUI"),
		Pkg.PackageSpec(name="Clp"),
		Pkg.PackageSpec(name="JuMP"),
		Pkg.PackageSpec(name="Distributions"),
		Pkg.PackageSpec(name="SparseArrays"),
		Pkg.PackageSpec(name="FFTW"),
		Pkg.PackageSpec(name="Wavelets"),
		Pkg.PackageSpec(name="Images"),
		Pkg.PackageSpec(name="TestImages")
    ])
end
=#

# ╔═╡ 32ab2dfc-4381-4388-80ff-bbdbda513180
TableOfContents(title="📚 Table of Contents", aside=true)

# ╔═╡ a5172a1d-a631-49b3-bc9a-21e98d4cf9e5
md"""
# Compressive Sensing

The task is to extract images or signals accurately and even exactly from a number of
samples which is far smaller than the desired resolution of the image/signal, e.g., the
number of pixels in the image. This new technique draws from results in several fields

Suppose we are given a sparse signal.

__Can we recover the signal with small number of measurements 
(far smaller than the desired resolution of the signal)?__

The answer is __YES, for some signals and carefully selected measurements using $l_1$ minimization.__


__Prerequisites__

The reader should be familiar to elementary concepts about signals, with linear algebra concepts, and linear programming.

__Competences__ 

The reader should be able to recover a signal from a small number of measurements.

__References__

For more details see

* [E. Candès and M. Wakin, An Introduction To Compressive Sampling](https://authors.library.caltech.edu/10092/1/CANieeespm08.pdf),
* [M. Davenport et al., Introduction to Compressed Sensing](http://www.ecs.umass.edu/~mduarte/images/IntroCS.pdf)
* [O. Holtz, Compressive sensing: a paradigm shift in signal processing](http://arxiv.org/abs/0812.3137), 
* [G. Kutyniok, Theory and Applications of Compressed Sensing](https://arxiv.org/abs/1203.3815)
* and the extensive list of [Compressive Sensing Resources](http://dsp.rice.edu/cs).



__Credits__: Daniel Bragg, an IASTE Intern, performed testing of some of the methods.
"""

# ╔═╡ e0b4e064-da18-45ae-8d96-d723b4cafac6
md"""
# Underdetermined systems

Let $A\in\mathbb{R}^{m\times n}$ with $m<n$, $x\in\mathbb{R}^n$ and $b\in\mathbb{R}^m$.

## Definitions

The system $Ax=b$ is __underdetermined__.

 $\|x\|_0$ is the number of nonzero entries of $x$ (__a quasi-norm__).

A matrix $A$ satisfies the __restricted isometry property__ (RIP) of order $k$ with constant $\delta_k\in(0,1)$ 
if 

$$
(1 − \delta_k )\|x\|_2^2 \leq \| Ax\|_2^2 \leq (1 + \delta_k)\| x\|_2^2$$

for any $x$ such that $\|x\|_0 \leq k$.

A __mutual incoherence__ of a matrix $A$ is 

$$
\mathcal{M}(A)= \max_{i \neq j} |[A^TA]_{ij}|,$$

that is, the absolutely maximal inner product of distinct columns of $A$. If the columns of $A$ have unit norms, $\mathcal{M}(A)\in[0,1]$.

The __spark__ of a given matrix $A$, $\mathop{\mathrm{spark}}(A)$, is the smallest number of columns of $A$ that are linearly dependant.

"""

# ╔═╡ ef27be74-2570-4dbf-9127-ca34aa3084b5
md"""
## Facts

1. An underdetermined system either has no solution or has infinitely many solutions. 

2. The typical problem is to choose the solution of some minimal norm. This problem can be reformulated as a constrained optimization problem

$$
\textrm{minimize}\ \|x\| \quad \textrm{subject to} \quad Ax=b.$$


3. For the 2-norm, the $l_2$ minimization problem is solved by SVD: let $\mathop{\mathrm{rank}}(A)=r$ and let $A=U\Sigma V^T$ be the SVD of $A$. Then 

$$
x=\sum_{k=1}^r \frac{U[:,k]^Tb}{\sigma_k} V[:,k].$$

2. For the 1-norm, the $l_1$ minimization problem is a __linear programming__ problem
    
$$\textrm{minimize}\ \ c^T x \quad \textrm{subject to} \quad Ax=b,\ x\geq 0,$$
    
for $c^T=\begin{bmatrix}1 & 1 & \cdots & 1 \end{bmatrix}$.
    
3. For the 0-norm, the $l_0$ problem (which appears in compressive sensing)
    
$$
\textrm{minimize}\ \|x\|_0 \quad \textrm{subject to} \quad Ax=b,$$

is NP-hard.

4. It holds $\mathop{\mathrm{spark}}(A)\in[2,m+1]$.

5. For any vector $b$, there exists at most one vector $x$ such that $\|x\|_0\leq k$ and $Ax=b$ if and only if $\mathop{\mathrm{spark}}(A) > 2k$. This implies that $m\geq 2k$, which is a good choice when we are computing solutions which are exactly sparse.

6. If 

$$k<\displaystyle \frac{1}{2} \left(1+\frac{1}{\mathcal{M}(A)}\right),$$

then for any vector $b$ there exists at most one vector $x$ such that $\|x\|_0\leq k$ and $Ax=b$.

7. If the solution $x$ of $l_0$ problem satisfies 

$\|x\|_0 < \displaystyle \frac{\sqrt{2}-1/2}{\mathcal{M}(A)},$

then the solution of $l_1$ problem is the solution of $l_0$ problem!

7. If $A$ has columns of unit-norm, then $A$ satisfies the RIP of order $k$ with 
$\delta_k = (k − 1)\mathcal{M}(A)$  for all $k < 1/\mathcal{M}(A)$.

8. If $A$ satisfies RIP of order $2k$ with $\delta_{2k}<\sqrt{2}-1$, then the solution of $l_1$ problem is the solution of $l_0$ problem! 

9. Checking whether the specific matrix has RIP is difficult. If $m ≥ C \cdot k \log\left(\displaystyle\frac{n}{k}\right)$, where $C$ is some constant depending on each instance, the following classes of matrices satisfy RIP with $\delta_{2k}<\sqrt{2}-1$ with overwhelming probability(the matrices are normalised to have columns with unit norms):
    
   1. Form $A$ by sampling at random $n$ column vectors on the unit sphere in $\mathbb{R}^m$.
   2. Form $A$ by sampling entries from the normal distribution with mean 0 and variance $1/ m$.
   3. Form $A$ by sampling entries from a symmetric Bernoulli distribution $P(A_{ij} = ±1/\sqrt{m}) = 1/2$.
   4. Form $A$ by sampling at random $m$ rows of the Fourier matrix.

10. The __compressive sensing__ interpretation is the following: the signal $x$ is reconstructed from samples with $m$ __functionals__ (the rows of $A$). 
 
"""

# ╔═╡ 2d1d9c68-228e-48e7-88b5-d3f3e439065e
md"""
## Examples 
### $l_2$ minimization
"""

# ╔═╡ 75d9b8f8-42c0-4a02-9683-878a351bb137
begin
	Random.seed!(678)
	A=rand(5,8)
	b=rand(5)
	x=A\b
end

# ╔═╡ f215166e-6a12-4859-920e-9856382c6f82
A

# ╔═╡ fcc27168-da40-446c-811e-4956ad27a11c
b

# ╔═╡ 17d946d6-458a-482c-9b28-d71c15c4027e
begin
	U,σ,V=svd(A)
	norm(A*x-b), norm( sum( [(U[:,k]'*b/σ[k])[1]*V[:,k]  for k=1:5])-x)
end

# ╔═╡ 4ae0b6d2-2e9b-4b09-aaef-caa899824853
md"""
### Small linear programming example

$$\begin{split}\min_{x,y}\, &-x\\
s.t.\quad          &2x + y \leq 1.5\\
& x \geq 0,\quad  y \geq 0\end{split}$$
"""

# ╔═╡ d498125b-54a0-401e-bd54-26a2b2d99ca5
begin
	model₀ = Model(with_optimizer(Clp.Optimizer))
	x₀, y₀, con = nothing, nothing, nothing
	@variable(model₀, 0 <= x₀)
	@variable(model₀, 0 <= y₀)
	@objective(model₀, Min, -x₀)
	@constraint(model₀, con, 2x₀ + 1y₀ <= 1.5)
	optimize!(model₀)
end

# ╔═╡ ac2841c5-8ec9-43de-a9c6-425524a40967
termination_status(model₀) == MOI.OPTIMAL

# ╔═╡ 0e5dbfb4-2c60-4a37-af42-a4788db0b281
objective_value(model₀)

# ╔═╡ 5525bba3-b96d-40e6-97fb-132c6767337d
value(x₀),value(y₀)

# ╔═╡ 35b0bd70-b0f8-4ca8-a7e7-805e31a032d8
md"""
### Exact sparse signal recovery

We recover randomly generated sparse signals "measured" with rows of the matrix $A$. 
The experiment is performed for types of matrices from Fact 9.

The $l_1$ minimization problem is solved using the package [JuMP.jl](https://github.com/JuliaOpt/JuMP.jl) with the linear programming solver from the package [Clp.jl](https://github.com/JuliaOpt/Clp.jl).

Random matrices are generated using the package [Distributions.jl](https://github.com/JuliaStats/Distributions.jl).
"""

# ╔═╡ 4c96769b-ec49-4134-91e3-df77545fcc61
function SamplingMatrix(m::Int,n::Int,kind::String)
    # kind is "Random" for Fact 9A, "Normal" for 9B, 
    # "Bernoulli" for 9C, and "Fourier" for 9D.
    if kind=="Random"
        A=svd(rand(m,n)).Vt
    elseif kind=="Normal"
        A=rand(Normal(0,1/m),m,n)
    elseif kind=="Bernoulli"
        A=2*(rand(Bernoulli(0.5),m,n).-0.5)
    elseif kind=="Fourier"
        # Elegant way of computing the Fourier matrix
        F=fft(Matrix(I,n,n),1)
        # Select m/2 random rows
        ind=Random.randperm(n)[1:div(m,2)]
        Fm=F[ind,:]
        # We need to work with real matrices
        A=[real(Fm); imag(Fm)]
    else
        return "Error"
    end
    # Normalize columns of A
    for i=1:size(A,2)
        normalize!(view(A,:,i))
    end
    return A
end

# ╔═╡ 9e02f055-6f3d-42f8-940e-f3bd944942fd
function recovery(A,b)
    model = Model(with_optimizer(Clp.Optimizer))
    @variable(model, 0<=x[1:size(A,2)])
    @objective(model, Min, sum(x))
    @constraint(model, con, A*x .== b)
    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        return value.(x)
    else
        return "Error"
    end
end

# ╔═╡ 9d5e6751-9e3e-4b0a-8fd3-35a43e079ee7
begin
	Random.seed!(4308)
	# Dimension of the sparse vector
	n=200 
	# Dimension of the sampled vector
	m=40
	# Sparsity
	k=12
	# Generate random sparse vector
	xₛ=sprand(n,k/n)
	# Generate sampling matrix, "Random", "Normal", "Bernoulli", "Fourier"
	kind="Random"
	Aₛ=SamplingMatrix(m,n,kind)
	# Check incoherence
	maximum(abs,Aₛ'*Aₛ-I),nnz(xₛ)
end

# ╔═╡ c17eac6a-1ab5-406a-b063-95c91dd81e46
# Sample x
bₛ=Aₛ*xₛ

# ╔═╡ acab203d-f158-4460-9452-7037895559bd
xᵣ=recovery(Aₛ,bₛ)

# ╔═╡ 372b7d99-b6e8-4656-aaa5-a5c489105e80
begin
	# Plot the solution
	scatter(xₛ,title="Exact Sparse Recovery",label="Original data")
	scatter!(xᵣ,label="Recovered data")
end

# ╔═╡ 8825ba12-98ff-48c9-b33a-a1d1c0da2eb3
md"""
# Recovery from noisy observations

In the presence of noise in observation, we want to recover a vector $x$ from $b=Ax + z$, where $z$ is a stochastic or deterministic unknown error term.

## Definition

The __hard thresholding operator__, $H_k(x)$, sets all but the $k$ entries of $x$ with largest magnitude to zero.

## Facts

1. The problem can be formulated as $l_1$ minimization problem

$$
\textrm{minimize}\ \|x\|_1 \quad \textrm{subject to} \quad \|Ax-b\|_2^2\leq\epsilon,$$

where $\epsilon$ bounds the amount of noise in the data.

2. Assume that $A$ satisfies RIP of order $2k$ with $\delta_{2k}< \sqrt{2}-1$. Then the solution $x^{\star}$ of the above problem satisfies 

$$
\|x^{\star}-x\|_2 \leq C_0 \displaystyle \frac{1}{\sqrt{k}}\|x-H_k(x)\|_1
+C_1\epsilon,$$

where $x$ is the original signal.

3. The $l_1$ problem is a convex programming problem and can be efficiently solved. The solution methods are beyond the scope of this course. 

4. If $k$ is known in advance, $A$ satisfies RIP with $\delta_{3k}<1/15$, and $\|A\|_2<1$, the $k$-sparse aproximation of $x$ can be computed by the __Iterative Hard Thresholding__ algorithm:
    1. __Initialization:__ $x=0$.
    2. __Iteration:__ repeat until convergence $x=H_k(x+A^T(b-Ax))$.
"""

# ╔═╡ 022cecdf-945c-4fa9-a26e-a13b11068ca7
# Iterative Hard Thresholding 
function H(x::Vector,k::Int)
    y=copy(x)
    ind=sortperm(abs.(y),rev=true)
    y[ind[k+1:end]].=0
    y
end

# ╔═╡ b275c776-47c4-4ed8-ba9d-c4ce1926d540
function IHT(A::Matrix, b::Vector,k::Int)
    # Tolerance
    τ=1e-12
    x=zeros(size(A,2))
    for i=1:50*m
        x=H(x+A'*(b-A*x),k)
    end
    x
end

# ╔═╡ 75621541-7741-4b23-92b4-56fda45ad6cf
md"""
##  Example

We construct the $k$ sparse $x$, form $b$, add noise, and recover it with the algorithm from Fact 4. The conditions on $A$ are rather restrictive, which means that $k$ must be rather small compared to $n$ and $m$ must be rather large. For convergence, we limit the number of iterations to $50m$.
"""

# ╔═╡ a2326335-6737-49e4-af42-55a114c66a35
begin
	Random.seed!(421)
	# Dimension of the sparse vector
	nₙ=300
	# Sparsity - kₙ is small compared to nₙ
	kₙ=8
	# Dimension of the sampled vector - mₙ is rather large
	mₙ=5*round(Int,kₙ*log(nₙ/kₙ))
	# Generate random sparse vector
	xₙ=10*sprand(nₙ,kₙ/nₙ)
	# Generate sampling matrix
	Aₙ=SamplingMatrix(mₙ,nₙ,"Normal")
	mₙ, nₙ
end

# ╔═╡ f4c0fff3-ab24-474d-8e28-3b77782d4588
begin
	# Sample xₙ
	bₙ=Aₙ*xₙ
	# Add noise
	noise=rand(mₙ)*1e-5
	bₙ.+=noise
end

# ╔═╡ 6477a3bd-2a9a-45e9-a59e-1ffdd8c21890
begin
	yₙ=IHT(Aₙ,bₙ,kₙ)
	norm(Aₙ*xₙ-bₙ)/norm(bₙ)
end

# ╔═╡ edb56b65-1ed2-4202-ad2d-02d7c4f026f1
SparseMatrixCSC(xₙ);

# ╔═╡ 3629a2a6-15c5-40fe-ba9a-2a605743ff5a
md"""
Let us try linear programing in the case of noisy observations.
"""

# ╔═╡ bd2cd42e-3dd4-453f-8046-d0292eae6896
zₙ=recovery(Aₙ,bₙ)

# ╔═╡ 1703a5e3-2985-4770-8ea9-e6a49b1f46d9
begin
	# Plot the solution
	scatter(xₙ,title="Noisy Sparse Recovery",label="Original data")
	scatter!(yₙ,label="Recovered data - IHT")
	scatter!(zₙ,label="Recovered data - l₁ minimization",legend=true)
end

# ╔═╡ cd074ea6-27b8-4f76-a025-6790b7a95ade
md"""
# Sensing images

Wavelet transformation of an image is essentially sparse, since only small number of cofficients is significant. This fact can be used for compression.

Wavelet transforms are implemented the package 
[Wavelets.jl](https://github.com/JuliaDSP/Wavelets.jl).

## Lena and R

The `tif` version of the image "Lena" has `65_798` bytes, the `png` version has `58_837` bytes, and the `jpeg` version has `26_214` bytes. We also test the algorithm on a simpler image of letter "R".
"""

# ╔═╡ a6b83c90-9431-471e-bef3-6de5a2b7d920
TestImages.remotefiles

# ╔═╡ 01fff994-6b74-45e9-ab65-b52c6339fe45
img=testimage("lena_gray_256.tif")
# img=map(Gray,load("./files/RDuarte.png"))

# ╔═╡ 9c0a1f14-f63d-4387-905d-809d9ffbf63a
typeof(img)

# ╔═╡ 52643e9d-9ac6-4b33-96d6-25803d8186d0
size(img)

# ╔═╡ a37a00b6-608b-4d99-be2c-a91cb1fa9bcd
show(img[1,1])

# ╔═╡ 5dd3690b-6957-49f2-95db-4d8ab223acf7
begin
	# Convert the image to 32 or 64 bit floats
	xₗ=map(Float32,img)
	" Number of matrix elements = ",prod(size(xₗ)), 
	" Bytes = ",sizeof(xₗ), size(xₗ)
end

# ╔═╡ 49c5d771-f06b-4cdb-982a-057639b3d22e
begin
	# Compute the wavelet transform of x or wavelet(WT.db3)
	wlet=wavelet(WT.Coiflet{4}(), WT.Filter, WT.Periodic)
	xₜ=dwt(xₗ,wlet,2);
end

# ╔═╡ 668ccf49-ba28-4a6d-ab95-a2a4c876b57f
colorview(Gray,xₜ)

# ╔═╡ 0c2f32af-0156-4e4a-a957-03032c10c4ce
md"""
We now set __all except__ the 10% or 5% absolutely largest coefficients to zero and reconstruct the image. The images are very similar, which illustrates that the wavelet transform of an image is essentially sparse.
"""

# ╔═╡ a2bff419-b8b7-44b7-834d-6104d187e31b
begin
	ind=sortperm(abs.(vec(xₜ)),rev=true)
	# 0.1 = 10%, try also 0.05 = 5%
	τ=0.05
	kₜ=round(Int,τ*length(ind))
	xsparse=copy(xₜ)
	xsparse[ind[kₜ+1:end]].=0;
end

# ╔═╡ 9823947d-6a81-40a4-9d4c-7ade1ac864af
nnz(sparse(xsparse))

# ╔═╡ a7fe5690-7b50-4922-b8cc-0e3e672ac25f
colorview(Gray,xsparse)

# ╔═╡ b2470310-2356-11eb-249f-a90af0c70a78
# Inverse wavelet transform of the sparse data
imgsparse=idwt(xsparse,wlet,2)

# ╔═╡ bfe752de-2356-11eb-02cc-8197d9ab259f
# Original and sparse image
img

# ╔═╡ e336cc06-c019-4613-8b49-e231e0f77726
colorview(Gray,imgsparse)

# ╔═╡ 9b06f8d8-0f8e-4eb3-997f-7fab1ede66bd
md"""
There are $k=6554$  $(3277)$ nonzero coefficients in a sparse wavelet representation. 

Is there the sensing matrix which can be used to sample and recover `xsparse`?

Actual algorithms are elaborate. For more details see [J. Romberg, Imaging via Compressive Sampling](https://authors.library.caltech.edu/10093/1/ROMieeespm08.pdf), IEEE Signal Processing Magazine, 25(2) (2008) 14-20., and [Duarte et al.,Single-Pixel Imaging via Compressive Sampling](http://www.wisdom.weizmann.ac.il/~vision/courses/2010_2/papers/csCamera-SPMag-web.pdf).
"""

# ╔═╡ 27f33bc4-c6c9-43f1-8b82-df5b4f710ad4
nnz(sparse(xsparse))

# ╔═╡ 1203652e-ee44-4f7b-af4c-e08d091d1c84
# Maximal number of nonzeros in a column 
maximum([nnz(sparse(xsparse[:,i])) for i=1:size(x,2)])

# ╔═╡ c2ecb04a-f354-4a8e-b85b-83fd59e19d08
begin
	# Dimensions
	nᵢ=size(xsparse,2)
	mᵢ=150
	Aᵢ=SamplingMatrix(mᵢ,nᵢ,"Normal")
	# Sampling (columnwise)
	bᵢ=Aᵢ*xsparse
end

# ╔═╡ 452464c2-eb0f-43e2-a833-ffa3c5724fd2
begin
	# Recovery columnwise
	xrecover=similar(xsparse)
	for l=1:size(xsparse,2)
	    zᵢ=recovery(Aᵢ,bᵢ[:,l])
	    xrecover[:,l]=zᵢ
	end
end

# ╔═╡ b108e499-5785-4f1a-ac15-098af656c134
xrecover

# ╔═╡ 12cb3942-6109-42a4-a8d8-aff3c261dff8
imgrecover=idwt(xrecover, wlet, 2)

# ╔═╡ eabace10-2357-11eb-204a-d76ba271bd7e
# imgrecover=idct(xrecover)
# Original sparse image and and recovered sparse image
colorview(Gray,imgsparse)

# ╔═╡ eabe9ea2-2357-11eb-0ea1-c32d6a429aa1
colorview(Gray,imgrecover)

# ╔═╡ 036e6c2e-c820-473f-9530-c7b74093b0eb
size(bᵢ)

# ╔═╡ Cell order:
# ╠═c806af6e-fdda-4e4c-8768-9e9b07e286c6
# ╠═53107773-07e7-4da5-9900-8ba5cefd7b68
# ╠═32ab2dfc-4381-4388-80ff-bbdbda513180
# ╟─a5172a1d-a631-49b3-bc9a-21e98d4cf9e5
# ╟─e0b4e064-da18-45ae-8d96-d723b4cafac6
# ╟─ef27be74-2570-4dbf-9127-ca34aa3084b5
# ╟─2d1d9c68-228e-48e7-88b5-d3f3e439065e
# ╠═75d9b8f8-42c0-4a02-9683-878a351bb137
# ╠═f215166e-6a12-4859-920e-9856382c6f82
# ╠═fcc27168-da40-446c-811e-4956ad27a11c
# ╠═17d946d6-458a-482c-9b28-d71c15c4027e
# ╟─4ae0b6d2-2e9b-4b09-aaef-caa899824853
# ╠═d498125b-54a0-401e-bd54-26a2b2d99ca5
# ╠═ac2841c5-8ec9-43de-a9c6-425524a40967
# ╠═0e5dbfb4-2c60-4a37-af42-a4788db0b281
# ╠═5525bba3-b96d-40e6-97fb-132c6767337d
# ╟─35b0bd70-b0f8-4ca8-a7e7-805e31a032d8
# ╠═4c96769b-ec49-4134-91e3-df77545fcc61
# ╠═9e02f055-6f3d-42f8-940e-f3bd944942fd
# ╠═9d5e6751-9e3e-4b0a-8fd3-35a43e079ee7
# ╠═c17eac6a-1ab5-406a-b063-95c91dd81e46
# ╠═acab203d-f158-4460-9452-7037895559bd
# ╠═372b7d99-b6e8-4656-aaa5-a5c489105e80
# ╟─8825ba12-98ff-48c9-b33a-a1d1c0da2eb3
# ╠═022cecdf-945c-4fa9-a26e-a13b11068ca7
# ╠═b275c776-47c4-4ed8-ba9d-c4ce1926d540
# ╟─75621541-7741-4b23-92b4-56fda45ad6cf
# ╠═a2326335-6737-49e4-af42-55a114c66a35
# ╠═f4c0fff3-ab24-474d-8e28-3b77782d4588
# ╠═6477a3bd-2a9a-45e9-a59e-1ffdd8c21890
# ╠═edb56b65-1ed2-4202-ad2d-02d7c4f026f1
# ╟─3629a2a6-15c5-40fe-ba9a-2a605743ff5a
# ╠═bd2cd42e-3dd4-453f-8046-d0292eae6896
# ╠═1703a5e3-2985-4770-8ea9-e6a49b1f46d9
# ╟─cd074ea6-27b8-4f76-a025-6790b7a95ade
# ╠═a6b83c90-9431-471e-bef3-6de5a2b7d920
# ╠═01fff994-6b74-45e9-ab65-b52c6339fe45
# ╠═9c0a1f14-f63d-4387-905d-809d9ffbf63a
# ╠═52643e9d-9ac6-4b33-96d6-25803d8186d0
# ╠═a37a00b6-608b-4d99-be2c-a91cb1fa9bcd
# ╠═5dd3690b-6957-49f2-95db-4d8ab223acf7
# ╠═49c5d771-f06b-4cdb-982a-057639b3d22e
# ╠═668ccf49-ba28-4a6d-ab95-a2a4c876b57f
# ╟─0c2f32af-0156-4e4a-a957-03032c10c4ce
# ╠═a2bff419-b8b7-44b7-834d-6104d187e31b
# ╠═9823947d-6a81-40a4-9d4c-7ade1ac864af
# ╠═a7fe5690-7b50-4922-b8cc-0e3e672ac25f
# ╠═b2470310-2356-11eb-249f-a90af0c70a78
# ╠═bfe752de-2356-11eb-02cc-8197d9ab259f
# ╠═e336cc06-c019-4613-8b49-e231e0f77726
# ╟─9b06f8d8-0f8e-4eb3-997f-7fab1ede66bd
# ╠═27f33bc4-c6c9-43f1-8b82-df5b4f710ad4
# ╠═1203652e-ee44-4f7b-af4c-e08d091d1c84
# ╠═c2ecb04a-f354-4a8e-b85b-83fd59e19d08
# ╠═452464c2-eb0f-43e2-a833-ffa3c5724fd2
# ╠═b108e499-5785-4f1a-ac15-098af656c134
# ╠═12cb3942-6109-42a4-a8d8-aff3c261dff8
# ╠═eabace10-2357-11eb-204a-d76ba271bd7e
# ╠═eabe9ea2-2357-11eb-0ea1-c32d6a429aa1
# ╠═036e6c2e-c820-473f-9530-c7b74093b0eb
