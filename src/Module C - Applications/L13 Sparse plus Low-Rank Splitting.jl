### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 0758937c-478f-45b7-a1d4-31810eecfd60
begin
	using PlutoUI
	PlutoUI.TableOfContents(aside=true)
end

# ╔═╡ 7bf123d1-dc37-44d0-9da9-982dbca9ec02
using LinearAlgebra

# ╔═╡ e0d1f2ca-ea96-4ae8-a3bd-cfabe92978d5
using SparseArrays

# ╔═╡ ea479efe-0f65-4365-a136-14afdfb180e2
using Images

# ╔═╡ 56d22088-733f-4b76-8b56-592229cd6a25
md"""
# Sparse + Low-Rank Splitting

Suppose we are given a data matrix $A$, and know that it has a form $$A=L+S,$$ where $L$ is of low-rank and $S$ is sparse, but we know neither the rank of $L$ nor the non-zero entries of $S$. 

_Can we recover $L$ and $S$?_

The answer is _YES, with high probability and with an efficient algorithm_. 

_Sparse + Low-rank_ splitting can be successfully applied to video surveilance, face recognition, latent semantic indexing, and ranking and collaborative filtering. 

__Prerequisites__

The reader should be familiar with linear algebra concepts, particularly SVD and its properties and algorithms.
 
__Competences__

The reader should be able to apply sparse + low-rank splitting to real problems.

__References__

For more details see [E. J. Candes, X. Li, Y. Ma, and J. Wright, Robust Principal Component Analysis?](http://arxiv.org/abs/0912.3599)

__Credits__: The author wishes to thank Dimitar Ninevski, a former IAESTE intern, for collecting and preparing some of the material. 


"""

# ╔═╡ 33d0a1b9-f2f4-4f48-998f-e058c9a69e14
md"""
## Definitions

Let $A\in\mathbb{R}^{m\times n}$ have rank $r$, and let $A=U\Sigma V^T$ be its SVD.

The __nuclear norm__ of $A$ is $\|A\|_*=\sum\limits_{i=1}^r \sigma_i(A)$.

Let $\|A\|_1=\sum\limits_{i,j} |A_{ij}|$ denote the $1$-norm of $A$ seen as a long vector.

Let $\|A\|_{\infty}=\max\limits_{i,j} |A_{ij}|$ denote the $\infty$-norm of $A$ seen as a long vector.

Given $\tau>0$, the __shrinkage operator__ $\mathcal{S}_{\displaystyle\tau}:\mathbb{R}\to \mathbb{R}$ is defined by
$\mathcal{S}_{\displaystyle\tau}(x)=\mathop{\mathrm{sign}}(x)\max\{|x|-\tau,0\}$,
and is extended to matrices by applying it to each element. 

Given $\tau>0$, the __singluar value thresholding operator__ is
$\mathcal{D}_{\displaystyle\tau}(A)=U \mathcal{S}_{\displaystyle\tau} (\Sigma) V^T$.

"""

# ╔═╡ 5c78225a-99b0-4587-87a7-d7ae505a9adc
md"""
### Example
"""

# ╔═╡ 7bb93306-9994-4302-9d03-43bb4404501e
# Shrinkage
function Shr(x::Array{T},τ::T) where T
    sign.(x).*max.(abs.(x).-τ,zero(T))
end

# ╔═╡ 19982690-3cf8-45f2-a19f-3a4004ec29f0
begin
	import Random
	Random.seed!(421)
	A₀=2*rand(3,5).-1
end

# ╔═╡ 35c93042-50a1-4d17-ba93-eb4abeb32bdc
Shr(A₀,0.5)

# ╔═╡ a5c7c4fb-4d28-4cdb-a825-81b422f5357c
# Singular value thresholding
function D(A::Array{T},τ::T) where T
    # U,σ,V=svd(A)
    # This can be replaced by a faster approach
    V=svd(A)
    S=Shr(V.S,τ)
    k=count(!iszero,S)
    return (V.U[:,1:k]*Diagonal(S[1:k]))*V.Vt[1:k,:]
end

# ╔═╡ 004407fe-d861-42b6-a061-4ef41176a289
svdvals(A₀)

# ╔═╡ 701fdeda-d5e3-4686-b57d-0c0dadba7c91
C₀=D(A₀,0.5)

# ╔═╡ 78fda2e8-1fbf-4aa1-bc4a-a05c262a410a
svdvals(C₀)

# ╔═╡ a29511fa-1deb-4bea-97aa-326bd366f6ab
md"""
## Facts

Let $A=L+S$ be the splitting that we are looking for.

1. The problem can be formulated as 

$\mathop{\textrm{arg min}}\limits_{\displaystyle \mathop{\mathrm{rank}}(L)\leq k} \|A-L\|_2$.

2. The problem makes sense if the __incoherence conditions__ 

$$
\max_{\displaystyle i} \| U_{:,1:r}^T e_i\|_2^2\leq \frac{\mu r}{m}, \quad
\max_{\displaystyle i} \| V_{:,1:r}^T e_i\|_2^2\leq \frac{\mu r}{n}, \quad
\|UV^T\|_{\infty} \leq \sqrt{\frac{\mu r}{m\cdot n}},$$

hold for some parameter $\mu$.

3. If the incoherence conditions are satisfied, the __Principal Component Pursuit estimate__,

$$
\mathop{\textrm{arg min}}\limits_{\displaystyle L+S=A} \|L\|_*+\lambda \|S\|_2,$$

exactly recovers $L$ and $S$.

4. __Principal Component Pursuit by Alternating Directions__ algorithm finds the above estimate
    1. _Initialization_: $S=0$, $Y=0$, $L=0$, $\mu>0$, $\delta=10^{-7}$.
    2. _Iterations_: while $\|A-L-S\|_F>\delta\|A\|_F$ repeat
        1. _SV Thresholding_: $L=\mathcal{D}_{\displaystyle \mu^{-1}}(A-S+\mu^{-1}Y)$
        2. _Shrinkage_: $S=\mathcal{S}_{\displaystyle \lambda \mu^{-1}}(A-L+\mu^{-1}Y)$
        3. _Updating_: $Y=Y+\mu(A-L-S)$
"""

# ╔═╡ 8f347422-2782-48ca-b165-f97219ab8677
function PCPAD(A::Array{T}) where T
    # Initialize
    δ=1.0e-7
    tol=δ*norm(A)
    m,n=size(A)
    S=zero(A)
    Y=zero(A)
    L=zero(A)
    T₁=zero(A)
    μ=(m*n)/(4*(norm(A[:],1)))
    μ₁=one(T)/μ
    λ=one(T)/√(max(m,n))
    λμ₁=λ*μ₁
    ν=1e20
    maxiter=200
    iterations=0
    # Iterate
    while (ν>tol) && iterations<maxiter
        # println(iterations," ",ν)
        iterations+=1
        L=D(A-S+μ₁*Y,μ₁)
        S=Shr(A-L+μ₁.*Y,λμ₁)
        T₁=(A-L-S)
        Y+=(μ.*T₁)
        ν=norm(T₁)
    end
    L,S, iterations
end

# ╔═╡ db4778a0-7948-43cc-b63a-3b2d2a924228
@time L₀,S₀,iter₀=PCPAD(A₀)

# ╔═╡ 0b8f097f-2327-467f-a1d5-481711773ae2
rank(L₀),norm(A₀-L₀-S₀)

# ╔═╡ e33dbeb8-cae9-418b-9e89-1e6563bd58a7
md"
## Examples

### Larger random matrix
"

# ╔═╡ e52bd27c-6263-4b65-9a60-5ddb31626881
begin
	# Now the real test
	# Dimensions of the matrix
	m=100
	n=100
	# Rank of the low-rank part L
	k=10
	# Generate L
	L=rand(m,k)*rand(k,n)
	rank(L)
end

# ╔═╡ 7c23f010-5b76-44e0-ab8e-dd1eb8a01588
begin
	# Sparsity of the sparse part S
	sparsity=0.1
	# Generate S
	S=10*sprand(m,n,sparsity)
	nnz(S)
end

# ╔═╡ e7a50a63-2312-43f0-b771-64452e33d808
# Generate the matrix A, it is a full matrix with full rank
A=L+S;

# ╔═╡ 42904941-1859-4a05-a08c-a7a650f6b188
rank(A)

# ╔═╡ 0bc47362-3b76-44a8-8c52-2d0fa640a984
# Decompose A into L₁ and S₁
@time L₁,S₁,iters=PCPAD(A);

# ╔═╡ b04a85a6-0fab-4a37-9b54-28dfbe9382aa
iters, rank(L₁), norm(L), norm(L-L₁), norm(S), norm(S-S₁)

# ╔═╡ f58a21b2-a940-49f8-a179-e0c2dd032579
md"""
Although there might be no convergence, the splitting is still good. 
"""

# ╔═╡ 1d16c2fa-d9b3-4e4c-bafc-9edbe408f516
S₁

# ╔═╡ 321cc936-0ed3-4fcd-bf95-504f8529ef1b
Matrix(S)

# ╔═╡ 8704e62c-278c-4cf2-9b14-063b6ab3abd3
md"""
### Face recognition

We will try to recover missing features. The images are chosen from the 
[Yale Face Database](http://vision.ucsd.edu/content/yale-face-database).

"""

# ╔═╡ 612b0d4d-0a74-473d-89c9-f89333817b09
# First a single image
img=load("files/17.jpg")

# ╔═╡ b5f358cd-6475-4187-b248-68cc62cf55af
show(img[1,1])

# ╔═╡ 0ef3370b-edfe-46f6-9fb6-81ce8e3d86b3
A₂=map(Float64,img);

# ╔═╡ 24c36ac7-7ba6-4df1-9a90-2381b97362a2
begin
	# Compute the splitting and show number of iterations
	@time L₂,S₂,iters₂=PCPAD(A₂)
	iters₂, rank(L₂), norm(A₂), norm(A₂-L₂-S₂)
end

# ╔═╡ d0874885-447b-4aaa-8b2a-9c59ede4ac78
colorview(Gray,L₂)

# ╔═╡ a1413dd4-12c0-44e8-8981-79cb10d308dc
# Try S+0.5
colorview(Gray,S₂.+0.5)  

# ╔═╡ e7714ae7-c473-44a6-9279-0374686495da
hcat(colorview(Gray,0.9*L₂+S₂.+0.2),img)

# ╔═╡ 481ab279-96bb-4a71-abb7-5df48b900e46
# Another image
img₃=load("files/19.jpg")

# ╔═╡ b8f4214c-2e11-4fba-a9a9-439a672516c8
begin
	A₃=map(Float64,img)
	L₃,S₃,iters₃=PCPAD(A₃)
	iters₃, rank(L₃), norm(A₃), norm(A₃-L₃-S₃)
end

# ╔═╡ c52d6a06-691f-49f1-9e5b-043429cba296
hcat(img₃,colorview(Gray,L₃),colorview(Gray,L₃+S₃.+0.3))

# ╔═╡ 0d8ddcb5-529e-4187-b257-9f8233abf656
md"""
### Multiple images

Each image of $168 \times 192$ pixels is converted to a vector of length 32256. 
All vectors are stacked in the columns of matrix $A$, and the low-rank + sparse splitting 
of $A$ is computed.
"""

# ╔═╡ 243bd190-6912-475f-870e-43f96656e9f6
begin
	# Load all images in the collection
	dir="./files/yaleB08/"
	files=readdir(dir)
end

# ╔═╡ 8b4577b3-8b23-4dbe-9d6f-9aec90849b60
begin
	n₄=length(files)-1
	images=Array{Any}(undef,n₄)
	B=Array{Any}(undef,n₄)
	for i=1:n₄
	    images[i]=Images.load(joinpath(dir,files[i+1]))
	    B[i]=map(Float64,images[i])
	end
end

# ╔═╡ db755a0a-0a04-4fc2-bdfe-641a0b933575
# Last 9 images are meaningless
images

# ╔═╡ c1fa7658-6c71-4891-b82e-dd676c2f1b37
begin
	# Form the big matrix - each image is converted to a column vector
	mi,ni=size(images[1])
	A₄=Array{Float64}(undef,mi*ni,n₄-9)
	for i=1:n₄-9
	    A₄[:,i]=vec(B[i])
	end
	size(A₄)
end

# ╔═╡ 8ca25ca4-3681-46d0-945a-2c99d20205b7
begin
	# Now the big SVDs - 1 minute (Tall and skinny)
	@time L₄,S₄,iters₄=PCPAD(A₄)
	iters₄, rank(L₄), norm(A₄), norm(A₄-L₄-S₄)
end

# ╔═╡ eaa6def3-06fc-4646-a228-1f9180050ffc
# For example
colorview(Gray,reshape(S₄[:,1],mi,ni).+0.3)

# ╔═╡ e32b8ed8-e50b-4b79-9f6d-119b6d2796dc
begin
	LowRank=similar(images)
	Sparse=similar(images)
	for i=1:n₄-9
	    LowRank[i]=colorview(Gray,reshape(L₄[:,i],mi,ni))
	    Sparse[i]=colorview(Gray,reshape(S₄[:,i],mi,ni).+0.3)
	end
	# hcat(images,LowRank,Sparse)
end

# ╔═╡ 66fd9e25-cb4a-410e-ae83-311fffb864c7
LowRank

# ╔═╡ adcb7667-cdff-4ba4-b59d-c2ece2f7ee31
Sparse

# ╔═╡ 80a6d992-0e72-4c9f-a593-18a0fed09e1c
begin
	i=29
	hcat(images[i],LowRank[i],Sparse[i],0.9*LowRank[i]+Sparse[i])
end

# ╔═╡ 85975780-6c17-4af9-a169-9fe906c1e369
md"""
### Mona Lisa's smile
"""

# ╔═╡ 0ad6a808-692d-47e6-9e8f-1f721a38524e
img₅=load("files/mona-lisa_1.jpg")

# ╔═╡ 8a7548de-6e1e-442c-b5a7-b65a6af0efa4
# As in the notebook S03
imgsep=map(Float64,channelview(img₅))

# ╔═╡ fbb0bf4c-2c91-47b2-8e61-c6c0a1085b27
size(imgsep)

# ╔═╡ 396542f6-737d-49ca-a3c9-41d319aa5cc7
begin
	# 1-2 minutes
	# Red
	@time RL,RS,Riter=PCPAD(imgsep[1,:,:])
	# Green
	GL,GS,Giter=PCPAD(imgsep[2,:,:])
	# Blue
	BL,BS,Biter=PCPAD(imgsep[3,:,:])
end

# ╔═╡ 1c13b5c0-26d5-4220-9fca-fecd7c945c57
Giter, rank(GL), norm(imgsep[2,:,:]), norm(imgsep[2,:,:]-GL-GS)

# ╔═╡ 062918a1-b84a-4c91-ba9e-7fe630582f63
# Mona Lisa's low-rank component
colorview(RGB,RL,GL,BL)

# ╔═╡ 6ccc341f-fae8-4f18-b3a5-3528a101a8c1
# Mona Lisa's sparse component
colorview(RGB,RS.+0.5,GS.+0.5,BS.+0.5)

# ╔═╡ 3bda0ed2-c721-4a8d-9beb-50fb0066b28e
md"""
### Mona Lisa's hands
"""

# ╔═╡ 94423d38-dc7b-4a26-b0f0-0ed532c80e72
img₆=load("files/Mona_Lisa_detail_hands.jpg")

# ╔═╡ a53efeb6-2676-4d13-b2b0-0da4089dbcc6
begin
	imgsep₁=map(Float64,channelview(img₆))
	@time RL₁,RS₁,Riter₁=PCPAD(imgsep₁[1,:,:])
	GL₁,GS₁,Giter₁=PCPAD(imgsep₁[2,:,:])
	BL₁,BS₁,Biter₁=PCPAD(imgsep₁[3,:,:])
	# Norm for Green
	Giter₁, rank(GL₁), norm(imgsep₁[2,:,:]), 
	norm(imgsep₁[2,:,:]-GL₁-GS₁)
end

# ╔═╡ fe9da092-9dd9-4304-96a7-1016cbdc4a5e
colorview(RGB,RL₁,GL₁,BL₁)

# ╔═╡ 26e90bdb-dfb3-4259-bb75-a037e86344fe
colorview(RGB,RS₁.+0.5,GS₁.+0.5,BS₁.+0.5)

# ╔═╡ Cell order:
# ╟─0758937c-478f-45b7-a1d4-31810eecfd60
# ╟─56d22088-733f-4b76-8b56-592229cd6a25
# ╟─33d0a1b9-f2f4-4f48-998f-e058c9a69e14
# ╟─5c78225a-99b0-4587-87a7-d7ae505a9adc
# ╠═7bb93306-9994-4302-9d03-43bb4404501e
# ╠═19982690-3cf8-45f2-a19f-3a4004ec29f0
# ╠═35c93042-50a1-4d17-ba93-eb4abeb32bdc
# ╠═7bf123d1-dc37-44d0-9da9-982dbca9ec02
# ╠═a5c7c4fb-4d28-4cdb-a825-81b422f5357c
# ╠═004407fe-d861-42b6-a061-4ef41176a289
# ╠═701fdeda-d5e3-4686-b57d-0c0dadba7c91
# ╠═78fda2e8-1fbf-4aa1-bc4a-a05c262a410a
# ╟─a29511fa-1deb-4bea-97aa-326bd366f6ab
# ╠═8f347422-2782-48ca-b165-f97219ab8677
# ╠═db4778a0-7948-43cc-b63a-3b2d2a924228
# ╠═0b8f097f-2327-467f-a1d5-481711773ae2
# ╟─e33dbeb8-cae9-418b-9e89-1e6563bd58a7
# ╠═e52bd27c-6263-4b65-9a60-5ddb31626881
# ╠═e0d1f2ca-ea96-4ae8-a3bd-cfabe92978d5
# ╠═7c23f010-5b76-44e0-ab8e-dd1eb8a01588
# ╠═e7a50a63-2312-43f0-b771-64452e33d808
# ╠═42904941-1859-4a05-a08c-a7a650f6b188
# ╠═0bc47362-3b76-44a8-8c52-2d0fa640a984
# ╠═b04a85a6-0fab-4a37-9b54-28dfbe9382aa
# ╟─f58a21b2-a940-49f8-a179-e0c2dd032579
# ╠═1d16c2fa-d9b3-4e4c-bafc-9edbe408f516
# ╠═321cc936-0ed3-4fcd-bf95-504f8529ef1b
# ╟─8704e62c-278c-4cf2-9b14-063b6ab3abd3
# ╠═ea479efe-0f65-4365-a136-14afdfb180e2
# ╠═612b0d4d-0a74-473d-89c9-f89333817b09
# ╠═b5f358cd-6475-4187-b248-68cc62cf55af
# ╠═0ef3370b-edfe-46f6-9fb6-81ce8e3d86b3
# ╠═24c36ac7-7ba6-4df1-9a90-2381b97362a2
# ╠═d0874885-447b-4aaa-8b2a-9c59ede4ac78
# ╠═a1413dd4-12c0-44e8-8981-79cb10d308dc
# ╠═e7714ae7-c473-44a6-9279-0374686495da
# ╠═481ab279-96bb-4a71-abb7-5df48b900e46
# ╠═b8f4214c-2e11-4fba-a9a9-439a672516c8
# ╠═c52d6a06-691f-49f1-9e5b-043429cba296
# ╟─0d8ddcb5-529e-4187-b257-9f8233abf656
# ╠═243bd190-6912-475f-870e-43f96656e9f6
# ╠═8b4577b3-8b23-4dbe-9d6f-9aec90849b60
# ╠═db755a0a-0a04-4fc2-bdfe-641a0b933575
# ╠═c1fa7658-6c71-4891-b82e-dd676c2f1b37
# ╠═8ca25ca4-3681-46d0-945a-2c99d20205b7
# ╠═eaa6def3-06fc-4646-a228-1f9180050ffc
# ╠═e32b8ed8-e50b-4b79-9f6d-119b6d2796dc
# ╠═66fd9e25-cb4a-410e-ae83-311fffb864c7
# ╠═adcb7667-cdff-4ba4-b59d-c2ece2f7ee31
# ╠═80a6d992-0e72-4c9f-a593-18a0fed09e1c
# ╟─85975780-6c17-4af9-a169-9fe906c1e369
# ╠═0ad6a808-692d-47e6-9e8f-1f721a38524e
# ╠═8a7548de-6e1e-442c-b5a7-b65a6af0efa4
# ╠═fbb0bf4c-2c91-47b2-8e61-c6c0a1085b27
# ╠═396542f6-737d-49ca-a3c9-41d319aa5cc7
# ╠═1c13b5c0-26d5-4220-9fca-fecd7c945c57
# ╠═062918a1-b84a-4c91-ba9e-7fe630582f63
# ╠═6ccc341f-fae8-4f18-b3a5-3528a101a8c1
# ╟─3bda0ed2-c721-4a8d-9beb-50fb0066b28e
# ╠═94423d38-dc7b-4a26-b0f0-0ed532c80e72
# ╠═a53efeb6-2676-4d13-b2b0-0da4089dbcc6
# ╠═fe9da092-9dd9-4304-96a7-1016cbdc4a5e
# ╠═26e90bdb-dfb3-4259-bb75-a037e86344fe
