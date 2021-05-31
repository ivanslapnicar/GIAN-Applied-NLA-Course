### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ cf46a0fe-43a4-4b6a-b4f8-a0c5fb4e9f03
# For binder, enable this cell ...
begin
	import Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
		Pkg.PackageSpec(name="PlutoUI"),
        Pkg.PackageSpec(name="Distributions"),
		Pkg.PackageSpec(name="SparseArrays")
    ])
end

# ╔═╡ e7af1703-0873-4b3f-8b8f-a8a2c874bcb1
begin
	using PlutoUI, LinearAlgebra, Distributions, SparseArrays
	import Random
end

# ╔═╡ cfaedd73-8e4c-4481-99d0-c04eafd79fe9
TableOfContents(title="📚 Table of Contents", aside=true)

# ╔═╡ f7c33c4f-b493-4d6b-84dd-d2a5087ffeb4
5*60,5*80

# ╔═╡ 00839532-0da0-4cbf-8093-01310a3799d0
md"""
# Randomized Linear Algebra

See [Yuxin Chen, Randomized linear algebra](https://www.princeton.edu/~yc5/orf570/randomized_linear_algebra.pdf).

"""

# ╔═╡ a6f4632f-5d64-4efb-a72f-706235cb3957
md"""
# Matrix multiplication

Let $A\in\mathbb{R}^{m\times n}$ and $B\in\mathbb{R}^{n\times p}$. Then,

$$
C=AB=\sum_{i=1}^{n} A_{:,i} B_{i,:}.$$

Assume, for simplicity $m=n=p$.

__Idea:__ approximate $C$ by randomly sampling $r$ rank-one components.


__Algorithm:__ for $l= 1,\cdots ,r$ pick $i_l\in\{1,···,n\}$ i.i.d. with probability $\mathbb{P}\{i_l=k\}=p_k$ and compute 

$$M=\sum_{l=1}^r \frac{1}{rp_{i_l}} A_{:,i_l} B_{i_l,:}$$

__Rationale:__ $M$ is an unbiased estimate of $C$,

$$\begin{aligned}
\mathbb{E}[M]&=\sum_{l=1}^r \sum_k \mathbb{P}\{i_l=k\} \frac{1}{r p_k} A_{:,k}B_{k,:}
=\sum_k A_{:,k}B_{k,:}=C.
\end{aligned}$$

__Importance sampling porbabilities__ $p_k$ are

* __Uniform sampling:__ $p_k=\displaystyle\frac{1}{n}$

* __Nonuniform sampling:__

$$
p_k=\frac{\|A_{:,k}\|_2 \|B_{k,:}\|_2}
{\sum_l \|A_{:,l}\|_2 \|B_{l,:}\|_2},  \tag{1}$$

which is computable in one-pass and requires $O(n)$ memory and $O(n^2)$ operations.

__Theorem.__ [Optimality] $\mathbb{E}[\|M-AB\|_F^2]$ is minimized for $p_k$ given by (1).

__Theorem.__ [Error] Choose $p_k\geq \displaystyle\frac{\beta \|A_{:,k}\|_2 \|B_{k,:}\|_2} {\sum_l \|A_{:,l}\|_2 \|B_{l,:}\|_2}$ for some 
$0<\beta \leq 1$. If $r\geq \displaystyle\frac{\log n}{\beta}$, then 


$$\|M-AB\|_F\leq \sqrt{\frac{\log n}{\beta r}}
\|A\|_F \|B\|_F$$

with probability exceeding $1-O(n^{-10})$.
"""

# ╔═╡ 25a893eb-2c23-40dc-8e24-f45ed8253697
md"
## Full matrix
"

# ╔═╡ 4dc397d7-5e07-4da9-94ea-fa22145d7222
begin
	Random.seed!(6789)
	n=3000
	A=rand(n,n)
	B=rand(n,n)
	β=1.0
	log(n)/β
end

# ╔═╡ 31305f95-1a9d-4981-bdff-bd8f8edc3802
begin
	# Uniform
	r=400
	iᵣ=rand(1:n,r)
	p=1/n
	@time M=A[:,iᵣ]*B[iᵣ,:]/(r*p)
	@time C=A*B;
end

# ╔═╡ 8422981a-2daf-4b93-b1a9-499f1f1f7eaa
norm(M-C), norm(C)

# ╔═╡ 477d5ccd-b9e8-482e-ad73-ae1c56eb4cfc
begin
	# Nonuniform
	pA=[norm(view(A,:,k)) for k=1:n]
	pB=[norm(view(B,k,:)) for k=1:n]
	s=pA⋅pB
	p₁=[pA[k]*pB[k]/s for k=1:n]
	sum(p₁)
end

# ╔═╡ f0733a4f-5a05-41b4-9313-678ecad74356
begin
	i₁=rand(Categorical(p₁),r)
	@time Mₙ=A[:,i₁]*inv(Diagonal(r*p₁[i₁]))*B[i₁,:]
	norm(Mₙ-C), norm(C), √(log(n)/(β*r))*norm(A)*norm(B)
end

# ╔═╡ b0908c15-d3f1-429e-8165-e40bf58fd991
md"
## Sparse matrix
"

# ╔═╡ 2ec3b0a7-d24f-4fb5-864d-65b7f1ec0e06
begin
	# Sparse, nonuniform
	Random.seed!(9751)
	nₛ=10000
	Aₛ=sprand(nₛ,nₛ,0.1)
	Bₛ=sprand(nₛ,nₛ,0.1)
	@time Cₛ=Aₛ*Bₛ
	βₛ=1.0
	log(nₛ)/βₛ
end

# ╔═╡ 1260c0a9-2e6c-4c46-b84d-533c19cc2f67
# Cₛ is full
nnz(Cₛ)/prod(size(Cₛ))

# ╔═╡ c906c80d-00e0-41da-8c6e-fa0d58c4ae60
begin
	# Nonuniform
	pAₛ=[norm(view(Aₛ,:,k)) for k=1:nₛ]
	pBₛ=[norm(view(Bₛ,k,:)) for k=1:nₛ]
	t=pAₛ⋅pBₛ
	pₛ=[pAₛ[k]*pBₛ[k]/t for k=1:nₛ]
	sum(pₛ)
end

# ╔═╡ 402e6d6d-502d-41bf-acff-1d9c09bd0bf0
begin
	rₛ=3000
	iₛ=rand(Categorical(pₛ),rₛ);
	@time Mₛ=Aₛ[:,iₛ]*inv(Diagonal(rₛ*pₛ[iₛ]))*Bₛ[iₛ,:]
	norm(Mₛ-Cₛ),√(log(nₛ)/(βₛ*rₛ))*norm(Aₛ)*norm(Bₛ), norm(Cₛ)
end

# ╔═╡ 3800b4ce-10fb-4300-9129-09a0ca744a89


# ╔═╡ Cell order:
# ╠═cf46a0fe-43a4-4b6a-b4f8-a0c5fb4e9f03
# ╠═e7af1703-0873-4b3f-8b8f-a8a2c874bcb1
# ╠═cfaedd73-8e4c-4481-99d0-c04eafd79fe9
# ╠═f7c33c4f-b493-4d6b-84dd-d2a5087ffeb4
# ╟─00839532-0da0-4cbf-8093-01310a3799d0
# ╟─a6f4632f-5d64-4efb-a72f-706235cb3957
# ╟─25a893eb-2c23-40dc-8e24-f45ed8253697
# ╠═4dc397d7-5e07-4da9-94ea-fa22145d7222
# ╠═31305f95-1a9d-4981-bdff-bd8f8edc3802
# ╠═8422981a-2daf-4b93-b1a9-499f1f1f7eaa
# ╠═477d5ccd-b9e8-482e-ad73-ae1c56eb4cfc
# ╠═f0733a4f-5a05-41b4-9313-678ecad74356
# ╟─b0908c15-d3f1-429e-8165-e40bf58fd991
# ╠═2ec3b0a7-d24f-4fb5-864d-65b7f1ec0e06
# ╠═1260c0a9-2e6c-4c46-b84d-533c19cc2f67
# ╠═c906c80d-00e0-41da-8c6e-fa0d58c4ae60
# ╠═402e6d6d-502d-41bf-acff-1d9c09bd0bf0
# ╠═3800b4ce-10fb-4300-9129-09a0ca744a89
