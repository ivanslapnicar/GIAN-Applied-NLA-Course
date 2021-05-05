### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ f8e2f7e1-d6a7-43ef-879c-db33c5f66924
begin
	using PlutoUI
	PlutoUI.TableOfContents(aside=true)
end

# ╔═╡ c44c6c6a-3b8c-46e4-93ad-365d03082226
using SymPy, LinearAlgebra

# ╔═╡ c226e28d-3a8a-42f6-8a51-2ac418023a27
md"""
# Singular Value Decomposition - Definitions and Facts


__Prerequisites__

The reader should be familiar with basic linear algebra concepts and notebooks related to eigenvalue decomposition.

__Competences__

The reader should be able to undestand and check the facts about singular value decomposition.

__Selected references__

There are many excellent books on the subject. Here we list a few:

* J.W. Demmel, _Applied Numerical Linear Algebra_, SIAM, Philadelphia, 1997.
* G. H. Golub and C. F. Van Loan, _Matrix Computations_, 4th ed., The John Hopkins University Press, Baltimore, MD, 2013.
* N. Higham, _Accuracy and Stability of Numerical Algorithms_, SIAM, Philadelphia, 2nd ed., 2002.
* L. Hogben, ed., _Handbook of Linear Algebra_, CRC Press, Boca Raton, 2014.
* B. N. Parlett, _The Symmetric Eigenvalue Problem_, Prentice-Hall, Englewood Cliffs, NJ, 1980, also SIAM, Philadelphia, 1998.
* G. W. Stewart, _Matrix Algorithms, Vol. II: Eigensystems_, SIAM, Philadelphia, 2001.
* L. N. Trefethen and D. Bau, III, _Numerical Linear Algebra_, SIAM, Philadelphia, 1997.
 
"""

# ╔═╡ ddbd255b-7ddd-4fba-ac4d-722e88801e59
md"""
# Singular value problems

For more details and the proofs of the Facts below, see [R. C. Li, Matrix Perturbation Theory, pp. 21.6-21.8](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and [R. Mathias, Singular Values and Singular Value Inequalities,pp. 24.1-24.17](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.

## Definitions

Let $A\in\mathbb{C}^{m\times n}$ and let $q=\min\{m,n\}$.

The __singular value decomposition__ (SVD) of $A$ is 

$$A=U\Sigma V^*,$$

where $U\in\mathbb{C}^{m\times m}$ and $V\in\mathbb{C}^{n\times n}$ are unitary, and $\Sigma=\mathop{\mathrm{diag}}(\sigma_1,\sigma_2,\ldots)\in\mathbb{R}^{m\times n}$ with all $\sigma_j\geq 0$.

Here $\sigma_j$ is the __singular value__, $u_j\equiv U_{:,j}$ is the corresponding __left singular vector__, and $v_j\equiv V_{:,j}$ is the corresponding __right singular vector__.

The __set of singular values__ is $sv(A)=\{\sigma_1,\sigma_2,\ldots,\sigma_{q}\}$.

We assume that singular values are ordered, $\sigma_1\geq\sigma_2\geq\cdots\sigma_q\geq 0$.

The __Jordan-Wielandt__ matrix is the Hermitian matrix 

$$
J=\begin{bmatrix}0 & A \\ A^* & 0 \end{bmatrix}
\in \mathbb{C}^{(m+n) \times (m+n)}.$$
"""

# ╔═╡ caa44e3e-3af5-4836-8ae3-dd3b9d68d222
md"""
## Facts

There are many facts related to the singular value problem for general matrices. We state some basic ones:

1. If $A\in\mathbb{R}^{m\times n}$, then $U$ and $V$ are real.

2. Singular values are unique (uniquely determined by the matrix).

3.  $\sigma_j(A^T)=\sigma_j(A^*)=\sigma_j(\bar A)=\sigma_j(A)$ for $j=1,2,\ldots,q$.

4.  $A v_j=\sigma_j u_{j}$ and $A^* u_{j}=\sigma_j v_{j}$ for $j=1,2,\ldots,q$.

5.  $A=\sigma_1 u_{1} v_{1}^* + \sigma_2 u_{2} v_{2}^* +\cdots + \sigma_q u_{q} v_{q}^*$.

6. __Unitary invariance.__ For any unitary $U\in\mathbb{C}^{m\times m}$ and $V\in\mathbb{C}^{n\times n}$, $sv(A)=sv(UAV)$.

7. There exist unitary matrices $U\in\mathbb{C}^{m\times m}$ and $V\in\mathbb{C}^{n\times n}$ such that $A=UBV$ if and only if  $sv(A)=sv(B)$.

8. SVD of $A$ is related to eigenvalue decompositions of Hermitian matrices $A^*A=V\Sigma^T\Sigma V^*$ and $AA^*=U\Sigma\Sigma^TU^*$. Thus, $\sigma_j^2(A)=\lambda_j(A^*A)=\lambda_j(AA^*)$ for $j=1,2,\ldots,q$.

9. The eigenvalues of Jordan-Wielandt matrix are $\pm \sigma_1(A), \pm\sigma_2(A), \cdots,\pm\sigma_q(A)$ together with $|m-n|$ zeros. The eigenvectors are obtained from an SVD of $A$. This relationship is used to deduce singular value results from the results for eigenvalues of Hermitian matrices.

10.  $\mathop{\mathrm{trace}}(|A|_{spr})=\sum_{i=1}^q \sigma_i$, where $|A|_{spr}=(A^*A)^{1/2}$.

11. If $A$ is square, then $|\det(A)|=\prod_{i=1}^n \sigma_i$.

12. If $A$ is square, then $A$ is singular $\Leftrightarrow$ $\sigma_j(A)=0$ for some $j$.

13. __Min-max Theorem.__ It holds:

$$
\begin{aligned}
\sigma_k& =\max_{\dim(W)=k}\min_{x\in W, \ \|x\|_2=1} \|Ax\|_2\\
& =\min_{\dim(W)=n-k+1}\max_{x\in W, \ \|x\|_2=1} \|Ax\|_2.
\end{aligned}$$

14.  $\|A\|_2=\sigma_1(A)$.

15. For $B\in\mathbb{C}^{m\times n}$,

$$
|\mathop{\mathrm{trace}}(AB^*)|\leq \sum_{j=1}^q \sigma_j(A)\sigma_j(B).$$

16. __Interlace Theorems.__ Let $B$ denote $A$ with the one of its rows _or_ columns deleted. Then

$$
\sigma_{j+1}(A)\leq \sigma_j(B)\leq \sigma_j(A),\quad j=1,\ldots,q-1.$$

Let $B$ denote $A$ with a row _and_ a column deleted. Then

$$
\sigma_{j+2}(A)\leq \sigma_j(B)\leq \sigma_j(A),\quad j=1,\ldots,q-2.$$

17. __Weyl Inequalities.__ For $B\in\mathbb{C}^{m\times n}$, it holds:

$$
\begin{aligned}
\sigma_{j+k-1}(A+B)& \leq \sigma_j(A)+\sigma_k(B), \quad  j+k\leq q+1,\\
\sum_{j=1}^k \sigma_j(A+B)& \leq \sum_{j=1}^k \sigma_j(A) + \sum_{j=1}^k \sigma_j(B), 
\quad k=1,\ldots,q.
\end{aligned}$$

"""

# ╔═╡ 4b764569-a32d-4170-a11d-2072d971846a
md"""
## Examples 

### Symbolic computation
"""

# ╔═╡ 772342da-a960-4aa7-b30b-7122b049eff9
A=[  3   2   1
 -5  -1  -4
  5   0   2]

# ╔═╡ 314c41e0-03e7-4f8e-851c-a4374e7f5092
x=symbols("x")

# ╔═╡ 4ab83338-0428-4942-979e-c1021b81c309
B=A'*A

# ╔═╡ bf02e40e-94f4-4c69-ab74-3496cbcd6cdf
# Characteristic polynomial p_B(λ)
p(x)=simplify(det(B-x*Matrix{Int}(I,3,3)))

# ╔═╡ 54264dd9-98cf-4c8f-b6f0-c6ee2ba11cdb
λ=solve(p(x),x)

# ╔═╡ 6cc74bec-e335-471d-87a0-9b9bad391a1d
map(Float64,λ)

# ╔═╡ 2d1ea310-24a8-448c-9a3d-0a4420bbc90c
begin
	V=Array{Any}(undef,3,3)
	for j=1:3
	    V[:,j]=nullspace(B-map(Float64,λ[j])*I)
	end
	V
end

# ╔═╡ 92b68122-5e9a-443a-aaa3-ad301cb1878a
begin
	U=Array{Any}(undef,3,3)
	for j=1:3
	    U[:,j]=nullspace(A*A'-map(Float64,λ[j])*I)
	end
	U
end

# ╔═╡ edaaa469-6a73-43e6-a8b3-b365bf265e04
σ=sqrt.(λ)

# ╔═╡ c1ac5d19-4971-461f-b73d-10aabb4eb56f
# Diagonalizing
V'*A'*A*V

# ╔═╡ 72352834-0bf5-41b7-86e2-0b2cb733d7f4
# Residual -what is wrong?
A-U*Diagonal(float(σ))*V'

# ╔═╡ ace071e5-764a-4c04-8916-7287ed53a01e
A-U*Diagonal(float(σ))*(V*Diagonal(sign.(U'*A*V)))'

# ╔═╡ 8d911867-9b90-4bff-9d0c-7358dfcc04ec
#?svd

# ╔═╡ 1b1b7ddd-f50d-4d65-8cc8-398955dcce34
# Using the structure SVD
S=svd(A)

# ╔═╡ b6663c47-1293-43ba-8c13-20858054c975
S.Vt

# ╔═╡ 74e584f4-a233-4f18-b04c-dfd1710515e1
# Residual
A-S.U*Diagonal(S.S)*S.Vt

# ╔═╡ 6ba22886-faa3-4f30-af96-366d9c001c94
md"""
Observe the signs of the columns!
"""

# ╔═╡ 80d341c5-43ae-4568-bcf3-ced1947662ce
S.U

# ╔═╡ 1245fe85-25a8-44cb-bbd7-3bc477859d04
U

# ╔═╡ abe3e648-65e6-423c-83de-673e96013670
md"
### Random complex matrix
"

# ╔═╡ 64535706-eaba-4155-b593-12cba7366127
begin
	import Random
	Random.seed!(421)
	m=5
	n=3
	q=min(m,n)
	A₁=rand(ComplexF64,m,n)
end

# ╔═╡ bd88592a-781f-46e7-a175-342b78906c45
S₁=svd(A₁,full=true)

# ╔═╡ f54a7c12-d175-45cf-b996-d13024910135
norm(A₁-S₁.U[:,1:q]*Diagonal(S₁.S)*S₁.Vt), norm(S₁.U'*S₁.U-I), norm(S₁.Vt*S₁.V-I)

# ╔═╡ b39c5987-d320-499a-816c-31608eb1d856
@bind k Slider(1:q,show_value=true)

# ╔═╡ e7aa4857-76c0-42da-964a-cecd428ec70e
# Fact 4
norm(A₁*S₁.V[:,k]-S₁.S[k]*S₁.U[:,k]), norm(A₁'*S₁.U[:,k]-S₁.S[k]*S₁.V[:,k])

# ╔═╡ da2e1129-474e-4a6a-9d66-7224f1f25b34
λ₁,V₁=eigen(A₁'*A₁)

# ╔═╡ b740803d-7bb2-4ce2-8338-4a53e333f9f9
sqrt.(λ₁)

# ╔═╡ 43316c8b-149b-4a52-90e9-ccb2c2ca4d67
S₁.V

# ╔═╡ 8a06f9fc-62d0-4394-8f5c-b861a63da947
abs.(S₁.Vt*V₁)

# ╔═╡ 879285b0-8fc9-43a2-b06d-30fd95556d61
md"""
__Explain non-uniqueness of $U$ and $V$!__
"""

# ╔═╡ ff0ea61d-7944-4316-a224-72461a808cdc
# Jordan-Wielandt matrix
J=[zero(A*A') A; A' zero(A'*A)]

# ╔═╡ 5d941c94-b186-49ba-9f03-1613a638ecb4
λJ,UJ=eigen(J)

# ╔═╡ 9909c421-607d-4883-9b35-705290ce8bf3
λJ

# ╔═╡ dda804cf-71c1-4735-bfc7-a97453698c8f
 float(σ)

# ╔═╡ f4b2ce9d-4783-420d-bc87-2ce27671c7e4
md"
### Random real matrix
"

# ╔═╡ 87de7452-77af-466b-920c-1a562ee05ea2
begin
	m₂=8
	n₂=5
	q₂=min(m₂,n₂)
	A₂=rand(-9:9,m₂,n₂)
end

# ╔═╡ 27f76447-12cf-4096-a1ab-ad997367b336
S₂=svd(A₂)

# ╔═╡ 63af09b0-1b32-478e-868f-95ddf2e951e8
# Fact 10
tr(sqrt(A₂'*A₂)), sum(S₂.S)

# ╔═╡ c30340fd-cc4e-4cbe-b68a-fb0b01f383fa
begin
	# Fact 11
	B₁=rand(n₂,n₂)
	det(B₁), prod(svdvals(B₁))
end

# ╔═╡ ecae308e-7444-4883-84bf-8106bb7106e9
# Fact 14
norm(A₂), opnorm(A₂), S₂.S[1]

# ╔═╡ eb15023f-f126-4177-bc2c-057543b8fcc1
begin
	# Fact 15
	B₂=rand(m₂,n₂)
	abs(tr(A₂*B₂')), svdvals(A₂)⋅svdvals(B₂)
end

# ╔═╡ 07f30928-36aa-4d59-bb53-fa8df06fdad0
@bind j Slider(1:q₂,show_value=true)

# ╔═╡ bf28d15d-c243-4229-8d41-0945c63e3baa
# Interlace Theorems (repeat several times)
S₂.S, svdvals(A₂[[1:j-1;j+1:m₂],:]), svdvals(A₂[:,[1:j-1;j+1:n₂]])

# ╔═╡ 69470862-8321-4953-b324-3a5b4e93d9c8
# Weyl Inequalities
# B=rand(m,n)
[svdvals(A₂+B₂) S₂.S svdvals(B₂)]

# ╔═╡ 35fead5b-4425-40b3-a64d-9cf7ebab18d2
@bind l Slider(1:q₂,show_value=true)

# ╔═╡ 61ecca0f-1e5e-4dec-90d3-924961b5f9c8
sum(svdvals(A₂+B₂)[1:l]),sum(svdvals(A₂)[1:l].+sum(svdvals(B₂)[1:l]))

# ╔═╡ 2d804bbd-c6ce-49ce-9197-d7850ccb7db5
md"""
# Matrix approximation

Let $A=U\Sigma V^*$, let $\tilde \Sigma$ be equal to $\Sigma$ except that $\tilde \Sigma_{jj}=0$ for $j>k$, and let $\tilde A=U\tilde \Sigma V^*$. Then $\mathop{\mathrm{rank}}(\tilde A)\leq k$ and

$$
\begin{aligned}
\min\{\|A-B\|_2: \mathop{\mathrm{rank}}(B)\leq k\} & =\|A-\tilde A\|_2=\sigma_{k+1}(A)\\
\min\{\|A-B\|_F: \mathop{\mathrm{rank}}(B)\leq k\} & =\|A-\tilde A\|_F=
\bigg(\sum_{j=k+1}^{q}\sigma_{j}^2(A)\bigg)^{1/2}.
\end{aligned}$$

This is the __Eckart-Young-Mirsky Theorem__.
"""

# ╔═╡ d7221da7-78fc-4d86-a693-95bab6d12d5a
A₂

# ╔═╡ 4d9640da-d85c-41ec-afcf-fa9548c25804
S₂.S

# ╔═╡ 997d0c9c-22a4-4a60-9730-0668eda26af4
@bind k₀ Slider(1:q₂,show_value=true)

# ╔═╡ 76bc9a58-c163-4df2-a315-5f9474e746aa
B₀=S₂.U*Diagonal([S₂.S[1:k₀];zeros(q₂-k₀)])*S₂.Vt

# ╔═╡ 36fb7249-ecc5-432a-9842-1c9cbf17c30a
opnorm(A₂-B₀), S₂.S[k₀+1]

# ╔═╡ 42ec2898-6676-4254-b553-215da8ec7490
norm(A₂-B₀),norm(S₂.S[k₀+1:q₂])

# ╔═╡ Cell order:
# ╟─f8e2f7e1-d6a7-43ef-879c-db33c5f66924
# ╟─c226e28d-3a8a-42f6-8a51-2ac418023a27
# ╟─ddbd255b-7ddd-4fba-ac4d-722e88801e59
# ╟─caa44e3e-3af5-4836-8ae3-dd3b9d68d222
# ╟─4b764569-a32d-4170-a11d-2072d971846a
# ╠═c44c6c6a-3b8c-46e4-93ad-365d03082226
# ╠═772342da-a960-4aa7-b30b-7122b049eff9
# ╠═314c41e0-03e7-4f8e-851c-a4374e7f5092
# ╠═4ab83338-0428-4942-979e-c1021b81c309
# ╠═bf02e40e-94f4-4c69-ab74-3496cbcd6cdf
# ╠═54264dd9-98cf-4c8f-b6f0-c6ee2ba11cdb
# ╠═6cc74bec-e335-471d-87a0-9b9bad391a1d
# ╠═2d1ea310-24a8-448c-9a3d-0a4420bbc90c
# ╠═92b68122-5e9a-443a-aaa3-ad301cb1878a
# ╠═edaaa469-6a73-43e6-a8b3-b365bf265e04
# ╠═c1ac5d19-4971-461f-b73d-10aabb4eb56f
# ╠═72352834-0bf5-41b7-86e2-0b2cb733d7f4
# ╠═ace071e5-764a-4c04-8916-7287ed53a01e
# ╠═8d911867-9b90-4bff-9d0c-7358dfcc04ec
# ╠═1b1b7ddd-f50d-4d65-8cc8-398955dcce34
# ╠═b6663c47-1293-43ba-8c13-20858054c975
# ╠═74e584f4-a233-4f18-b04c-dfd1710515e1
# ╟─6ba22886-faa3-4f30-af96-366d9c001c94
# ╠═80d341c5-43ae-4568-bcf3-ced1947662ce
# ╠═1245fe85-25a8-44cb-bbd7-3bc477859d04
# ╟─abe3e648-65e6-423c-83de-673e96013670
# ╠═64535706-eaba-4155-b593-12cba7366127
# ╠═bd88592a-781f-46e7-a175-342b78906c45
# ╠═f54a7c12-d175-45cf-b996-d13024910135
# ╠═b39c5987-d320-499a-816c-31608eb1d856
# ╠═e7aa4857-76c0-42da-964a-cecd428ec70e
# ╠═da2e1129-474e-4a6a-9d66-7224f1f25b34
# ╠═b740803d-7bb2-4ce2-8338-4a53e333f9f9
# ╠═43316c8b-149b-4a52-90e9-ccb2c2ca4d67
# ╠═8a06f9fc-62d0-4394-8f5c-b861a63da947
# ╟─879285b0-8fc9-43a2-b06d-30fd95556d61
# ╠═ff0ea61d-7944-4316-a224-72461a808cdc
# ╠═5d941c94-b186-49ba-9f03-1613a638ecb4
# ╠═9909c421-607d-4883-9b35-705290ce8bf3
# ╠═dda804cf-71c1-4735-bfc7-a97453698c8f
# ╟─f4b2ce9d-4783-420d-bc87-2ce27671c7e4
# ╠═87de7452-77af-466b-920c-1a562ee05ea2
# ╠═27f76447-12cf-4096-a1ab-ad997367b336
# ╠═63af09b0-1b32-478e-868f-95ddf2e951e8
# ╠═c30340fd-cc4e-4cbe-b68a-fb0b01f383fa
# ╠═ecae308e-7444-4883-84bf-8106bb7106e9
# ╠═eb15023f-f126-4177-bc2c-057543b8fcc1
# ╠═07f30928-36aa-4d59-bb53-fa8df06fdad0
# ╠═bf28d15d-c243-4229-8d41-0945c63e3baa
# ╠═69470862-8321-4953-b324-3a5b4e93d9c8
# ╠═35fead5b-4425-40b3-a64d-9cf7ebab18d2
# ╠═61ecca0f-1e5e-4dec-90d3-924961b5f9c8
# ╟─2d804bbd-c6ce-49ce-9197-d7850ccb7db5
# ╠═d7221da7-78fc-4d86-a693-95bab6d12d5a
# ╠═4d9640da-d85c-41ec-afcf-fa9548c25804
# ╠═997d0c9c-22a4-4a60-9730-0668eda26af4
# ╠═76bc9a58-c163-4df2-a315-5f9474e746aa
# ╠═36fb7249-ecc5-432a-9842-1c9cbf17c30a
# ╠═42ec2898-6676-4254-b553-215da8ec7490
