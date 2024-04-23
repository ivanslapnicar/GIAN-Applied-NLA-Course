### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 220ac3b2-e4e9-422f-b0a9-7dfc8a32b327
using PlutoUI, Random, LinearAlgebra, SymPyPythonCall

# ╔═╡ 34979e8f-0d24-4577-aaa2-1462201276dc
PlutoUI.TableOfContents(aside=true)

# ╔═╡ c226e28d-3a8a-42f6-8a51-2ac418023a27
md"""
# Singular Value Decomposition - Definitions and Facts


__Prerequisites__

The reader should be familiar with basic linear algebra concepts and notebooks related to eigenvalue decomposition.

__Competences__

The reader should be able to understand and check the facts about singular value decomposition.

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
A-U*Diagonal(float.(σ))*V'

# ╔═╡ ace071e5-764a-4c04-8916-7287ed53a01e
A-U*Diagonal(float.(σ))*(V*Diagonal(sign.(U'*A*V)))'

# ╔═╡ 8d911867-9b90-4bff-9d0c-7358dfcc04ec
#?svd

# ╔═╡ 1b1b7ddd-f50d-4d65-8cc8-398955dcce34
# Using the structure SVD
S=svd(A)

# ╔═╡ b6663c47-1293-43ba-8c13-20858054c975
S.Vt

# ╔═╡ 4c626917-0ded-4f45-88b2-1d8f2d792b33
S.V

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
	Random.seed!(421)
	m=5
	n=3
	q=min(m,n)
	A₁=randn(ComplexF64,m,n)
end

# ╔═╡ bd88592a-781f-46e7-a175-342b78906c45
S₁=svd(A₁,full=true)

# ╔═╡ 867af0c2-6ac1-4a06-99da-9e573e2069a4
S₁.U

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
Eⱼ=eigen(J)

# ╔═╡ 9909c421-607d-4883-9b35-705290ce8bf3
Eⱼ.values

# ╔═╡ dda804cf-71c1-4735-bfc7-a97453698c8f
 float.(σ)

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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[compat]
PlutoUI = "~0.7.58"
SymPyPythonCall = "~0.2.5"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "89544e0289f47d7b6eda625e83cbaf0eb02a2dca"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "0f748c81756f2e5e6854298f11ad8b2dfae6911a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.CommonEq]]
git-tree-sha1 = "6b0f0354b8eb954cdba708fb262ef00ee7274468"
uuid = "3709ef60-1bee-4518-9f2f-acd86f176c50"
version = "0.2.1"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.CondaPkg]]
deps = ["JSON3", "Markdown", "MicroMamba", "Pidfile", "Pkg", "Preferences", "TOML"]
git-tree-sha1 = "e81c4263c7ef4eca4d645ef612814d72e9255b41"
uuid = "992eb4ea-22a4-4c89-a5bb-47a3300528ab"
version = "0.2.22"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "eb3edce0ed4fa32f75a0a11217433c31d56bd48b"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.0"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cad560042a7cc108f5a4c24ea1431a9221f22c1b"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.2"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.6.4+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.MicroMamba]]
deps = ["Pkg", "Scratch", "micromamba_jll"]
git-tree-sha1 = "011cab361eae7bcd7d278f0a7a00ff9c69000c51"
uuid = "0b3b1443-0f03-428d-bdfb-f27f9c1191ea"
version = "0.1.14"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.1.10"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.23+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PythonCall]]
deps = ["CondaPkg", "Dates", "Libdl", "MacroTools", "Markdown", "Pkg", "REPL", "Requires", "Serialization", "Tables", "UnsafePointers"]
git-tree-sha1 = "0fe6664f742903eab8929586af78e10a51b33577"
uuid = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
version = "0.9.19"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ca4bccb03acf9faaf4137a9abc1881ed1841aa70"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.SymPyCore]]
deps = ["CommonEq", "CommonSolve", "Latexify", "LinearAlgebra", "Markdown", "RecipesBase", "SpecialFunctions"]
git-tree-sha1 = "4c5a53625f0e53ce1e726a6dab1c870017303728"
uuid = "458b697b-88f0-4a86-b56b-78b75cfb3531"
version = "0.1.16"

    [deps.SymPyCore.extensions]
    SymPyCoreSymbolicUtilsExt = "SymbolicUtils"

    [deps.SymPyCore.weakdeps]
    SymbolicUtils = "d1185830-fcd6-423d-90d6-eec64667417b"

[[deps.SymPyPythonCall]]
deps = ["CommonEq", "CommonSolve", "CondaPkg", "LinearAlgebra", "PythonCall", "SpecialFunctions", "SymPyCore"]
git-tree-sha1 = "1948385c5c0f0659ca3abcdea214318d691b1770"
uuid = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
version = "0.2.5"

    [deps.SymPyPythonCall.extensions]
    SymPyPythonCallSymbolicsExt = "Symbolics"

    [deps.SymPyPythonCall.weakdeps]
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnsafePointers]]
git-tree-sha1 = "c81331b3b2e60a982be57c046ec91f599ede674a"
uuid = "e17b2a0c-0bdf-430a-bd0c-3a23cae4ff39"
version = "1.0.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.micromamba_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "66d07957bcf7e4930d933195aed484078dd8cbb5"
uuid = "f8abcde7-e9b7-5caa-b8af-a437887ae8e4"
version = "1.4.9+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═220ac3b2-e4e9-422f-b0a9-7dfc8a32b327
# ╠═34979e8f-0d24-4577-aaa2-1462201276dc
# ╟─c226e28d-3a8a-42f6-8a51-2ac418023a27
# ╟─ddbd255b-7ddd-4fba-ac4d-722e88801e59
# ╟─caa44e3e-3af5-4836-8ae3-dd3b9d68d222
# ╟─4b764569-a32d-4170-a11d-2072d971846a
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
# ╠═4c626917-0ded-4f45-88b2-1d8f2d792b33
# ╠═74e584f4-a233-4f18-b04c-dfd1710515e1
# ╟─6ba22886-faa3-4f30-af96-366d9c001c94
# ╠═80d341c5-43ae-4568-bcf3-ced1947662ce
# ╠═1245fe85-25a8-44cb-bbd7-3bc477859d04
# ╟─abe3e648-65e6-423c-83de-673e96013670
# ╠═64535706-eaba-4155-b593-12cba7366127
# ╠═bd88592a-781f-46e7-a175-342b78906c45
# ╠═867af0c2-6ac1-4a06-99da-9e573e2069a4
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
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
