### A Pluto.jl notebook ###
# v0.19.20

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
using PlutoUI, Random, LinearAlgebra, SymPy

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

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[compat]
PlutoUI = "~0.7.38"
SymPy = "~1.1.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "bd787ad51a8ab5d0b73840f32474f07613d21302"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.CommonEq]]
git-tree-sha1 = "d1beba82ceee6dc0fce8cb6b80bf600bbde66381"
uuid = "3709ef60-1bee-4518-9f2f-acd86f176c50"
version = "0.2.0"

[[deps.CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "b153278a25dd42c65abbf4e62344f9d22e59191b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.43.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6e47d11ea2776bc5627421d59cdcc1296c058071"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.7.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

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

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "6f14549f7760d84b2db7a9b10b88cd3cc3025730"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.14"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a970d55c2ad8084ca317a4658ba6ce99b7523571"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.12"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "670e559e5c8e191ded66fa9ea89c97f10376bb4c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.38"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "1fc929f47d7c151c839c5fc1375929766fb8edcc"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.93.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

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

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.SymPy]]
deps = ["CommonEq", "CommonSolve", "Latexify", "LinearAlgebra", "Markdown", "PyCall", "RecipesBase", "SpecialFunctions"]
git-tree-sha1 = "e1865ba3c44551087a04295ddc40c10edf1b24a0"
uuid = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
version = "1.1.6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VersionParsing]]
git-tree-sha1 = "58d6e80b4ee071f5efd07fda82cb9fbe17200868"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.3.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
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
