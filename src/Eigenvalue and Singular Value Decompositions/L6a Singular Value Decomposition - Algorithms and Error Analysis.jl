### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 3496436d-ea6b-4dda-8e06-1140e1de83dc
using PlutoUI, LinearAlgebra, Random

# ╔═╡ 39afa4c0-b02e-40cb-9086-a93bd26a8521
TableOfContents(aside=true)

# ╔═╡ 43d24d37-df93-4ee6-8e48-89f9ca6f0f92
md"""
# Singular Value Decomposition - Algorithms and Error Analysis

We study only algorithms for real matrices, which are most commonly used in the applications described in this course. 


For more details, see 
[A. Kaylor Cline and I. Dhillon, Computation of the Singular Value Decomposition, pp. 58.1-58.13](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.

__Prerequisites__

The reader should be familiar with facts about the singular value decomposition and perturbation theory and algorithms for the symmetric eigenvalue decomposition.

 
__Competences__

The reader should be able to apply an adequate algorithm to a given problem and assess the accuracy of the solution.
"""

# ╔═╡ 64ca893a-a1ee-4172-bd20-cf4aca1c3d41
md"""
# Basics

## Definitions 

The __singular value decomposition__ (SVD) of $A\in\mathbb{R}^{m\times n}$ is $A=U\Sigma V^T$, where $U\in\mathbb{R}^{m\times m}$ is orthogonal, $U^TU=UU^T=I_m$, $V\in\mathbb{R}^{n\times n}$ is orthogonal, $V^TV=VV^T=I_n$, and $\Sigma \in\mathbb{R}^{m\times n}$ is diagonal with singular values $\sigma_1,\ldots,\sigma_{\min\{m,n\}}$ on the diagonal. 

If $m>n$, the __thin SVD__ of $A$ is $A=U_{1:m,1:n} \Sigma_{1:n,1:n} V^T$.

## Facts

1. Algorithms for computing SVD of $A$ are modifications of algorithms for the symmetric eigenvalue decomposition of the matrices $AA^T$, $A^TA$ and $\begin{bmatrix} 0 & A\\ A^T & 0 \end{bmatrix}$.

2. Most commonly used approach is the three-step algorithm:
    1. Reduce $A$ to bidiagonal matrix $B$ by orthogonal transformations, $X^TAY=B$.
    2. Compute the SVD of $B$ with QR iterations, $B=W\Sigma Z^T$.
    3. Multiply $U=XW$ and $V=YZ$.

3. If $m\geq n$, the overall operation count for this algorithm is $O(mn^2)$ operations.

4. __Error bounds.__ Let $U\Sigma V^T$ and $\tilde U \tilde \Sigma \tilde V^T$ be the exact and the computed SVDs of $A$, respectively. The algorithms generally compute the SVD with errors bounded by

$$
|\sigma_i-\tilde \sigma_i|\leq \phi \epsilon\|A\|_2,
\qquad
\|u_i-\tilde u_i\|_2, \| v_i-\tilde v_i\|_2 \leq \psi\epsilon \frac{\|A\|_2}
{\min_{j\neq i} 
|\sigma_i-\tilde \sigma_j|},$$

where $\epsilon$ is machine precision and $\phi$ and $\psi$ are slowly growing polynomial functions of $n$ which depend upon the algorithm used (typically $O(n)$ or $O(n^2)$). These bounds are obtained by combining perturbation bounds with the floating-point error analysis of the algorithms.
"""

# ╔═╡ 38b90f14-4c98-4a9c-a54a-5fabacb252ea
md"""
#  Bidiagonalization


## Facts

1. The reduction of $A$ to bidiagonal matrix can be performed by applying $\min\{m-1,n\}$ Householder reflections $H_L$ from the left and $n-2$ Householder reflections $H_R$ from the right. In the first step, $H_L$ is chosen to annihilate all elements of the first column below the diagonal and $H_R$ is chosen to annihilate all elements of the first row right of the first super-diagonal. Applying this procedure recursively yields the bidiagonal matrix.

2.  $H_L$ and $H_R$ do not depend on the normalization of the respective Householder vectors $v_L$ and $v_R$. With the normalization $[v_L]_1=[V_R]_1=1$, the vectors $v_L$ are stored in the lower-triangular part of $A$, and the vectors $v_R$ are stored in the upper-triangular part of $A$ above the super-diagonal. 

3. The matrices $H_L$ and $H_R$ are not formed explicitly - given $v_L$ and $v_R$, $A$ is overwritten with $H_L A H_R$ in $O(mn)$ operations by using matrix-vector multiplication and rank-one updates.

4. Instead of performing rank-one updates, $p$ transformations can be accumulated and then applied. This __block algorithm__ is rich in matrix-matrix multiplications (roughly one-half of the operations is performed using BLAS 3 routines), but it requires extra workspace.

5. If the matrices $X$ and $Y$ are needed explicitly, they can be computed from the stored Householder vectors. To minimize the operation count, the computation starts from the smallest matrix, and the size is gradually increased.

6. The backward error bounds for the bidiagonalization are as follows: The computed matrix $\tilde B$ is equal to the matrix which would be obtained by exact bidiagonalization of some perturbed matrix $A+\Delta A$, where $\|\Delta A\|_2 \leq \psi \varepsilon \|A\|_2$ and $\psi$ is a slowly increasing function of $n$. The computed matrices $\tilde X$ and $\tilde Y$ satisfy $\tilde X=X+\Delta X$ and $\tilde Y=Y+\Delta Y$, where $\|\Delta X \|_2,\|\Delta Y\|_2\leq \phi \varepsilon$ and $\phi$ is a slowly increasing function of $n$.

7. The bidiagonal reduction is implemented in the [LAPACK](http://www.netlib.org/lapack) subroutine [DGEBRD](http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational.html#ga9c735b94f840f927f8085fd23f3ee2e6). The computation of $X$ and $Y$ is implemented in the subroutine [DORGBR](http://www.netlib.org/lapack/lapack-3.1.1/html/dorgtr.f.html), which is not yet wrapped in Julia.

8. Bidiagonalization can also be performed using Givens rotations. Givens rotations act more selectively than Householder reflectors and are useful if $A$ has some special structure, for example, if $A$ is a banded matrix. Error bounds for function `BidiagG()` are the same as above but with slightly different functions $\psi$ and $\phi$.
"""

# ╔═╡ 875fce0a-1b1b-4332-a150-6ec5f20aa7dc
md"
## Example
"

# ╔═╡ 3c8c847e-978d-4d43-99ea-773d915e03ea
begin
	m=8
	n=5
	Random.seed!(421)
	A=rand(-9.0:9,m,n)
end

# ╔═╡ fa7b3f29-d9bc-4a64-90ae-16a6a3f8e3a6
# Householder matrix from vector
function H(x)
    v=copy(x)
    v[1]+=sign(x[1])*norm(x)
    # display(v/v[1])
    I-(2/(v⋅v))*v*v'
end

# ╔═╡ 78d6a3a1-9fd5-4eec-94dd-2baf31cf312e
H₁=H(A[:,1])

# ╔═╡ 2447bb48-334b-4554-b7ca-5b4f74f68be9
# Orthogonality
norm(H₁'*H₁-I)

# ╔═╡ ed9bfaff-ed9f-442d-b5a5-c0435e22d2ef
B₁=H₁*A

# ╔═╡ 8856d77a-f2fb-4d7e-aca3-bde649508c8a
H¹=cat(1,H(B₁[1,2:5]),dims=(1,2))

# ╔═╡ 5d742283-df40-416b-8067-1a21f89f6e71
H₁*A*H¹

# ╔═╡ fd73c193-a5bb-4eb9-a0ce-aca44b7ef7ee
# ?LAPACK.gebrd!

# ╔═╡ 783deb44-2d8f-4a40-80d5-cc65bcf9018c
# We need copy()
O=LAPACK.gebrd!(copy(A))

# ╔═╡ cfba6cc8-451c-4716-95a2-9a3f49009253
O[4]

# ╔═╡ 2f92ce80-72d0-4424-949e-a4f07065102c
B=Bidiagonal(O[2],O[3][1:end-1],'U')

# ╔═╡ e132f765-d232-40e3-9d41-1ec07528dbae
[svdvals(A) svdvals(B)]

# ╔═╡ 583548a3-b3ad-493f-8061-fae21138208a
function BidiagX(H::Matrix)
	# Extract matrix X from the matrix computed by LAPACK.gebrd!()
    m,n=size(H)
    T=typeof(H[1,1])
    X = Matrix{T}(I,m,n)
    v = Array{T}(undef,m)
    for j = n : -1 : 1
        v[j] = one(T)
        v[j+1:m] = H[j+1:m, j]
        γ = -2 / (v[j:m]⋅v[j:m])
        w = γ * X[j:m, j:n]'*v[j:m]
        X[j:m, j:n] = X[j:m, j:n] + v[j:m]*w'
    end
    X
end

# ╔═╡ 75d77c3b-649e-4d20-a1cf-5cc4d1335896
function BidiagY(H::AbstractMatrix)
	# Extract matrix Y from the matrix computed by LAPACK.gebrd!()
    n,m=size(H)
    T=typeof(H[1,1])
    Y = Matrix{T}(I,n,n)
    v = Array{T}(undef,n)
    for j = n-2 : -1 : 1
        v[j+1] = one(T)
        v[j+2:n] = H[j+2:n, j]
        γ = -2 / (v[j+1:n]⋅v[j+1:n])
        w = γ * Y[j+1:n, j+1:n]'*v[j+1:n]
        Y[j+1:n, j+1:n] = Y[j+1:n, j+1:n] + v[j+1:n]*w'
    end
    Y
end

# ╔═╡ 28a70c10-3900-4a22-9c93-41abe15b5ce8
X=BidiagX(O[1])

# ╔═╡ a0065053-51a6-4351-b7e8-da5e37b3eb94
Y=BidiagY(O[1]')

# ╔═╡ 583eec81-0a99-4ccc-9e7e-0625a942afbc
# Orthogonality
norm(X'*X-I), norm(Y'*Y-I)

# ╔═╡ 0b4010cc-5956-4a52-879f-dbc3dd6cc519
# Residual error
norm(A*Y-X*B)

# ╔═╡ 7d6da96f-c272-46e8-979c-14cc2a92d071
# Bidiagonalization using Givens rotations
function BidiagG(A1::Matrix)
    A=deepcopy(A1)
    m,n=size(A)
    T=typeof(A[1,1])
    X=Matrix{T}(I,m,m)
    Y=Matrix{T}(I,n,n)
    for j = 1 : n        
        for i = j+1 : m
            G,r=givens(A,j,i,j)
            # Use the faster in-place variant
            # A=G*A
            lmul!(G,A)
            # X=G*X
            lmul!(G,X)
            # display(A)
        end
        for i=j+2:n
            G,r=givens(A',j+1,i,j)
            # A*=adjoint(G)
            rmul!(A,adjoint(G))
            # Y*=adjoint(G)
            rmul!(Y,adjoint(G))
            # display(A)
        end
    end
    X',Bidiagonal(diag(A),diag(A,1),'U'), Y
end

# ╔═╡ 5496e8da-6899-4792-8c20-46a5f8843354
X₁, B¹, Y₁ = BidiagG(A)

# ╔═╡ 1caedf0a-23d5-460e-8e61-f533f7f731a0
# Orthogonality
norm(X₁'*X₁-I), norm(Y₁'*Y₁-I)

# ╔═╡ e5ccbecf-5bf3-4c0e-a036-2f8fab7d0a3f
# Diagonalization
X₁'*A*Y₁

# ╔═╡ c5820b28-6e62-4b87-af13-6cbf3a3369a5
# X, Y and B are not unique
B

# ╔═╡ c2552861-ffc9-45e1-aac4-e44b30f28ee3
B¹

# ╔═╡ 28ee391d-2ae4-4d35-9832-973071e55681
md"""
# Bidiagonal QR method

Let $B$ be a real upper-bidiagonal matrix of order $n$ and let $B=W\Sigma Z^T$ be its SVD.

All methods for computing the SVD of a bidiagonal matrix are derived from the methods for computing the EVD of the tridiagonal matrix $T=B^T B$.


## Facts

1. The shift $\mu$ is the eigenvalue of the $2\times 2$ matrix $T_{n-1:n,n-1:n}$ which is closer to $T_{n,n}$. The first Givens rotation from the right is the one which annihilates the element $(1,2)$ of the shifted $2\times 2$ matrix $T_{1:2,1:2}-\mu I$. Applying this rotation to $B$ creates the bulge at the element $B_{2,1}$. This bulge is subsequently chased out by applying adequate Givens rotations alternating from the left and the right. This is the __Golub-Kahan algorithm__.

2. The computed SVD satisfies error bounds from Fact 4 above.

3. The special variant of the zero-shift QR algorithm (the __Demmel-Kahan algorithm__) computes the singular values with high relative accuracy. 

4. The tridiagonal divide-and-conquer method, bisection and inverse iteration, and MRRR method can also be adapted for bidiagonal matrices. 

5. Zero shift QR algorithm for bidiagonal matrices is implemented in the LAPACK routine [DBDSQR](http://www.netlib.org/lapack/explore-html/db/dcc/dbdsqr_8f.html). It is also used in the function `svdvals()`. Divide-and-conquer algorithm for bidiagonal matrices is implemented in the LAPACK routine [DBDSDC](http://www.netlib.org/lapack/explore-html/d9/d08/dbdsdc_8f.html). However, this algorithm also calls zero-shift QR to compute singular values.
"""

# ╔═╡ e64fc85e-2861-4e42-9789-ee12b6f9950f
md"
## Examples
"

# ╔═╡ a0d540ea-d116-499a-bcad-d6bfe3e52470
W,σ,Z=svd(B)

# ╔═╡ 2b900980-6ccb-441c-9ef5-e7dc68ff843e
@which svd(B)

# ╔═╡ 90970845-d921-4b88-9681-d344451d4c3c
σ₁=svdvals(B)

# ╔═╡ 860a3476-b257-4384-b255-4a8bd68728c9
@which svdvals!(B)

# ╔═╡ 2c1943df-95d8-43bb-bf6a-8fe13424a2c9
norm(σ-σ₁)

# ╔═╡ 1a79b153-6560-45d7-8113-e39101d17460
# ?LAPACK.bdsqr!

# ╔═╡ 7fafeb92-568a-4270-b4fb-4d44c655bcfb
B.dv

# ╔═╡ 99109bd8-e2ca-435c-a242-5772ea4432c0
one(B)

# ╔═╡ ef17e375-670b-4b7d-878a-bc2c06760651
begin
	V₀=Matrix(one(B))
	U₀=Matrix(one(B))
	C₀=Matrix(one(B))
	σ₂,Z₂,W₂,C = LAPACK.bdsqr!('U',copy(B.dv),copy(B.ev),V₀,U₀,C₀)
end

# ╔═╡ 613c0e7b-7288-4dac-a6be-731cf4f8be63
W₂'*B*Z₂'

# ╔═╡ e8119506-bd02-47e2-890c-c088fd883532
#?LAPACK.bdsdc!

# ╔═╡ ecf74ee7-97e3-4749-b63d-2cf200e4451e
σ₃,ee,W₃,Z₃,rest=LAPACK.bdsdc!('U','I',copy(B.dv),copy(B.ev))

# ╔═╡ cbed834b-4935-431c-96f6-3630534ba2eb
W₃'*B*Z₃'

# ╔═╡ ffa18a0d-c576-473a-9851-f7bd70a843ba
md"""
Functions `svd()`, `LAPACK.bdsqr!()` and `LAPACK.bdsdc!()` use the same algorithm to compute singular values.
"""

# ╔═╡ beec8bad-9621-4c66-88d3-4f48dbcf5566
[σ₃-σ₂ σ₃-σ]

# ╔═╡ 569d2241-b9a8-4501-8b17-9527e1777cdc
md"
Let us compute some timings. We observe $O(n^2)$ operations.
"

# ╔═╡ cfd21881-5cfe-430e-b60a-a783feac2cdb
begin
	n₀=1000
	Bₙ=Bidiagonal(rand(n₀),rand(n₀-1),'U')
	B₂ₙ=Bidiagonal(rand(2*n₀),rand(2*n₀-1),'U')
	println(" ")
	@time svdvals(Bₙ)
	@time svdvals(B₂ₙ)
	@time LAPACK.bdsdc!('U','N',copy(Bₙ.dv),copy(Bₙ.ev))
	@time svd(Bₙ)
	@time svd(B₂ₙ)
	1
end

# ╔═╡ 5ef6758d-51f9-49c5-a736-413f8be3e734
md"""
# QR method

Final algorithm is obtained by combining bidiagonalization and bidiagonal SVD methods. Standard method is implemented in the LAPACK routine [DGESVD](http://www.netlib.org/lapack/explore-html/d8/d2d/dgesvd_8f.html). Divide-and-conquer method is implemented in the LAPACK routine [DGESDD](http://www.netlib.org/lapack/explore-html/db/db4/dgesdd_8f.html).

The functions `svd()`, `svdvals()`, and `svdvecs()`  use `DGESDD`. Wrappers for `DGESVD` and `DGESDD` give more control about output of eigenvectors.
"""

# ╔═╡ 416c8bca-215c-491d-b1f1-1abf0793b65b
# The built-in algorithm
U,σA,V=svd(A)

# ╔═╡ 17d90d71-0776-498d-8c09-9d651ee3ea1f
Sv=svd(A)

# ╔═╡ c8e602f7-9ccf-4753-9aa5-a07df141cf40
Sv.Vt

# ╔═╡ 4ef07f50-e9aa-4075-960a-dce2ee9f9d9e
begin
	# With our building blocks
	U₁=X*W
	V₁=Y*Z
	U₁'*A*V₁
end

# ╔═╡ bb4e96f8-e61a-486c-b385-72dd2542e7fc
#?LAPACK.gesvd!

# ╔═╡ 7c5cea63-5273-46df-ae5e-58a7b4312360
# DGESVD
LAPACK.gesvd!('A','A',copy(A))

# ╔═╡ dd07def6-6ee3-4f48-ba11-340674fbc22d
#?LAPACK.gesdd!

# ╔═╡ 7e8aebb8-7621-4fd2-afeb-bdc71b091cd1
LAPACK.gesdd!('N',copy(A))

# ╔═╡ 08f390ca-5122-4a38-b7c0-631e7d9ea726
md"""
Let us perform some timings. We observe $O(n^3)$ operations.
"""

# ╔═╡ f425bd0a-ebde-42a5-8585-270b7e86f27f
begin
	n₁=1000
	Aₙ=rand(n₁,n₁)
	A₂ₙ=rand(2*n₁,2*n₁)
	println(" ")
	@time Uₙ,σₙ,Vₙ=svd(Aₙ)
	@time svd(A₂ₙ)
	@time LAPACK.gesvd!('A','A',copy(Aₙ))
	@time LAPACK.gesdd!('A',copy(Aₙ))
	@time LAPACK.gesdd!('A',copy(A₂ₙ))
	1
end

# ╔═╡ ed98fe9c-1ee8-4333-8ace-958223df6552
# Residual
norm(Aₙ*Vₙ-Uₙ*Diagonal(σₙ))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
PlutoUI = "~0.7.58"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "1867d9ce1bd88115b124f124b5d7cd866c186b11"

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

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

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

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

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

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

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

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

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

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

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

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

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
# ╠═3496436d-ea6b-4dda-8e06-1140e1de83dc
# ╠═39afa4c0-b02e-40cb-9086-a93bd26a8521
# ╟─43d24d37-df93-4ee6-8e48-89f9ca6f0f92
# ╟─64ca893a-a1ee-4172-bd20-cf4aca1c3d41
# ╟─38b90f14-4c98-4a9c-a54a-5fabacb252ea
# ╟─875fce0a-1b1b-4332-a150-6ec5f20aa7dc
# ╠═3c8c847e-978d-4d43-99ea-773d915e03ea
# ╠═fa7b3f29-d9bc-4a64-90ae-16a6a3f8e3a6
# ╠═78d6a3a1-9fd5-4eec-94dd-2baf31cf312e
# ╠═2447bb48-334b-4554-b7ca-5b4f74f68be9
# ╠═ed9bfaff-ed9f-442d-b5a5-c0435e22d2ef
# ╠═8856d77a-f2fb-4d7e-aca3-bde649508c8a
# ╠═5d742283-df40-416b-8067-1a21f89f6e71
# ╠═fd73c193-a5bb-4eb9-a0ce-aca44b7ef7ee
# ╠═783deb44-2d8f-4a40-80d5-cc65bcf9018c
# ╠═cfba6cc8-451c-4716-95a2-9a3f49009253
# ╠═2f92ce80-72d0-4424-949e-a4f07065102c
# ╠═e132f765-d232-40e3-9d41-1ec07528dbae
# ╠═583548a3-b3ad-493f-8061-fae21138208a
# ╠═75d77c3b-649e-4d20-a1cf-5cc4d1335896
# ╠═28a70c10-3900-4a22-9c93-41abe15b5ce8
# ╠═a0065053-51a6-4351-b7e8-da5e37b3eb94
# ╠═583eec81-0a99-4ccc-9e7e-0625a942afbc
# ╠═0b4010cc-5956-4a52-879f-dbc3dd6cc519
# ╠═7d6da96f-c272-46e8-979c-14cc2a92d071
# ╠═5496e8da-6899-4792-8c20-46a5f8843354
# ╠═1caedf0a-23d5-460e-8e61-f533f7f731a0
# ╠═e5ccbecf-5bf3-4c0e-a036-2f8fab7d0a3f
# ╠═c5820b28-6e62-4b87-af13-6cbf3a3369a5
# ╠═c2552861-ffc9-45e1-aac4-e44b30f28ee3
# ╟─28ee391d-2ae4-4d35-9832-973071e55681
# ╟─e64fc85e-2861-4e42-9789-ee12b6f9950f
# ╠═a0d540ea-d116-499a-bcad-d6bfe3e52470
# ╠═2b900980-6ccb-441c-9ef5-e7dc68ff843e
# ╠═90970845-d921-4b88-9681-d344451d4c3c
# ╠═860a3476-b257-4384-b255-4a8bd68728c9
# ╠═2c1943df-95d8-43bb-bf6a-8fe13424a2c9
# ╠═1a79b153-6560-45d7-8113-e39101d17460
# ╠═7fafeb92-568a-4270-b4fb-4d44c655bcfb
# ╠═99109bd8-e2ca-435c-a242-5772ea4432c0
# ╠═ef17e375-670b-4b7d-878a-bc2c06760651
# ╠═613c0e7b-7288-4dac-a6be-731cf4f8be63
# ╠═e8119506-bd02-47e2-890c-c088fd883532
# ╠═ecf74ee7-97e3-4749-b63d-2cf200e4451e
# ╠═cbed834b-4935-431c-96f6-3630534ba2eb
# ╟─ffa18a0d-c576-473a-9851-f7bd70a843ba
# ╠═beec8bad-9621-4c66-88d3-4f48dbcf5566
# ╟─569d2241-b9a8-4501-8b17-9527e1777cdc
# ╠═cfd21881-5cfe-430e-b60a-a783feac2cdb
# ╟─5ef6758d-51f9-49c5-a736-413f8be3e734
# ╠═416c8bca-215c-491d-b1f1-1abf0793b65b
# ╠═17d90d71-0776-498d-8c09-9d651ee3ea1f
# ╠═c8e602f7-9ccf-4753-9aa5-a07df141cf40
# ╠═4ef07f50-e9aa-4075-960a-dce2ee9f9d9e
# ╠═bb4e96f8-e61a-486c-b385-72dd2542e7fc
# ╠═7c5cea63-5273-46df-ae5e-58a7b4312360
# ╠═dd07def6-6ee3-4f48-ba11-340674fbc22d
# ╠═7e8aebb8-7621-4fd2-afeb-bdc71b091cd1
# ╟─08f390ca-5122-4a38-b7c0-631e7d9ea726
# ╠═f425bd0a-ebde-42a5-8585-270b7e86f27f
# ╠═ed98fe9c-1ee8-4333-8ace-958223df6552
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
