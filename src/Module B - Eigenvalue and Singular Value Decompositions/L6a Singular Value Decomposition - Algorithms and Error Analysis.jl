### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ bfe42bab-98e8-4460-a8ba-12968297d227
begin
	using PlutoUI, LinearAlgebra
	PlutoUI.TableOfContents(aside=true)
end

# ╔═╡ 43d24d37-df93-4ee6-8e48-89f9ca6f0f92
md"""
# Singular Value Decomposition - Algorithms and Error Analysis

We study only algorithms for real matrices, which are most commonly used in the applications described in this course. 


For more details, see 
[A. Kaylor Cline and I. Dhillon, Computation of the Singular Value Decomposition, pp. 58.1-58.13](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.

__Prerequisites__

The reader should be familiar with facts about the singular value decomposition and perturbation theory and algorithms for the symmetric eigenvalue decomposition.

 
__Competences__

The reader should be able to apply an adequate algorithm to a given problem, and to assess the accuracy of the solution.
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

1. The reduction of $A$ to bidiagonal matrix can be performed by applying $\min\{m-1,n\}$ Householder reflections $H_L$ from the left and $n-2$ Householder reflections $H_R$ from the right. In the first step, $H_L$ is chosen to annihilate all elements of the first column below the diagonal, and $H_R$ is chosen to annihilate all elements of the first row right of the first super-diagonal. Applying this procedure recursively yields the bidiagonal matrix.

2.  $H_L$ and $H_R$ do not depend on the normalization of the respective Householder vectors $v_L$ and $v_R$. With the normalization $[v_L]_1=[V_R]_1=1$, the vectors $v_L$ are stored in the lower-triangular part of $A$, and the vectors $v_R$ are stored in the upper-triangular part of $A$ above the super-diagonal. 

3. The matrices $H_L$ and $H_R$ are not formed explicitly - given $v_L$ and $v_R$, $A$ is overwritten with $H_L A H_R$ in $O(mn)$ operations by using matrix-vector multiplication and rank-one updates.

4. Instead of performing rank-one updates, $p$ transformations can be accumulated, and then applied. This __block algorithm__ is rich in matrix-matrix multiplications (roughly one half of the operations is performed using BLAS 3 routines), but it requires extra workspace.

5. If the matrices $X$ and $Y$ are needed explicitly, they can be computed from the stored Householder vectors. In order to minimize the operation count, the computation starts from the smallest matrix and the size is gradually increased.

6. The backward error bounds for the bidiagonalization are as follows: The computed matrix $\tilde B$ is equal to the matrix which would be obtained by exact bidiagonalization of some perturbed matrix $A+\Delta A$, where $\|\Delta A\|_2 \leq \psi \varepsilon \|A\|_2$ and $\psi$ is a slowly increasing function of $n$. The computed matrices $\tilde X$ and $\tilde Y$ satisfy $\tilde X=X+\Delta X$ and $\tilde Y=Y+\Delta Y$, where $\|\Delta X \|_2,\|\Delta Y\|_2\leq \phi \varepsilon$ and $\phi$ is a slowly increasing function of $n$.

7. The bidiagonal reduction is implemented in the [LAPACK](http://www.netlib.org/lapack) subroutine [DGEBRD](http://www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational.html#ga9c735b94f840f927f8085fd23f3ee2e6). The computation of $X$ and $Y$ is implemented in the subroutine [DORGBR](http://www.netlib.org/lapack/lapack-3.1.1/html/dorgtr.f.html), which is not yet wrapped in Julia.

8. Bidiagonalization can also be performed using Givens rotations. Givens rotations act more selectively than Householder reflectors, and are useful if $A$ has some special structure, for example, if $A$ is a banded matrix. Error bounds for function `BidiagG()` are the same as above, but with slightly different functions $\psi$ and $\phi$.
"""

# ╔═╡ 875fce0a-1b1b-4332-a150-6ec5f20aa7dc
md"
## Example
"

# ╔═╡ 3c8c847e-978d-4d43-99ea-773d915e03ea
begin
	m=8
	n=5
	import Random
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
# Extract X
function BidiagX(H::Matrix)
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
# Extract Y
function BidiagY(H::AbstractMatrix)
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

All metods for computing the SVD of bidiagonal matrix are derived from the methods for computing the EVD of the tridiagonal matrix $T=B^T B$.


## Facts

1. The shift $\mu$ is the eigenvalue of the $2\times 2$ matrix $T_{n-1:n,n-1:n}$ which is closer to $T_{n,n}$. The first Givens rotation from the right is the one which annihilates the element $(1,2)$ of the shifted $2\times 2$ matrix $T_{1:2,1:2}-\mu I$. Applying this rotation to $B$ creates the bulge at the element $B_{2,1}$. This bulge is subsequently chased out by applying adequate Givens rotations alternating from the left and from the right. This is the __Golub-Kahan algorithm__.

2. The computed SVD satisfes error bounds from the Fact 4 above.

3. The special variant of zero-shift QR algorithm (the __Demmel-Kahan algorithm__) computes the singular values with high relative accuracy. 

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
σ-σ₁

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

# ╔═╡ Cell order:
# ╟─bfe42bab-98e8-4460-a8ba-12968297d227
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
