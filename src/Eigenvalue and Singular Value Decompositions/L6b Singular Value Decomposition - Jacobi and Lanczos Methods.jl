### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ fdbecdd6-3cdf-45ae-82c1-10e7016e7880
using PlutoUI, LinearAlgebra, Arpack

# ╔═╡ 8db1c1b2-8b5f-402c-8441-768c1a943c38
using SparseArrays

# ╔═╡ 5a428e03-2145-4767-ac0b-95f30a76af5c
import LinearAlgebra.Givens

# ╔═╡ 1ac3f71a-36bb-4d4a-90ed-de3e70c75cfe
PlutoUI.TableOfContents(aside=true)

# ╔═╡ f04cb86f-94aa-418d-82f9-d9ef12ca90fd
md"""
# Singular Value Decomposition - Jacobi and Lanczos Methods

Since computing the SVD of $A$ can be seen as computing the EVD of the symmetric matrices $A^*A$, $AA^*$, or $\begin{bmatrix}0 & A \\ A^* & 0 \end{bmatrix}$, simple modifications of the corresponding EVD algorithms yield version for computing the SVD.

For more details on one-sided Jacobi method, see [Z. Drmač, Computing Eigenvalues and Singular Values to High Relative Accuracy, pp. 59.1-59.21](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.

__Prerequisites__

The reader should be familiar with concepts of singular values and vectors, related perturbation theory, and algorithms, and Jacobi and Lanczos methods for the symmetric eigenvalue decomposition.

__Competences__

The reader should be able to recognise matrices which warrant high relative accuracy and to apply Jacobi method to them. The reader should be able to recognise matrices to which Lanczos method can be efficiently applied and do so.
"""

# ╔═╡ 2eff5f1f-0989-4292-8f04-80983c1bd133
md"""
# One-sided Jacobi method

Let $A\in\mathbb{R}^{m\times n}$ with $\mathop{\mathrm{rank}}(A)=n$ (therefore, $m\geq n$) and $A=U\Sigma V^T$ its thin SVD.

## Definition

Let $A=BD$, where $D=\mathop{\mathrm{diag}} (\| A_{:,1}\|_2, \ldots, \|A_{:,n}\|_2)$ is a __diagonal scaling__ , and $B$ is the __scaled matrix__ of $A$ from the right.  Then $[B^T B]_{i,i}=1$.

## Facts

1. Let $\tilde U$, $\tilde V$ and $\tilde \Sigma$ be the approximations of $U$, $V$ and $\Sigma$, respectively, computed by a backward stable algorithm as $A+\Delta A=\tilde U\tilde \Sigma \tilde V^T$. Since the orthogonality of $\tilde U$ and $\tilde V$ cannot be guaranteed, this product in general does not represent and SVD. There exist nearby orthogonal matrices $\hat U$ and $\hat V$ such that $(I+E_1)(A+\Delta A)(I+E_2)=\hat U \tilde \Sigma \hat V^T$, where departures from orthogonalithy, $E_1$ and $E_2$, are small in norm.

2. Standard algorithms compute the singular values with backward error $\| \Delta A\|\leq \phi\varepsilon \|A\|_2$, where $\varepsilon$ is machine precision and $\phi$ is a slowly growing function of $n$. The best error bound for the singular values is $|\sigma_j-\tilde \sigma_j|\leq \| \Delta A\|_2$, and the best relative error bound is

$$
\max_j \frac{|\sigma_j-\tilde\sigma_j|}{\sigma_j}\leq \frac{\| \Delta A \|_2}{\sigma_j} \leq \phi \varepsilon \kappa_2(A).$$

3. Let $\|[\Delta A]_{:,j}\|_2\leq \varepsilon \| A_{:,j}\|_2$ for all $j$. Then $A+\Delta A=(B+\Delta B)D$ and $\|\Delta B\|_F\leq \sqrt{n}\varepsilon$, and

$$
\max_j \frac{|\sigma_j-\tilde\sigma_j|}{\sigma_j}\leq 
\| (\Delta B) B^{\dagger} \|_2\leq
\sqrt{n}\varepsilon \| B^{\dagger}\|_2.$$

This is Fact 3 from the [Relative perturbation theory](https://ivanslapnicar.github.io/GIAN-Applied-NLA-Course/L5b%20Singular%20Value%20Decomposition%20-%20Perturbation%20Theory%20.jl.html).

4. It holds

$$
\| B^{\dagger} \| \leq \kappa_2(B) \leq \sqrt{n} \min_{S=\mathop{\mathrm{diag}}} 
\kappa_2(A\cdot S)\leq \sqrt{n}\,\kappa_2(A).$$

Therefore, numerical algorithm with column-wise small backward error computes singular values more accurately than an algorithm with small norm-wise backward error.

5. In each step, one-sided Jacobi method computes the Jacobi rotation matrix from the pivot submatrix of the current Gram matrix $A^TA$. Afterwards, $A$ is multiplied with the computed rotation matrix from the right (only two columns are affected). Convergence of the Jacobi method for the symmetric matrix $A^TA$ to a diagonal matrix, implies that the matrix $A$ converges to the matrix $AV$ with orthogonal columns and $V^TV=I$. Then $AV=U\Sigma$, $\Sigma=\mathop{\mathrm{diag}}(\| A_{:,1}\|_2, \ldots, \| A_{:,n}\|_2)$, $U=AV\Sigma^{-1}$, and  $A=U\Sigma V^T$ is the SVD of $A$.

6. One-sided Jacobi method computes the SVD with error bound from Facts 2 and 3, provided that the condition of the intermittent scaled matrices does not grow much. There is overwhelming numerical evidence for this. Alternatively, if $A$ is square, the one-sided Jacobi method can be applied to the transposed matrix $A^T=DB^T$ and the same error bounds apply, but the condition of the scaled matrix  (_this time from the left_) does not change. This approach is slower.

7. One-sided Jacobi method can be preconditioned by applying one QR factorization with full pivoting and one QR factorization withour pivoting to $A$, to obtain faster convergence, without sacrifying accuracy. This method is implemented in the LAPACK routine [DGESVJ](http://www.netlib.org/lapack/explore-html-3.3.1/d1/d5e/dgesvj_8f_source.html). _Writing the wrapper for `DGESVJ` is a tutorial assignment._

"""

# ╔═╡ 21541d5a-d787-438f-ada3-56e07e62ae4e
md"
## Examples

### Random matrix
"

# ╔═╡ 9fd7b6a0-ed2f-486b-99ec-806c1b19f033
function JacobiR(A₁::AbstractMatrix)
    A=deepcopy(A₁)
    m,n=size(A)
    T=typeof(A[1,1])
    V=Matrix{T}(I,n,n)
    # Tolerance for rotation
    tol=√n*eps(T)
    # Counters
    p=n*(n-1)/2
    sweep=0
    pcurrent=0
    # First criterion is for standard accuracy, second one is for relative accuracy
    # while sweep<30 && vecnorm(A-diagm(diag(A)))>tol
    while sweep<30 && pcurrent<p
        sweep+=1
        # Row-cyclic strategy
        for i = 1 : n-1 
            for j = i+1 : n
                # Compute the 2 x 2 sumbatrix of A'*A
                # F=A[:,[i,j]]'*A[:,[i,j]]
				F=view(A,:,[i,j])'*view(A,:,[i,j])
                # Check the tolerance - the first criterion is standard,
                # the second one is for relative accuracy               
                # if A[i,j]!=zero(T)
                # 
                if abs(F[1,2])>tol*sqrt(F[1,1]*F[2,2])
                    # Compute c and s
                    τ=(F[2,2]-F[1,1])/(2*F[1,2])
                    t=sign(τ)/(abs(τ)+√(1+τ^2))
                    c=1/sqrt(1+t^2)
                    s=c*t
                    G=LinearAlgebra.Givens(i,j,c,s)
                    # A*=G'
                    rmul!(A,G)
                    # V*=G'
                    rmul!(V,G)
                    pcurrent=0
                else
                    pcurrent+=1
                end
            end
        end
    end
    σ=[norm(A[:,k]) for k=1:n]
    for k=1:n
        A[:,k]./=σ[k]
    end
    # A, σ, V
    SVD(A,σ,adjoint(V))
end

# ╔═╡ 04b83647-fde5-40ad-a0e1-aebfd99c40b5
begin
	m=8
	n=5
	import Random
	Random.seed!(432)
	A=rand(-9.0:9,m,n)
end

# ╔═╡ 751c0242-28cf-4e85-a28c-adc2b096511d
U,σ,V=JacobiR(A)

# ╔═╡ 3b813d5b-b27f-41bd-8c28-b7a4f3591ca8
# Residual and orthogonality
norm(A*V-U*Diagonal(σ)), norm(U'*U-I),norm(V'*V-I)

# ╔═╡ 25b11022-ddd4-48c7-91a8-8306c8f04d08
md"""
### Strongly scaled matrix
"""

# ╔═╡ 90059aec-9467-4548-9ef3-05a36d8e730e
begin
	m₁=20
	n₁=15
	B₁=rand(m₁,n₁)
	D₁=exp.(50*(rand(n₁).-0.5))
	A₁=B₁*Diagonal(D₁)
end

# ╔═╡ d71f46c3-0265-4e9b-9e11-bad11ae617ca
cond(B₁), cond(A₁)

# ╔═╡ 9b015091-c8b6-416f-bfa0-cd8742feae53
U₁,σ₁,V₁=JacobiR(A₁);

# ╔═╡ fb528048-4886-476d-acf6-bb0d1105f3c3
[sort(σ₁,rev=true) svdvals(A₁)]

# ╔═╡ 2c628908-f9e2-4b04-a7b8-ebe6ff97d207
# Relative errors
(sort(σ₁,rev=true)-svdvals(A₁))./sort(σ₁,rev=true)

# ╔═╡ f1734a00-d452-4adf-8125-dfbcee74d338
# Residual
norm(A₁*V₁-U₁*Diagonal(σ₁))/norm(A₁)

# ╔═╡ caa171b1-e8cd-4fbe-a064-964c1a896f7a
# Diagonalization
U₁'*A₁*V₁

# ╔═╡ 98fa4641-2d3f-48f5-ae1d-a9f3b96f798e
md"""
In the alternative approach, we first apply QR factorization with column pivoting to obtain the square matrix.
"""

# ╔═╡ 19f3f447-ca93-4a6a-8b3f-403afd273fdd
# ?qr

# ╔═╡ 9dc37b20-3b94-4cbe-9c96-0bf12bfbe005
Q=qr(A₁,Val(true))

# ╔═╡ 38954e60-3f69-407e-8000-8a6e4477c245
# Residual
norm(Q.Q*Q.R-A₁[:,Q.p])

# ╔═╡ bda5ca94-e567-4615-a3f4-bce3c88cfb39
S=JacobiR(Q.R')

# ╔═╡ aeb0105f-53e6-437b-a1f0-376ffcfdb5d7
(sort(σ₁)-sort(S.S))./sort(σ₁)

# ╔═╡ 7c3ffb94-6403-43da-81fa-6fd8ef8a24b8
md"
Now $QRP^T=A$ and $R^T=U_R\Sigma_R V^T_R$, so 

$$
A=(Q V_R) \Sigma_R (U_R^T P^T)$$ 

is an SVD of $A$.
"

# ╔═╡ 11d8a586-361f-42f4-bd9a-77e483fc7a74
# Residual
norm(A₁*S.U[invperm(Q.p),:]-Q.Q*S.V*Diagonal(S.S))

# ╔═╡ e1dffb13-ed1b-44b4-9cb1-bd5227962cc9
md"
# Lanczos method

The function `svds()` is based on the Lanczos method for symmetric matrices. Input can be matrix, but also an operator which defines the product of the given matrix with a vector.

## Examples

### Random matrix
"

# ╔═╡ e6f4eb1b-2a3c-4785-8645-df5c72aa1ad7
# ?svds

# ╔═╡ 8c66ea99-3a59-4923-b73c-4ae3c9d7062e
begin
	m₂=20
	n₂=15
	A₂=rand(m₂,n₂)
end

# ╔═╡ 8af3203b-9e58-460a-a7e0-81c60d6dcad4
σ₂=svdvals(A₂)

# ╔═╡ 5f2c6284-21c4-4bdc-8545-50bf4944f1ef
begin
	# Some largest singular values
	k₂=6
	S₂,rest=svds(A₂,nsv=k₂)
	(σ₂[1:k₂]-S₂.S)./σ₂[1:k₂]
end

# ╔═╡ 559e2a09-70d4-4d1e-8941-45ed9ccbae6b
md"
### Large matrix
"

# ╔═╡ 91952aea-a239-4e2d-80f1-df6aefc58c4c
begin
	m₃=2000
	n₃=1500
	A₃=randn(m₃,n₃)
end

# ╔═╡ a7068313-39ae-4e83-a5cd-68aec37712c9
@time U₃,σ₃,V₃=svd(A₃);

# ╔═╡ 2e1d70eb-9751-4f4e-826f-e2aa05612cca
begin
	# This is rather slow
	k₃=10
	@time S₃,rest₃=svds(A₃,nsv=k₃);
end

# ╔═╡ 11a45bbe-a9f9-44e5-a7c2-559bddebefae
(σ₃[1:k₃]-S₃.S)./σ₃[1:k₃]

# ╔═╡ 5a75f57f-1293-4c76-9095-4d32e00cc093
md"""
### Very large sparse matrix
"""

# ╔═╡ 37f296ac-8853-480d-b424-b9037a7e50bc
#$ ?sprand

# ╔═╡ 151e13fb-3c48-40fb-8699-17ebb1569f2d
A₄=sprand(10000,3000,0.05)

# ╔═╡ 017a66c9-bbc2-464a-905b-b5316c49292a
begin
	# No vectors, this takes about 5 sec.
	k₄=10
	@time S₄,rest₄=svds(A₄,nsv=k₄,ritzvec=false)
end

# ╔═╡ 435d9693-996e-4f35-ac8d-e1f70077ce94
# Full matrix, no vectors, about 19 sec
@time σ₄=svdvals(Matrix(A₄))

# ╔═╡ 1b45c928-f1a4-437d-9980-a70a8e1896db
maximum(abs,(S₄.S-σ₄[1:k₄])./σ₄[1:k₄])

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Arpack = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
Arpack = "~0.5.3"
PlutoUI = "~0.7.38"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "91ca22c4b8437da89b030f08d71db55a379ce958"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.3"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

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

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "1285416549ccfcdf0c50d4997a94331e88d68413"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.3.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

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

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═fdbecdd6-3cdf-45ae-82c1-10e7016e7880
# ╠═5a428e03-2145-4767-ac0b-95f30a76af5c
# ╠═1ac3f71a-36bb-4d4a-90ed-de3e70c75cfe
# ╟─f04cb86f-94aa-418d-82f9-d9ef12ca90fd
# ╟─2eff5f1f-0989-4292-8f04-80983c1bd133
# ╟─21541d5a-d787-438f-ada3-56e07e62ae4e
# ╠═9fd7b6a0-ed2f-486b-99ec-806c1b19f033
# ╠═04b83647-fde5-40ad-a0e1-aebfd99c40b5
# ╠═751c0242-28cf-4e85-a28c-adc2b096511d
# ╠═3b813d5b-b27f-41bd-8c28-b7a4f3591ca8
# ╟─25b11022-ddd4-48c7-91a8-8306c8f04d08
# ╠═90059aec-9467-4548-9ef3-05a36d8e730e
# ╠═d71f46c3-0265-4e9b-9e11-bad11ae617ca
# ╠═9b015091-c8b6-416f-bfa0-cd8742feae53
# ╠═fb528048-4886-476d-acf6-bb0d1105f3c3
# ╠═2c628908-f9e2-4b04-a7b8-ebe6ff97d207
# ╠═f1734a00-d452-4adf-8125-dfbcee74d338
# ╠═caa171b1-e8cd-4fbe-a064-964c1a896f7a
# ╟─98fa4641-2d3f-48f5-ae1d-a9f3b96f798e
# ╠═19f3f447-ca93-4a6a-8b3f-403afd273fdd
# ╠═9dc37b20-3b94-4cbe-9c96-0bf12bfbe005
# ╠═38954e60-3f69-407e-8000-8a6e4477c245
# ╠═bda5ca94-e567-4615-a3f4-bce3c88cfb39
# ╠═aeb0105f-53e6-437b-a1f0-376ffcfdb5d7
# ╟─7c3ffb94-6403-43da-81fa-6fd8ef8a24b8
# ╠═11d8a586-361f-42f4-bd9a-77e483fc7a74
# ╟─e1dffb13-ed1b-44b4-9cb1-bd5227962cc9
# ╠═e6f4eb1b-2a3c-4785-8645-df5c72aa1ad7
# ╠═8c66ea99-3a59-4923-b73c-4ae3c9d7062e
# ╠═8af3203b-9e58-460a-a7e0-81c60d6dcad4
# ╠═5f2c6284-21c4-4bdc-8545-50bf4944f1ef
# ╟─559e2a09-70d4-4d1e-8941-45ed9ccbae6b
# ╠═91952aea-a239-4e2d-80f1-df6aefc58c4c
# ╠═a7068313-39ae-4e83-a5cd-68aec37712c9
# ╠═2e1d70eb-9751-4f4e-826f-e2aa05612cca
# ╠═11a45bbe-a9f9-44e5-a7c2-559bddebefae
# ╟─5a75f57f-1293-4c76-9095-4d32e00cc093
# ╠═8db1c1b2-8b5f-402c-8441-768c1a943c38
# ╠═37f296ac-8853-480d-b424-b9037a7e50bc
# ╠═151e13fb-3c48-40fb-8699-17ebb1569f2d
# ╠═017a66c9-bbc2-464a-905b-b5316c49292a
# ╠═435d9693-996e-4f35-ac8d-e1f70077ce94
# ╠═1b45c928-f1a4-437d-9980-a70a8e1896db
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002