### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 7be5652e-3e5a-4199-a099-40f2da28053c
using LinearAlgebra, Arpack, LinearMaps, SparseArrays, PlutoUI, Random

# ╔═╡ 12bae146-12e7-4a78-afac-f5c2d6b86b66
PlutoUI.TableOfContents(aside=true)

# ╔═╡ f647d0dd-1fe4-42cf-b55c-38baa12f2db8
md"""
# Symmetric Eigenvalue Decomposition - Lanczos Method


If the matrix $A$ is large and sparse and/or if only some eigenvalues and their eigenvectors are desired, iterative methods are the methods of choice. For example, the power method can be useful to compute the eigenvalue with the largest modulus. The basic operation in the power method is matrix-vector multiplication, and this can be performed very fast if $A$ is sparse. Moreover, $A$ need not be stored in the computer -- the input for the algorithm can be just a function which, given some vector $x$, returns the product $Ax$.

An _improved_ version of the power method, which efficiently computes some eigenvalues (either largest in modulus or near some target value $\mu$) and the corresponding eigenvectors, is the Lanczos method.

For more details, see [I. Slapničar, Symmetric Matrix Eigenvalue Techniques, pp. 55.1-55.25](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.

__Prerequisites__

The reader should be familiar with concepts of eigenvalues and eigenvectors, related perturbation theory, and algorithms. 

 
__Competences__

The reader should be able to recognize matrices that warrant the use of the Lanczos method, to apply the method, and to assess the accuracy of the solution.
"""

# ╔═╡ d5f270bd-94c1-4da6-a8e6-a53337488020
md"""
# Lanczos method

 $A$ is a real symmetric matrix of order $n$.

## Definitions

Given a nonzero vector $x$ and an index $k<n$, the __Krylov matrix__ is defined as

$$
K_k=\begin{bmatrix} x & Ax & A^2 x &\cdots & A^{k-1}x \end{bmatrix}.$$

__Krilov subspace__ is the subspace spanned by the columns of $K_k$.

## Facts

1. The Lanczos method is based on the following observation. If $K_k=XR$ is the $QR$ factorization of the matrix $K_k$, then the $k\times k$ matrix $T=X^T A X$ is tridiagonal. The matrices $X$ and $T$ can be computed by using only matrix-vector products in $O(kn)$ operations.

2. Let $T=Q\Lambda Q^T$ be the EVD of $T$. Then $\lambda_i$ approximates well some of the largest and smallest eigenvalues of $A$, and the columns of the matrix $U=XQ$ approximate the corresponding eigenvectors.

3. As $k$ increases, the largest (smallest) eigenvalues of the matrix $T_{1:k,1:k}$ converge towards some of the largest (smallest) eigenvalues of $A$ (due to the Cauchy interlace property). The algorithm can be redesigned to compute only the largest or smallest eigenvalues. Also, by using the shift and invert strategy, the method can be used to compute eigenvalues near some specified value. To obtain better approximations, $k$ should be greater than the number of required eigenvalues. On the other side, to obtain better accuracy and efficacy, $k$ should be as small as possible.

4. The last computed element, $\mu=T_{k+1,k}$, provides information about accuracy:

$$
\begin{aligned}
\|AU-U\Lambda\|_2&=\mu, \\
\|AU_{:,i}-\lambda_i U_{:,i}\|_2&=\mu |Q_{ki}|, \quad  i=1,\ldots,k.
\end{aligned}$$

Further, there are $k$ eigenvalues $\tilde\lambda_1,\ldots,\tilde\lambda_k$ of $A$ such that $|\lambda_i-\tilde\lambda_i|\leq \mu$, and for the corresponding eigenvectors, we have 

$$\sin2\Theta(U_{:,i},\tilde U_{:,i}) \leq \frac{2\mu}{\min_{j\neq i} |\lambda_i-\tilde \lambda_j|}.$$ 

5. In practical implementations, $\mu$ is usually used to determine the index $k$. 

6. The Lanczos method has inherent numerical instability in the floating-point arithmetic: since the Krylov vectors are generated by the power method, they converge towards an eigenvector of $A$. Thus, as $k$ increases, the Krylov vectors become more and more parallel, the recursion in the function `Lanczos()` becomes numerically unstable and the computed columns of $X$ cease to be sufficiently orthogonal. This affects both the convergence and the accuracy of the algorithm. For example, several eigenvalues of $T$ may converge towards a simple eigenvalue of $A$ (the so-called, __ghost eigenvalues__).

7. The loss of orthogonality is dealt with by using the __full reorthogonalization__ procedure: in each step, the new ${\bf z}$ is orthogonalized against all previous columns of $X$, that is, in function `Lanczos()`, the formula 
```
z=z-Tr.dv[i]*X[:,i]-Tr.ev[i-1]*X[:,i-1]
```
is replaced by
```
z=z-sum(dot(z,Tr.dv[i])*X[:,i]-Tr.ev[i-1]*X[:,i-1]
```
  
To obtain better orthogonality, the latter formula is usually executed twice. The full reorthogonalization raises the operation count to $O(k^2n)$.

8. The __selective reorthogonalization__ is the procedure in which the current $z$ is orthogonalized against some selected columns of $X$, to attain sufficient numerical stability and not increase the operation count too much. The details are very subtle and can be found in the references.

9. > Efficient implementation of the method - speeding the convergence and selections of desired eigenvalues (smallest in modulus, largest in modulus, or contained in an interval) - requires the usage of shifts and implicit restart of the method. These are fast but elaborate procedures. Please consider many available references for details.
  
10. The Lanczos method is usually used for sparse matrices. Sparse matrix $A$ is stored in the sparse format in which only values and indices of nonzero elements are stored. The number of operations required to multiply some vector by $A$ is also proportional to the number of nonzero elements.
  
11. The function `eigs()` implements Lanczos method real for symmetric matrices and more general Arnoldi method for general matrices.
"""

# ╔═╡ 97e0dbf8-b9be-4503-af5d-4cf6e86311eb
function Lanczos(A::Array{T}, x::Vector{T}, k::Int) where T
    n=size(A,1)
    X=Array{T}(undef,n,k)
    dv=Array{T}(undef,k)
    ev=Array{T}(undef,k-1)
    X[:,1]=x/norm(x)
    for i=1:k-1
        z=A*X[:,i]
        dv[i]=X[:,i]⋅z
        # Three-term recursion
        if i==1
            z=z-dv[i]*X[:,i]
        else
            # z=z-dv[i]*X[:,i]-ev[i-1]*X[:,i-1]
            # Full reorthogonalization - once or even twice
            z=z-sum([(z⋅X[:,j])*X[:,j] for j=1:i])
            z=z-sum([(z⋅X[:,j])*X[:,j] for j=1:i])
        end
        μ=norm(z)
        if μ==0
            Tr=SymTridiagonal(dv[1:i-1],ev[1:i-2])
            return eigvals(Tr), X[:,1:i-1]*eigvecs(Tr), X[:,1:i-1], μ
        else
            ev[i]=μ
            X[:,i+1]=z/μ
        end
    end
    # Last step
    z=A*X[:,end]
    dv[end]=X[:,end]⋅z
    z=z-dv[end]*X[:,end]-ev[end]*X[:,end-1]
    μ=norm(z)
    Tr=SymTridiagonal(dv,ev)
    eigvals(Tr), X*eigvecs(Tr), X, μ
end

# ╔═╡ 4b63c1d9-f043-448f-9e05-911f52d4227d
begin
	Random.seed!(421)
	n=100
	A=Matrix(Symmetric(rand(n,n)))
	# Or: A = rand(5,5) |> t -> t + t'
	x=rand(n)
	k=10
end

# ╔═╡ 219ce78b-8bd3-4df3-93df-c08fad30e33f
λ,U,X,μ=Lanczos(A,x,10)

# ╔═╡ 1ef82905-e422-4644-ad00-26c448cb0e3a
# Orthogonality of X
norm(X'*X-I)

# ╔═╡ 595b22a5-4456-41fb-b4d6-861833bc6d47
# Tridiagonalization
X'*A*X

# ╔═╡ b2e762d4-650c-4846-ae30-32614b516fe3
# Residual
norm(A*U-U*Diagonal(λ)), μ

# ╔═╡ bd2b8d59-0525-4db5-99cd-e2f98f735eda
U'*A*U

# ╔═╡ 6791903f-0cf7-4bbc-adfb-ff9e80b373dc
# Orthogonality of U
norm(U'*U-I)

# ╔═╡ 218a2a21-e391-4a90-a96a-71bbb0d2f895
# Full eigenvalue decomposition
λeigen,Ueigen=eigen(A)

# ╔═╡ e23c1bc6-c4d8-4c5d-9a83-2b00c5d93b25
# ?eigs

# ╔═╡ fc00c272-d66f-42be-9964-7a934b32015c
# Lanczos method from Arpack.jl
λeigs,Ueigs=eigs(A; nev=k, which=:LM, ritzvec=true, v0=x)

# ╔═╡ 91eec1a8-413c-4960-bb52-8da2435e9e4a
[λ λeigs λeigen[sortperm(abs.(λeigen),rev=true)[1:k]] ]

# ╔═╡ aa20cf19-2bc8-4831-ae26-fb883614c63f
md"""
We see that `eigs()` computes `k` eigenvalues with largest modulus. What eigenvalues did `Lanczos()` compute?
"""

# ╔═╡ 9ce7a012-723d-4184-8ef3-00f468e61281
begin
	println("Evals")
	for i=1:k
	    println(minimum(abs,λeigen.-λ[i]))
	end
end

# ╔═╡ ed7d1400-e6d9-44e2-a633-7f6b3df74272
md"""
Conslusion is that the naive implementation of Lanczos is not enough. However, it is fine, when all eigenvalues are computed. Why?
"""

# ╔═╡ d2f31110-5587-4dd3-a5cf-cc3e046e1ee0
λall,Uall,Xall,μall=Lanczos(A,x,100)

# ╔═╡ 04763c96-3a2b-4092-8969-99c18cc8fd54
# Residual and relative errors 
norm(A*Uall-Uall*Diagonal(λall)), norm((λeigen-λall)./λeigen)

# ╔═╡ 95ef94d2-129f-4dbb-b705-9a2c8660e22e
md"""
# Operator version

We can use Lanczos method with operator which, given vector `x`, returns the product `A*x`. We use the function `LinearMap()` from the package [LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl)
"""

# ╔═╡ b8716ffe-8405-4018-bdb9-29c8cd2243dc
# ?LinearMap

# ╔═╡ 3bd1283d-91fa-4fdd-bcf8-946ac83b848c
# Operator from the matrix
C=LinearMap(A, issymmetric=true)

# ╔═╡ 0f8596c1-ee32-4b0d-b13b-d36fb611c99e
begin
	λC,UC=eigs(C; nev=k, which=:LM, ritzvec=true, v0=x)
	λC
end

# ╔═╡ 05cd7b4e-18cf-485c-8113-4855f22ac8a0
md"""
Here is an example of `LinearMap()` with the function. 
"""

# ╔═╡ a8aeb3ce-5a67-48c7-9c15-8cc11e7ec39b
f(x)=A*x

# ╔═╡ 86bcfb51-6508-4644-ab3d-ee5d9675e1d6
D=LinearMap(f,n,issymmetric=true)

# ╔═╡ 5c08501f-010a-4d17-96b7-e339b0299262
begin
	λD,UD=eigs(D, nev=k, which=:LM, ritzvec=true, v0=x)
	λD
end

# ╔═╡ 778a4da1-af10-4a7a-8a04-00890d61f101
norm(λeigs-λC), norm(λeigs-λD)

# ╔═╡ 8a8d59cb-0a69-412e-948e-8110f04419fe
md"""
# Sparse matrices
"""

# ╔═╡ f63b3be6-f5d4-49c7-b8c7-a03826f21ad4
# ?sprand

# ╔═╡ 7beaf848-ad66-47ff-9411-b4ea94476f38
# Generate a sparse symmetric matrix
C₁=sprand(n,n,0.05) |> t -> t+t'

# ╔═╡ d468bf54-54ad-4f07-87d7-ce3d666471aa
issymmetric(C₁)

# ╔═╡ 0f9a4d39-c00c-44b2-8831-eeca1b34e79c
eigs(C₁; nev=k, which=:LM, ritzvec=true, v0=x)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Arpack = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LinearMaps = "7a12625a-238d-50fd-b39a-03d52299707e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[compat]
Arpack = "~0.5.4"
LinearMaps = "~3.11.2"
PlutoUI = "~0.7.58"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "505084f379dc11bde924788d1db56f6dedc6d31f"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

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
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

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
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ee79c3208e55786de58f8dcccca098ced79f743f"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.3"

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

    [deps.LinearMaps.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

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
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

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
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═7be5652e-3e5a-4199-a099-40f2da28053c
# ╠═12bae146-12e7-4a78-afac-f5c2d6b86b66
# ╟─f647d0dd-1fe4-42cf-b55c-38baa12f2db8
# ╟─d5f270bd-94c1-4da6-a8e6-a53337488020
# ╠═97e0dbf8-b9be-4503-af5d-4cf6e86311eb
# ╠═4b63c1d9-f043-448f-9e05-911f52d4227d
# ╠═219ce78b-8bd3-4df3-93df-c08fad30e33f
# ╠═1ef82905-e422-4644-ad00-26c448cb0e3a
# ╠═595b22a5-4456-41fb-b4d6-861833bc6d47
# ╠═b2e762d4-650c-4846-ae30-32614b516fe3
# ╠═bd2b8d59-0525-4db5-99cd-e2f98f735eda
# ╠═6791903f-0cf7-4bbc-adfb-ff9e80b373dc
# ╠═218a2a21-e391-4a90-a96a-71bbb0d2f895
# ╠═e23c1bc6-c4d8-4c5d-9a83-2b00c5d93b25
# ╠═fc00c272-d66f-42be-9964-7a934b32015c
# ╠═91eec1a8-413c-4960-bb52-8da2435e9e4a
# ╟─aa20cf19-2bc8-4831-ae26-fb883614c63f
# ╠═9ce7a012-723d-4184-8ef3-00f468e61281
# ╟─ed7d1400-e6d9-44e2-a633-7f6b3df74272
# ╠═d2f31110-5587-4dd3-a5cf-cc3e046e1ee0
# ╠═04763c96-3a2b-4092-8969-99c18cc8fd54
# ╟─95ef94d2-129f-4dbb-b705-9a2c8660e22e
# ╠═b8716ffe-8405-4018-bdb9-29c8cd2243dc
# ╠═3bd1283d-91fa-4fdd-bcf8-946ac83b848c
# ╠═0f8596c1-ee32-4b0d-b13b-d36fb611c99e
# ╟─05cd7b4e-18cf-485c-8113-4855f22ac8a0
# ╠═a8aeb3ce-5a67-48c7-9c15-8cc11e7ec39b
# ╠═86bcfb51-6508-4644-ab3d-ee5d9675e1d6
# ╠═5c08501f-010a-4d17-96b7-e339b0299262
# ╠═778a4da1-af10-4a7a-8a04-00890d61f101
# ╟─8a8d59cb-0a69-412e-948e-8110f04419fe
# ╠═f63b3be6-f5d4-49c7-b8c7-a03826f21ad4
# ╠═7beaf848-ad66-47ff-9411-b4ea94476f38
# ╠═d468bf54-54ad-4f07-87d7-ce3d666471aa
# ╠═0f9a4d39-c00c-44b2-8831-eeca1b34e79c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
