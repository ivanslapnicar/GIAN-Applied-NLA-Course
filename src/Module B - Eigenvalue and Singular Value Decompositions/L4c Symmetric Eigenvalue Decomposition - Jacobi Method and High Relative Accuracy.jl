### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 2c797491-f5aa-4fde-aa6e-8533a29ae008
using LinearAlgebra, PlutoUI, Random

# ╔═╡ 83e156d6-6745-433f-be54-f2b30b803394
import LinearAlgebra.Givens

# ╔═╡ 0c08032c-d90e-41df-a352-11f815dd7aac
PlutoUI.TableOfContents(aside=true)

# ╔═╡ 70c2fde9-110c-4f20-9324-0e0601934c2c
md"""
# Symmetric Eigenvalue Decomposition - Jacobi Method and High Relative Accuracy


The Jacobi method is the oldest method for EVD computations, dating back from 1864. The method does not require tridiagonalization. Instead, the method computes a sequence of orthogonally similar matrices which converge to a diagonal matrix of eigenvalues. In each step a simple plane rotation which sets one off-diagonal element to zero is performed. 

For positive definite matrices, the method computes eigenvalues with high relative accuracy.

For more details, see [I. Slapničar, Symmetric Matrix Eigenvalue Techniques, pp. 55.1-55.25](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and [Z. Drmač, Computing Eigenvalues and Singular Values to High Relative Accuracy, pp. 59.1-59.21](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.


__Prerequisites__

The reader should be familiar with concepts of eigenvalues and eigenvectors, related perturbation theory, and algorithms. 

 
__Competences__

The reader should be able to recognise matrices which warrant high relative accuracy and to apply  Jacobi method to them.

"""

# ╔═╡ abc0c46c-ec25-4fcb-8a8f-cefeda20418d
md"""
# Jacobi method

 $A$ is a real symmetric matrix of order $n$ and $A= U \Lambda  U^T$ is its EVD.

## Definitions

The __Jacobi method__ forms a sequence of matrices,

$$
A_0=A, \qquad A_{k+1}=G(c,s,i_k,j_k) \,A_k \,G(c,s,i_k,j_k)^T, \qquad
k=1,2,\ldots,$$

where $G(c,s,i_k,j_k)$ is the orthogonal __plane rotation matrix__.
The parameters $c$ and $s$ are chosen such that 

$$
[A_{k+1}]_{i_k j_k}=[A_{k+1}]_{j_k i_k}=0.$$

The plane rotation is also called __Jacobi rotation__. 

The __off-norm__ of $A$ is 

$$
\| A\|_{\mathrm{off}}=\big(\sum_{i}\sum_{j\neq i} a_{ij}^2\big)^{1/2},$$

that is, off-norm is the Frobenius norm of the
matrix consisting of all off-diagonal elements of $A$.

The choice of __pivot elements__ $[A_k]_{i_kj_k}$ is called the 
__pivoting strategy__.

The __optimal pivoting strategy__, originally used by Jacobi, chooses pivoting
elements such that 

$$
|[A_k]_{i_k j_k}|=\max_{i<j} |[A_k]_{ij}|.$$

The __row-cyclic__ pivoting strategy chooses pivot elements
  in the systematic row-wise order,

$$
(1,2), (1,3), \ldots,(1,n),(2,3),
(2,4),\ldots,(2,n),(3,4),\ldots,(n-1,n).$$

Similarly, the column-cyclic strategy chooses pivot elements column-wise.

One pass through all matrix elements is called __cycle__ or __sweep__.
"""

# ╔═╡ 1760ea70-45a2-426c-a9dd-23377711f3c9
md"""
## Facts

1. The Jacobi rotations parameters $c$ and $s$ are computed as follows: if $[A_k]_{i_kj_k}=0$, then $c=1$ and $s=0$, otherwise

$$
\begin{aligned}
& \tau=\frac{[A_k]_{i_ki_k}-[A_k]_{j_kj_k} }{2[A_k]_{i_kj_k} },\qquad
t=\frac{\mathop{\mathrm{sign}}(\tau)}{|\tau|+\sqrt{1+\tau^2}},\\
& c=\frac{1}{\sqrt{1+t^2}},\qquad s=c\cdot t.
\end{aligned}$$

2. After each rotation, the off-norm decreases,

$$
\|A_{k+1}\|_{\mathrm{off}}^2=\|A_{k}\|_{\mathrm{off}}^2-2[A_k]_{i_kj_k}^2.$$

With the appropriate pivoting strategy, the method converges in the sense that

$$
\|A_{k}\|_{\mathrm{off}}\to 0,\qquad A_k\to\Lambda, \qquad 
\prod_{k=1}^{\infty} G(i_k,j_k,c,s)^T \to U.$$

3. For the optimal pivoting strategy the square of the pivot element is greater than the average squared element,

$$
[A_k]_{i_kj_k}^2\geq \frac{1}{n(n-1)}\,
\|A_k\|_{\mathrm{off}}^2.$$

Thus,

$$
\|A_{k+1}\|_{\mathrm{off}}^2\leq\left(1-\frac{2}{n(n-1)}\right)\|A_{k}\|_{\mathrm{off}}^2$$

and the method converges.

4. For the row cyclic and the column cyclic pivoting strategies, the method converges. The convergence is ultimately __quadratic__ in the sense that

$$
\|A_{k+n(n-1)/2}\|_{\mathrm{off}} \leq\ const\cdot  \|A_{k}\|_{\mathrm{off}}^2,$$
provided $\|A_{k}\|_{\mathrm{off}}$ is sufficiently small.

5. The EVD computed by the Jacobi method satisfies the standard error bounds.

6. The Jacobi method is suitable for parallel computation. There exist convergent parallel strategies which enable simultaneous execution of several rotations.
  
7. The Jacobi method is simple, but it is slower than the methods based on tridiagonalization. It is conjectured that standard implementations require $O(n^3\log n)$ operations. More precisely, each cycle clearly requires $O(n^3)$ operations and it is conjectured that $\log n$ cycles are needed until convergence.
 
8. If $A$ is positive definite, the method can be modified such that it reaches the speed of the methods based on tridiagonalization and at the same time computes the EVD with high relative accuracy.
"""

# ╔═╡ 972f4040-d91c-4b13-a885-2ba40e505c7f
md"""
## Examples

$\begin{bmatrix} c & s\\-s&  c\end{bmatrix}^T \begin{bmatrix} a & b\\ b & d\end{bmatrix}
\begin{bmatrix} c & s\\-s&  c\end{bmatrix} = \begin{bmatrix} \tilde a & 0 \\ 0 &\tilde b\end{bmatrix}$

"""

# ╔═╡ 5cc14464-8fc4-495d-97d4-f106570ed942
function Jacobi(A::Array{T}) where T<:Real
    n=size(A,1)
    U=Matrix{T}(I,n,n)
    # Tolerance for rotation
    tol=√n*eps(T)
    # Counters
    p=n*(n-1)/2
    sweep=0
    pcurrent=0
    # First criterion is for standard accuracy, second one is for relative accuracy
    while sweep<10 && norm(A-Diagonal(A))>tol
    # while sweep<30 && pcurrent<p
        sweep+=1
        # Row-cyclic strategy
        for i = 1 : n-1 
            for j = i+1 : n
                # Check for the tolerance - the first criterion is standard,
                # the second one is for relative accuracy for PD matrices               
                if A[i,j]!=zero(T)
                # if abs(A[i,j])>tol*√(abs(A[i,i]*A[j,j]))
                    # Compute c and s
                    τ=(A[j,j]-A[i,i])/(2*A[i,j])
                    t=sign(τ)/(abs(τ)+√(1+τ^2))
                    c=one(T)/√(one(T)+t^2)
                    s=c*t
                    G=Givens(i,j,c,s)
                    A=G'*A
                    A*=G
                    A[i,j]=zero(T)
                    A[j,i]=zero(T)
                    U*=G
                    pcurrent=0
                    # To observe convergence
                    # display(A)
                else
                    pcurrent+=1
                end
            end
        end
        # display(A)
    end
    Eigen(diag(A),U)
end

# ╔═╡ cfd753ba-7102-4f84-8733-56b229d8a46d
 methodswith(Givens)

# ╔═╡ d1d5fea3-f5e2-4a73-9db7-6dc6f550a88f
begin
	Random.seed!(516)
	n=4
	A=Matrix(Symmetric(rand(n,n)))
end

# ╔═╡ b12ff4b3-6cb9-497f-9990-40aa3bcf6665
E=Jacobi(A)

# ╔═╡ a995f1e7-10cb-4291-826b-37431237bd8f
# Orthogonality and residual
norm(E.vectors'*E.vectors-I), norm(A*E.vectors-E.vectors*Diagonal(E.values))

# ╔═╡ 963e458d-4020-418b-bba5-0c9513c8e52d
begin
	# Positive definite matrix
	A₁=rand(100,100)
	A₁=Matrix(Symmetric(A₁'*A₁))
	@time E₁=Jacobi(A₁)
	norm(E₁.vectors'*E₁.vectors-I), norm(A₁*E₁.vectors-E₁.vectors*Diagonal(E₁.values))
end

# ╔═╡ 21ac1147-b940-4172-a57a-04e0fe74ccd7
begin
	# Now the standard QR method
	@time λₛ,Uₛ=eigen(A₁)
	norm(Uₛ'*Uₛ-I),norm(A₁*Uₛ-Uₛ*Diagonal(λₛ))
end

# ╔═╡ 3c7034f9-a477-470e-8d71-88e3bda9340b
md"""
`Jacobi()` is accurate but very slow. Notice the extremely high memory allocation.

The two key elements to reducing the allocations are: 
1. make sure variables don't change type within a function, and  
2. reuse arrays in hot loops.

Here we will simply use the in-place multiplication routines which are in Julia denoted by `!`.
"""

# ╔═╡ 46c32fa9-e5a8-4f94-a314-115f7b234e7d
function Jacobi₁(A₁::Array{T}) where T<:Real
    A=copy(A₁)
    n=size(A,1)
    U=Matrix{T}(I,n,n)
    # Tolerance for rotation
    tol=√n*eps(T)
    # Counters
    p=n*(n-1)/2
    sweep=0
    pcurrent=0
    # First criterion is for standard accuracy, second one is for relative accuracy
    # while sweep<30 && norm(A-Diagonal(diag(A)))>tol
    while sweep<30 && pcurrent<p
        sweep+=1
        # Row-cyclic strategy
        for i = 1 : n-1 
            for j = i+1 : n
                # Check for the tolerance - the first criterion is standard,
                # the second one is for relative accuracy for PD matrices               
                # if A[i,j]!=zero(T)
                if abs(A[i,j])>tol*√(abs(A[i,i]*A[j,j]))
                    # Compute c and s
                    τ=(A[j,j]-A[i,i])/(2*A[i,j])
                    t=sign(τ)/(abs(τ)+√(1+τ^2))
                    c=1/√(1+t^2)
                    s=c*t
                    G=Givens(i,j,c,s)
                    # A=G*A
                    lmul!(adjoint(G),A)
                    # A*=G
                    rmul!(A,G)
                    A[i,j]=zero(T)
                    A[j,i]=zero(T)
                    # U*=G
                    rmul!(U,G)
                    pcurrent=0
                else
                    pcurrent+=1
                end
            end
        end
    end
    Eigen(diag(A),U)
end

# ╔═╡ 182805c0-0ab1-40ef-a795-adee10713cca
@time Jacobi₁(A₁)

# ╔═╡ 5c5e6429-63e5-4844-8f21-07e1423ada47
md"""
# Relative perturbation theory

 $A$  is a real symmetric PD matrix of order $n$  and $A=U\Lambda U^T$ is its EVD.

## Definition

The __scaled matrix__ of the matrix $A$ is the matrix

$$
A_S=D^{-1} A D^{-1}, \quad D=\mathop{\mathrm{diag}}(\sqrt{A_{11}},\sqrt{A_{22}},\ldots,\sqrt{A_{nn}}).$$

## Facts

1. The above diagonal scaling is nearly optimal (van der Sluis):

$$
\kappa_2(A_S)\leq  n \min\limits_{D=\mathrm{diag}} \kappa(DAD) \leq n\kappa_2(A).$$

2. Let $A$ and $\tilde A=A+\Delta A$ both be positive definite, and let their eigenvalues have the same ordering. Then

$$
\frac{|\lambda_i-\tilde\lambda_i|}{\lambda_i}\leq 
\frac{\| D^{-1} (\Delta A) D^{-1}\|_2}{\lambda_{\min} (A_S)}\equiv
\|A_S^{-1}\|_2 \| \Delta A_S\|_2.$$

If $\lambda_i$ and $\tilde\lambda_i$ are simple, then

$$
\|U_{:,i}-\tilde U_{:,i}\|_2 \leq \frac{\| A_S^{-1}\|_2 \|\Delta A_S\|_2}
{\displaystyle\min_{j\neq i}\frac{|\lambda_i-\lambda_j|}{\sqrt{\lambda_i\lambda_j}}}.$$

These bounds are much sharper than the standard bounds for matrices for which $\kappa_2(A_S)\ll \kappa_2(A)$.

3. The Jacobi method with the relative stopping criterion 

$$
|A_{ij}|\leq tol \sqrt{A_{ii}A_{jj}}, \quad \forall i\neq j,$$

and some user defined tolerance $tol$ (usually $tol=n\varepsilon$), computes the EVD with small scaled  backward error

$$
\|\Delta A_S\|\leq \varepsilon\, O(\|A_S\|_2)\leq O(n)\varepsilon,$$

_provided_ that $\kappa_2([A_k]_S)$  does not grow much during the iterations. There is overwhelming numerical evidence that the scaled condition does not grow much, and the growth can be monitored, as well.

The proofs of the above facts are in [J. Demmel and K. Veselić, Jacobi's method is more accurate than QR](http://www.netlib.org/lapack/lawnspdf/lawn15.pdf).  
"""

# ╔═╡ 733d322f-8d91-4f70-a01b-0fa8f0edf3ab
md"
## Example of a scaled matrix
"

# ╔═╡ 2fb8d573-5c61-4406-bf67-042e6adb86b5
begin
	Random.seed!(431)
	n₂=6
	A₂=rand(n₂,n₂)
	A₂=Matrix(Symmetric(A₂'*A₂));
	Aₛ=[A₂[i,j]/√(A₂[i,i]*A₂[j,j]) for i=1:n₂, j=1:n₂]
end

# ╔═╡ 1f1b8df4-09a6-46c4-b08a-621461652aa7
A₂

# ╔═╡ 5b8fd806-78e6-4aef-8800-df5c0feba3bb
cond(Aₛ), cond(A₂)

# ╔═╡ 9dba38b0-b08e-4928-9e6f-de9ce3c27106
# We add a strong scaling
D=exp.(50*(rand(n₂).-0.5))

# ╔═╡ e526191c-7091-4170-af9f-f73e8a6dc734
H=Diagonal(D)*Aₛ*Diagonal(D)

# ╔═╡ 795c80a3-3573-454e-a945-adfb03a258e2
# Now we scale again
Hₛ=inv(√Diagonal(H))*H*inv(√Diagonal(H))

# ╔═╡ 3baf9fc0-b0a3-4bc5-9112-a9ccd982f998
cond(Hₛ),cond(H)

# ╔═╡ 9e2588e1-379f-482b-afe5-e57f9d0ae001
# Jacobi method
λ₂,U₂=Jacobi₁(H)

# ╔═╡ 5dd05357-7bd0-4857-b562-4c46f06a502d
# Orthogonality and relative residual 
norm(U₂'*U₂-I),norm(H*U₂-U₂*Diagonal(λ₂))/norm(H)

# ╔═╡ 2cf0b11f-b01e-4eb5-a617-be0eff673d89
# Standard QR method
λ₃,U₃=eigen(H)

# ╔═╡ 079159ab-f391-41df-a2e4-990f80243c6f
# Compare
[sort(λ₂) sort(λ₃)]

# ╔═╡ 011b7cd7-8692-4a29-9e97-ea223d98c761
# Check with BigFloat
Jacobi₁(map(BigFloat,H))

# ╔═╡ 1a57313f-d4c4-476c-92be-ad45be008edf
λ₂[1]

# ╔═╡ 0055c941-e98a-40dd-802a-98ceebedc374
md"""
# Indefinite matrices

## Definition

__Spectral absolute value__ of the matrix $A$ is the matrix 

$$
|A|_{\mathrm{spr}}=(A^2)^{1/2}.$$

This is positive definite part of the polar decomposition of $A$.

## Facts

1. The above perturbation bounds for positive definite matrices essentially hold with $A_S$ replaced by $[|A|_{\mathrm{spr}}]_S$.

2. Jacobi method can be modified to compute the EVD with small backward error $\| \Delta [|A|_{\mathrm{spr}}]_S\|_2$.

The details of the indefinite case are beyond the scope of this course, and the reader should consider references.
"""

# ╔═╡ fb368704-0f09-49aa-b870-b4378f1c5424
C=Symmetric(rand(5,5).-0.5)

# ╔═╡ 43fa4cac-6254-489e-b70c-9c8d7a45a7ad
G=eigen(C)

# ╔═╡ c86490af-ecf3-48e4-a4e3-99d5471081ac
Cₐ=√(C*C)

# ╔═╡ 210d21f3-077b-4fea-9d43-d12d33e4312f
eigvals(Cₐ)

# ╔═╡ 5f671402-af57-45d0-bc95-357d709d59ff
G.vectors*Diagonal(abs.(G.values))*G.vectors'

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
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
# ╠═2c797491-f5aa-4fde-aa6e-8533a29ae008
# ╠═83e156d6-6745-433f-be54-f2b30b803394
# ╠═0c08032c-d90e-41df-a352-11f815dd7aac
# ╟─70c2fde9-110c-4f20-9324-0e0601934c2c
# ╟─abc0c46c-ec25-4fcb-8a8f-cefeda20418d
# ╟─1760ea70-45a2-426c-a9dd-23377711f3c9
# ╟─972f4040-d91c-4b13-a885-2ba40e505c7f
# ╠═5cc14464-8fc4-495d-97d4-f106570ed942
# ╠═cfd753ba-7102-4f84-8733-56b229d8a46d
# ╠═d1d5fea3-f5e2-4a73-9db7-6dc6f550a88f
# ╠═b12ff4b3-6cb9-497f-9990-40aa3bcf6665
# ╠═a995f1e7-10cb-4291-826b-37431237bd8f
# ╠═963e458d-4020-418b-bba5-0c9513c8e52d
# ╠═21ac1147-b940-4172-a57a-04e0fe74ccd7
# ╟─3c7034f9-a477-470e-8d71-88e3bda9340b
# ╠═46c32fa9-e5a8-4f94-a314-115f7b234e7d
# ╠═182805c0-0ab1-40ef-a795-adee10713cca
# ╟─5c5e6429-63e5-4844-8f21-07e1423ada47
# ╟─733d322f-8d91-4f70-a01b-0fa8f0edf3ab
# ╠═1f1b8df4-09a6-46c4-b08a-621461652aa7
# ╠═2fb8d573-5c61-4406-bf67-042e6adb86b5
# ╠═5b8fd806-78e6-4aef-8800-df5c0feba3bb
# ╠═9dba38b0-b08e-4928-9e6f-de9ce3c27106
# ╠═e526191c-7091-4170-af9f-f73e8a6dc734
# ╠═795c80a3-3573-454e-a945-adfb03a258e2
# ╠═3baf9fc0-b0a3-4bc5-9112-a9ccd982f998
# ╠═9e2588e1-379f-482b-afe5-e57f9d0ae001
# ╠═5dd05357-7bd0-4857-b562-4c46f06a502d
# ╠═2cf0b11f-b01e-4eb5-a617-be0eff673d89
# ╠═079159ab-f391-41df-a2e4-990f80243c6f
# ╠═011b7cd7-8692-4a29-9e97-ea223d98c761
# ╠═1a57313f-d4c4-476c-92be-ad45be008edf
# ╟─0055c941-e98a-40dd-802a-98ceebedc374
# ╠═fb368704-0f09-49aa-b870-b4378f1c5424
# ╠═43fa4cac-6254-489e-b70c-9c8d7a45a7ad
# ╠═c86490af-ecf3-48e4-a4e3-99d5471081ac
# ╠═210d21f3-077b-4fea-9d43-d12d33e4312f
# ╠═5f671402-af57-45d0-bc95-357d709d59ff
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
