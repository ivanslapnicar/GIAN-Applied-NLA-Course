### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ 2cd8aee2-d752-47fd-8203-55148bd75002
begin
	using PlutoUI, LinearAlgebra
	PlutoUI.TableOfContents(aside=true)
end

# ╔═╡ 69545c6e-1af7-49fc-b3ae-a8fb0cdc600e
md"""
# Symmetric Eigenvalue Decomposition - Jacobi Method and High Relative Accuracy


The Jacobi method is the oldest method for EVD computations, dating back from 1864.  The method does not require tridiagonalization. Instead, the method computes a sequence of orthogonally similar  matrices which converge to a diagonal matrix of eigenvalues. In each step a simple plane rotation which sets one off-diagonal element to zero is performed. 

For positive definite matrices, the method computes eigenvalues with high relative accuracy.

For more details, see [I. Slapničar, Symmetric Matrix Eigenvalue Techniques, pp. 55.1-55.25](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and [Z. Drmač, Computing Eigenvalues and Singular Values to High Relative Accuracy, pp. 59.1-59.21](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.



__Prerequisites__

The reader should be familiar with concepts of eigenvalues and eigenvectors, related perturbation theory, and algorithms. 

 
__Competences__

The reader should be able to recognise matrices which warrant high relative accuracy and to apply Jacobi method to them.

"""

# ╔═╡ 21e8ee7c-3394-44a6-ae15-3d2a5548cbb5
md"""
# Jacobi method

 $A$ is a real symmetric matrix of order $n$ and $A= U \Lambda  U^T$ is its EVD.

## Definitions

The __Jacobi method__ forms a sequence of matrices,

$$
A_0=A, \qquad A_{k+1}=G^T(c,s,i_k,j_k)\ A_k\ G(c,s,i_k,j_k), \qquad
k=1,2,\ldots,$$

where $G(c,s,i_k,j_k)$ is the orthogonal __plane rotation matrix__. The parameters $c$ and $s$ are chosen such that 

$$
[A_{k+1}]_{i_k j_k}=[A_{k+1}]_{j_k i_k}=0.$$

The plane rotation is also called __Jacobi rotation__. 

The __off-norm__ of $A$ is 

$$
\| A\|_{\mathrm{off}}=\big(\sum_{i}\sum_{j\neq i} a_{ij}^2\big)^{1/2},$$

that is, off-norm is the Frobenius norm of the matrix consisting of all off-diagonal elements of $A$.

The choice of __pivot elements__ $[A_k]_{i_kj_k}$ is called the __pivoting strategy__.

The __optimal pivoting strategy__, originally used by Jacobi, chooses pivoting elements such that 

$$
|[A_k]_{i_k j_k}|=\max_{i<j} |[A_k]_{ij}|.$$

The __row-cyclic__ pivoting strategy chooses pivot elements in the systematic row-wise order,

$$
(1,2), (1,3), \ldots,(1,n),(2,3),
(2,4),\ldots,(2,n),(3,4),\ldots,(n-1,n).$$

Similarly, the column-cyclic strategy chooses pivot elements column-wise.

One pass through all matrix elements is called __cycle__ or __sweep__.
"""

# ╔═╡ c7642876-0a2b-40af-b7da-a0ffcde1a304
md"""
## Facts

1. The Jacobi rotations parameters $c$ and $s$ are computed as follows: If $[A_k]_{i_kj_k}=0$, then $c=1$ and $s=0$, otherwise

$$\begin{aligned}
& \tau=\frac{[A_k]_{j_kj_k}-[A_k]_{i_ki_k} }{2[A_k]_{i_kj_k} },\qquad
t=\frac{\mathop{\mathrm{sign}}(\tau)}{|\tau|+\sqrt{1+\tau^2}},\\
& c=\frac{1}{\sqrt{1+t^2}},\qquad s=c\cdot t.
\end{aligned}$$

2. After each rotation, the off-norm decreases,

$$
\|A_{k+1}\|_{\mathrm{off}}^2=\|A_{k}\|_{\mathrm{off}}^2-2[A_k]_{i_kj_k}^2.$$

With the appropriate pivoting strategy, the method converges in the sense that

$$\|A_{k}\|_{\mathrm{off}}\to 0,\qquad A_k\to\Lambda, \qquad 
\prod_{k=1}^{\infty} G(i_k,j_k,c,s) \to U.$$

3. For the optimal pivoting strategy the square of the pivot element is greater than the average squared element,

$$[A_k]_{i_kj_k}^2\geq \frac{1}{n(n-1)}\,
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

# ╔═╡ 8dd8a3e6-9488-4cbc-9a4e-f6819dba7112
md"
## Examples
"

# ╔═╡ 0be1d6d8-dffc-403b-b3b3-1b209cacb09c
md"""

$\begin{bmatrix} c & s\\ -s&  c\end{bmatrix}^T \begin{bmatrix} a & b\\ b & d\end{bmatrix}
\begin{bmatrix} c & s\\-s&  c\end{bmatrix} = \begin{bmatrix} \tilde a & 0 \\ 0 &\tilde b\end{bmatrix}$
"""

# ╔═╡ ff4f7ae3-7044-4c13-b079-c4cd4e8d39e6
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
	while sweep<10 && norm(A-Diagonal(diag(A)))>tol
	    # while sweep<30 && pcurrent<p
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
	                c=one(T)/√(one(T)+t^2)
	                s=c*t
	                G=LinearAlgebra.Givens(i,j,c,s)
	                A=G'*A*G
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
	Eigen(diag(A), U)
end

# ╔═╡ 6aae1498-f7b6-4a44-b073-85af98370116
 methodswith(LinearAlgebra.Givens);

# ╔═╡ d33323b9-eddc-4b46-bb15-d7437511ae42
begin
	import Random
	Random.seed!(516)
	n=4
	A=Matrix(Symmetric(rand(n,n)))
end

# ╔═╡ cdc58669-88f1-4b68-bac6-b3d809926cee
E=Jacobi(A)

# ╔═╡ 902ab03a-576b-4222-8eb1-d21615885d32
# Orthogonality and residual
norm(E.vectors'*E.vectors-I),
norm(A*E.vectors-E.vectors*Diagonal(E.values))

# ╔═╡ 8def3e64-cd9e-4195-82ce-4be64b6c86d2
begin
	# Positive definite matrix
	n₁=100
	A₁=rand(n₁,n₁)
	A₁=Matrix(Symmetric(A₁'*A₁))
end

# ╔═╡ 4cc668c2-a428-45f8-b3cd-279e27c91731
@time E₁=Jacobi(A₁)

# ╔═╡ a20d11c7-ed77-4328-9148-8df93cfc6d0a
# Orthogonality and residual
norm(E₁.vectors'*E₁.vectors-I),
norm(A₁*E₁.vectors-E₁.vectors*Diagonal(E₁.values))

# ╔═╡ 2777a09f-04c3-431d-adae-65f7227705a6
cond(A₁)

# ╔═╡ 40e61ea2-b363-42d5-b9e5-cdc0fda2c1d4
# Now the standard QR method
Eₛ=eigen(A₁);

# ╔═╡ a934f125-cb2b-4b62-8b0a-b2f199efcd53
norm(Eₛ.vectors'*Eₛ.vectors-I),
norm(A₁*Eₛ.vectors-Eₛ.vectors*Diagonal(Eₛ.values))

# ╔═╡ 27fe06a6-19b2-46a7-8a3b-dba02d9fc08a
md"""
`Jacobi()` is accurate, but slow. Notice the extremely high memory allocation.

The two key elements to reducing the allocations are: 
1. make sure variables don't change type within a function, and  
2. reuse arrays in hot loops.

Here we will simply use the in-place multiplication routines which are in Julia denoted by `!`.
"""

# ╔═╡ 68423c13-da3f-427a-a4b3-3c3906622979
@time eigen(A₁);

# ╔═╡ 2d0ba3db-a62e-4220-b56d-0524b634f40b
@time Jacobi(A₁);

# ╔═╡ c3f15544-4919-49a7-8735-a7fdb42968fa
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
                    c=1.0/√(1+t^2)
                    s=c*t
                    G=LinearAlgebra.Givens(i,j,c,s)
                    # A=G*A
                    lmul!(G',A)
                    # A*=G'
                    rmul!(A,G)
                    A[i,j]=zero(T)
                    A[j,i]=zero(T)
                    # U*=G'
                    rmul!(U,G)
                    pcurrent=0
                else
                    pcurrent+=1
                end
            end
        end
    end
    Eigen(diag(A), U)
end

# ╔═╡ 09e641cf-8541-4322-9a9d-76912fc5ed8b
@time E₂=Jacobi₁(A₁);

# ╔═╡ 53608bf9-ba8b-4dcb-b04b-5fcc8c281334
# Orthogonality and residual
norm(E₂.vectors'*E₂.vectors-I),norm(A₁*E₂.vectors-E₂.vectors*Diagonal(E₂.values))

# ╔═╡ 384ec404-05f8-487f-86fb-1c9affda3d7e
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

## Scaled matrix

"""

# ╔═╡ e9d96500-0f41-4138-868b-eae1a4e9d89f
# D=Diagonal([1,2,3,4,1000])

# ╔═╡ 217eefa9-fe09-4336-909b-583c056a9a43
begin
	Random.seed!(431)
	n₂=6
	A₂=rand(n,n)
	A₂=Matrix(Symmetric(A₂'*A₂));
end

# ╔═╡ 455e70f0-10e9-43a0-94f9-1704a3838f1d
Aₛ=[A[i,j]/sqrt(A[i,i]*A[j,j]) for i=1:n, j=1:n]

# ╔═╡ 7a7fd27e-bb03-40e9-a40c-1fc659ef0b8c
cond(Aₛ), cond(A)

# ╔═╡ fa29c15f-d09c-45e7-861b-b84e4981b9bf
begin
	# We add a strong scaling
	Random.seed!(5621)
	D=exp.(50*(rand(n).-0.5))
end

# ╔═╡ 96e7f4d8-bfb1-4be3-8061-375e50ecc9fa
H=Diagonal(D)*Aₛ*Diagonal(D)

# ╔═╡ 1186661c-bbc1-4a16-8bbc-60fcd62f5e42
# Now we scale again
Hₛ=[H[i,j]/sqrt(H[i,i]*H[j,j]) for i=1:n, j=1:n]

# ╔═╡ a4014941-8b37-4d15-bb7d-99ab1ed29495
cond(Hₛ),cond(H)

# ╔═╡ 2cefe20f-3c9b-438c-9ab3-9ee65b60c1a7
# Jacobi method
λ,U=Jacobi(H)

# ╔═╡ c73763d3-b310-4b20-93bc-800e5f8c9d5d
# Standard QR method
λ₁,U₁=eigen(H)

# ╔═╡ 2ad7a58c-184a-402d-94c9-b952623ebec6
# Compare
[sort(λ) sort(λ₁)]

# ╔═╡ 34bcad49-bd18-42d4-ae59-0d53021a8e3e
λ[1]

# ╔═╡ f1e072d7-45cc-4aac-bc0b-b1b2a5fb1b67
sort(λ)-sort(λ₁)

# ╔═╡ c297e58d-d572-4b41-9840-49d820c428ca
(sort(λ)-sort(λ₁))./sort(λ)

# ╔═╡ 10d26475-2ca4-4f2a-9efd-61b27087be39
# Check with BigFloat
λ₂,U₂=Jacobi(map(BigFloat,H))

# ╔═╡ 1ff0ddf9-b340-4de5-a3d9-e9a3cd81bff7
# Relative error is eps()*cond(AS)
map(Float64,(sort(λ₂)-sort(λ))./sort(λ₂))

# ╔═╡ 770c0f98-6fdf-47bc-b474-fe808c56aa19
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

# ╔═╡ eeab54c2-6cdc-4071-80a7-c1734f61057d


# ╔═╡ Cell order:
# ╠═2cd8aee2-d752-47fd-8203-55148bd75002
# ╟─69545c6e-1af7-49fc-b3ae-a8fb0cdc600e
# ╟─21e8ee7c-3394-44a6-ae15-3d2a5548cbb5
# ╟─c7642876-0a2b-40af-b7da-a0ffcde1a304
# ╟─8dd8a3e6-9488-4cbc-9a4e-f6819dba7112
# ╟─0be1d6d8-dffc-403b-b3b3-1b209cacb09c
# ╠═ff4f7ae3-7044-4c13-b079-c4cd4e8d39e6
# ╠═6aae1498-f7b6-4a44-b073-85af98370116
# ╠═d33323b9-eddc-4b46-bb15-d7437511ae42
# ╠═cdc58669-88f1-4b68-bac6-b3d809926cee
# ╠═902ab03a-576b-4222-8eb1-d21615885d32
# ╠═8def3e64-cd9e-4195-82ce-4be64b6c86d2
# ╠═4cc668c2-a428-45f8-b3cd-279e27c91731
# ╠═a20d11c7-ed77-4328-9148-8df93cfc6d0a
# ╠═2777a09f-04c3-431d-adae-65f7227705a6
# ╠═40e61ea2-b363-42d5-b9e5-cdc0fda2c1d4
# ╠═a934f125-cb2b-4b62-8b0a-b2f199efcd53
# ╟─27fe06a6-19b2-46a7-8a3b-dba02d9fc08a
# ╠═68423c13-da3f-427a-a4b3-3c3906622979
# ╠═2d0ba3db-a62e-4220-b56d-0524b634f40b
# ╠═c3f15544-4919-49a7-8735-a7fdb42968fa
# ╠═09e641cf-8541-4322-9a9d-76912fc5ed8b
# ╠═53608bf9-ba8b-4dcb-b04b-5fcc8c281334
# ╟─384ec404-05f8-487f-86fb-1c9affda3d7e
# ╠═e9d96500-0f41-4138-868b-eae1a4e9d89f
# ╠═217eefa9-fe09-4336-909b-583c056a9a43
# ╠═455e70f0-10e9-43a0-94f9-1704a3838f1d
# ╠═7a7fd27e-bb03-40e9-a40c-1fc659ef0b8c
# ╠═fa29c15f-d09c-45e7-861b-b84e4981b9bf
# ╠═96e7f4d8-bfb1-4be3-8061-375e50ecc9fa
# ╠═1186661c-bbc1-4a16-8bbc-60fcd62f5e42
# ╠═a4014941-8b37-4d15-bb7d-99ab1ed29495
# ╠═2cefe20f-3c9b-438c-9ab3-9ee65b60c1a7
# ╠═c73763d3-b310-4b20-93bc-800e5f8c9d5d
# ╠═2ad7a58c-184a-402d-94c9-b952623ebec6
# ╠═34bcad49-bd18-42d4-ae59-0d53021a8e3e
# ╠═f1e072d7-45cc-4aac-bc0b-b1b2a5fb1b67
# ╠═c297e58d-d572-4b41-9840-49d820c428ca
# ╠═10d26475-2ca4-4f2a-9efd-61b27087be39
# ╠═1ff0ddf9-b340-4de5-a3d9-e9a3cd81bff7
# ╟─770c0f98-6fdf-47bc-b474-fe808c56aa19
# ╠═eeab54c2-6cdc-4071-80a7-c1734f61057d
