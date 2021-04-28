### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ 8911d755-8ad6-4eb2-82f0-44a403f2ef60
begin
	using LinearAlgebra, PlutoUI
	PlutoUI.TableOfContents(aside=true)
end

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
                    τ=(A[i,i]-A[j,j])/(2*A[i,j])
                    t=sign(τ)/(abs(τ)+√(1+τ^2))
                    c=one(T)/√(one(T)+t^2)
                    s=c*t
                    G=LinearAlgebra.Givens(i,j,c,s)
                    A=G*A
                    A*=G'
                    A[i,j]=zero(T)
                    A[j,i]=zero(T)
                    U*=G'
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
 methodswith(LinearAlgebra.Givens);

# ╔═╡ d1d5fea3-f5e2-4a73-9db7-6dc6f550a88f
begin
	import Random
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
	λₛ,Uₛ=eigen(A₁)
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
                    τ=(A[i,i]-A[j,j])/(2*A[i,j])
                    t=sign(τ)/(abs(τ)+√(1+τ^2))
                    c=1/√(1+t^2)
                    s=c*t
                    G=LinearAlgebra.Givens(i,j,c,s)
                    # A=G*A
                    lmul!(G,A)
                    # A*=G'
                    rmul!(A,adjoint(G))
                    A[i,j]=zero(T)
                    A[j,i]=zero(T)
                    # U*=G'
                    rmul!(U,adjoint(G))
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

# ╔═╡ 5b8fd806-78e6-4aef-8800-df5c0feba3bb
cond(Aₛ), cond(A₂)

# ╔═╡ 9dba38b0-b08e-4928-9e6f-de9ce3c27106
# We add a strong scaling
D=exp.(50*(rand(n₂).-0.5))

# ╔═╡ e526191c-7091-4170-af9f-f73e8a6dc734
H=Diagonal(D)*Aₛ*Diagonal(D)

# ╔═╡ 795c80a3-3573-454e-a945-adfb03a258e2
# Now we scale again
Hₛ=[H[i,j]/ √(H[i,i]*H[j,j]) for i=1:n₂, j=1:n₂]

# ╔═╡ 3baf9fc0-b0a3-4bc5-9112-a9ccd982f998
cond(Hₛ),cond(H)

# ╔═╡ 9e2588e1-379f-482b-afe5-e57f9d0ae001
# Jacobi method
λ₂,U₂=Jacobi(H)

# ╔═╡ 5dd05357-7bd0-4857-b562-4c46f06a502d
# Orthogonality and residual 
norm(U₂'*U₂-I),norm(H*U₂-U₂*Diagonal(λ₂)) # /norm(H)

# ╔═╡ 2cf0b11f-b01e-4eb5-a617-be0eff673d89
# Standard QR method
λ₃,U₃=eigen(H)

# ╔═╡ 079159ab-f391-41df-a2e4-990f80243c6f
# Compare
[sort(λ₂) sort(λ₃)]

# ╔═╡ 011b7cd7-8692-4a29-9e97-ea223d98c761
# Check with BigFloat
Jacobi(map(BigFloat,H))

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

# ╔═╡ Cell order:
# ╟─8911d755-8ad6-4eb2-82f0-44a403f2ef60
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
