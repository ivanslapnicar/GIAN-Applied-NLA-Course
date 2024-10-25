### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 893a2edb-d744-4d7f-b81d-fcf916922b81
using PlutoUI,Random, LinearAlgebra

# ╔═╡ 44758c8b-25bd-47d4-b728-90b4e98b6baf
TableOfContents(aside=true)

# ╔═╡ e355fccd-fb92-49d4-b978-c4f1680fbb8d
md"""
# Symmetric Eigenvalue Decomposition - Algorithms and Error Analysis

We study mainly algorithms for real symmetric matrices, which are most commonly used in the applications described in this course. 

For more details, see 
[I. Slapničar, Symmetric Matrix Eigenvalue Techniques, pp. 55.1-55.25](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.


__Prerequisites__

The reader should be familiar with basic linear algebra concepts and facts on eigenvalue decomposition and perturbation theory

 
__Competences__

The reader should be able to apply adequate algorithm to a given problem, and to assess accuracy of the solution.

"""

# ╔═╡ 4fe7e0b0-bbd5-4746-9f78-ac4dd1abf4b6
md"""
# Backward error and stability

## Definitions

If the value of a function $f(x)$ is computed with an algorithm  $\mathrm{alg(x)}$, the 
__algorithm error__ is 

$$
\|\mathrm{alg(x)}-f(x)\|,$$

and the __relative algorithm error__ is

$$
\frac{\| \mathrm{alg}(x)-f(x)\|}{\| f(x) \|},$$

in respective norms. Therse errors can be hard or even impossible to estimate directly. 

In this case, assume that $f(x)$ computed by $\mathrm{alg}(x)$ is equal to exact value of the function for a perturbed argument,

$$
\mathrm{alg}(x)=f(x+\delta x),$$

for some  __backward error__ $\delta x$.

Algoritam is __stable__ is the above equality always holds for small $\delta x$.
"""

# ╔═╡ 143d2bd9-6873-4ff1-ae84-37ee1dc53749
begin
	B=rand(ComplexF64,5,5)
	B=B+B'
end

# ╔═╡ 104e51a1-2d72-497d-8556-0f6ad97552d4
begin
	C=copy(B)
	for i=1:100
		Q,R=qr(C)
		C=R*Q
	end
end

# ╔═╡ 527035ab-9b51-497b-9cb9-4a5d7c82f725
abs.(C)

# ╔═╡ 98ce0894-8849-4d94-9c8c-4b693eb67825
C

# ╔═╡ e7e45287-7e3a-4334-beee-a0cc45dd6275
C[1,1]

# ╔═╡ d59b2c3d-ac34-4b0c-9ecf-691141f5f3a1
eigvals(B)

# ╔═╡ 9993ff53-f07e-46cf-b871-a4b4a2806c24
md"""

# Basic methods

## Definitions

The eigenvalue decomposition (EVD) of a real symmetric matrix $A=[a_{ij}]$ is $A=U\Lambda U^T$, where $U$ is a $n\times n$ real orthonormal matrix, $U^TU=UU^T=I_n$, and $\Lambda=\mathop{\mathrm{diag}} (\lambda_1,\dots, \lambda_n)$ is a real diagonal matrix.

The numbers $\lambda_i$ are the eigenvalues of $A$, the vectors $U_{:i}$,
$i=1,\dots,n$, are the eigenvectors of $A$, and
$AU_{:i}=\lambda_i U_{:i}$, $i=1,\dots,n$.

If $|\lambda_1|> |\lambda_2| \geq \cdots \geq |\lambda_n|$, we say that $\lambda_1$ is the __dominant eigenvalue__.

__Deflation__ is a process of reducing the size of the matrix whose
EVD is to be determined, given that one eigenvector is known.

The __shifted matrix__ of the matrix $A$ is the matrix $A-\mu I$, where
$\mu$ is the __shift__.

__Power method__ starts from unit vector $x_0$ and computes the sequences

$$\nu_k=x_k^T A x_k, \qquad x_{k+1}= A x_k / \| A x_k \|, \qquad k=0,1,2,\dots,$$

until convergence. Normalization of $x_k$ can be performed in any norm and serves
the numerical stability of the algorithm (avoiding overflow or underflow).

__Inverse iteration__ is the power method applied to the inverse of
a shifted matrix:

$$\nu_k=x_k^T A x_k, \quad 
x_{k+1}= (A-\mu I)^{-1} x_k, \quad 
x_{k+1} = x_{k+1}/\|x_{k+1}\|, \quad k=0,1,2,\dots.$$

__QR iteration__ starts from the matrix $A_0=A$ and forms the sequence of matrices

$$A_k=Q_kR_k \quad \textrm{(QR factorization)}, \qquad A_{k+1}=R_kQ_k,\qquad k=0,1,2,\ldots$$


__Shifted QR iteration__ is the QR iteration applied to a shifted matrix:

$$A_k-\mu I=Q_kR_k \quad \textrm{(QR factorization)}, \quad A_{k+1}=R_kQ_k+\mu I ,\quad k=0,1,2,\ldots$$


"""

# ╔═╡ 2e05504e-3e54-437f-a736-9190be08c332
md"""

## Facts

1. If $\lambda_1$ is the dominant eigenvalue, $x_0$ is not orthogonal to $U_{:1}$, and the norm is $2$-norm, then $\nu_k\to \lambda_1$ and $x_k\to U_{:1}$. In other words, the power method converges to the dominant eigenvalue and its eigenvector.

2. The convergence is linear in the sense that 
$$
|\lambda_1-\nu_k|\approx \left|\frac{c_2}{c_1}\right| \left|
 \frac{\lambda_2}{\lambda_1}\right|^k,\qquad
\|U_{:1}-x_k\|_2 =O\bigg(\bigg|
 \frac{\lambda_2}{\lambda_1}\bigg|^k\bigg)\!,$$
where $c_i$ is the coefficient of the $i$-th eigenvector in the linear combination expressing the starting vector $x_0$.
 
3. Since $\lambda_1$ is not available, the convergence is determined using residuals:  if $\| Ax_k-\nu_k x_k\|_2\leq tol$, where $tol$ is a user prescribed stopping criterion, then $|\lambda_1-\nu_k|\leq tol$.

4. After computing the dominant eigenpair, we can perform deflation to reduce the given EVD for $A$ to the one of size $n-1$ for $A_1$:

$$\begin{bmatrix} U_{:1} & X \end{bmatrix}^T A  \begin{bmatrix} U_{:1} & X \end{bmatrix}= \begin{bmatrix} \lambda_1 & \\ & A_1 \end{bmatrix}, \quad \begin{bmatrix} U_{:1} & X \end{bmatrix} \textrm{orthonormal}, \quad A_1=X^TAX.$$

5. The EVD of the shifted matrix $A-\mu I$ is $U(\Lambda-\mu I) U^T$.  

6. Inverse iteration requires solving the system of linear equations $(A-\mu I)x_{k+1}= x_k$ for $x_{k+1}$ in each step. At the beginning, LU factorization of $A-\mu I$ needs to be computed, which requires $2n^3/3$ operations. In each subsequent step, two triangular systems need to be solved, which requires $2n^2$ operations.

7. If $\mu$ is close to some eigenvalue of $A$, the eigenvalues of the inverse of the shifted matrix satisfy $|1/\lambda_1|\gg |1/\lambda_2|\geq\cdots\geq |1/\lambda_n|$, so the convergence of the inverse iteration method is fast.
  
8. If $\mu$ is very close to some eigenvalue of $A$, then the matrix $A-\mu I$ is nearly singular, so the solutions of linear systems may have large errors. However, these errors are almost entirely in the direction of the dominant eigenvector so the inverse iteration method is both fast and accurate!!
  
9. We can further increase the speed of convergence of inverse iterations by substituting the shift $\mu$ with the Rayleigh quotient $\nu_k$ at the cost of computing new LU factorization.
  
10. Matrices $A_k$ and $A_{k+1}$ from both QR iterations are orthogonally similar, $A_{k+1}=Q_k^TA_kQ_k$.

11. The QR iteration method is essentially equivalent to the power method and the shifted QR iteration method is essentially equivalent to the inverse power method on the shifted matrix. 
  
12. The straightforward application of the QR iteration requires $O(n^3)$ operations per step, so better implementation is needed.

"""

# ╔═╡ 30ee06d5-2b99-423b-93d2-773acbf41723
md"
## Examples

In order to keep the programs simple, in the examples below we do not compute full matrix of eigenvectors.
"

# ╔═╡ 3994bd3e-c2b2-486c-924f-c38896760bff
function Power(A::Matrix,x::Vector,tol::Real=1e-10)
    y=A*x
    ν=x⋅y
    steps=1
    while norm(y-ν*x)>tol && steps<500
        x=y/norm(y)
        y=A*x
        ν=x⋅y
        # println(x)
        steps+=1
    end
    ν, y/norm(y), steps
end

# ╔═╡ 5450fb80-ab1f-4212-ad6f-aa0483bf74d6
begin
	Random.seed!(421)
	n=6
	A=Matrix(Symmetric(rand(-9:9,n,n)))
end

# ╔═╡ ab54ea2a-efbd-44d8-953e-cdd86b2e783d
ν,x,steps=Power(A,ones(n),1e-10)

# ╔═╡ 7e0320e6-1d38-4ce4-85fb-e3c7c7022420
qx,rx=qr(x)

# ╔═╡ 6e49df99-931c-47ea-8f4f-efe4a5273fd3
qx'*A*qx

# ╔═╡ 2c3d8a7f-c986-4277-9470-c5abab058ee4
eigvals(A)

# ╔═╡ 302316e7-df55-4c65-9593-93d04fdb239f
(21.0721/22.1025)^500

# ╔═╡ 5b8c1f0f-e396-40ee-8018-e0ea53af0b2a
(18.7/23.27)^131

# ╔═╡ 290024ee-b36d-4a71-8adf-d433a6daf6ad
ν-eigvals(A)[6]

# ╔═╡ 71107f6d-4c6e-4bc1-a27b-51a99a172b87
# Speed of convergence
(18.7/23)^131

# ╔═╡ c1de7100-40f0-4775-857b-856c24dc8be1
function InvIter(A::Matrix,x::Vector,μ::Real,tol::Real)
    y=(A-μ*I)\x
    ν=x⋅y
    steps=1
    while norm(y-ν*x)>tol && steps<501
        x=y/norm(y)
        y=(A-μ*I)\x
        ν=x⋅y
        # println(x)
        steps+=1
    end
    1/ν+μ, y/norm(y), steps
end

# ╔═╡ 2ca58fc6-4691-4c40-9fb2-8ef3636ea0d3
InvIter(A,ones(n),12.0,1e-10)

# ╔═╡ df6a18ea-a3ef-42ea-85af-7636aa1afc51
begin
	D=Diagonal([5,-5,3,2,1])
	Dq,Dr=qr(rand(5,5))
	D=Dq'*D*Dq
end

# ╔═╡ d3535056-04e2-43db-9073-1e3704ea6d47
eigvals(D)

# ╔═╡ 11b95202-73be-4403-ba45-1dda9fe1e48b
Power(D,ones(5),1e-10)

# ╔═╡ b6db0848-4142-4d88-98ca-1cb415b92742
# Eigenvector is fine
[eigvecs(A)[:,end] x]

# ╔═╡ e9285938-1fc7-41f7-b809-0785f925e72d
# Deflation
function Deflation(A::Matrix,x::Vector)
    X,R=qr(x)
    # To make sure the returned matrix symmetric use
    # full(Symmetric(X[:,2:end]'*A*X[:,2:end]))
    # display(X'*A*X)
    X[:,2:end]'*A*X[:,2:end]
end

# ╔═╡ 3e464ca7-d32c-4e72-96d3-07e48d0c9b34
A₁=Deflation(A,x)

# ╔═╡ 3adb99d9-288d-4d27-b283-456306d54aae
eigvals(A₁)

# ╔═╡ 2dc7c262-098d-4433-ab17-8e8d3ca00bd7
# Compute the second alsolutely largest eigenvalue of A
Power(A₁,ones(size(A₁,1)),1e-10)

# ╔═╡ e4f1b71f-2809-4136-8e56-aa9389d4d674
# Put it all together - eigenvectors are omitted for the sake of simplicty
function PowerMethod(A::Matrix{T}, tol::Real) where T
    n=size(A,1)
    S =  T==Int ? Float64 : T
    λ=Vector{S}(undef,n)
    for i=1:n
        λ[i],x,steps=Power(A,ones(n-i+1),tol)
        A=Deflation(A,x)
    end
    λ
end

# ╔═╡ a9a8b682-e673-4fbe-84a1-138cb1220808
l=PowerMethod(A,1e-10)

# ╔═╡ ff4866d8-32cd-4d23-ac40-ec410eb56ced
InvIter(A,ones(size(A,1)),-13.0018,1e-10)

# ╔═╡ e4587338-6727-49f4-9036-831fdf3bf172
# QR iteration
function QRIteration(A::Matrix, tol::Real)
    steps=1
    while norm(tril(A,-1))>tol
        Q,R=qr(A)
        A=R*Q
        steps+=1
    end
    A,steps
end

# ╔═╡ edb2bd46-9c83-46b4-be8f-ffb02411a690
QRIteration(A,1e-10)

# ╔═╡ 13366c0b-ae11-4bd5-8844-f33694bbd16c
md"""
#  Tridiagonalization

The following implementation of $QR$ iteration requires a total of $O(n^3)$ operations:

1. Reduce $A$ to tridiagonal form $T$ by orthogonal similarities, $X^TAX=T$.
2. Compute the EVD of $T$ with QR iterations, $T=Q\Lambda Q^T$.
3. Multiply $U=XQ$.

One step of QR iterations on $T$ requires $O(n)$ operations if only $\Lambda$ is computed, and $O(n^2)$ operations if
$Q$ is accumulated, as well.

## Definitions

Given vector $v$, __Householder reflector__ is a symmetric orthogonal matrix 

$$
H= I - 2\, \frac{v v^T}{v^Tv}.$$

Given $c=\cos\varphi$, $s=\sin\varphi$, and indices $i<j$, __Givens rotation matrix__ is an orthogonal matrix
$G(c,s,i,j)$ which is equal to the identity matrix except for elements

$$
G_{ii}=G_{jj}=c,\qquad G_{ij}=-G_{ji}=s.$$
"""

# ╔═╡ f0f2aeb5-bfcb-4c6e-a720-e96941f1ed1d
md"""
## Facts

1. Given vector $x$, choosing $v=x+\mathrm{\mathop{sign}}(x_{1})\,\|x\|_2\, e_1$ yields the Householder reflector which performs the QR factorization of $x$:

$$
Hx=-\mathop{\mathrm{sign}}(x_{1})\, \|x\|_2\, e_1.$$

2. Given a 2-dimensional vector $\begin{bmatrix}x\\y\end{bmatrix}$, choosing $r= \sqrt{x^2+y^2}$, $c=\frac{x}{r}$ and $s=\frac{y}{r}$, gives the Givens roatation matrix such that

$$
G(c,s,1,2)\cdot \begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}r\\ 0 \end{bmatrix}.$$

The hypotenuse $r$ is computed using the `hypot()` function in order to avoid underflow or overflow.

3. Tridiagonal form is not unique.

4. The reduction of $A$ to tridiagonal matrix by Householder reflections is performed as follows. Let $A=\begin{bmatrix} \alpha & a^T  \\ a & B \end{bmatrix}$. Let $v=a+\mathrm{\mathop{sign}}(a_{1})\,\|a\|_2\, e_1$, let $H$ be the corresponding Householder reflector and set $H_1=\begin{bmatrix} 1 & \\ & H \end{bmatrix}$. Then

$$
H_1AH_1=
\begin{bmatrix} \alpha & a^T H \\ Ha & HBH \end{bmatrix} =
\begin{bmatrix} \alpha & \nu e_1^T \\ \nu e_1 & A_1 \end{bmatrix},
\quad  \nu=-\mathop{\mathrm{sign}}(a_{1})\, \|a\|_2.$$

This step annihilates all elements in the first column below the first subdiagonal and all elements in the first row to the right of the first subdiagonal. Applying this procedure recursively yields the tridiagonal matrix $T=X^T AX$, where $X=H_1H_2\cdots H_{n-2}$.

5.  $H$ does not depend on the normalization of $v$. With the normalization $v_1=1$, $a_{2:n-1}$ can be overwritten by $v_{2:n-1}$, so $v_1$ does not need to be stored.

6.  $H$ is not formed explicitly - given $v$, $B$ is overwritten with $HBH$ in $O(n^2)$ operations by using one matrix-vector multiplication and two rank-one updates.

7. When symmetry is exploited in performing rank-2 update, tridiagonalization  requires $4n^3/3$ operations. Instead of performing rank-2 update on $B$, one can accumulate $p$ transformations and perform rank-$2p$ update. This __block algorithm__ is rich in matrix--matrix multiplications (roughly one half of the operations is performed using BLAS 3 routines), but it requires extra workspace for $U$ and $V$.

8. If $X$ is needed explicitly, it can be computed from the stored Householder vectors $v$. In order to minimize the operation count, the computation starts from the smallest matrix and the size is gradually increased:

$$
H_{n-2},\quad H_{n-3}H_{n-2},\dots,\quad X=H_1\cdots H_{n-2}.$$

A column-oriented version is possible as well, and the operation count in both cases is $4n^3/3$. If the Householder reflectors $H_i$ are accumulated in the order in which they are generated, the operation count is $2n^3$.

9. The backward error bounds for functions `Tridiag()` and `TridiagX()` are as follows: The computed matrix $\tilde T$ is equal to the matrix which would be obtained by exact tridiagonalization of some perturbed matrix $A+\Delta A$, where $\|\Delta A\|_2 \leq \psi \varepsilon \|A\|_2$ and $\psi$ is a slowly increasing function of $n$. The computed matrix $\tilde X$ satisfies $\tilde X=X+\Delta X$, where $\|\Delta X \|_2\leq \phi \varepsilon$ and $\phi$ is a slowly increasing function of $n$.

10. Tridiagonalization using Givens rotations requires $\displaystyle\frac{(n-1)(n-2)}{2}$ plane rotations, which amounts to $4n^3$ operations if symmetry is properly exploited. The operation count is reduced to $8n^3/3$ if fast rotations are used. Fast rotations are obtained by factoring out absolutely larger of $c$ and $s$ from $G$.

11. Givens rotations in the function `TridiagG()`  can be performed in different orderings. For example, the elements in the first column and row can be annihilated by rotations in the planes $(n-1,n)$, $(n-2,n-1)$, $\ldots$, $(2,3)$. Givens rotations act more selectively than Householder reflectors, and are useful if $A$ has some special structure, for example, if $A$ is a banded matrix. 

12. Error bounds for function `TridiagG()` are the same as above, but with slightly different functions $\psi$ and $\phi$.

12. The block version of tridiagonal reduction is implemented in the  [LAPACK](http://www.netlib.org/lapack) subroutine [DSYTRD](http://www.netlib.org/lapack/explore-3.1.1-html/dsytrd.f.html). The computation of $X$ is implemented in the subroutine [DORGTR](http://www.netlib.org/lapack/lapack-3.1.1/html/dorgtr.f.html). The size of the required extra workspace (in elements) is  $lwork=nb*n$, where $nb$ is the optimal block size (here, $nb=64)$, and it is determined automatically by the subroutines. The subroutine [DSBTRD](http://www.netlib.org/lapack/explore-html/d0/d62/dsbtrd_8f.html) tridiagonalizes a symmetric band matrix by using Givens rotations. There are no Julia wappers for these routines yet!

"""

# ╔═╡ 90615988-0945-4fc9-b8d5-fff49fb49d55
md"

## Examples

### Householder vectors
"

# ╔═╡ e1afe4dc-d237-4b49-ae1f-123bc59aa7fc
function Tridiag(A::Matrix)
    # Normalized Householder vectors are stored in the lower 
    # triangular part of A below the first subdiagonal
    n=size(A,1)
    T=Float64
    A=map(T,A)
    v=Vector{T}(undef,n)
    Trid=SymTridiagonal(zeros(n),zeros(n-1))
    for j = 1 : n-2
        μ = sign(A[j+1,j])*norm(A[j+1:n, j])
        if μ != zero(T)
            β =A[j+1,j]+μ
            v[j+2:n] = A[j+2:n,j] / β
        end
        A[j+1,j]=-μ
        A[j,j+1]=-μ
        v[j+1] = one(T)
        γ = -2 / (v[j+1:n]⋅v[j+1:n])
        w = γ* A[j+1:n, j+1:n]*v[j+1:n]
        q = w + γ * v[j+1:n]*(v[j+1:n]⋅w) / 2 
        A[j+1:n, j+1:n] = A[j+1:n,j+1:n] + v[j+1:n]*q' + q*v[j+1:n]'
        A[j+2:n, j] = v[j+2:n]
    end
    SymTridiagonal(diag(A),diag(A,1)), tril(A,-2)
end

# ╔═╡ a752ab01-7008-46ee-815e-40bc7d8e07ea
A

# ╔═╡ a78d3cc7-42dc-4d5e-86ef-e0b6efc5ab18
T,H=Tridiag(A)

# ╔═╡ bdd46cff-bae9-47bb-b868-3045520ec8be
[eigvals(A) eigvals(T)]

# ╔═╡ 289e5149-bdd4-4ece-abfa-53bb0413ba48
# How is H stored
v₁=[1;H[3:6,1]]

# ╔═╡ dd920abd-55bc-4eed-895c-b68b1f6393a1
H₁=cat([1],I-2*v₁*v₁'/(v₁⋅v₁),dims=(1,2))

# ╔═╡ 14924dfe-1a0c-480d-80e1-b77ea965e789
H₁*A*H₁

# ╔═╡ b0256b1a-b748-44d0-991e-bc6d12f619e6
function TridiagX(H::Matrix)
	# Extract X from the matrix returned by Tridiag()
    n=size(H,1)
	T=Float64
    X = Matrix{T}(I,n,n)
    v=Vector{T}(undef,n)
    for j = n-2 : -1 : 1
        v[j+1] = one(T)
        v[j+2:n] = H[j+2:n, j]
        γ = -2 / (v[j+1:n]⋅v[j+1:n])
        w = γ * X[j+1:n, j+1:n]'*v[j+1:n]
        X[j+1:n, j+1:n] = X[j+1:n, j+1:n] + v[j+1:n]*w'
    end
    X
end

# ╔═╡ f0cc09a5-2bec-47e5-957f-f7e4bc11396d
X=TridiagX(H)

# ╔═╡ 46c42818-3d23-470d-b05d-033bd5bc2d88
# Fact 7: norm(ΔX)<ϕ*eps()
X'*X

# ╔═╡ 8f84d695-ccf9-48d2-a82f-337bb626dff1
# Tridiagonalization
X'*A*X

# ╔═╡ 9a7a6c5a-d11e-46f7-8906-fcf907d2cd97
md"
### Givens rotations
"

# ╔═╡ fcc9ff8b-1549-4782-89ca-13c0f867af00
function TridiagG(A::Matrix)
	# Tridiagonalization using Givens rotations
    n=size(A,1)
    X=Matrix{Float64}(I,n,n)
    for j = 1 : n-2
        for i = j+2 : n
            G,r=givens(A,j+1,i,j)
            A=(G*A)*G'
            # display(A)
            X*=G'
        end
    end
    SymTridiagonal(diag(A),diag(A,1)), X
end

# ╔═╡ 61c41ad7-df89-421f-a9a5-2a0ee606b9f6
#?givens

# ╔═╡ 04ee850c-a475-476c-803b-43d7132d2632
methodswith(LinearAlgebra.Givens)

# ╔═╡ 8c21dd54-6a32-40a4-a576-0e837915cd02
A

# ╔═╡ 68f9bdd0-4059-424c-8ee8-cd46ab9ecc29
T₁,X₁=TridiagG(float(A))

# ╔═╡ 4b53ee99-6b20-4153-af8e-79b13c1b4ee3
# Orthogonality
X₁'*X₁

# ╔═╡ 4a484bf9-001f-4ec6-a4cd-d6bf2fc5b617
# Tridiagonalization
X₁'*A*X₁

# ╔═╡ 278d1e53-d815-470f-8908-3ddf08c09938
md"
There may be differences in signs: compare T (computed using Householder reflectors) and T₁ (computed using Givens rotations).
"

# ╔═╡ def69e92-5a19-434f-b584-4841a4049c27
T

# ╔═╡ ac500be6-6160-4294-8748-70bbcaf9be1b
T₁

# ╔═╡ 50681bb6-6db0-4e07-b52d-94d3b6662526
# One step of QR method
Q,R=qr(Matrix(T))

# ╔═╡ e024fae1-ceea-4f44-9692-00a203afdf3a
# Triangulararity and symmetricity is preserved
R*Q

# ╔═╡ 6df580a7-0376-4264-9fa7-540d6bab6f9d
md"""
# Tridiagonal QR method

Let $T$ be a real symmetric tridiagonal matrix of order $n$ and $T=Q\Lambda Q^T$ be its EVD.

Each step of the shifted QR iterations can be elegantly implemented without explicitly computing the shifted matrix  $T-\mu I$.


## Definition

__Wilkinson's shift__ $\mu$ is the eigenvalue of the bottom right $2\times 2$ submatrix of $T$, which is closer to $T_{n,n}$. 


## Facts

1. The stable formula for the Wilkinson's shift is

$$
\mu=T_{n,n}-
\frac{T_{n,n-1}^2}{\tau+\mathop{\mathrm{sign}}(\tau)\sqrt{\tau^2+T_{n,n-1}^2}},\qquad
\tau=\frac{T_{n-1,n-1}-T_{n,n}}{2}.$$

2. Wilkinson's shift is the most commonly used shift. With Wilkinson's shift, the algorithm always converges in the sense that $T_{n-1,n}\to 0$. The convergence is quadratic, that is, $|[T_{k+1}]_{n-1,n}|\leq c |[T_{k}]_{n-1,n}|^2$ for some constant $c$, where $T_k$ is the matrix after the $k$-th sweep. Even more, the convergence is usually cubic. However, it can also happen that some $T_{i,i+i}$, $i\neq n-1$, becomes sufficiently small before $T_{n-1,n}$, so the practical program has to check for deflation at each step.

3. __Chasing the Bulge.__ The plane rotation parameters at the start of the sweep are computed as if the shifted $T-\mu I$ has been formed. Since the rotation is applied to the original $T$ and not to $T-\mu I$, this creates new nonzero elements at the positions $(3,1)$ and $(1, 3)$, the so-called __bulge__.  The subsequent rotations simply chase the bulge out of the lower right corner of the matrix. The rotation in the $(2,3)$ plane sets the elements $(3,1)$ and $(1,3)$ back to zero, but it generates two new nonzero elements at positions $(4,2)$ and $(2,4)$; the rotation in the $(3,4)$ plane sets the elements $(4,2)$ and $(2,4)$ back to zero, but it generates two new nonzero elements at positions $(5,3)$ and $(3,5)$, etc.

4. __Implicit__ $Q$ __Theorem.__ The effect of this procedure is the following. At the end of the first sweep, the resulting matrix $T_1$ is equal to the the matrix that would have been obtained by factorizing $T-\mu I=QR$ and computing $T_1=RQ+\mu I$.

5. Since the convergence of the function `TridEigQR()` is quadratic (or even cubic), an eigenvalue is isolated after just a few steps, which requires $O(n)$ operations. This means that $O(n^2)$ operations are needed to compute all eigenvalues. 

6. If the eigenvector matrix $Q$ is desired, the plane rotations need to be accumulated similarly to the accumulation of $X$ in the function `TridiagG()`. This accumulation requires $O(n^3)$ operations. Another, faster, algorithm to  first compute only $\Lambda$ and then compute $Q$ using inverse iterations. Inverse iterations on a tridiagonal matrix are implemented in the LAPACK routine [DSTEIN](http://www.netlib.org/lapack/explore-html/d8/d35/dstein_8f.html).

7. __Error bounds.__ Let $U\Lambda U^T$ and $\tilde U \tilde \Lambda \tilde U^T$ be the exact and the computed EVDs of $A$, respectively, such that the diagonals of $\Lambda$ and $\tilde \Lambda$ are in the same order. Numerical methods generally compute the EVD with the errors bounded by 

$$
|\lambda_i-\tilde \lambda_i|\leq \phi \epsilon\|A\|_2,
\qquad
\|u_i-\tilde u_i\|_2\leq \psi\epsilon \frac{\|A\|_2}
{\min_{j\neq i} 
|\lambda_i-\tilde \lambda_j|},$$

where $\epsilon$ is machine precision and $\phi$ and $\psi$ are slowly growing polynomial functions of $n$ which depend upon the algorithm used (typically $O(n)$ or $O(n^2)$). Such bounds are obtained by combining perturbation bounds with the floating-point error analysis of the respective algorithms.

8. The eigenvalue decomposition $T=Q\Lambda Q^T$ computed by `TridEigQR()` satisfies the error bounds from fact 7. with $A$ replaced by $T$ and $U$ replaced by $Q$. The deflation criterion implies $|T_{i,i+1}|\leq \epsilon \|T\|_F$, which is within these bounds.
   
9. The EVD computed by function `SymEigQR()` satisfies the error bounds given in Fact 7. However, the algorithm tends to perform better on matrices, which are graded downwards, that is, on matrices that exhibit systematic decrease in the size of the matrix elements as we move along the diagonal. For such matrices the tiny eigenvalues can usually be computed with higher relative accuracy (although counterexamples can be easily constructed). If the tiny eigenvalues are of interest, it should be checked whether there exists a symmetric permutation that moves larger elements to the upper left corner, thus converting the given matrix to the one that is graded downwards.

10. The function `TridEigQR()` is implemented in the LAPACK subroutine [DSTEQR](http://www.netlib.org/lapack/explore-html/d9/d3f/dsteqr_8f.html). This routine can compute just the eigenvalues, or both eigenvalues and eigenvectors.

11. The function `SymEigQR()` is  Algorithm 5 is implemented in the functions `eigen()`, `eigvals()` and `eigvecs()`,  and in the  LAPACK routine [DSYEV](http://www.netlib.org/lapack/explore-html/dd/d4c/dsyev_8f.html). To compute only eigenvalues, DSYEV calls DSYTRD and DSTEQR without the eigenvector option. To compute both eigenvalues and eigenvectors, DSYEV calls DSYTRD, DORGTR, and DSTEQR with the eigenvector option.
"""

# ╔═╡ 6cd72968-b9fc-47fb-bf7a-b87c9d64b8dc
md"
## Example
"

# ╔═╡ a0e1250f-c543-4186-8a3d-5a0ac8e1747a
function TridEigQR(A₁::SymTridiagonal)
    A=copy(A₁)
    n=length(A.dv)
    T=Float64
    λ=Vector{T}(undef,n)
    B=Matrix{T}
    if n==1
        return map(T,A.dv)
    end
    if n==2
        τ=(A.dv[end-1]-A.dv[end])/2
        μ=A.dv[end]-A.ev[end]^2/(τ+sign(τ)*sqrt(τ^2+A.ev[end]^2))
        # Only rotation
        B=A[1:2,1:2]
        G,r=givens(B-μ*I,1,2,1)
        B=(G*B)*G'
        return diag(B)[1:2]
    end
    steps=1
    k=0
    while k==0 && steps<=10
        # Shift
        τ=(A.dv[end-1]-A.dv[end])/2
        μ=A.dv[end]-A.ev[end]^2/(τ+sign(τ)*sqrt(τ^2+A.ev[end]^2))
        # First rotation
        B=A[1:3,1:3]
        G,r=givens(B-μ*I,1,2,1)
        B=(G*B)*G'
        A.dv[1:2]=diag(B)[1:2]
        A.ev[1:2]=diag(B,-1)
        bulge=B[3,1]
        # Bulge chasing
        for i = 2 : n-2
            B=A[i-1:i+2,i-1:i+2]
            B[3,1]=bulge
            B[1,3]=bulge
            G,r=givens(B,2,3,1)
            B=(G*B)*G'
            A.dv[i:i+1]=diag(B)[2:3]
            A.ev[i-1:i+1]=diag(B,-1)
            bulge=B[4,2]
        end
        # Last rotation
        B=A[n-2:n,n-2:n]
        B[3,1]=bulge
        B[1,3]=bulge
        G,r=givens(B,2,3,1)
        B=(G*B)*G'
        A.dv[n-1:n]=diag(B)[2:3]
        A.ev[n-2:n-1]=diag(B,-1)
        steps+=1
        # Deflation criterion
        k=findfirst(abs.(A.ev) .< sqrt.(abs.(A.dv[1:n-1].*A.dv[2:n]))*eps(T))
        k=k==nothing ? 0 : k
        # display(A)
    end
    λ[1:k]=TridEigQR(SymTridiagonal(A.dv[1:k],A.ev[1:k-1]))
    λ[k+1:n]=TridEigQR(SymTridiagonal(A.dv[k+1:n],A.ev[k+1:n-1]))
    return λ
end

# ╔═╡ 55b1969a-16f5-459d-a9ef-da7066851b93
T

# ╔═╡ de28e7ec-6171-499a-a87b-7109c31860b2
# Built-in function
λ=eigvals(T)

# ╔═╡ 129314c9-da01-41ed-bb00-577059c3ee1d
md"""
#  Computing the eigenvectors 

Once the eigenvalues are computed, the eigenvectors can be efficiently computed with inverse iterations.Inverse iterations for tridiagonal matrices are implemented in the LAPACK routine [DSTEIN](http://www.netlib.org/lapack/explore-html/d8/d35/dstein_8f.html).
"""

# ╔═╡ ae5e4cb9-562a-4bf2-bf6f-6b4c3e8d10d4
#?LAPACK.stein!

# ╔═╡ 892a6a42-df98-4088-b0bd-c406daf0eecb
begin
	# Comapare faster version using stein! with standard eigen
	nb=2000
	Tbig=SymTridiagonal(rand(nb),rand(nb-1))
	println("Timings for n = ",nb)
	@time λbig=eigvals(Tbig);
	@time Ub=LAPACK.stein!(Tbig.dv,Tbig.ev,λbig);
	@time λb,Xb=eigen(Tbig);
	# Residual
	norm(Tbig*Ub-Ub*Diagonal(λbig)), norm(Tbig*Xb-Xb*Diagonal(λb))
end

# ╔═╡ dd74443c-eba6-464d-99cc-612884b105e5
md"""
Alternatively, the rotations in `TridEigQR()` can be accumulated to compute the eigenvectors. This is not optimal, but is instructive. We keep the name of the function, using Julia's __multiple dispatch__ feature.
"""

# ╔═╡ 030c2a11-c204-4c67-9412-d745b1826a50
function TridEigQR(A₁::SymTridiagonal,U::Matrix)
    # U is either the identity matrix or the output from myTridiagX()
    A=deepcopy(A₁)
    n=length(A.dv)
    T=Float64
    λ=Vector{T}(undef,n)
    B=Matrix{T}
    if n==1
        return map(T,A.dv), U
    end
    if n==2
        τ=(A.dv[end-1]-A.dv[end])/2
        μ=A.dv[end]-A.ev[end]^2/(τ+sign(τ)*sqrt(τ^2+A.ev[end]^2))
        # Only rotation
        B=A[1:2,1:2]
        G,r=givens(B-μ*I,1,2,1)
        B=(G*B)*G'
        U*=G'
        return diag(B)[1:2], U
    end
    steps=1
    k=0
    while k==0 && steps<=10
        # Shift
        τ=(A.dv[end-1]-A.dv[end])/2
        μ=A.dv[end]-A.ev[end]^2/(τ+sign(τ)*sqrt(τ^2+A.ev[end]^2))
        # First rotation
        B=A[1:3,1:3]
        G,r=givens(B-μ*I,1,2,1)
        B=(G*B)*G'
        U[:,1:3]*=G'
        A.dv[1:2]=diag(B)[1:2]
        A.ev[1:2]=diag(B,-1)
        bulge=B[3,1]
        # Bulge chasing
        for i = 2 : n-2
            B=A[i-1:i+2,i-1:i+2]
            B[3,1]=bulge
            B[1,3]=bulge
            G,r=givens(B,2,3,1)
            B=(G*B)*G'
            U[:,i-1:i+2]=U[:,i-1:i+2]*G'
            A.dv[i:i+1]=diag(B)[2:3]
            A.ev[i-1:i+1]=diag(B,-1)
            bulge=B[4,2]
        end
        # Last rotation
        B=A[n-2:n,n-2:n]
        B[3,1]=bulge
        B[1,3]=bulge
        G,r=givens(B,2,3,1)
        B=(G*B)*G'
        U[:,n-2:n]*=G'
        A.dv[n-1:n]=diag(B)[2:3]
        A.ev[n-2:n-1]=diag(B,-1)
        steps+=1
        # Deflation criterion
        k=findfirst(abs.(A.ev) .< sqrt.(abs.(A.dv[1:n-1].*A.dv[2:n]))*eps())
        k=k==nothing ? 0 : k
    end
    λ[1:k], U[:,1:k]=TridEigQR(SymTridiagonal(A.dv[1:k],A.ev[1:k-1]),U[:,1:k])
    λ[k+1:n], U[:,k+1:n]=TridEigQR(SymTridiagonal(A.dv[k+1:n],A.ev[k+1:n-1]),U[:,k+1:n])
    λ, U
end

# ╔═╡ 2b575a6a-7a14-449b-9e77-90bbf49633a6
λ₁=TridEigQR(T)

# ╔═╡ ada3a004-5262-4d3f-af82-409c5cae5f28
# Relative errors
(sort(λ)-sort(λ₁))./sort(λ)

# ╔═╡ ccf6e2e8-5183-46fa-886d-d62fff02a444
U₁=LAPACK.stein!(T.dv,T.ev,sort(λ₁))

# ╔═╡ 8f6f840b-d822-4a46-a4ed-71d217c3b816
# Orthogonality
norm(U₁'*U₁-I)

# ╔═╡ 593617c3-729c-472c-9fb4-fb921eeddd82
# Residual
norm(T*U₁-U₁*Diagonal(sort(λ₁)))

# ╔═╡ d8c56876-1cfa-4a3a-98f3-aef736bc984f
begin
	# Some timings - n=100, 200, 400
	n₂=100
	T₂=SymTridiagonal(rand(n₂),rand(n₂-1))
	println("Timings for n = ",n₂)
	@time λ₂=TridEigQR(T₂);
	@time μ₂=eigvals(T₂);
	@time U₂=LAPACK.stein!(T₂.dv,T₂.ev,λ₂);
	# Residual
	norm(T₂*U₂-U₂*Diagonal(sort(λ₂)))
end

# ╔═╡ 764db6d6-62eb-4b30-9cc5-f8e74de215d3
TridEigQR(T,Matrix{Float64}(I,size(T)))

# ╔═╡ 3024fda5-7021-43df-a88c-534eba8cb5d0
md"""
# Symmetric QR method

Combining `Tridiag()`, `TridiagX()` and `TridEigQR()`, we get the method for computing symmetric EVD.
"""

# ╔═╡ 3bb5b460-f4ce-4867-bea0-ad280ccffa1a
function SymEigQR(A::Matrix)
    T,H=Tridiag(A)
    X=TridiagX(H)
    return TridEigQR(T,X)
end

# ╔═╡ 03b0397c-669a-4546-a792-e2ccd82d1a6b
A

# ╔═╡ 5fc4fef5-cd3d-43b2-8c21-a989932bf0aa
λ₃,U₃=SymEigQR(float(A))

# ╔═╡ 0950e002-06b7-4f4f-b4a4-248c2b924ecf
# Orthogonality 
norm(U₃'*U₃-I)

# ╔═╡ d85d8841-e9a5-48d2-a8cd-e39493a96ffc
# Residual
norm(A*U₃-U₃*Diagonal(λ₃))

# ╔═╡ 1806c500-36ee-4e48-a279-dc0be1a6e574
md"""
# Unsymmetric matrices

The $QR$ iterations for unsymmetric matrices are implemented as follows:

1. Reduce $A$ to Hessenberg form form $H$ by orthogonal similarities, $X^TAX=H$.
2. Compute the EVD of $H$ with QR iterations, $H=Q\Lambda Q^*$.
3. Multiply $U=XQ$.

The algorithm requires of $O(n^3)$ operations. For more details, see 
[D. Watkins, Unsymmetric Matrix Eigenvalue Techniques, pp. 56.1-56.12](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897).
"""

# ╔═╡ e0c94c27-5df9-49ce-9497-04e91ade2901
A₄=rand(-9:9,5,5)

# ╔═╡ 04092dec-0d14-417f-bf9e-250a678901a8
E₄=eigen(A₄)

# ╔═╡ d722ddfc-6478-4a89-bf17-0187674b186d
X

# ╔═╡ 67811f28-3a80-4dbd-a434-880b5e5c29be
# Residual 
norm(A₄*E₄.vectors-E₄.vectors*Diagonal(E₄.values))

# ╔═╡ b35fc2bf-cc5b-4168-a394-0d634cab8385
md"""
## Hessenberg factorization
"""

# ╔═╡ 979a20a7-90a0-40c5-9ae1-0acec9d9991e
#?hessenberg

# ╔═╡ faaec20c-160a-4259-abec-b8f225df4a32
H₄=hessenberg(A₄)

# ╔═╡ d3e06e0f-a0be-4aff-836b-1701408487ea
H₄.H

# ╔═╡ fd066fd1-0910-40b4-96e1-a703ec23e834
H₄.Q

# ╔═╡ 403ad3f2-d761-4361-98c9-a94dedcc73bd
H₄.Q'*A₄*H₄.Q

# ╔═╡ 3be5de14-d041-40c6-86a9-048eca55699d
eigvals(Matrix(H₄.H))

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
# ╠═893a2edb-d744-4d7f-b81d-fcf916922b81
# ╠═44758c8b-25bd-47d4-b728-90b4e98b6baf
# ╟─e355fccd-fb92-49d4-b978-c4f1680fbb8d
# ╟─4fe7e0b0-bbd5-4746-9f78-ac4dd1abf4b6
# ╠═143d2bd9-6873-4ff1-ae84-37ee1dc53749
# ╠═104e51a1-2d72-497d-8556-0f6ad97552d4
# ╠═527035ab-9b51-497b-9cb9-4a5d7c82f725
# ╠═98ce0894-8849-4d94-9c8c-4b693eb67825
# ╠═e7e45287-7e3a-4334-beee-a0cc45dd6275
# ╠═d59b2c3d-ac34-4b0c-9ecf-691141f5f3a1
# ╟─9993ff53-f07e-46cf-b871-a4b4a2806c24
# ╟─2e05504e-3e54-437f-a736-9190be08c332
# ╟─30ee06d5-2b99-423b-93d2-773acbf41723
# ╠═3994bd3e-c2b2-486c-924f-c38896760bff
# ╠═5450fb80-ab1f-4212-ad6f-aa0483bf74d6
# ╠═ab54ea2a-efbd-44d8-953e-cdd86b2e783d
# ╠═7e0320e6-1d38-4ce4-85fb-e3c7c7022420
# ╠═6e49df99-931c-47ea-8f4f-efe4a5273fd3
# ╠═2c3d8a7f-c986-4277-9470-c5abab058ee4
# ╠═302316e7-df55-4c65-9593-93d04fdb239f
# ╠═5b8c1f0f-e396-40ee-8018-e0ea53af0b2a
# ╠═290024ee-b36d-4a71-8adf-d433a6daf6ad
# ╠═71107f6d-4c6e-4bc1-a27b-51a99a172b87
# ╠═c1de7100-40f0-4775-857b-856c24dc8be1
# ╠═2ca58fc6-4691-4c40-9fb2-8ef3636ea0d3
# ╠═df6a18ea-a3ef-42ea-85af-7636aa1afc51
# ╠═d3535056-04e2-43db-9073-1e3704ea6d47
# ╠═11b95202-73be-4403-ba45-1dda9fe1e48b
# ╠═b6db0848-4142-4d88-98ca-1cb415b92742
# ╠═e9285938-1fc7-41f7-b809-0785f925e72d
# ╠═3e464ca7-d32c-4e72-96d3-07e48d0c9b34
# ╠═3adb99d9-288d-4d27-b283-456306d54aae
# ╠═2dc7c262-098d-4433-ab17-8e8d3ca00bd7
# ╠═e4f1b71f-2809-4136-8e56-aa9389d4d674
# ╠═a9a8b682-e673-4fbe-84a1-138cb1220808
# ╠═ff4866d8-32cd-4d23-ac40-ec410eb56ced
# ╠═e4587338-6727-49f4-9036-831fdf3bf172
# ╠═edb2bd46-9c83-46b4-be8f-ffb02411a690
# ╟─13366c0b-ae11-4bd5-8844-f33694bbd16c
# ╟─f0f2aeb5-bfcb-4c6e-a720-e96941f1ed1d
# ╟─90615988-0945-4fc9-b8d5-fff49fb49d55
# ╠═e1afe4dc-d237-4b49-ae1f-123bc59aa7fc
# ╠═a752ab01-7008-46ee-815e-40bc7d8e07ea
# ╠═a78d3cc7-42dc-4d5e-86ef-e0b6efc5ab18
# ╠═bdd46cff-bae9-47bb-b868-3045520ec8be
# ╠═289e5149-bdd4-4ece-abfa-53bb0413ba48
# ╠═dd920abd-55bc-4eed-895c-b68b1f6393a1
# ╠═14924dfe-1a0c-480d-80e1-b77ea965e789
# ╠═b0256b1a-b748-44d0-991e-bc6d12f619e6
# ╠═f0cc09a5-2bec-47e5-957f-f7e4bc11396d
# ╠═46c42818-3d23-470d-b05d-033bd5bc2d88
# ╠═8f84d695-ccf9-48d2-a82f-337bb626dff1
# ╟─9a7a6c5a-d11e-46f7-8906-fcf907d2cd97
# ╠═fcc9ff8b-1549-4782-89ca-13c0f867af00
# ╠═61c41ad7-df89-421f-a9a5-2a0ee606b9f6
# ╠═04ee850c-a475-476c-803b-43d7132d2632
# ╠═8c21dd54-6a32-40a4-a576-0e837915cd02
# ╠═68f9bdd0-4059-424c-8ee8-cd46ab9ecc29
# ╠═4b53ee99-6b20-4153-af8e-79b13c1b4ee3
# ╠═4a484bf9-001f-4ec6-a4cd-d6bf2fc5b617
# ╟─278d1e53-d815-470f-8908-3ddf08c09938
# ╠═def69e92-5a19-434f-b584-4841a4049c27
# ╠═ac500be6-6160-4294-8748-70bbcaf9be1b
# ╠═50681bb6-6db0-4e07-b52d-94d3b6662526
# ╠═e024fae1-ceea-4f44-9692-00a203afdf3a
# ╟─6df580a7-0376-4264-9fa7-540d6bab6f9d
# ╟─6cd72968-b9fc-47fb-bf7a-b87c9d64b8dc
# ╠═a0e1250f-c543-4186-8a3d-5a0ac8e1747a
# ╠═55b1969a-16f5-459d-a9ef-da7066851b93
# ╠═de28e7ec-6171-499a-a87b-7109c31860b2
# ╠═2b575a6a-7a14-449b-9e77-90bbf49633a6
# ╠═ada3a004-5262-4d3f-af82-409c5cae5f28
# ╟─129314c9-da01-41ed-bb00-577059c3ee1d
# ╠═ae5e4cb9-562a-4bf2-bf6f-6b4c3e8d10d4
# ╠═ccf6e2e8-5183-46fa-886d-d62fff02a444
# ╠═8f6f840b-d822-4a46-a4ed-71d217c3b816
# ╠═593617c3-729c-472c-9fb4-fb921eeddd82
# ╠═d8c56876-1cfa-4a3a-98f3-aef736bc984f
# ╠═892a6a42-df98-4088-b0bd-c406daf0eecb
# ╟─dd74443c-eba6-464d-99cc-612884b105e5
# ╠═030c2a11-c204-4c67-9412-d745b1826a50
# ╠═764db6d6-62eb-4b30-9cc5-f8e74de215d3
# ╟─3024fda5-7021-43df-a88c-534eba8cb5d0
# ╠═3bb5b460-f4ce-4867-bea0-ad280ccffa1a
# ╠═03b0397c-669a-4546-a792-e2ccd82d1a6b
# ╠═5fc4fef5-cd3d-43b2-8c21-a989932bf0aa
# ╠═0950e002-06b7-4f4f-b4a4-248c2b924ecf
# ╠═d85d8841-e9a5-48d2-a8cd-e39493a96ffc
# ╟─1806c500-36ee-4e48-a279-dc0be1a6e574
# ╠═e0c94c27-5df9-49ce-9497-04e91ade2901
# ╠═04092dec-0d14-417f-bf9e-250a678901a8
# ╠═d722ddfc-6478-4a89-bf17-0187674b186d
# ╠═67811f28-3a80-4dbd-a434-880b5e5c29be
# ╟─b35fc2bf-cc5b-4168-a394-0d634cab8385
# ╠═979a20a7-90a0-40c5-9ae1-0acec9d9991e
# ╠═faaec20c-160a-4259-abec-b8f225df4a32
# ╠═d3e06e0f-a0be-4aff-836b-1701408487ea
# ╠═fd066fd1-0910-40b4-96e1-a703ec23e834
# ╠═403ad3f2-d761-4361-98c9-a94dedcc73bd
# ╠═3be5de14-d041-40c6-86a9-048eca55699d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
