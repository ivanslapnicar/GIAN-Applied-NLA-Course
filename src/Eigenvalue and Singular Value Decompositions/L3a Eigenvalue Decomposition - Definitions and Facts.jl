### A Pluto.jl notebook ###
# v0.19.23

using Markdown
using InteractiveUtils

# ╔═╡ 973abf88-9125-41c7-8372-61efb2976fc2
using PlutoUI,SymPy, Random, LinearAlgebra, Polynomials, ToeplitzMatrices, SpecialMatrices , Statistics

# ╔═╡ ec1fe200-8d73-11eb-0ba3-e593aa79dab2
PlutoUI.TableOfContents(aside=true)

# ╔═╡ 10910df9-240f-4e6b-8076-1a9be4d66ba1
md"""
# Eigenvalue Decomposition - Definitions and Facts


__Prerequisites__

The reader should be familiar with basic linear algebra concepts. 

 
__Competences__

The reader should be able to understand and check the facts about eigenvalue decomposition.

__Selected references__

There are many excellent books on the subject. Here we list a few:

* J.W. Demmel, _Applied Numerical Linear Algebra_, SIAM, Philadelphia, 1997.
* G. H. Golub and C. F. Van Loan, _Matrix Computations_, 4th ed., The John Hopkins University Press, Baltimore, MD, 2013.
* N. Higham, _Accuracy and Stability of Numerical Algorithms_, SIAM, Philadelphia, 2nd ed., 2002.
* L. Hogben, ed., _Handbook of Linear Algebra_, CRC Press, Boca Raton, 2014.
* B. N. Parlett, _The Symmetric Eigenvalue Problem_, Prentice-Hall, Englewood Cliffs, NJ, 1980, also SIAM, Philadelphia, 1998.
* G. W. Stewart, _Matrix Algorithms, Vol. II: Eigensystems_, SIAM, Philadelphia, 2001.
* L. N. Trefethen and D. Bau, III, _Numerical Linear Algebra_, SIAM, Philadelphia, 1997.
* J. H. Wilkinson, _The Algebraic Eigenvalue Problem_, Clarendon Press, Oxford, U.K.,  1965.
    
"""

# ╔═╡ b87354b9-ee7b-4ce0-bf26-b4e71c6be6a9
md"""
# General matrices

For more details and the proofs of the Facts below, see 
[L. M. DeAlba, Determinants and Eigenvalues, pp. 4.1-1.15](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.


## Definitions

We state the basic definitions:

Let $\mathbb{F}=\mathbb{R}$ or $F=\mathbb{C}$ and let $A\in \mathbb{F}^{n\times n}$ with elements $a_{ij}\in \mathbb{F}$.

An element $\lambda \in \mathbb{F}$ is an __eigenvalue__ of $A$ if
$\exists x\in \mathbb{F}^n$, $x\neq 0$ such that

$$Ax=\lambda x,$$

and $x$ is an __eigenvector__ of $\lambda$.

__Characteristic polynomial__ of $A$ is $p_A(x)=\det(A-xI)$.

__Algebraic multiplicty__, $\alpha(\lambda)$, is the multiplicity of $\lambda$ as a root of $p_A(x)$.

__Spectrum of $A$__, $\sigma(A)$, is the multiset of all eigenvalues of $A$, with each eigenvalue appearing $\alpha(A)$ times.

__Spectral radius__ of $A$ is 

$$\rho(A)=\max\{|\lambda|, \lambda \in \sigma(A)\}.$$

__Eigenspace__ of $\lambda$ is 

$$
E_{\lambda}(A)=\ker(A-\lambda I).$$

__Geometric multiplicity__ of $\lambda$ is 

$$\gamma(\lambda)=\dim(E_{\lambda}(A)).$$

 $\lambda$ is __simple__ if $\alpha(\lambda)=1$.

 $\lambda$ is __semisimple__ if $\alpha(\lambda)=\gamma(\lambda)$.

 $A$ is __nonderogatory__ if $\gamma(\lambda)=1$ for all $\lambda$.

 $A$ is __nondefective__ if every $\lambda$ is semisimple.

 $A$ is __diagonalizable__ if there exists nonsingular $B$ matrix and diagonal matrix $D$ such that $A=BDB^{-1}$.

__Trace__ of $A$ is 

$$\mathop{\mathrm{tr}}(A)=\sum_i a_{ii}.$$

 $Q\in\mathbb{C}^{n\times n}$ is __unitary__ if $Q^*Q=QQ^*=I$,
where $Q^*=(\bar Q)^T$.

__Schur decomposition__ of $A$ is $A=QTQ^*$,
where $Q$ is unitary and $T$ is upper triangular.

 $A$ and $B$ are __similar__ if $B=QAQ^{-1}$ for some nonsingular matrix $Q$.

 $A$ is __normal__ if $AA^*=A^*A$. 
"""

# ╔═╡ 31a71b9e-1dba-4a69-b87e-9202292eb3ed
md"""
## Facts

There are many facts related to the eigenvalue problem for general matrices. We state some basic ones:

1.  $\lambda\in\sigma(A) \Leftrightarrow p_A(\lambda)=0$.

1. __Cayley-Hamilton Theorem.__ $p_A(A)=0$. 

2. A simple eigenvalue is semisimple.

3.  $\mathop{\mathrm{tr}}(A)=\sum_{i=1}^n \lambda_i$.
 
4.  $\det(A)=\prod_{i=1}^n \lambda_i$.

5.  $A$ is singular $\Leftrightarrow$ $\det(A)=0$ $\Leftrightarrow$ $0\in\sigma(A)$.

7. If $A$ is triangular, $\sigma(A)=\{a_{11},a_{22},\ldots,a_{nn}\}$.

6. For $A\in\mathbb{C}^{n\times n}$, $\lambda\in\sigma(A)$ $\Leftrightarrow$ $\bar\lambda\in\sigma(A^*)$.

8. __Corollary of the Fundamental theorem of algebra.__ For $A\in\mathbb{R}^{n\times n}$, $\lambda\in\sigma(A)$ $\Leftrightarrow$ $\bar\lambda\in\sigma(A)$. 

9. If $(\lambda,x)$ is an eigenpair of a nonsingular $A$, then $(1/\lambda,x)$ is an eigenpair of $A^{-1}$.

10. Eigenvectors corresponding to distinct eigenvalues are linearly independent.

11.  $A$ is diagonalizable $\Leftrightarrow$ $A$ is nondefective $\Leftrightarrow$ $A$ has $n$ linearly independent eigenvectors. 

12. Every $A$ has Schur decomposition. Moreover, $T_{ii}=\lambda_i$.  _(For proof see GVL p.351 (375).)_

13. If $A$ is normal, matrix $T$ from its Schur decomposition is normal. Consequently:

    *  $T$ is diagonal, and has eigenvalues of $A$ on diagonal,
    * matrix $Q$ of the Schur decomposition is the unitary matrix of eigenvectors,
    * all eigenvalues of $A$ are semisimple and $A$ is nondefective.

14. Real matrix has a __real Schur decomposition__, $Q^TAQ=T$, where $Q$ is real orthogonal and $T$ is upper block-triangular with $1\times 1$ and $2 \times 2$ blocks on the diagonal. The $1\times 1$ blocks correspond to real eigenvalues, and the $2\times 2$ blocks correspond to pairs of complex conjugate eigenvalues.  

15. If $A$ and $B$ are similar, $\sigma(A)=\sigma(B)$. Consequently, $\mathop{\mathrm{tr}}(A)=\mathop{\mathrm{tr}}(B)$ and $\det(A)=\det(B)$.

16. Eigenvalues and eigenvectors are continous and differentiable: if $\lambda$ is a simple eigenvalue of $A$ and $A(\varepsilon)=A+\varepsilon E$ for some $E\in F^{n\times n}$, for small $\varepsilon$ there exist differentiable functions $\lambda(\varepsilon)$ and $x(\varepsilon)$ such that $A(\varepsilon) x(\varepsilon) = \lambda(\varepsilon) x(\varepsilon)$.

17. Classical motivation for the eigenvalue problem is the following: consider the system of linear differential equations with constant coefficients, $\dot y(t)=Ay(t)$. If the solution is $y=e^{\lambda t} x$ for some constant vector $x$, then

$$\lambda e^{\lambda t} x=Ae^{\lambda t} x \quad \textrm{or} \quad Ax=\lambda x.$$

"""

# ╔═╡ 2f5aee03-3c2e-4920-9d10-c509b06be627
md"""
## Examples

We shall illustrate above Definitions and Facts on several small examples, using symbolic computation.
"""

# ╔═╡ 37d24090-8d78-11eb-059d-737014bb6588
md"
### Defective eigenvalue
"

# ╔═╡ 184bc4ec-c566-4f21-99b3-b6e8cd328b47
A=[-3 7 -1; 6 8 -2; 72 -28 19]

# ╔═╡ 5d0aac9e-4f48-4f09-956f-5033a1fb8f11
x=symbols("x")

# ╔═╡ 73ece9ec-97dd-46bc-8e17-10ce7bfc18bc
begin
	eye(n)=Matrix{Int}(I,n,n)
	A-x*eye(3)
end

# ╔═╡ ce83f04c-e915-4599-beeb-d4a0a681bc17
# Characteristic polynomial p_A(λ)
p(x)=det(A-x*eye(3))

# ╔═╡ ff5044c0-8d76-11eb-3951-6bb4b5caa7d0
p(x)

# ╔═╡ 55171a69-3557-4ba5-8a4a-813f38c03dd3
# Characteristic polynomial in nicer form
factor(p(x))

# ╔═╡ 6df42190-a29e-4c68-8684-07d8d0c9d200
λ=solve(p(x),x)

# ╔═╡ b94f7fcd-c6f4-4f4f-924a-20f00e82d94e
md"""
The eigenvalues are $\lambda_1=-6$ and $\lambda_2=15$ with algebraic multiplicities
$\alpha(\lambda_1)=1$ and $\alpha(\lambda_2)=2$.
"""

# ╔═╡ 1b90d50a-bf97-4b1a-9580-d729b03c914e
nullspace(float.(A-λ[1]*I))

# ╔═╡ 15b60cd5-eb30-43bb-bbcf-dd533e29c35f
nullspace(float.(A-λ[2]*I))

# ╔═╡ 3619fca4-1374-4adc-b29c-8058b2e147e0
md"""
The geometric multiplicities are $\gamma(\lambda_1)=1$ and $\gamma(\lambda_2)=1$. Thus, $\lambda_2$ is not semisimple, therefore $A$ is defective and not diagonalizable.
"""

# ╔═╡ d46018c0-a5f8-4dd0-8f3d-73322be792e9
# Trace and determinant
tr(A), λ[1]+λ[2]+λ[2]

# ╔═╡ d387b26f-bec1-47a9-b1c6-f20c43f0bde8
det(A), λ[1]*λ[2]*λ[2]

# ╔═╡ 703da5c4-cd09-4832-b64e-5fda75099db5
# Schur decomposition
S=schur(A)

# ╔═╡ e26f113c-dd98-4e97-b495-581de1b431af
fieldnames(typeof(S))

# ╔═╡ 62779bfc-ea5d-4eec-94ad-2641e11fc44d
S.T

# ╔═╡ bdf079a6-beee-44f5-8200-613e54b3a8dc
S.Z

# ╔═╡ e1f5027c-1080-4c95-81cc-a75dde641b88
# Orthogonality
S.Z'*S.Z

# ╔═╡ c71a7760-b4c6-404e-b5ef-f4879cb787cd
begin
	# Similar matrices
	M=rand(-5:5,3,3)
	eigvals(M*A*inv(M)), tr(M*A*inv(M)), det(M*A*inv(M))
end

# ╔═╡ 343c8f45-6226-4361-a149-5313a47c355c
md"
### Real Schur decomposition
"

# ╔═╡ 41a05c50-f6df-4b93-bd8a-50412829105e
begin
	# Fact 14
	Random.seed!(326)
	A₀=rand(-9:9,6,6)
end

# ╔═╡ cd8fb666-d2f2-423c-ac39-405c842e97ba
S₀=schur(A₀)

# ╔═╡ 0fe16884-eac0-496c-9ee4-9a40acb019fe
eigvals(S₀.T[1:2,1:2])

# ╔═╡ 7aa7ac89-837c-4ee8-b9f1-8ad0256f25e1
md"""
### Diagonalizable matrix

This matrix is nondefective and diagonalizable.
"""

# ╔═╡ c3724c4f-142b-4863-8c04-4a0c15411f40
A₁=[57 -21 21; -14 22 -7; -140 70 -55]

# ╔═╡ 792f632e-fdc2-4016-baaa-b366704f02fd
p₁(x)=factor(det(A₁-x*eye(3)))

# ╔═╡ a8664a40-8d78-11eb-325e-41c6497be282
factor(p₁(x))

# ╔═╡ 1d42245a-2ed6-4cc7-8cbd-2b6aa4e64861
λ₁=solve(p₁(x),x)

# ╔═╡ f7aae752-70c4-4a10-b96e-206c13a6ced1
nullspace(float.(A₁-λ₁[2]*I))

# ╔═╡ 60525ba5-2c76-406e-8eb3-63ced2e00bdd
# F=schur(A)
schur(A₁).T

# ╔═╡ 189544ef-122c-4e30-bfd6-cfc126fc0320
md"""
### Symbolic computation for $n=4$

Let us try some random examples of dimension $n=4$ (the largest $n$ for which we can compute eigevalues symbolically).
"""

# ╔═╡ da340240-f845-4591-94c3-9ffce6a33d8a
begin
	Random.seed!(123)
	A₄=rand(-4:4,4,4)
end

# ╔═╡ e72bf173-7e55-4476-91ff-789847dc53cf
p₄(x)=factor(det(A₄-x*eye(4)))

# ╔═╡ da4a5d70-92d8-11eb-0635-11c530ff7d21
p₄(x)

# ╔═╡ ec7fc5c0-92d8-11eb-16df-af1133ccb42a
f(x)=x^4+5*x^3-17*x^2-4*x-31*x^0

# ╔═╡ 29de5fd0-92d9-11eb-09c0-0356001caaef
# Cayley-Hamilton teorem
f(A₄)

# ╔═╡ d43480a3-9247-48a4-8a3b-e006ab58539e
λ₄=solve(p₄(x),x)

# ╔═╡ 5995cd31-a75d-4718-8808-4ada01920344
λ₄[1]

# ╔═╡ 57e2742e-8d79-11eb-0a33-f7250ac02840
λ₄[2]

# ╔═╡ 5e069120-8d79-11eb-3c00-0b5249d1f2dc
λ₄[3]

# ╔═╡ 63167eee-8d79-11eb-29c2-2febbd7cbd93
λ₄[4]

# ╔═╡ e051d2eb-3972-4c97-b7eb-8a18629f6171
md"""
Since all eigenvalues are distinct, they are all simple and the matrix is diagonalizable. 
With high probability, all eigenvalues of a random matrix are simple.

Do not try to use `nullspace()` here.

Symbolic computation does not work well with floating-point numbers - in the following example, the determinant of $A-xI$ is a rational function:
"""

# ╔═╡ 22f0a627-beb2-4b06-aa94-ee1e90fe8a6c
begin
	A₂=rand(4,4)
    factor(det(A₂-x*eye(4)))
end

# ╔═╡ da6baba0-92d9-11eb-1603-7961ad1f49c8
A₂

# ╔═╡ c6ab93ab-b336-4bb4-9dda-b725108f7827
md"""
Let us try `Rational` numbers:
"""

# ╔═╡ ffcb919f-9754-4650-92f1-7dd163c5df25
Aᵣ=map(Rational,A₂)

# ╔═╡ 32ce58b9-783b-4739-9001-0fe581b2953d
begin
	pᵣ(x)=factor(det(Aᵣ-x*eye(4)))
	pᵣ(x)
end

# ╔═╡ a7e8d711-57a6-477a-a641-b929789dbadf
λᵣ=solve(pᵣ(x),x)

# ╔═╡ 5781508c-388a-4df0-a6f9-0fe312a81335
# For example:
λᵣ[1]

# ╔═╡ 90b4341d-7964-4e41-a268-8a047a058286
md"""
### Circulant matrix

For more details, see 
[A. Böttcher and I. Spitkovsky, Special Types of Matrices, pp. 22.1-22.20](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.


Given $a_0,a_1,\ldots,a_{n-1}\in \mathbb{C}$, the __circulant matrix__ is

$$C(a_0,a_1,\ldots,a_{n-1})=\begin{bmatrix}
a_0 & a_{n-1} & a_{n-2} & \cdots & a_{1} \\
a_1 & a_0 & a_{n-1} & \cdots & a_{2} \\
a_2 & a_1 & a_{0} & \cdots & a_{3} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
a_{n-1} & a_{n-2} & a_{n-3} & \cdots & a_{0}
\end{bmatrix}.$$

Let $a(z)=a_0+a_1z+a_2z^2+\cdots +a_{n-1}z^{n-1}$ be the associated complex polynomial.

Let $\omega_n=\exp\big(\displaystyle\frac{2\pi i}{n}\big)$. The __Fourier matrix__ of order $n$ is

$$F_n=\frac{1}{\sqrt{n}} \bigg[ \omega_n^{(j-1)(k-1)}\bigg]_{j,k=1}^n=
\frac{1}{\sqrt{n}} \begin{bmatrix} 
1 & 1 & \cdots & 1 \\
1& \omega_n & \omega_n^2 & \cdots \omega_n^{n-1} \\
1& \omega_n^2 & \omega_n^4 & \cdots \omega_n^{2(n-1)} \\
\vdots & \vdots & \ddots & \vdots \\
1& \omega_n^{n-1} & \omega_n^{2(n-1)} & \cdots \omega_n^{(n-1)(n-1)}
\end{bmatrix}.$$

Fourier matrix is unitary. 
Fourier matrix is a Vandermonde matrix, 

$$F_n=\displaystyle\frac{1}{\sqrt{n}} V(1,\omega_n,\omega_n^2,\ldots, \omega_n^{n-1}).$$

Circulant matrix is normal and, thus, unitarily diagonalizable, with the eigenvalue decomposition

$$
C(a_0,a_1,\ldots,a_{n-1})=F_n^*\mathop{\mathrm{diag}}[(a(1),a(\omega_n),a(\omega_n^2),\ldots, 
a(\omega_n^{n-1})]F_n.$$

We shall use the packages [ToeplitzMatrices.jl](https://github.com/JuliaMatrices/ToeplitzMatrices.jl) and [SpecialMatrices.jl](https://github.com/JuliaMatrices/SpecialMatrices.jl).
"""

# ╔═╡ 76ffa073-59b8-4118-aa32-522f4ff05979
varinfo(ToeplitzMatrices)

# ╔═╡ f85ce269-7555-48ed-a2fd-1193d3f33f40
begin
	Random.seed!(124)
	n=6
	a=rand(-9:9,n)
end

# ╔═╡ e6544939-c8c2-464c-83a0-a6bf1751efa1
C=Circulant(a)

# ╔═╡ b559d56b-02bb-47d3-8f0f-8dae09d918d8
# Check for normality
C*C'-C'*C

# ╔═╡ 5e0adf59-29d3-47ac-8f3f-52929cc4e1c1
pᵪ=Polynomial(a)

# ╔═╡ 83ba2d1b-66de-41bf-8ded-7070bed15587
# Polynomials are easy to manipulate
pᵪ*pᵪ

# ╔═╡ b0255c92-3242-442a-a670-1bfef38e955c
# n-th complex root of unity
ω=exp(2*π*im/n)

# ╔═╡ c6116576-5b25-45b2-bc71-a889f302b5ac
begin
	v=[ω^k for k=0:n-1]
	Fₙ=Vandermonde(v)/√n
end

# ╔═╡ 3de2dc38-ba95-479e-89bc-77cb520daf2f
Λᵪ=Fₙ*C*Fₙ'

# ╔═╡ ad32c51c-7cbb-42f9-abc8-561828df1379
# Check
[diag(Λᵪ) pᵪ.(v) eigvals(Matrix(C))]

# ╔═╡ a7999464-7060-4345-b396-e2e15fcbcaac
md"""

# Hermitian and real symmetric matrices

For more details and the proofs of the Facts below, see 
[W. Barrett, Hermitian and Positive Definite Matrices, pp. 9.1-9.13](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.

## Definitions

Matrix $A\in \mathbb{C}^{n\times n}$ is __Hermitian__ or __self-adjoint__ if $A^*=A$, 
or element-wise, $\bar a_{ij}=a_{ji}$. We say $A\in\mathcal{H}_n$.

Matrix $A\in \mathbb{R}^{n\times n}$ is __symmetric__ if $A^T=A$, or element-wise, $a_{ij}=a_{ji}$.
We say $A\in\mathcal{S}_n$.

__Rayleigh qoutient__ of $A\in\mathcal{H}_n$ and nonzero vector $x\in\mathbb{C}^n$ is

$$
R_A(x)=\frac{x^*Ax}{x^*x}.$$

Matrices $A,B \in\mathcal{H}_n$ are __congruent__ if there exists nonsingular matrix $C$ such that 
$B=C^*AC$.

__Inertia__ of $A\in\mathcal{H}_n$ is the ordered triple 

$$\mathop{\mathrm{in}}(A)=(\pi(A),\nu(A),\zeta(A)),$$

where $\pi(A)$ is the number of positive eigenvalues of $A$,  $\nu(A)$ is the number of negative eigenvalues of $A$,
and $\zeta(A)$ is the number of zero eigenvalues of $A$.

__Gram matrix__ of a set of vectors $x_1,x_2,\ldots,x_k\in\mathbb{C}^{n}$ is the matrix $G$ with entries $G_{ij}=x_i^*x_j$. 
"""

# ╔═╡ 01e2e27b-506e-439a-ba9a-bef034328482
md"""
## Facts

Assume $A$ is Hermitian and $x\in\mathbb{C}^n$ is nonzero. Then

1. Real symmetric matrix is Hermitian, and real Hermitian matrix is symmetric.

2. Hermitian and real symmetric matrices are normal.

3.  $A+A^*$, $A^*A$, and $AA^*$ are Hermitian.

4. If $A$ is nonsingular, $A^{-1}$ is Hermitian.

5. Main diagonal entries of $A$ are real.

6. Matrix $T$ from the Schur decomposition of $A$ is Hermitian. Consequently:
   
    *  $T$ is diagonal and real, and has eigenvalues of $A$ on diagonal,
    * matrix $Q$ of the Schur decomposition is the unitary matrix of eigenvectors,
    * all eigenvalues of $A$ are semisimple and $A$ is nondefective,
    * eigenvectors corresponding to distinct eigenvalues are orthogonal.
    
7. __Spectral Theorem.__ To summarize:

    * if $A\in\mathcal{H}_n$, there is a unitary matrix $U$ and real diagonal matrix $\Lambda$ such that $A=U\Lambda U^*$. The diagonal entries of $\Lambda$ are the eigenvalues of $A$, and the columns of $U$ are the corresponding eigenvectors.
    * if $A\in\mathcal{S}_n$, the same holds with orthogonal matrix $U$, $A=U\Lambda U^T$.
    * if $A\in\mathcal{H}_n$ with eigenpairs $(\lambda_i,u_i)$, then $A=\lambda_1u_1u_1^*+\lambda_2 u_2u_2^* +\cdots +\lambda_n u_nu_n^*$.
    
    * similarly, if  $A\in\mathcal{S}_n$, then $A=\lambda_1u_1u_1^T+\lambda_2 u_2u_2^T +\cdots +\lambda_n u_nu_n^T$.
    
8. Since all eigenvalues of $A$ are real, they can be ordered: $\lambda_1\geq \lambda_2\geq \cdots \geq \lambda_n$.

9. __Rayleigh-Ritz Theorem.__ It holds:

$$\begin{aligned}
\lambda_n &\leq \frac{x^*Ax}{x^*x} \leq \lambda_1, \\
\lambda_1&=\max_x\frac{x^*Ax}{x^*x} =\max_{\|x\|_2=1} x^*Ax, \\
\lambda_n&=\min_x\frac{x^*Ax}{x^*x} =\min_{\|x\|_2=1} x^*Ax.
\end{aligned}$$

10. __Courant-Fischer Theorem.__ It holds:

$$\begin{aligned}
\lambda_k& =\max_{\dim(W)=k}\min_{x\in W} \frac{x^*Ax}{x^*x}\\
& =\min_{\dim(W)=n-k+1}\max_{x\in W} \frac{x^*Ax}{x^*x}.
\end{aligned}$$

11. __Cauchy Interlace Theorem.__ Let $B$ be the principal submatrix of $A$ obtained by deleting the $i$-th row and the $i$-th column of $A$. Let $\mu_1\geq \mu_2\geq \cdots \geq \mu_{n-1}$ be the eignvalues of $B$. Then

$$\lambda_1\geq \mu_1\geq \lambda_2\geq \mu_2\geq \lambda_3\geq\cdots\geq \lambda_{n-1}\geq\mu_{n-1}\geq\lambda_n.$$

12. __Weyl Inequalities.__ For $A,B\in\mathcal{H}_n$, it holds:

$$\begin{aligned}
\lambda_{j+k-1}(A+B)& \leq \lambda_j(A)+\lambda_k(B), & \textrm{for} \ j+k\leq n+1,\\
\lambda_{j+k-n}(A+B)& \geq \lambda_j(A)+\lambda_k(B), & \textrm{for} \ j+k\geq n+1,\\
\lambda_j(A)+\lambda_n(B) & \leq \lambda_j(A+B) \leq \lambda_j(A)+\lambda_1(B), & \textrm{for} \ j=1,2,\ldots,n.
\end{aligned}$$

13.  $\pi(A)+\nu(A)+\zeta(A)=n$.

14.  $\mathop{\mathrm{rank}}(A)=\pi(A)+\nu(A)$.

15. If $A$ is nonsingular, $\mathop{\mathrm{in}}(A)=\mathop{\mathrm{in}}(A^{-1})$.

16. If $A,B \in\mathcal{H}_n$ are similar,  $\mathop{\mathrm{in}}(A)=\mathop{\mathrm{in}}(B)$.

17. __Sylvester's Law of Inertia.__ $A,B \in\mathcal{H}_n$ are congruent if and only if $\mathop{\mathrm{in}}(A)=\mathop{\mathrm{in}}(B)$.

18. __Subadditivity of Inertia.__ For $A,B \in\mathcal{H}_n$, 

$$\pi(A+B)\leq \pi(A)+\pi(B), \qquad \nu(A+B)\leq \nu(A)+\nu(B).$$

19. Gram matrix is Hermitian.
"""

# ╔═╡ 40c67883-7c5a-4033-96df-090c58b72f65
md"""
## Examples
### Hermitian matrix
"""

# ╔═╡ 43ee6302-cd9e-472e-8c25-9c079d90bd1c
begin
	# Generating Hermitian matrix
	Random.seed!(431)
	nₕ=6
	Aₕ=rand(ComplexF64,nₕ,nₕ)
	Aₕ=Aₕ+adjoint(Aₕ)
end

# ╔═╡ f2b96941-4a21-4c1a-8c91-cd97786ff285
ishermitian(Aₕ)

# ╔═╡ 7a4e98d5-40ba-4ff1-9644-df99163e250d
# Diagonal entries
Diagonal(Aₕ)

# ╔═╡ 669c86fe-c5ac-4b6e-ac38-be628d2d38f3
# Schur decomposition
schur(Aₕ)

# ╔═╡ c19945f2-1fcc-4a33-acdf-615cc9621b12
λₕ,Uₕ=eigen(Aₕ)

# ╔═╡ e98a08fc-58af-4e07-ad77-80492f28490e
# Spectral theorem
norm(Aₕ-Uₕ*Diagonal(λₕ)*Uₕ')

# ╔═╡ 96f6750c-422d-4c2f-9719-32a3c8042181
# Spectral theorem
Aₕ-sum([λₕ[i]*Uₕ[:,i]*Uₕ[:,i]' for i=1:nₕ])

# ╔═╡ 189b6d8f-a080-473c-a9a7-0507707707c8
λₕ

# ╔═╡ a6339443-0249-4b12-b94b-1f4b2b1b6d10
begin
	# Cauchy Interlace Theorem (repeat several times)
	@show i=rand(1:nₕ)
	eigvals(Aₕ[[1:i-1;i+1:end],[1:i-1;i+1:end]])
end

# ╔═╡ b42042c0-5592-4793-b3ff-72ff269dcf6e
# Inertia
inertia(A)=[sum(eigvals(A).>0), sum(eigvals(A).<0), sum(eigvals(A).==0)]

# ╔═╡ 73eb2ab0-92f7-4d74-8960-0b11e8773f4b
inertia(Aₕ)

# ╔═╡ b6adec3b-d812-4268-9d13-9e82dcf30f22
begin
	# Similar matrices
	Mₕ=rand(ComplexF64,nₕ,nₕ)
	eigvals(Mₕ*Aₕ*inv(Mₕ))
end

# ╔═╡ 3181a27c-8360-4a08-bcb0-f6669a9ce062
md"""
This did not work numerically due to rounding errors!
"""

# ╔═╡ e439d2b6-b581-4b87-9f07-ee0e93338a5f
# Congruent matrices - this does not work either, without some preparation
inertia(A)==inertia(Mₕ'*Aₕ*Mₕ)

# ╔═╡ e7a79e8d-d7d7-4280-88b7-58d6d2f228b8
# However, 
inertia(Aₕ)==inertia(Hermitian(Mₕ'*Aₕ*Mₕ))

# ╔═╡ 1549b463-2849-4994-a2a3-45c71efda75e
# or, 
inertia((Mₕ'*Aₕ*Mₕ+(Mₕ'*Aₕ*Mₕ)')/2)

# ╔═╡ de537b01-876b-40a3-a941-4b5d8bcd2e0f
begin
	# Weyl's Inequalities
	B=rand(ComplexF64,nₕ,nₕ)
	Bₕ=(B+B')/10
	μₕ=eigvals(Bₕ)
	γₕ=eigvals(Aₕ+Bₕ)
	μₕ,γₕ
end

# ╔═╡ 5e49f18c-d633-469e-af8a-1e3877ebf07a
begin
	# Theorem uses different order!
	j=rand(1:nₕ)
	k=rand(1:nₕ)
	@show j,k
	if j+k<=n+1
	    @show sort(γₕ,rev=true)[j+k-1], sort(λₕ,rev=true)[j]+sort(μₕ,rev=true)[k]
	end
	if j+k>=n+1
	    @show sort(γₕ,rev=true)[j+k-n], sort(λₕ,rev=true)[j]+sort(μₕ,rev=true)[k]
	end
end

# ╔═╡ 1866e2e0-8766-4ebd-b045-1f7dcae30b24
sort(λₕ,rev=true)

# ╔═╡ 80c270bc-e313-4b78-94d0-2fcd9aa29a89
md"""
### Real symmetric matrix
"""

# ╔═╡ da3207e5-1cdb-474f-a5f0-11b7c7ea5640
begin
	# Generating real symmetric matrix
	Random.seed!(531)
	nₛ=5
	Aₛ=rand(-9:9,nₛ,nₛ)
	Aₛ=Aₛ+Aₛ'
end

# ╔═╡ e44e5b0c-83e0-4dec-be0d-c5d9bebc7d55
issymmetric(Aₛ)

# ╔═╡ f804922d-b411-4d6b-a973-0e9486756279
schur(Aₛ)

# ╔═╡ 157fd857-d4c8-4433-9e41-9aba559ce0ea
cond(A)

# ╔═╡ dc690338-6ba8-463d-b35b-e57b4e8b0159
inertia(Aₛ)

# ╔═╡ f8334f36-4e46-423f-8028-8a51caff6fc4
begin
	Cₛ=rand(nₛ,nₛ)
	inertia(Cₛ'*Aₛ*Cₛ)
end

# ╔═╡ 53637040-356d-4310-970e-6a9b9c1260c9
md"""
# Positive definite matrices

These matrices are an important subset of Hermitian or real symmteric matrices.

## Definitions

Matrix $A\in\mathcal{H}_n$ is __positive definite__ (PD) if $x^*Ax>0$ for all nonzero $x\in\mathbb{C}^n$.

Matrix $A\in\mathcal{H}_n$ is __positive semidefinite__ (PSD) if $x^*Ax\geq 0$ for all nonzero $x\in\mathbb{C}^n$.

## Facts

1.  $A\in\mathcal{S}_n$ is PD if $x^TAx>0$ for all nonzero $x\in \mathbb{R}^n$, and is PSD if $x^TAx\geq 0$ for all $x\in \mathbb{R}^n$.
 
2. If $A,B\in \mathrm{PSD}_n$, then $A+B\in \mathrm{PSD}_n$. If, in addition, $A\in \mathrm{PD}_n$, then $A+B\in \mathrm{PD}_n$.

3. If $A\in \mathrm{PD}_n$, then $\mathop{\mathrm{tr}}(A)>0$ and $\det(A)>0$. 

3. If $A\in \mathrm{PSD}_n$, then $\mathop{\mathrm{tr}}(A)\geq 0$ and $\det(A)\geq 0$.

4. Any principal submatrix of a PD matrix is PD. Any principal submatrix of a PSD matrix is PSD. Consequently, all minors are positive or nonnegative, respectively.

5.  $A\in\mathcal{H}_n$ is PD iff __every leading__ principal minor of $A$ is positive. $A\in\mathcal{H}_n$ is PSD iff __every__ principal minor is nonnegative.

6. For $A\in \mathrm{PSD}_n$, there exists unique PSD $k$-th __root__, $A^{1/k}=U\Lambda^{1/k} U^*$.

7. __Cholesky Factorization.__ $A\in\mathcal{H}_n$ if PD iff there is an invertible lower triangular matrix $L$ with positive diagonal entries such that $A=LL^*$.

8. Gram matrix is PSD. If the vectors are linearly independent, Gram matrix is PD.
"""

# ╔═╡ 5082d501-8e0f-45ec-8927-ad1a0be51ae3
md"""
## Examples
### Positive definite matrix
"""

# ╔═╡ e00702a4-2ad5-4dee-8d0c-7d15c4ffdeb8
begin
	# Generating positive definite matrix as a Gram matrix
	nₚ=5
	Aₚ=rand(ComplexF64,nₚ,nₚ)
	Aₚ=Aₚ*Aₚ'
end

# ╔═╡ a587ec16-fbb5-4b79-b5c8-9986f62a451e
Z=eigen(Aₚ)

# ╔═╡ 6a4a1377-ec2c-4043-9246-6d371c85a482
exp(Aₚ)

# ╔═╡ 1fe5b23c-2ee8-473d-9876-283ad9862f43
Z.vectors*Diagonal(exp.(Z.values))*Z.vectors'

# ╔═╡ fb844ff8-1696-45a8-97a9-a08ae0ace55b
ishermitian(Aₚ)

# ╔═╡ c6d47f33-1c78-4568-a0f6-e9d1deb669ca
eigvals(Aₚ)

# ╔═╡ ada85c66-6268-4664-a458-4213d7d7111f
# Positivity of principal leading minors
[det(Aₚ[1:i,1:i]) for i=1:nₚ]

# ╔═╡ a43a9532-f865-4166-b0c6-124045030e64
begin
	# Matrix function - square root
	λₚ,Uₚ=eigen(Aₚ)
	A2=Uₚ*√Diagonal(λₚ)*Uₚ'
	norm(Aₚ-A2^2)
end

# ╔═╡ 8a4c37e0-a442-46da-929c-24cdc8280313
# Cholesky factorization - lower triangular factor
Lₚ=cholesky(Aₚ).L

# ╔═╡ 1ad5826e-f849-4cef-82a1-2bd97cf38731
norm(Aₚ-Lₚ*Lₚ')

# ╔═╡ 8b3db768-e5d1-49cf-8320-7cb36a400ce1
md"
### Positive semidefinite matrix
"

# ╔═╡ 18457042-90d2-4180-a999-8d3454134528
begin
	# Generate positive semidefinite matrix as a Gram matrix, try it several times
	A₅=rand(-9:9,6,4)
	A₅=A₅*A₅'
end

# ╔═╡ 853b9c26-a15b-4109-81ee-5ce6a2dd42bd
# There are rounding errors!
eigvals(A₅)

# ╔═╡ ce5c03dd-f8a8-453d-a3b8-d618aa33be9e
rank(A₅)

# ╔═╡ 710b83d0-8d8c-11eb-39c2-e97b1880a16c
md"
How does function `rank()` work?
"

# ╔═╡ 854dfa30-8d8c-11eb-0d82-7d17e7d40d98
@which rank(A₅)

# ╔═╡ 5a229e53-a1e0-4d0f-b479-22bbd4dfaa06
s₅=svdvals(A₅)

# ╔═╡ 7056a7af-f3cf-4fa6-aa4e-b891b6444aa6
s₅[1]*eps()*n

# ╔═╡ f4b34f27-082b-4bbe-8ee0-c1066de09a9b
# Cholesky factorization can fail
cholesky(A₅)

# ╔═╡ 0528ccb5-04a8-4b8e-a0c8-d845ca716c64
md"
### Covariance and correlation matrices 

[Covariance and correlation matrices](https://en.wikipedia.org/wiki/Covariance_matrix) are PSD.

Correlation matrix is diagonally scaled covariance matrix.
"

# ╔═╡ 9daeb277-5cbb-406e-b42f-c21ddaf4366f
begin
	Random.seed!(651)
	y=rand(10,5)
end

# ╔═╡ 69346220-2310-4015-b29b-5b7186791fa8
Cov=cov(y)

# ╔═╡ 070a4816-6fe3-4a29-b1ec-b68d1500f48d
mean(y,dims=1)

# ╔═╡ 15c93185-d758-426f-a94d-cba68a1c84a1
y.-mean(y,dims=1)

# ╔═╡ 688efeb3-07da-4d75-a35b-a79f9440da76
# Covariance matrix is a Gram matrix
Cov₁=(y.-mean(y,dims=1))'*(y.-mean(y,dims=1))/(size(y,1)-1)

# ╔═╡ eb371182-ea27-4334-8cde-3fd8bf810148
norm(Cov-Cov₁)

# ╔═╡ f28e0e9a-23c5-4114-a853-2da285a24641
# Correlation matrix 
Cor=cor(y)

# ╔═╡ 867926b1-3d3e-4c7f-95cf-54ed6b54c112
# Diagonal scaling of ovariance matrix
D=inv(√Diagonal(Cov))

# ╔═╡ c5fe8288-4650-4a38-a322-38780978095d
D*Cov*D

# ╔═╡ f6be3df8-6368-4535-8085-537a80dd3ae0
eigvals(Cov)

# ╔═╡ 808904ca-b1d9-4eca-ae40-bcd53451fa83
eigvals(Cor)

# ╔═╡ c35bd54f-0f0f-4979-b842-a800eca12a6a
md"
Let us check the case when there are mor obervations than data:
"

# ╔═╡ 24fc1b19-97b4-4535-9276-c9db11549fff
z=Matrix(transpose(y))

# ╔═╡ 34001f70-e2db-4529-a315-caf38ecb87b0
Covₜ=cov(z)

# ╔═╡ aee1f31c-4650-4ead-b25e-8a728419b18c
Cov₂=(z.-mean(z,dims=1))'*(z.-mean(z,dims=1))/(size(z,1)-1)

# ╔═╡ ee1c130c-9cb1-46f3-8416-d1997d118f4d
norm(Covₜ-Cov₂)

# ╔═╡ 80a7d977-99e7-4d51-9cc9-4e6b7c410a6a
eigvals(Covₜ)

# ╔═╡ c74eb635-2948-4db3-bd53-c7d7a5b2a454
inertia(Covₜ)

# ╔═╡ cc0422bf-2e9b-463a-9b1a-468b09bb370e
rank(Covₜ)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Polynomials = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialMatrices = "928aab9d-ef52-54ac-8ca1-acd7ca42c160"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
ToeplitzMatrices = "c751599d-da0a-543b-9d20-d0a503d91d24"

[compat]
PlutoUI = "~0.7.50"
Polynomials = "~2.0.25"
SpecialMatrices = "~2.0.0"
SymPy = "~1.1.4"
ToeplitzMatrices = "~0.7.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "e9443138597cccf0a5a79156bff88223717b02f2"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "6f1d9bc1c08f9f4a8fa92e3ea3cb50153a1b40d4"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.1.0"

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
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

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
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6e47d11ea2776bc5627421d59cdcc1296c058071"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.7.0"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

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

[[deps.ExprTools]]
git-tree-sha1 = "56559bbef6ca5ea0c0818fa5c90320398a6fbf8d"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.8"

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
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "61feba885fac3a407465726d0c330b3055df897f"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.1.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Intervals]]
deps = ["Dates", "Printf", "RecipesBase", "Serialization", "TimeZones"]
git-tree-sha1 = "323a38ed1952d30586d0fe03412cde9399d3618b"
uuid = "d8418881-c3e1-53bb-8760-2df7ec849ed5"
version = "1.5.0"

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

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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
git-tree-sha1 = "58f25e56b706f95125dcb796f39e1fb01d913a71"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.10"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

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

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Mocking]]
deps = ["Compat", "ExprTools"]
git-tree-sha1 = "29714d0a7a8083bba8427a4fbfb00a540c681ce7"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.3"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "ba8c0f8732a24facba709388c74ba99dcbfdda1e"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.0"

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

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "621f4f3b4977325b9128d5fae7a8b4829a0c2222"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.4"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "5bb5129fdd62a2bbbe17c2756932259acf467386"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.50"

[[deps.Polynomials]]
deps = ["Intervals", "LinearAlgebra", "MutableArithmetics", "RecipesBase"]
git-tree-sha1 = "a1f7f4e41404bed760213ca01d7f384319f717a5"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "2.0.25"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

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

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[deps.SpecialMatrices]]
deps = ["LinearAlgebra", "Polynomials"]
git-tree-sha1 = "08c7b8ef9cbf1a33df2408756bb15b491cf5b372"
uuid = "928aab9d-ef52-54ac-8ca1-acd7ca42c160"
version = "2.0.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.SymPy]]
deps = ["CommonEq", "CommonSolve", "Latexify", "LinearAlgebra", "Markdown", "PyCall", "RecipesBase", "SpecialFunctions"]
git-tree-sha1 = "1763d267a68a4e58330925b7ce8b9ea2ec06c882"
uuid = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
version = "1.1.4"

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

[[deps.TimeZones]]
deps = ["Dates", "Downloads", "InlineStrings", "LazyArtifacts", "Mocking", "Printf", "RecipesBase", "Serialization", "Unicode"]
git-tree-sha1 = "2d4b6de8676b34525ac518de36006dc2e89c7e2e"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.7.2"

[[deps.ToeplitzMatrices]]
deps = ["AbstractFFTs", "LinearAlgebra", "StatsBase"]
git-tree-sha1 = "b61dc0269afe4c4e6109cee4d4098121bf59a8d0"
uuid = "c751599d-da0a-543b-9d20-d0a503d91d24"
version = "0.7.0"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

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
# ╠═973abf88-9125-41c7-8372-61efb2976fc2
# ╠═ec1fe200-8d73-11eb-0ba3-e593aa79dab2
# ╟─10910df9-240f-4e6b-8076-1a9be4d66ba1
# ╟─b87354b9-ee7b-4ce0-bf26-b4e71c6be6a9
# ╟─31a71b9e-1dba-4a69-b87e-9202292eb3ed
# ╟─2f5aee03-3c2e-4920-9d10-c509b06be627
# ╟─37d24090-8d78-11eb-059d-737014bb6588
# ╠═184bc4ec-c566-4f21-99b3-b6e8cd328b47
# ╠═5d0aac9e-4f48-4f09-956f-5033a1fb8f11
# ╠═73ece9ec-97dd-46bc-8e17-10ce7bfc18bc
# ╠═ce83f04c-e915-4599-beeb-d4a0a681bc17
# ╠═ff5044c0-8d76-11eb-3951-6bb4b5caa7d0
# ╠═55171a69-3557-4ba5-8a4a-813f38c03dd3
# ╠═6df42190-a29e-4c68-8684-07d8d0c9d200
# ╟─b94f7fcd-c6f4-4f4f-924a-20f00e82d94e
# ╠═1b90d50a-bf97-4b1a-9580-d729b03c914e
# ╠═15b60cd5-eb30-43bb-bbcf-dd533e29c35f
# ╟─3619fca4-1374-4adc-b29c-8058b2e147e0
# ╠═d46018c0-a5f8-4dd0-8f3d-73322be792e9
# ╠═d387b26f-bec1-47a9-b1c6-f20c43f0bde8
# ╠═703da5c4-cd09-4832-b64e-5fda75099db5
# ╠═e26f113c-dd98-4e97-b495-581de1b431af
# ╠═62779bfc-ea5d-4eec-94ad-2641e11fc44d
# ╠═bdf079a6-beee-44f5-8200-613e54b3a8dc
# ╠═e1f5027c-1080-4c95-81cc-a75dde641b88
# ╠═c71a7760-b4c6-404e-b5ef-f4879cb787cd
# ╟─343c8f45-6226-4361-a149-5313a47c355c
# ╠═41a05c50-f6df-4b93-bd8a-50412829105e
# ╠═cd8fb666-d2f2-423c-ac39-405c842e97ba
# ╠═0fe16884-eac0-496c-9ee4-9a40acb019fe
# ╟─7aa7ac89-837c-4ee8-b9f1-8ad0256f25e1
# ╠═c3724c4f-142b-4863-8c04-4a0c15411f40
# ╠═792f632e-fdc2-4016-baaa-b366704f02fd
# ╠═a8664a40-8d78-11eb-325e-41c6497be282
# ╠═1d42245a-2ed6-4cc7-8cbd-2b6aa4e64861
# ╠═f7aae752-70c4-4a10-b96e-206c13a6ced1
# ╠═60525ba5-2c76-406e-8eb3-63ced2e00bdd
# ╟─189544ef-122c-4e30-bfd6-cfc126fc0320
# ╠═da340240-f845-4591-94c3-9ffce6a33d8a
# ╠═e72bf173-7e55-4476-91ff-789847dc53cf
# ╠═da4a5d70-92d8-11eb-0635-11c530ff7d21
# ╠═ec7fc5c0-92d8-11eb-16df-af1133ccb42a
# ╠═29de5fd0-92d9-11eb-09c0-0356001caaef
# ╠═d43480a3-9247-48a4-8a3b-e006ab58539e
# ╠═5995cd31-a75d-4718-8808-4ada01920344
# ╠═57e2742e-8d79-11eb-0a33-f7250ac02840
# ╠═5e069120-8d79-11eb-3c00-0b5249d1f2dc
# ╠═63167eee-8d79-11eb-29c2-2febbd7cbd93
# ╟─e051d2eb-3972-4c97-b7eb-8a18629f6171
# ╠═22f0a627-beb2-4b06-aa94-ee1e90fe8a6c
# ╠═da6baba0-92d9-11eb-1603-7961ad1f49c8
# ╟─c6ab93ab-b336-4bb4-9dda-b725108f7827
# ╠═ffcb919f-9754-4650-92f1-7dd163c5df25
# ╠═32ce58b9-783b-4739-9001-0fe581b2953d
# ╠═a7e8d711-57a6-477a-a641-b929789dbadf
# ╠═5781508c-388a-4df0-a6f9-0fe312a81335
# ╟─90b4341d-7964-4e41-a268-8a047a058286
# ╠═76ffa073-59b8-4118-aa32-522f4ff05979
# ╠═f85ce269-7555-48ed-a2fd-1193d3f33f40
# ╠═e6544939-c8c2-464c-83a0-a6bf1751efa1
# ╠═b559d56b-02bb-47d3-8f0f-8dae09d918d8
# ╠═5e0adf59-29d3-47ac-8f3f-52929cc4e1c1
# ╠═83ba2d1b-66de-41bf-8ded-7070bed15587
# ╠═b0255c92-3242-442a-a670-1bfef38e955c
# ╠═c6116576-5b25-45b2-bc71-a889f302b5ac
# ╠═3de2dc38-ba95-479e-89bc-77cb520daf2f
# ╠═ad32c51c-7cbb-42f9-abc8-561828df1379
# ╟─a7999464-7060-4345-b396-e2e15fcbcaac
# ╟─01e2e27b-506e-439a-ba9a-bef034328482
# ╟─40c67883-7c5a-4033-96df-090c58b72f65
# ╠═43ee6302-cd9e-472e-8c25-9c079d90bd1c
# ╠═f2b96941-4a21-4c1a-8c91-cd97786ff285
# ╠═7a4e98d5-40ba-4ff1-9644-df99163e250d
# ╠═669c86fe-c5ac-4b6e-ac38-be628d2d38f3
# ╠═c19945f2-1fcc-4a33-acdf-615cc9621b12
# ╠═e98a08fc-58af-4e07-ad77-80492f28490e
# ╠═96f6750c-422d-4c2f-9719-32a3c8042181
# ╠═189b6d8f-a080-473c-a9a7-0507707707c8
# ╠═a6339443-0249-4b12-b94b-1f4b2b1b6d10
# ╠═b42042c0-5592-4793-b3ff-72ff269dcf6e
# ╠═73eb2ab0-92f7-4d74-8960-0b11e8773f4b
# ╠═b6adec3b-d812-4268-9d13-9e82dcf30f22
# ╟─3181a27c-8360-4a08-bcb0-f6669a9ce062
# ╠═e439d2b6-b581-4b87-9f07-ee0e93338a5f
# ╠═e7a79e8d-d7d7-4280-88b7-58d6d2f228b8
# ╠═1549b463-2849-4994-a2a3-45c71efda75e
# ╠═de537b01-876b-40a3-a941-4b5d8bcd2e0f
# ╠═5e49f18c-d633-469e-af8a-1e3877ebf07a
# ╠═1866e2e0-8766-4ebd-b045-1f7dcae30b24
# ╟─80c270bc-e313-4b78-94d0-2fcd9aa29a89
# ╠═da3207e5-1cdb-474f-a5f0-11b7c7ea5640
# ╠═e44e5b0c-83e0-4dec-be0d-c5d9bebc7d55
# ╠═f804922d-b411-4d6b-a973-0e9486756279
# ╠═157fd857-d4c8-4433-9e41-9aba559ce0ea
# ╠═dc690338-6ba8-463d-b35b-e57b4e8b0159
# ╠═f8334f36-4e46-423f-8028-8a51caff6fc4
# ╟─53637040-356d-4310-970e-6a9b9c1260c9
# ╟─5082d501-8e0f-45ec-8927-ad1a0be51ae3
# ╠═e00702a4-2ad5-4dee-8d0c-7d15c4ffdeb8
# ╠═a587ec16-fbb5-4b79-b5c8-9986f62a451e
# ╠═6a4a1377-ec2c-4043-9246-6d371c85a482
# ╠═1fe5b23c-2ee8-473d-9876-283ad9862f43
# ╠═fb844ff8-1696-45a8-97a9-a08ae0ace55b
# ╠═c6d47f33-1c78-4568-a0f6-e9d1deb669ca
# ╠═ada85c66-6268-4664-a458-4213d7d7111f
# ╠═a43a9532-f865-4166-b0c6-124045030e64
# ╠═8a4c37e0-a442-46da-929c-24cdc8280313
# ╠═1ad5826e-f849-4cef-82a1-2bd97cf38731
# ╟─8b3db768-e5d1-49cf-8320-7cb36a400ce1
# ╠═18457042-90d2-4180-a999-8d3454134528
# ╠═853b9c26-a15b-4109-81ee-5ce6a2dd42bd
# ╠═ce5c03dd-f8a8-453d-a3b8-d618aa33be9e
# ╟─710b83d0-8d8c-11eb-39c2-e97b1880a16c
# ╠═854dfa30-8d8c-11eb-0d82-7d17e7d40d98
# ╠═5a229e53-a1e0-4d0f-b479-22bbd4dfaa06
# ╠═7056a7af-f3cf-4fa6-aa4e-b891b6444aa6
# ╠═f4b34f27-082b-4bbe-8ee0-c1066de09a9b
# ╟─0528ccb5-04a8-4b8e-a0c8-d845ca716c64
# ╠═9daeb277-5cbb-406e-b42f-c21ddaf4366f
# ╠═69346220-2310-4015-b29b-5b7186791fa8
# ╠═070a4816-6fe3-4a29-b1ec-b68d1500f48d
# ╠═15c93185-d758-426f-a94d-cba68a1c84a1
# ╠═688efeb3-07da-4d75-a35b-a79f9440da76
# ╠═eb371182-ea27-4334-8cde-3fd8bf810148
# ╠═f28e0e9a-23c5-4114-a853-2da285a24641
# ╠═867926b1-3d3e-4c7f-95cf-54ed6b54c112
# ╠═c5fe8288-4650-4a38-a322-38780978095d
# ╠═f6be3df8-6368-4535-8085-537a80dd3ae0
# ╠═808904ca-b1d9-4eca-ae40-bcd53451fa83
# ╟─c35bd54f-0f0f-4979-b842-a800eca12a6a
# ╠═24fc1b19-97b4-4535-9276-c9db11549fff
# ╠═34001f70-e2db-4529-a315-caf38ecb87b0
# ╠═aee1f31c-4650-4ead-b25e-8a728419b18c
# ╠═ee1c130c-9cb1-46f3-8416-d1997d118f4d
# ╠═80a7d977-99e7-4d51-9cc9-4e6b7c410a6a
# ╠═c74eb635-2948-4db3-bd53-c7d7a5b2a454
# ╠═cc0422bf-2e9b-463a-9b1a-468b09bb370e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
