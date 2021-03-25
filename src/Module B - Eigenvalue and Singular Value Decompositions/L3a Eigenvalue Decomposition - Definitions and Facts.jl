### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ ec1fe200-8d73-11eb-0ba3-e593aa79dab2
begin
	using PlutoUI
	PlutoUI.TableOfContents(aside=true)
end

# ╔═╡ 973abf88-9125-41c7-8372-61efb2976fc2
using SymPy

# ╔═╡ da340240-f845-4591-94c3-9ffce6a33d8a
begin
	using Random
	Random.seed!(123)
	A₄=rand(-4:4,4,4)
end

# ╔═╡ 60d351a5-0ef2-47cd-a5a5-5ecf18c4e898
begin
	using SpecialMatrices
	using Polynomials
end

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
## General matrices

For more details and the proofs of the Facts below, see 
[L. M. DeAlba, Determinants and Eigenvalues, pp. 4.1-1.15](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.


### Definitions

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
### Facts

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

12. Every $A$ has Schur decomposition. Moreover, $T_{ii}=\lambda_i$.

14. If $A$ is normal, matrix $T$ from its Schur decomposition is normal. Consequently:

    *  $T$ is diagonal, and has eigenvalues of $A$ on diagonal,
    * matrix $Q$ of the Schur decomposition is the unitary matrix of eigenvectors,
    * all eigenvalues of $A$ are semisimple and $A$ is nondefective.

13. If $A$ and $B$ are similar, $\sigma(A)=\sigma(B)$. Consequently, $\mathop{\mathrm{tr}}(A)=\mathop{\mathrm{tr}}(B)$ and $\det(A)=\det(B)$.

11. Eigenvalues and eigenvectors are continous and differentiable: if $\lambda$ is a simple eigenvalue of $A$ and $A(\varepsilon)=A+\varepsilon E$ for some $E\in F^{n\times n}$, for small $\varepsilon$ there exist differentiable functions $\lambda(\varepsilon)$ and $x(\varepsilon)$ such that $A(\varepsilon) x(\varepsilon) = \lambda(\varepsilon) x(\varepsilon)$.

16. Classical motivation for the eigenvalue problem is the following: consider the system of linear differential equations with constant coefficients, $\dot y(t)=Ay(t)$. If the solution is $y=e^{\lambda t} x$ for some constant vector $x$, then

$$\lambda e^{\lambda t} x=Ae^{\lambda t} x \quad \textrm{or} \quad Ax=\lambda x.$$

"""

# ╔═╡ 2f5aee03-3c2e-4920-9d10-c509b06be627
md"""
### Examples

We shall illustrate above Definitions and Facts on several small examples, using symbolic computation.
"""

# ╔═╡ 37d24090-8d78-11eb-059d-737014bb6588
md"
#### Defective eigenvalue
"

# ╔═╡ 184bc4ec-c566-4f21-99b3-b6e8cd328b47
A=[-3 7 -1; 6 8 -2; 72 -28 19]

# ╔═╡ 73ece9ec-97dd-46bc-8e17-10ce7bfc18bc
begin
	using LinearAlgebra
	eye(n)=Matrix{Int}(I,n,n)
	A-x*eye(3)
end

# ╔═╡ 5d0aac9e-4f48-4f09-956f-5033a1fb8f11
@vars x

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
nullspace(map(Float64,A-λ[1]*I))

# ╔═╡ 15b60cd5-eb30-43bb-bbcf-dd533e29c35f
nullspace(map(Float64,A-λ[2]*I))

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

# ╔═╡ 7aa7ac89-837c-4ee8-b9f1-8ad0256f25e1
md"""
#### Diagonalizable matrix

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
nullspace(map(Float64,A₁-λ₁[2]*I))

# ╔═╡ 60525ba5-2c76-406e-8eb3-63ced2e00bdd
# F=schur(A)
schur(A₁).T

# ╔═╡ 189544ef-122c-4e30-bfd6-cfc126fc0320
md"""
#### Symbolic computation for $n=4$

Let us try some random examples of dimension $n=4$ (the largest $n$ for which we can compute eigevalues symbolically).
"""

# ╔═╡ e72bf173-7e55-4476-91ff-789847dc53cf
p₄(x)=factor(det(A₄-x*eye(4)))

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

Symbolic computation does not work well with floating-point numbers - in the following example, the degree of $p_A(x)$ is 8 instead of 4:
"""

# ╔═╡ 22f0a627-beb2-4b06-aa94-ee1e90fe8a6c
begin
	A₂=rand(4,4)
    factor(det(A₂-x*eye(4)))
end

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
#### Circulant matrix

For more details, see 
[A. B&ouml;ttcher and I. Spitkovsky, Special Types of Matrices, pp. 22.1-22.20](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.


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

We shall use the package [SpecialMatrices.jl](https://github.com/JuliaMatrices/SpecialMatrices.jl).
"""

# ╔═╡ 16795c64-2994-4fca-89f3-f035fb6a2e45
# using Pkg; Pkg.add(PackageSpec(name="SpecialMatrices",rev="master"))

# ╔═╡ 76ffa073-59b8-4118-aa32-522f4ff05979
varinfo(SpecialMatrices)

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

## Hermitian and real symmetric matrices

For more details and the proofs of the Facts below, see 
[W. Barrett, Hermitian and Positive Definite Matrices, pp. 9.1-9.13](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.

### Definitions

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
### Facts

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
###  Example of a Hermitian matrix
"""

# ╔═╡ 43ee6302-cd9e-472e-8c25-9c079d90bd1c
begin
	# Generating Hermitian matrix
	Random.seed!(432)
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
### Example of a real symmetric matrix
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
## Positive definite matrices

These matrices are an important subset of Hermitian or real symmteric matrices.

### Definitions

Matrix $A\in\mathcal{H}_n$ is __positive definite__ (PD) if $x^*Ax>0$ for all nonzero $x\in\mathbb{C}^n$.

Matrix $A\in\mathcal{H}_n$ is __positive semidefinite__ (PSD) if $x^*Ax\geq 0$ for all nonzero $x\in\mathbb{C}^n$.

### Facts

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
### Example of a positive definite matrix
"""

# ╔═╡ e00702a4-2ad5-4dee-8d0c-7d15c4ffdeb8
begin
	# Generating positive definite matrix as a Gram matrix
	nₚ=5
	Aₚ=rand(ComplexF64,nₚ,nₚ)
	Aₚ=Aₚ*Aₚ'
end

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
	Aₚ-A2^2
end

# ╔═╡ 8a4c37e0-a442-46da-929c-24cdc8280313
# Cholesky factorization - lower triangular factor
Lₚ=cholesky(Aₚ).L

# ╔═╡ 1ad5826e-f849-4cef-82a1-2bd97cf38731
norm(Aₚ-Lₚ*Lₚ')

# ╔═╡ 8b3db768-e5d1-49cf-8320-7cb36a400ce1
md"
### Example of a positive semidefinite matrix
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
s₅[1]*eps()

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
begin
	using Statistics
	Cov=cov(y)
end

# ╔═╡ 688efeb3-07da-4d75-a35b-a79f9440da76
# Covariance matrix is a Gram matrix
(y.-mean(y,dims=1))'*(y.-mean(y,dims=1))/(size(y,1)-1)-Cov

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

# ╔═╡ 34001f70-e2db-4529-a315-caf38ecb87b0
Covₜ=cov(y')

# ╔═╡ 80a7d977-99e7-4d51-9cc9-4e6b7c410a6a
eigvals(Covₜ)

# ╔═╡ c74eb635-2948-4db3-bd53-c7d7a5b2a454
inertia(Covₜ)

# ╔═╡ cc0422bf-2e9b-463a-9b1a-468b09bb370e
rank(Covₜ)

# ╔═╡ Cell order:
# ╟─10910df9-240f-4e6b-8076-1a9be4d66ba1
# ╟─ec1fe200-8d73-11eb-0ba3-e593aa79dab2
# ╟─b87354b9-ee7b-4ce0-bf26-b4e71c6be6a9
# ╟─31a71b9e-1dba-4a69-b87e-9202292eb3ed
# ╟─2f5aee03-3c2e-4920-9d10-c509b06be627
# ╠═973abf88-9125-41c7-8372-61efb2976fc2
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
# ╠═d43480a3-9247-48a4-8a3b-e006ab58539e
# ╠═5995cd31-a75d-4718-8808-4ada01920344
# ╠═57e2742e-8d79-11eb-0a33-f7250ac02840
# ╠═5e069120-8d79-11eb-3c00-0b5249d1f2dc
# ╠═63167eee-8d79-11eb-29c2-2febbd7cbd93
# ╟─e051d2eb-3972-4c97-b7eb-8a18629f6171
# ╠═22f0a627-beb2-4b06-aa94-ee1e90fe8a6c
# ╟─c6ab93ab-b336-4bb4-9dda-b725108f7827
# ╠═ffcb919f-9754-4650-92f1-7dd163c5df25
# ╠═32ce58b9-783b-4739-9001-0fe581b2953d
# ╠═a7e8d711-57a6-477a-a641-b929789dbadf
# ╠═5781508c-388a-4df0-a6f9-0fe312a81335
# ╟─90b4341d-7964-4e41-a268-8a047a058286
# ╠═16795c64-2994-4fca-89f3-f035fb6a2e45
# ╠═60d351a5-0ef2-47cd-a5a5-5ecf18c4e898
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
# ╠═688efeb3-07da-4d75-a35b-a79f9440da76
# ╠═f28e0e9a-23c5-4114-a853-2da285a24641
# ╠═867926b1-3d3e-4c7f-95cf-54ed6b54c112
# ╠═c5fe8288-4650-4a38-a322-38780978095d
# ╠═f6be3df8-6368-4535-8085-537a80dd3ae0
# ╠═808904ca-b1d9-4eca-ae40-bcd53451fa83
# ╠═34001f70-e2db-4529-a315-caf38ecb87b0
# ╠═80a7d977-99e7-4d51-9cc9-4e6b7c410a6a
# ╠═c74eb635-2948-4db3-bd53-c7d7a5b2a454
# ╠═cc0422bf-2e9b-463a-9b1a-468b09bb370e
