### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 973abf88-9125-41c7-8372-61efb2976fc2
using PlutoUI, SymPyPythonCall, Random, LinearAlgebra, Polynomials, ToeplitzMatrices, SpecialMatrices , Statistics

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

# ╔═╡ ef7b4ce4-1d60-4d09-9b98-fc762fc92618
float(A-λ[1]*I)

# ╔═╡ 1b90d50a-bf97-4b1a-9580-d729b03c914e
nullspace(float(A-λ[1]*I))

# ╔═╡ 15b60cd5-eb30-43bb-bbcf-dd533e29c35f
nullspace(float(A-λ[2]*I))

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
nullspace(float(A₁-λ₁[2]*I))

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
	Fₙ=Vandermonde(v)'/√n
end

# ╔═╡ 3de2dc38-ba95-479e-89bc-77cb520daf2f
Λᵪ=Fₙ'*C*Fₙ

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
SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
ToeplitzMatrices = "c751599d-da0a-543b-9d20-d0a503d91d24"

[compat]
PlutoUI = "~0.7.58"
Polynomials = "~3.2.13"
SpecialMatrices = "~3.0.0"
SymPyPythonCall = "~0.2.5"
ToeplitzMatrices = "~0.8.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.10.2"
manifest_format = "2.0"
project_hash = "f4ee287a0546795afbc316c0efd7e08fa6fa38e0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

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

[[deps.CommonEq]]
git-tree-sha1 = "6b0f0354b8eb954cdba708fb262ef00ee7274468"
uuid = "3709ef60-1bee-4518-9f2f-acd86f176c50"
version = "0.2.1"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "c955881e3c981181362ae4088b35995446298b80"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.14.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.0+0"

[[deps.CondaPkg]]
deps = ["JSON3", "Markdown", "MicroMamba", "Pidfile", "Pkg", "Preferences", "TOML"]
git-tree-sha1 = "e81c4263c7ef4eca4d645ef612814d72e9255b41"
uuid = "992eb4ea-22a4-4c89-a5bb-47a3300528ab"
version = "0.2.22"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "f7f4319567fe769debfcf7f8c03d8da1dd4e2fb0"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.9"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Format]]
git-tree-sha1 = "f3cf88025f6d03c194d73f5d13fee9004a108329"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.6"

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

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5fdf2fe6724d8caabf43b557b84ce53f3b7e2f6b"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.0.2+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "eb3edce0ed4fa32f75a0a11217433c31d56bd48b"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.0"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "cad560042a7cc108f5a4c24ea1431a9221f22c1b"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.2"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "18144f3e9cbe9b15b070288eef858f71b291ce37"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.27"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "72dc3cf284559eb8f53aa593fe62cb33f83ed0c0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.0.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+1"

[[deps.MicroMamba]]
deps = ["Pkg", "Scratch", "micromamba_jll"]
git-tree-sha1 = "011cab361eae7bcd7d278f0a7a00ff9c69000c51"
uuid = "0b3b1443-0f03-428d-bdfb-f27f9c1191ea"
version = "0.1.14"

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

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pidfile]]
deps = ["FileWatching", "Test"]
git-tree-sha1 = "2d8aaf8ee10df53d0dfb9b8ee44ae7c04ced2b03"
uuid = "fa939f87-e72e-5be4-a000-7fc836dbe307"
version = "1.3.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.10.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "71a22244e352aa8c5f0f2adde4150f62368a3f2e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.58"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase"]
git-tree-sha1 = "3aa2bb4982e575acd7583f01531f241af077b163"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "3.2.13"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

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

[[deps.PythonCall]]
deps = ["CondaPkg", "Dates", "Libdl", "MacroTools", "Markdown", "Pkg", "REPL", "Requires", "Serialization", "Tables", "UnsafePointers"]
git-tree-sha1 = "0fe6664f742903eab8929586af78e10a51b33577"
uuid = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
version = "0.9.19"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

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

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.10.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.SpecialMatrices]]
deps = ["LinearAlgebra", "Polynomials"]
git-tree-sha1 = "8fd75ee3d16a83468a96fd29a1913fb170d2d2fd"
uuid = "928aab9d-ef52-54ac-8ca1-acd7ca42c160"
version = "3.0.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.10.0"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "ca4bccb03acf9faaf4137a9abc1881ed1841aa70"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.10.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.2.1+1"

[[deps.SymPyCore]]
deps = ["CommonEq", "CommonSolve", "Latexify", "LinearAlgebra", "Markdown", "RecipesBase", "SpecialFunctions"]
git-tree-sha1 = "4c5a53625f0e53ce1e726a6dab1c870017303728"
uuid = "458b697b-88f0-4a86-b56b-78b75cfb3531"
version = "0.1.16"

    [deps.SymPyCore.extensions]
    SymPyCoreSymbolicUtilsExt = "SymbolicUtils"

    [deps.SymPyCore.weakdeps]
    SymbolicUtils = "d1185830-fcd6-423d-90d6-eec64667417b"

[[deps.SymPyPythonCall]]
deps = ["CommonEq", "CommonSolve", "CondaPkg", "LinearAlgebra", "PythonCall", "SpecialFunctions", "SymPyCore"]
git-tree-sha1 = "1948385c5c0f0659ca3abcdea214318d691b1770"
uuid = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"
version = "0.2.5"

    [deps.SymPyPythonCall.extensions]
    SymPyPythonCallSymbolicsExt = "Symbolics"

    [deps.SymPyPythonCall.weakdeps]
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ToeplitzMatrices]]
deps = ["AbstractFFTs", "DSP", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "df4e499f6321e72f801aab45336ba76ed06e97db"
uuid = "c751599d-da0a-543b-9d20-d0a503d91d24"
version = "0.8.3"

    [deps.ToeplitzMatrices.extensions]
    ToeplitzMatricesStatsBaseExt = "StatsBase"

    [deps.ToeplitzMatrices.weakdeps]
    StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

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

[[deps.UnsafePointers]]
git-tree-sha1 = "c81331b3b2e60a982be57c046ec91f599ede674a"
uuid = "e17b2a0c-0bdf-430a-bd0c-3a23cae4ff39"
version = "1.0.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+1"

[[deps.micromamba_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "66d07957bcf7e4930d933195aed484078dd8cbb5"
uuid = "f8abcde7-e9b7-5caa-b8af-a437887ae8e4"
version = "1.4.9+0"

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
# ╠═ef7b4ce4-1d60-4d09-9b98-fc762fc92618
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
