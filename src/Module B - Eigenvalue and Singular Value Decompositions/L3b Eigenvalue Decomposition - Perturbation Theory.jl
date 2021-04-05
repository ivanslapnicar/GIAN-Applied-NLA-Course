### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

# ╔═╡ 4ea41550-9313-11eb-1cdf-df6c5527f80f
begin
	using PlutoUI
	PlutoUI.TableOfContents(aside=true)
end

# ╔═╡ 4c360009-f01e-45bb-b458-29b15f8cac8f
begin
	using LinearAlgebra, SpecialMatrices, Statistics
	import Random
	Random.seed!(125)
	x=rand(-9:9,5)
end

# ╔═╡ 7d3ed16e-50ea-4ede-b968-e32baa56f76f
md"""
# Eigenvalue Decomposition - Perturbation Theory


__Prerequisites__

The reader should be familiar with basic linear algebra concepts and facts about eigenvalue decomposition. 

__Competences__ 

The reader should be able to understand and check the facts about perturbations of eigenvalues and eigenvectors.

"""

# ╔═╡ da0d7094-fa3e-4f1e-a6f1-9d1b9316f65a
md"""
# Norms

In order to measure changes, we need to define norms. For more details and the proofs of the Facts below, see 
[R. Byers and B. N. Datta, Vector and Matrix Norms, Error Analysis, Efficiency, and Stability, pp. 50.1-50.24](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.

## Definitions

__Norm__ on a vector space $X$ is a real-valued function $\| \phantom{x} \| : X\to \mathbb{R}$ with the following properties:

1. __Positive definiteness.__ $\| x\|\geq 0$ and $\|x\|=0$ if and only if $x$ is the zero vector.  
2. __Homogeneity.__ $\| \lambda x\|=|\lambda| \|x\|$  
3. __Triangle inequality.__ $\| x+y\| \leq \|x\|+\|y\|$ 

Commonly encountered vector norms for $x\in\mathbb{C}^n$ are:

* __Hölder norm__ or $p$-__norm__: for $p\geq 1$, $\|x\|_p=\big(|x_1|^p+|x_2|^p+\cdots |x_n|^p)^{1/p}$,
* __Sum norm__ or $1$-__norm__: $\|x\|_1=|x_1|+|x_2|+\cdots |x_n|$,
* __Euclidean norm__ or $2$-__norm__: $\|x\|_2=\sqrt{|x_1|^2+|x_2|^2+\cdots |x_n|^2}$,
* __Sup-norm__ or $\infty$-__norm__: $\|x\|_\infty = \max\limits_{i=1,\ldots,n} |x_i|$.

Vector norm is __absolute__ if $\||x|\|=\|x\|$.

Vector norm is __monotone__ if $|x|\leq |y|$ implies $\|x\|\leq \|y\|$. 

From every vector norm we can derive a corresponding __induced__ matrix norm (also, __operator norm__ or __natural norm__):

$$\|A\| = \max\limits_{x\neq 0} \frac{\|Ax\|}{\|x\|}=\max\limits_{\|x\|=1} \|Ax\|.$$

For matrix $A\in\mathbb{C}^{m\times n}$ we define:

* __Maximum absolute column sum norm__: $\|A\|_1=\max\limits_{1\leq j \leq n} \sum_{i=1}^m |a_{ij}|$,
* __Spectral norm__: $\|A\|_2=\sqrt{\rho(A^*A)}=\sigma_{\max}(A)$  (largest singular value of $A$),
* __Maximum absolute row sum norm__: $\|A\|_{\infty}=\max\limits_{1\leq i \leq m} \sum_{j=1}^n |a_{ij}|$,
* __Euclidean norm__ or __Frobenius norm__: 
$\|A\|_F =\sqrt{\sum_{i,j} |a_{ij}|^2}=\sqrt{\mathop{\mathrm{tr}}(A^*A)}$.

Matrix norm is __consistent__ if $\|A\cdot B\|\leq \|A\| \cdot \| B\|$, where $A$ and $B$ are compatible for matrix multiplication.

Matrix norm is __absolute__ if $\||A|\|=\|A\|$.
"""

# ╔═╡ f8ba4230-9313-11eb-170c-ab58597826f8
md"
### Examples
"

# ╔═╡ 7bd6f898-b86c-4670-96e8-9016666ee14f
norm(x,1), norm(x), norm(x,Inf)

# ╔═╡ 8e1ced3a-d23d-4f5f-8c65-dc13ab392504
A=rand(-4:4,5,4)

# ╔═╡ d9ef7b4b-4954-464a-96ac-e3f49800e0b7
norm(A,1), norm(A), norm(A,2), norm(A,Inf), 
opnorm(A), opnorm(A,1), opnorm(A,Inf)

# ╔═╡ 90fcd9f5-d98c-4db6-bf30-21e377209681
md"""
## Facts


1.  $\|x\|_1$, $\|x\|_2$, $\|x\|_\infty$ and $\|x\|_p$ are absolute and monotone vector norms.
2. A vector norm is absolute iff it is monotone.
3. __Convergence.__ $x_k\to x_*$ iff for any vector norm $\|x_k-x_*\|\to 0$.
4. Any two vector norms are equivalent in the sense that, for all $x$ and some $\alpha,\beta>0$

$$\alpha \|x\|_\mu \leq \|x\|_\nu \leq \beta \|x\|_\mu.$$ 

In particular:

   *  $\|x\|_2 \leq \|x\|_1\leq \sqrt{n}\|x\|_2$,
   *  $\|x\|_\infty \leq \|x\|_2\leq \sqrt{n}\|x\|_\infty$,
   *  $\|x\|_\infty \leq \|x\|_1\leq n\|x\|_\infty$.

5. __Cauchy-Schwartz inequality.__ $|x^*y|\leq \|x\|_2\|y\|_2$.
6. __Hölder inequality.__ If $p,q\geq 1$ and $\displaystyle\frac{1}{p}+\frac{1}{q}=1$, then 

$$|x^*y|\leq \|x\|_p\|y\|_q.$$

7.  $\|A\|_1$, $\|A\|_2$ and $\|A\|_\infty$ are induced by the corresponding vector norms.
8.  $\|A\|_F$ is not an induced norm.
9.  $\|A\|_1$, $\|A\|_2$, $\|A\|_\infty$ and $\|A\|_F$ are consistent.
10.  $\|A\|_1$, $\|A\|_\infty$ and $\|A\|_F$ are absolute. However, $\||A|\|_2\neq \|A\|_2$.
11. Any two matrix norms are equivalent in the sense that, for all $A$ and some $\alpha,\beta>0$

$$\alpha \|A\|_\mu \leq \|A\|_\nu \leq \beta \|A\|_\mu.$$

In particular:
   *  $\frac{1}{\sqrt{n}}\|A\|_\infty \leq \|A\|_2\leq \sqrt{m}\|A\|_\infty$,
   *  $\|A\|_2 \leq \|A\|_F\leq \sqrt{n}\|A\|_2$,
   *  $\frac{1}{\sqrt{m}}\|A\|_1 \leq \|A\|_2\leq \sqrt{n}\|A\|_1$.

12.  $\|A\|_2\leq \sqrt{\|A\|_1 \|A\|_\infty}$.
13.  $\|AB\|_F\leq \|A\|_F\|B\|_2$ and $\|AB\|_F\leq \|A\|_2\|B\|_F$.
14. If $A=xy^*$, then $\|A\|_2=\|A\|_F=\|x\|_2\|y\|_2$.
15.  $\|A^*\|_2=\|A\|_2$ and $\|A^*\|_F=\|A\|_F$.
16. For a unitary matrix $U$ of compatible dimension,

$$\|AU\|_2=\|A\|_2,\quad \|AU\|_F=\|A\|_F,\quad
\|UA\|_2=\|A\|_2,\quad  \|UA\|_F=\|A\|_F.$$

17. For $A$ square, $\rho(A)\leq\|A\|$.
18. For $A$ square, $A^k\to 0$ iff $>\rho(A)<1$.
"""

# ╔═╡ 8261b5ce-9315-11eb-2378-af63c3eb4c02
md"
### Random matrix
"

# ╔═╡ bb6eee61-c3ee-4b6a-ab44-a638fba776b8
A

# ╔═╡ 060b8d2f-e6f3-4625-9cf6-d4cc504145fa
# Absolute norms. Spectral norm is not absolute.
opnorm(A,1), opnorm(abs.(A),1), opnorm(A,Inf), opnorm(abs.(A),Inf), norm(A),
norm(abs.(A)),  opnorm(A),opnorm(abs.(A))

# ╔═╡ f55a3c60-9315-11eb-027a-dd008031f6f9
m,n=size(A)

# ╔═╡ c4a1b888-9497-4eb4-aa1c-23671123f426
# Equivalence of norms
opnorm(A,Inf)\sqrt(n),opnorm(A), sqrt(m)*opnorm(A,Inf)

# ╔═╡ b46cbb6f-36f4-4a66-b970-8fe9b0506b6a
opnorm(A), norm(A), sqrt(n)*opnorm(A)

# ╔═╡ 254b2df4-f57e-4284-a735-6fd5d4546c52
opnorm(A,1)\sqrt(m),opnorm(A), sqrt(n)*opnorm(A,1)

# ╔═╡ 09fd0923-1987-4a2e-be28-78451aa2c11a
# Fact 12
opnorm(A), sqrt(opnorm(A,1)*opnorm(A,Inf))

# ╔═╡ dcba6892-b90e-41ad-8d58-53e72d40e882
begin
	# Fact 13
	B=rand(n,rand(1:9))
	norm(A*B), norm(A)*opnorm(B), opnorm(A)*norm(B)
end

# ╔═╡ c765c02e-9316-11eb-146c-ff5d3ba81323
x

# ╔═╡ d948741f-6748-44d7-ae8c-cf074f7549bd
begin
	# Fact 14
	y=rand(ComplexF64,10)
	opnorm(x*y'), norm(x*y'), norm(x)*norm(y)
end

# ╔═╡ ec0746a1-41bd-4d61-9deb-8e227106e592
begin
	# Fact 15
	A₁=rand(-4:4,7,5)+im*rand(-4:4,7,5)
	opnorm(A₁), opnorm(A₁'), norm(A₁), norm(A₁')
end

# ╔═╡ d92799f7-1813-450b-8353-9d03078f69f0
# Unitary invariance - generate random unitary matrix U
U=qr(rand(ComplexF64,size(A₁)));

# ╔═╡ 0d58d4c4-16c6-4a5b-bdbd-f3f599d0d648
opnorm(A₁), opnorm(U.Q*A₁), norm(A₁), norm(U.Q*A₁)

# ╔═╡ b1a90c59-8c3e-4530-83ef-6b3e6c6fa687
begin
	# Spectral radius
	A₂=rand(ComplexF64,7,7)
	maximum(abs,eigvals(A₂)), opnorm(A₂,Inf), opnorm(A₂,1), opnorm(A₂), norm(A₂)
end

# ╔═╡ 31e73f82-93e7-448c-940b-5c9f5e573e5c
begin
	# Fact 18
	B₂=A₂/(maximum(abs,eigvals(A₂))+2)
	maximum(abs,eigvals(B₂)), norm(B₂^100)
end

# ╔═╡ 909033d9-bb27-4f0e-b07d-574f7fcee700
md"""
# Errors and condition numbers

We want to answer the question:

__How much the value of a function changes with respect to the change of its argument?__

## Definitions

For function $f(x)$ and argument $x$, the __absolute error__ with respect to the __perturbation__ of the argument 
$\delta x$ is 

$$
\| f(x+\delta x)-f(x)\| = \frac{\| f(x+\delta x)-f(x)\|}{\| \delta x \|} \|\delta x\| \equiv \kappa \|\delta x\|.$$

The  __condition__ or  __condition number__ $\kappa$ tells how much does the perturbation of the argument increase. (Its form resembles derivative.)

Similarly, the __relative error__ with respect to the relative perturbation of the argument is

$$
\frac{\| f(x+\delta x)-f(x)\|}{\| f(x) \|}= \frac{\| f(x+\delta x)-f(x)\|\cdot  \|x\| }{\|\delta x\| \cdot\| f(x)\|}
\cdot \frac{\|\delta x\|}{\|x\|} \equiv \kappa_{rel} \frac{\|\delta x\|}{\|x\|}.$$
"""

# ╔═╡ 7ef162c0-5c80-4e96-bcc3-125b302b1d5e
md"""
# Peturbation bounds

## Definitions

Let $A\in\mathbb{C}^{n\times n}$.

Pair $(\lambda,x)\in\mathbb{C}\times\mathbb{C}^{n\times n}$ is an __eigenpair__ of $A$ if $x\neq 0$ and $Ax=\lambda x$.

Triplet $(y,\lambda,x)\in\times\mathbb{C}^{n}\times\mathbb{C}\times\mathbb{C}^{n}$ is an __eigentriplet__ of $A$ if $x,y\neq 0$ and $Ax=\lambda x$ and $y^*A=\lambda y^*$.

__Eigenvalue matrix__ is a diagonal matrix $\Lambda=\mathop{\mathrm{diag}}(\lambda_1,\lambda_2,\ldots,\lambda_n)$.

If all eigenvalues are real, they can be increasingly ordered. $\Lambda^\uparrow$ is the eigenvalue matrix of increasingly ordered eigenvalues.

 $\tau$ is a __permutation__ of $\{1,2,\ldots,n\}$.

 $\tilde A=A+\Delta A$ is a __perturbed matrix__, where $\Delta A$ is __perturbation__. $(\tilde \lambda,\tilde x)$ are the eigenpairs of $\tilde A$.

__Condition number__ of a nonsingular matrix $X$ is $\kappa(X)=\|X\| \|X^{-1}\|$.

Let $X,Y\in\mathbb{C}^{n\times k}$ with $\mathop{\mathrm{rank}}(X)=\mathop{\mathrm{rank}}(Y)=k$. The __canonical angles__ between their column spaces, $\theta_i$, are defined by $\cos \theta_i=\sigma_i$, where $\sigma_i$ are the singular values of the matrix

$$(Y^*Y)^{-1/2}Y^*X(X^*X)^{-1/2}.$$ 

The __canonical angle matrix__ between $X$ and $Y$ is 

$$\Theta(X,Y)=\mathop{\mathrm{diag}}(\theta_1,\theta_2,\ldots,\theta_k).$$
    
"""

# ╔═╡ 3c8f71a5-0c2f-43a0-bf71-db9df707f84e
md"""
## Facts

Bounds become more strict as matrices have more structure. 
Many bounds have versions in spectral norm and Frobenius norm.
For more details and the proofs of the Facts below, see 
[R.-C. Li, Matrix Perturbation Theory, pp 21.1-21.20](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897), and the references therein.

1. There exists $\tau$ such that

$$\|\Lambda- \tilde\Lambda_\tau\|_2\leq 4(\|A\|_2+\|\tilde A\|_2)^{1-1/n}\|\Delta A\|_2^{1/n}.$$

2. __First-order perturbation bounds.__ Let $(y,\lambda,x)$ be an eigentriplet of a simple $\lambda$. $\Delta A$ changes $\lambda$ to $\tilde\lambda=\lambda+ \delta\lambda$, where

$$\delta\lambda=\frac{y^*(\Delta A)x}{y^*x}+O(\|\Delta A\|_2^2).$$

3. Let $\mu$ be a semisimple eigenvalue of $A$ with multiplicitiy $k$, and let $X,Y\in \mathbb{C}^{n\times k}$ be the matrices of the corresponding right and left eigenvectors, that is, $AX=\lambda X$ and $Y^*A=\lambda Y^*$, such that $Y^*X=I_k$. $\Delta A$ changes the $k$ copies of $\mu$ to $\tilde \mu=\mu+\delta\mu_i$, where $\delta\mu_i$ are the eigenvalues of $Y^*(\Delta A) X$ up to $O(\|\Delta A\|_2^2)$.

4. Perturbations and the inverse: if $\|A\|_p<1$, then $I-A$ is nonsingular and _(for proof see GVL p. 74 (98))_

$$(I-A)^{-1}=\sum\limits_{k=0}^\infty A^k,$$

$$\|(I-A)^{-1}\|_p \leq \frac{1}{1-\|A\|_p},\qquad 
\|(I-A)^{-1}-I\|_p \leq \frac{\|A\|_p}{1-\|A\|_p}.$$

5. __Geršgorin Circle Theorem.__ If $X^{-1} A X=D+F$, where $D=\mathop{\mathrm{diag}}(d_1,\ldots,d_n)$ and $F$ has zero diagonal entries, then _(for proof see GVL p.357 (381))_

$$\sigma(A) \subseteq \bigcup\limits_{i=1}^n D_i,$$

where 

$$D_i=\big\{z\in\mathbb{C} : |z-d_i| \leq \sum\limits_{j=1}^n |f_{ij}| \big\}.$$

Moreover, by continuity, if a connected component of $D$ consists of $k$ circles, it contains $k$ eigenvalues.

6. __Bauer-Fike Theorem.__ If $A$ is diagonalizable and $A=X\Lambda X^{-1}$ is its eigenvalue decomposition, then _(for proof see GVL p.357 (381))_

$$\max_i\min_j |\tilde \lambda_i -
\lambda_j|\leq \|X^{-1}(\Delta A)X\|_p\leq \kappa_p(X)\|\Delta A\|_p.$$

7. If $A$ and $\tilde A$ are diagonalizable, there exists $\tau$ such that

$$\|\Lambda-\tilde\Lambda_\tau\|_F\leq \sqrt{\kappa_2(X)\kappa_2(\tilde X)}\|\Delta A\|_F.$$ 

If $\Lambda$ and  $\tilde\Lambda$ are real, then

$$\|\Lambda^\uparrow-\tilde\Lambda^\uparrow\|_{2,F} \leq \sqrt{\kappa_2(X)\kappa_2(\tilde X)}\|\Delta A\|_{2,F}.$$

8. If $A$ is normal, there exists $\tau$ such that $\|\Lambda-\tilde\Lambda_\tau\|_F\leq\sqrt{n}\|\Delta A\|_F$.

9. __Hoffman-Wielandt Theorem.__ If $A$ and $\tilde A$ are normal, there exists $\tau$ such that $\|\Lambda-\tilde\Lambda_\tau\|_F\leq\|\Delta A\|_F$.

10. If $A$ and $\tilde A$ are Hermitian, for any unitarily invariant norm $\|\Lambda^\uparrow-\tilde\Lambda^\uparrow\| \leq \|\Delta A\|$. In particular,

$$\begin{aligned}
\max_i|\lambda^\uparrow_i-\tilde\lambda^\uparrow_i|&\leq \|\Delta A\|_2,\\ 
\sqrt{\sum_i(\lambda^\uparrow_i-\tilde\lambda^\uparrow_i)^2}&\leq \|\Delta A\|_F.
\end{aligned}$$

11. __Residual bounds.__ Let $A$ be Hermitian. For some $\tilde\lambda\in\mathbb{R}$ and $\tilde x\in\mathbb{C}^n$ with $\|\tilde x\|_2=1$, define __residual__ $r=A\tilde x-\tilde\lambda\tilde x$. Then $|\tilde\lambda-\lambda|\leq \|r\|_2$ for some $\lambda\in\sigma(A)$.

12. Let, in addition,  $\tilde\lambda=\tilde x^* A\tilde x$, let $\lambda$ be closest to $\tilde\lambda$ and $x$ be its unit eigenvector, and let 

$$\eta=\mathop{\mathrm{gap}}(\tilde\lambda)= \min_{\lambda\neq\mu\in\sigma(A)}|\tilde\lambda-\mu|.$$

If $\eta>0$, then

$$|\tilde\lambda-\lambda|\leq \frac{\|r\|_2^2}{\eta},\quad \sin\theta(x,\tilde x)\leq \frac{\|r\|_2}{\eta}.$$

13. Let $A$ be Hermitian, $X\in\mathbb{C}^{n\times k}$ have full column rank, and $M\in\mathcal{H}_k$ having eigenvalues $\mu_1\leq\mu_2\leq\cdots\leq\mu_k$. Set $R=AX-XM$. Then there exist $\lambda_{i_1}\leq\lambda_{i_2}\leq\cdots\leq\lambda_{i_k}\in\sigma(A)$ such that

$$
\begin{aligned}    
\max_{1\leq j\leq k} |\mu_j-\lambda_{i_j}|& \leq \frac{\|R\|_2}{\sigma_{\min}(X)},\\
\sqrt{\sum_{j=1}^k (\mu_j-\lambda_{i_j})^2}&\leq \frac{\|R\|_F}{\sigma_{\min}(X)}.
\end{aligned}$$

(The indices $i_j$ need not be the same in the above formulae.)

14. If, additionally, $X^*X=I$ and $M=X^*AX$, and if all but $k$ of $A$'s eigenvalues differ from every one of $M$'s eigenvalues by at least $\eta>0$, then

$$\sqrt{\sum_{j=1}^k (\mu_j-\lambda_{i_j})^2}\leq \frac{\|R\|_F^2}{\eta\sqrt{1-\|R\|_F^2/\eta^2}}.$$

15. Let $A=\begin{bmatrix} M & E^* \\ E & H \end{bmatrix}$ and $\tilde A=\begin{bmatrix} M & 0 \\ 0 & H \end{bmatrix}$ be Hermitian, and set $\eta=\min |\mu-\nu|$ over all $\mu\in\sigma(M)$ and $\nu\in\sigma(H)$. Then

$$\max |\lambda_j^\uparrow -\tilde\lambda_j^\uparrow| \leq \frac{2\|E\|_2^2}{\eta+\sqrt{\eta^2+4\|E\|_2^2}}.$$

16. Let 

$$\begin{bmatrix} X_1^*\\ X_2^* \end{bmatrix} A \begin{bmatrix} X_1 & X_2 \end{bmatrix}=
\begin{bmatrix} A_1 &  \\ & A_2 \end{bmatrix}, \quad \begin{bmatrix} X_1 & X_2 \end{bmatrix} \quad \textrm{unitary},
\quad X_1\in\mathbb{C}^{n\times k}.$$

Let $Q\in\mathbb{C}^{n\times k}$ have orthonormal columns and for a Hermitian $k\times k$ matrix $M$ set
$R=AQ-QM$. Let $\eta=\min|\mu-\nu|$ over all $\mu\in\sigma(M)$ and $\nu\in\sigma(A_2)$. If $\eta > 0$, then

$$\|\sin\Theta(X_1,Q)\|_F\leq \frac{\|R\|_F}{\eta}.$$
"""

# ╔═╡ 98fade4c-cdcb-495e-9328-96e86ca8e90e
md"""
## Examples 

### Nondiagonalizable matrix
"""

# ╔═╡ 8f064645-cf1f-4226-add4-53bc72665bf6
A₃=[-3 7 -1; 6 8 -2; 72 -28 19]

# ╔═╡ 568366c7-3eee-4948-a096-7bf87537412d
# (Right) eigenvectors
X₃=eigen(A₃)

# ╔═╡ fe46c02b-74bd-4303-8314-3b6e96b9e51a
cond(X₃.vectors)

# ╔═╡ c48aa813-8012-409f-9b08-ce9805da9cb3
# Left eigenvectors
Y₃=eigen(Matrix(A₃'))

# ╔═╡ f0bcadb5-882c-4e46-9079-b36097650435
cond(Y₃.vectors)

# ╔═╡ 21b412f3-f460-4f60-8b31-85c839566d62
begin
	# Try k=2,3
	k=1
	Y₃.vectors[:,k]'*A₃-Y₃.values[k]*Y₃.vectors[:,k]'
end

# ╔═╡ 6654b8d5-0c31-4a44-a16a-3f71e714ce31
begin
	ΔA₃=rand(3,3)/20
	B₃=A₃+ΔA₃
end

# ╔═╡ 3fbcbecd-6e69-49d0-94bf-06a4f6704858
norm(ΔA₃)

# ╔═╡ 210137a6-1b7b-4741-84d2-9d72e18cd999
Z₃=eigen(B₃)

# ╔═╡ e946019f-5e75-4195-8991-da371300e9a2
begin
	# Fact 2
	l=1
	Z₃.values[l]-X₃.values[l], Y₃.vectors[:,l]'*ΔA₃*X₃.vectors[:,l] /(Y₃.vectors[:,l]⋅X₃.vectors[:,l])
end

# ╔═╡ 4723a4f9-af98-4f2d-8628-0ffc58488e62
md"""
### Jordan form
"""

# ╔═╡ f3d63eb6-0174-458c-aeef-46ed6d62db7f
begin
	n₄=6
	c=0.5
	J=Bidiagonal(c*ones(n₄),ones(n₄-1),'U')
end

# ╔═╡ b9770791-4fa6-4d3f-9688-ac79b7ae7f22
# Accurately defined eigenvalues
λ₄=eigvals(J)

# ╔═╡ 8112bf44-3146-4b9f-953a-54de6f4c8c5b
# Only one eigenvector
eigvecs(J)

# ╔═╡ 6b26afc1-e427-496e-b9a5-4618551aeede
begin
	x₄=eigvecs(J)[:,1]
	y₄=eigvecs(J')[:,1]
end

# ╔═╡ b2a07659-cb08-4c53-99ac-32f817485062
y₄'*J-0.5*y₄'

# ╔═╡ 989b7c60-c673-4b77-ada8-909db2904fa2
# Just one perturbed element in the lower left corner
ΔJ=sqrt(eps())*[zeros(n₄-1);1]*Matrix(I,1,n₄)

# ╔═╡ 1da2bfee-6792-4aed-8798-91cc058cd540
μ₄=eigvals(J+ΔJ)

# ╔═╡ a6769e47-59f7-430c-9c20-a121a98f19bf
# Fact 2
maximum(abs,λ₄-μ₄)

# ╔═╡ fab42c40-7d01-4787-a491-3d37cae1768d
y₄'*ΔJ*x₄/(y₄⋅x₄)

# ╔═╡ 2f783e6e-5b29-477f-8b45-00461013c66f
md"""
However, since $J+\Delta J$ is diagonalizable, we can apply Bauer-Fike theorem to it: 
"""

# ╔═╡ 9385e7e0-17d6-4840-98a2-fa69d53ebea3
Y₄=eigvecs(J+ΔJ)

# ╔═╡ 18e6478b-bb8e-482c-bd02-0c85bca6fc7f
cond(Y₄)

# ╔═╡ ae524f6a-3e64-49ea-92e3-671559f85a5f
opnorm(inv(Y₄)*ΔJ*Y₄), cond(Y₄)*opnorm(ΔJ)

# ╔═╡ 9c0c9fcf-9581-479f-8ec5-d885dd3e6665
md"""
### Normal matrix
"""

# ╔═╡ 300e0ddd-3ab0-4803-9528-665e754e54ca
begin
	n₅=5
	C=Circulant(rand(-5:5,n₅))
end

# ╔═╡ fdecade7-4d5c-460f-a566-d52868c75ec0
eigvals(Matrix(C))

# ╔═╡ 01250371-f8ab-4a0a-b283-ffbcffdc2929
ΔC=randn(n₅,n₅)*0.0001

# ╔═╡ 0787f637-44bf-4c5c-b8d2-92df842a91b5
opnorm(ΔC), eigvals(C+ΔC)

# ╔═╡ 5645f7ea-50e5-43f7-afc3-1c46fbdd7c5b
md"""
### Hermitian matrix
"""

# ╔═╡ 51cd2fc5-e295-4b76-9fd1-fdd2b92ce05a
begin
	# Random matrix with strong column scaling
	m₆=10
	n₆=6
	D₆=Diagonal(exp.(10*randn(n₆)))
	A₆=cor(randn(m₆,n₆)*D₆)
end

# ╔═╡ bc5a0ee0-931f-11eb-204c-b9e03584c787
D₆

# ╔═╡ 1eb12479-0662-410b-9569-c4e8939a291f
ΔA₆=cor(rand(m₆,n₆)*D₆)*1e-5

# ╔═╡ 821f6b8b-0e12-404b-b1dc-57d809ebf79b
begin
	λ₆,U₆=eigen(A₆) 
	μ₆=eigvals(A₆+ΔA₆)
	[λ₆ μ₆]
end

# ╔═╡ ac3d3677-1160-4a79-a25f-8faa7fd89940
norm(ΔA₆)

# ╔═╡ f4051f63-a0d3-4b43-8c61-2bbd08338a72
# ?round

# ╔═╡ 612870ba-75df-4729-a16e-432f70707fb4
begin
	# Residual bounds - how close is μ, y to λ[2],X[:,2]
	k₆=3
	ζ₆=round(λ₆[k₆],sigdigits=2)
	y₆=round.(U₆[:,k₆],sigdigits=2)
	normalize!(y₆)
end

# ╔═╡ 3ba55b4b-5445-48a3-b327-b7d3af28397a
ζ₆

# ╔═╡ 6f3da676-c953-4d23-8b9a-bb62b8b8ac94
# Fact 9
r₆=A₆*y₆-ζ₆*y₆

# ╔═╡ 886ce4f8-73a0-4b57-8d5e-082bcfc1f9aa
minimum(abs,ζ₆.-λ₆), norm(r₆)

# ╔═╡ 82a4adaa-3663-4200-9f15-d09fd4cd0259
begin
	# Fact 10 - μ is Rayleigh quotient
	ξ₆=y₆⋅(A₆*y₆)
	ρ₆=A₆*y₆-ξ₆*y₆
end

# ╔═╡ b1025690-4244-4933-9693-9bbd3ad8aedb
η₆=min(abs(ξ₆-λ₆[k₆-1]),abs(ξ₆-λ₆[k₆+1]))

# ╔═╡ 5e69cab3-712c-4754-b6b1-579bc3c8e114
ξ₆-λ₆[k₆], norm(ρ₆)^2/η₆

# ╔═╡ 3792c478-fbaa-4683-849a-ca0d0cb751f3
begin
	# Eigenvector bound
	# cos(θ)
	cosθ=dot(y₆,U₆[:,k₆])
	# sin(θ)
	sinθ=sqrt(1-cosθ^2)
	sinθ,norm(ρ₆)/η₆
end

# ╔═╡ 0dd080f1-575d-43e0-9285-20279f390d2f
begin
	# Residual bounds - Fact 13
	Q₆=round.(U₆[:,1:3],sigdigits=2)
	# Orthogonalize
	F=qr(Q₆)
	X=Matrix(F.Q)
	# Make sure M₆ is hermitian
	M₆=Hermitian(X'*A₆*X)
	μₘ=eigvals(M₆)
	R₆=A₆*X-X*M₆
end

# ╔═╡ ea07eea3-bab7-4add-90cc-43acd7c2ec89
λ₆

# ╔═╡ 53f2860a-db33-49cd-b95e-3086f76c2592
μₘ

# ╔═╡ 8cdffcc2-c8c1-410e-95db-6327931782a1
M₆

# ╔═╡ 0e871e72-47ee-4492-9f73-fcfd2ef1aada
# The entries of μ are not ordered - which algorithm was called?
issymmetric(M₆)

# ╔═╡ 108f73bc-b1cf-47ed-80f0-281cb5ebd048
ηₘ=λ₆[4]-λ₆[3]

# ╔═╡ dbc219f9-9fcf-4aac-b7d5-feb8fcafe3e9
norm(λ₆[1:3]-μₘ), norm(R₆)^2/ηₘ

# ╔═╡ 7cae5501-8a51-4874-8b23-5825f79624e3
A₆

# ╔═╡ 7cb49041-f326-4b2c-90f8-99c4d58178eb
begin
	# Fact 15
	M₇=A₆[1:3,1:3]
	H=A₆[4:6,4:6]
	E=A₆[4:6,1:3]
	# Block-diagonal matrix
	B₇=cat(M₇,H,dims=(1,2))
end

# ╔═╡ 75c00141-e691-4f9c-8b7a-ed262b88a62d
begin
	η₇=minimum(abs,eigvals(M₇)-eigvals(H))
	μ₇=eigvals(B₇)
	[λ₆ μ₇]
end

# ╔═╡ 433a0652-912f-4f59-8494-4d0381e22115
2*norm(E)^2/(η₇+sqrt(η₇^2+4*norm(E)^2))

# ╔═╡ ad6ca597-65b2-4403-a6aa-1394399c3ed8
begin
	# Eigenspace bounds - Fact 16
	B₆=A₆+ΔA₆
	μ,V=eigen(B₆)
end

# ╔═╡ 4727c0ba-124e-41a4-861a-57eb5a6d849d
begin
	# sin(Θ(U[:,1:3],V[:,1:3]))
	X₆=U₆[:,1:3]
	Q=V[:,1:3]
	cosθ₆=svdvals(sqrt(Q'*Q)*Q'*X₆*sqrt(X₆'*X₆))
	sinθ₆=sqrt.(1 .-cosθ₆.^2)
end

# ╔═╡ dd00cbef-e1f6-43bc-95b0-1b6694c23583
# Bound
M₈=Q'*A₆*Q

# ╔═╡ 99f44a10-b3e1-442c-a042-b2f0afad1797
R₈=A₆*Q-Q*M₈

# ╔═╡ b6f8f48e-2e80-4999-95a9-701d491411bd
eigvals(M₈), λ₆

# ╔═╡ 7a6554bb-0518-4068-9784-678f846d3b3f
begin
	η₈=abs(eigvals(M₈)[3]-λ₆[4])
	norm(sinθ₆), norm(R₈)/η₈
end

# ╔═╡ Cell order:
# ╟─4ea41550-9313-11eb-1cdf-df6c5527f80f
# ╟─7d3ed16e-50ea-4ede-b968-e32baa56f76f
# ╟─da0d7094-fa3e-4f1e-a6f1-9d1b9316f65a
# ╟─f8ba4230-9313-11eb-170c-ab58597826f8
# ╠═4c360009-f01e-45bb-b458-29b15f8cac8f
# ╠═7bd6f898-b86c-4670-96e8-9016666ee14f
# ╠═8e1ced3a-d23d-4f5f-8c65-dc13ab392504
# ╠═d9ef7b4b-4954-464a-96ac-e3f49800e0b7
# ╟─90fcd9f5-d98c-4db6-bf30-21e377209681
# ╟─8261b5ce-9315-11eb-2378-af63c3eb4c02
# ╠═bb6eee61-c3ee-4b6a-ab44-a638fba776b8
# ╠═060b8d2f-e6f3-4625-9cf6-d4cc504145fa
# ╠═f55a3c60-9315-11eb-027a-dd008031f6f9
# ╠═c4a1b888-9497-4eb4-aa1c-23671123f426
# ╠═b46cbb6f-36f4-4a66-b970-8fe9b0506b6a
# ╠═254b2df4-f57e-4284-a735-6fd5d4546c52
# ╠═09fd0923-1987-4a2e-be28-78451aa2c11a
# ╠═dcba6892-b90e-41ad-8d58-53e72d40e882
# ╠═c765c02e-9316-11eb-146c-ff5d3ba81323
# ╠═d948741f-6748-44d7-ae8c-cf074f7549bd
# ╠═ec0746a1-41bd-4d61-9deb-8e227106e592
# ╠═d92799f7-1813-450b-8353-9d03078f69f0
# ╠═0d58d4c4-16c6-4a5b-bdbd-f3f599d0d648
# ╠═b1a90c59-8c3e-4530-83ef-6b3e6c6fa687
# ╠═31e73f82-93e7-448c-940b-5c9f5e573e5c
# ╟─909033d9-bb27-4f0e-b07d-574f7fcee700
# ╟─7ef162c0-5c80-4e96-bcc3-125b302b1d5e
# ╟─3c8f71a5-0c2f-43a0-bf71-db9df707f84e
# ╟─98fade4c-cdcb-495e-9328-96e86ca8e90e
# ╠═8f064645-cf1f-4226-add4-53bc72665bf6
# ╠═568366c7-3eee-4948-a096-7bf87537412d
# ╠═fe46c02b-74bd-4303-8314-3b6e96b9e51a
# ╠═c48aa813-8012-409f-9b08-ce9805da9cb3
# ╠═f0bcadb5-882c-4e46-9079-b36097650435
# ╠═21b412f3-f460-4f60-8b31-85c839566d62
# ╠═6654b8d5-0c31-4a44-a16a-3f71e714ce31
# ╠═3fbcbecd-6e69-49d0-94bf-06a4f6704858
# ╠═210137a6-1b7b-4741-84d2-9d72e18cd999
# ╟─e946019f-5e75-4195-8991-da371300e9a2
# ╟─4723a4f9-af98-4f2d-8628-0ffc58488e62
# ╠═f3d63eb6-0174-458c-aeef-46ed6d62db7f
# ╠═b9770791-4fa6-4d3f-9688-ac79b7ae7f22
# ╠═8112bf44-3146-4b9f-953a-54de6f4c8c5b
# ╠═6b26afc1-e427-496e-b9a5-4618551aeede
# ╠═b2a07659-cb08-4c53-99ac-32f817485062
# ╠═989b7c60-c673-4b77-ada8-909db2904fa2
# ╠═1da2bfee-6792-4aed-8798-91cc058cd540
# ╠═a6769e47-59f7-430c-9c20-a121a98f19bf
# ╠═fab42c40-7d01-4787-a491-3d37cae1768d
# ╟─2f783e6e-5b29-477f-8b45-00461013c66f
# ╠═9385e7e0-17d6-4840-98a2-fa69d53ebea3
# ╠═18e6478b-bb8e-482c-bd02-0c85bca6fc7f
# ╠═ae524f6a-3e64-49ea-92e3-671559f85a5f
# ╟─9c0c9fcf-9581-479f-8ec5-d885dd3e6665
# ╠═300e0ddd-3ab0-4803-9528-665e754e54ca
# ╠═fdecade7-4d5c-460f-a566-d52868c75ec0
# ╠═01250371-f8ab-4a0a-b283-ffbcffdc2929
# ╠═0787f637-44bf-4c5c-b8d2-92df842a91b5
# ╟─5645f7ea-50e5-43f7-afc3-1c46fbdd7c5b
# ╠═51cd2fc5-e295-4b76-9fd1-fdd2b92ce05a
# ╠═bc5a0ee0-931f-11eb-204c-b9e03584c787
# ╠═1eb12479-0662-410b-9569-c4e8939a291f
# ╠═821f6b8b-0e12-404b-b1dc-57d809ebf79b
# ╠═ac3d3677-1160-4a79-a25f-8faa7fd89940
# ╠═f4051f63-a0d3-4b43-8c61-2bbd08338a72
# ╠═612870ba-75df-4729-a16e-432f70707fb4
# ╠═3ba55b4b-5445-48a3-b327-b7d3af28397a
# ╠═6f3da676-c953-4d23-8b9a-bb62b8b8ac94
# ╠═886ce4f8-73a0-4b57-8d5e-082bcfc1f9aa
# ╠═82a4adaa-3663-4200-9f15-d09fd4cd0259
# ╠═b1025690-4244-4933-9693-9bbd3ad8aedb
# ╠═5e69cab3-712c-4754-b6b1-579bc3c8e114
# ╠═3792c478-fbaa-4683-849a-ca0d0cb751f3
# ╠═0dd080f1-575d-43e0-9285-20279f390d2f
# ╠═ea07eea3-bab7-4add-90cc-43acd7c2ec89
# ╠═53f2860a-db33-49cd-b95e-3086f76c2592
# ╠═8cdffcc2-c8c1-410e-95db-6327931782a1
# ╠═0e871e72-47ee-4492-9f73-fcfd2ef1aada
# ╠═108f73bc-b1cf-47ed-80f0-281cb5ebd048
# ╠═dbc219f9-9fcf-4aac-b7d5-feb8fcafe3e9
# ╠═7cae5501-8a51-4874-8b23-5825f79624e3
# ╠═7cb49041-f326-4b2c-90f8-99c4d58178eb
# ╠═75c00141-e691-4f9c-8b7a-ed262b88a62d
# ╠═433a0652-912f-4f59-8494-4d0381e22115
# ╠═ad6ca597-65b2-4403-a6aa-1394399c3ed8
# ╠═4727c0ba-124e-41a4-861a-57eb5a6d849d
# ╠═dd00cbef-e1f6-43bc-95b0-1b6694c23583
# ╠═99f44a10-b3e1-442c-a042-b2f0afad1797
# ╠═b6f8f48e-2e80-4999-95a9-701d491411bd
# ╠═7a6554bb-0518-4068-9784-678f846d3b3f
