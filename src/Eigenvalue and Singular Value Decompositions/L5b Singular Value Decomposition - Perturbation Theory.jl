### A Pluto.jl notebook ###
# v0.19.41

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 4d0ff51d-5cd6-4f47-9c48-66055deabde5
using PlutoUI, LinearAlgebra, Random

# ╔═╡ 7d4b46fd-de3f-4211-a27a-3b92b2a1e873
TableOfContents(aside=true)

# ╔═╡ b34474bc-3dd2-491d-a9aa-8d9402e2e880
md"""
# Singular Value Decomposition - Perturbation Theory

__Prerequisites__

The reader should be familiar with eigenvalue decomposition, singular value decompostion, and perturbation theory for eigenvalue decomposition.

__Competences__

The reader should be able to understand and check the facts about perturbations of singular values and vectors.
"""

# ╔═╡ 872db817-6bee-42dd-b50d-32c82534f993
md"""
# Peturbation bounds

For more details and the proofs of the Facts below, see [R.-C. Li, Matrix Perturbation Theory, pp. 21.6-21.8](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.

## Definitions

Let $A\in\mathbb{C}^{m\times n}$ and let $A=U\Sigma V^*$ be its SVD.

The set of $A$'s singular values is $sv(B)=\{\sigma_1,\sigma_2,\ldots)$, with  $\sigma_1\geq \sigma_2\geq \cdots\geq 0$, and let $sv_{ext}(B)=sv(B)$ unless $m>n$ for which $sv_{ext}(B)=sv(B)\cup \{0,\ldots,0\}$ (additional $|m-n|$ zeros).

Triplet $(u,\sigma,v)\in\times\mathbb{C}^{m}\times\mathbb{R}\times\mathbb{C}^{n}$ is a __singular triplet__ of $A$ if $\|u\|_2=1$, $\|v\|_2=1$, $\sigma\geq 0$, and $Av=\sigma u$ and $A^*u=\sigma v$.

 $\tilde A=A+\Delta A$ is a __perturbed matrix__, where $\Delta A$ is __perturbation__. _The same notation is adopted to $\tilde A$, except all symbols are with tildes._

__Spectral condition number__ of $A$ is $\kappa_2(A)=\sigma_{\max}(A)/ \sigma_{\min}(A)$.

Let $X,Y\in\mathbb{C}^{n\times k}$ with $\mathop{\mathrm{rank}}(X)=\mathop{\mathrm{rank}}(Y)=k$. The __canonical angles__ between their column spaces are $\theta_i=\cos^{-1}\sigma_i$, where $\sigma_i$ are the singular values of 
$(Y^*Y)^{-1/2}Y^*X(X^*X)^{-1/2}$. The __canonical angle matrix__ between $X$ and $Y$ is 

$$
\Theta(X,Y)=\mathop{\mathrm{diag}}(\theta_1,\theta_2,\ldots,\theta_k).$$
    
"""

# ╔═╡ 9f6cf19d-a3b4-44a9-b93c-294ba71526df
md"""
## Facts

1. __Mirsky Theorem.__ $\|\Sigma-\tilde\Sigma\|_2\leq \|\Delta A\|_2$ and $\|\Sigma-\tilde\Sigma\|_F\leq \|\Delta A\|_F$.

2. __Residual bounds.__ Let $\|\tilde u\|_2=\|\tilde v\|_2=1$ and $\tilde \mu=\tilde u^* A \tilde v$. Let residuals $r=A\tilde v-\tilde \mu \tilde u$ and $s=A^*\tilde u - \tilde \mu \tilde v$, and let $\varepsilon=\max\{\|r\|_2,\|s\|_2\}$. Then $|\tilde \mu -\mu|\leq \varepsilon$ for some singular value $\mu$ of $A$. 

3. The smallest error matrix $\Delta A$ for which $(\tilde u, \tilde \mu, \tilde v)$ is a singular triplet of $\tilde A$ satisfies $\| \Delta A\|_2=\varepsilon$.

4. Let $\mu$ be the closest singular value in $sv_{ext}(A)$ to $\tilde \mu$ and $(u,\mu,v)$ be the associated singular triplet, and let

$$
\eta=\mathop{\mathrm{gap}}(\tilde\mu)= \min_{\mu\neq\sigma\in sv_{ext}(A)}|\tilde\mu-\sigma|.$$

If $\eta>0$, then

$$
\begin{aligned}
|\tilde\mu-\mu |&\leq \frac{\varepsilon^2}{\eta},\\
\sqrt{\sin^2\theta(u,\tilde u)+ \sin^2\theta(v,\tilde v)} & \leq 
\frac{\sqrt{\|r\|_2^2 + \|s\|_2^2}}{\eta}.
\end{aligned}$$

5. Let

$$
A=\begin{bmatrix} M & E \\ F & H \end{bmatrix}, \quad 
\tilde A=\begin{bmatrix} M & 0 \\ 0 & H \end{bmatrix},$$

where $M\in\mathbb{C}^{k\times k}$, and set $\eta=\min |\mu-\nu|$ over all $\mu\in sv(M)$ and $\nu\in sv_{ext}(H)$, and $\varepsilon =\max \{ \|E\|_2,\|F\|_2 \}$. Then

$$
\max |\sigma_j -\tilde\sigma_j| \leq \frac{2\varepsilon^2}{\eta+\sqrt{\eta^2+4\varepsilon^2}}.$$

6. Let $m\geq n$ and let

$$
\begin{bmatrix} U_1^*\\ U_2^* \end{bmatrix} A \begin{bmatrix} V_1 & V_2 \end{bmatrix}=
\begin{bmatrix} A_1 &  \\ & A_2 \end{bmatrix}, \quad 
\begin{bmatrix} \tilde U_1^*\\ \tilde U_2^* \end{bmatrix} \tilde A \begin{bmatrix} \tilde V_1 & \tilde V_2 \end{bmatrix}=
\begin{bmatrix} \tilde A_1 &  \\ & \tilde A_2 \end{bmatrix},
$$

where $\begin{bmatrix} U_1 & U_2 \end{bmatrix}$, $\begin{bmatrix} V_1 & V_2 \end{bmatrix}$, $\begin{bmatrix} \tilde U_1 & \tilde U_2 \end{bmatrix}$, and $\begin{bmatrix} \tilde V_1 & \tilde V_2 \end{bmatrix}$ are unitary, and $U_1,\tilde U_1\in \mathbb{C}^{m\times k}$, $V_1,\tilde V_1\in \mathbb{C}^{n\times k}$. Set

$$
R=A\tilde V_1-\tilde U_1\tilde A_1,\quad 
S=A^*\tilde U_1-\tilde V_1 \tilde A_1.$$

Let $\eta=\min|\tilde \mu-\nu|$ over all $\tilde \mu\in sv(\tilde A_1)$ and $\nu\in sv_{ext}(A_2)$. If $\eta > 0$, then

$$
\sqrt{\|\sin\Theta(U_1,\tilde U_1)\|_F^2 +
\|\sin\Theta(V_1,\tilde V_1)\|_F^2}
\leq \frac{\sqrt{\|R\|_F^2 + \|S\|_F^2 }}{\eta}.$$

"""

# ╔═╡ e9f05c46-8d4c-4666-8ec4-05f6b7f986e5
md"""
## Example
"""

# ╔═╡ 034bc13c-8d64-4deb-a6ee-8076880f8de9
begin
	Random.seed!(421)
	m=8
	n=5
	k=min(m,n)
	A=rand(-9:9,m,n)
end

# ╔═╡ f03032b0-1617-4d8e-a7b6-ce4b70334166
begin
	ΔA=rand(m,n)/100
	B=A+ΔA
end

# ╔═╡ c8f1b7df-c164-4077-a7e2-0471d6db7660
begin
	S=svd(A)
	S₁=svd(B)
	σ=S.S
	σ₁=S₁.S
end

# ╔═╡ f487e102-294a-447c-a48e-316070c7f184
# Mirsky's Theorems
maximum(abs,σ-σ₁),opnorm(Diagonal(σ)-Diagonal(σ₁)),
opnorm(ΔA), norm(σ-σ₁), norm(ΔA)

# ╔═╡ 83d33c37-acb2-4fbe-a116-3510e2ea7f60
@bind j Slider(2:k-1,show_value=true)

# ╔═╡ 948a85ef-4fd8-4197-b916-c617369b7de9
begin
	# Residual bounds - how close is (x,ζ,y) to (U[:,j],σ[j],V[:,j])
	x=round.(S.U[:,j],digits=3)
	y=round.(S.V[:,j],digits=3)
	x=x/norm(x)
	y=y/norm(y)
	ζ=(x'*A*y)[]
	σ, j, ζ-σ[j]
end

# ╔═╡ 1cb8d9ff-7356-46a0-b932-b5d00d2bc3d0
begin
	# Fact 2
	r=A*y-ζ*x
	s=A'*x-ζ*y
	ϵ=max(norm(r),norm(s))
end

# ╔═╡ 6c79949f-e690-49d0-90e8-5e8ac3eb0d98
minimum(abs,σ.-ζ), ϵ

# ╔═╡ 2f99ff26-feda-4aaf-b84b-63ff83824992
# Fact 4
η=min(abs(ζ-σ[j-1]),abs(ζ-σ[j+1]))

# ╔═╡ 4df6a5f9-6851-4938-9540-3b19af75fb93
ζ-σ[j], ϵ^2/η

# ╔═╡ 2fdcd958-67f8-431d-880a-f640ee01d071
begin
	# Eigenvector bound
	# cos(θ)
	cosθU=dot(x,S.U[:,j])
	cosθV=dot(y,S.V[:,j])
	# Bound
	√(1-cosθU^2+1-cosθV^2), √(norm(r)^2+norm(s)^2)/η
end

# ╔═╡ 6265561d-2db5-4389-8cb5-b801223e49a2
begin
	# Fact 5 - we create small off-diagonal block perturbation
	j₁=3
	M=A[1:j₁,1:j₁]
	H=A[j₁+1:m,j₁+1:n]
	B₁=cat(M,H,dims=(1,2))
end

# ╔═╡ 49c1a292-b554-4b65-a569-76beb97e9810
begin
	E=rand(Float64,size(A[1:j₁,j₁+1:n]))/100
	F=rand(Float64,size(A[j₁+1:m,1:j₁]))/100
	C=float(B₁)
	C[1:j₁,j₁+1:n]=E
	C[j₁+1:m,1:j₁]=F
	C
end

# ╔═╡ a4fc10bb-e16a-475d-a634-0cfa269975df
svdvals(M)

# ╔═╡ 701183c4-c0a4-42ff-a7b2-739a97ff9f68
svdvals(H)'

# ╔═╡ 83555dd6-0b4b-4baf-b648-e5a0d4fb52fb
svdvals(M).-svdvals(H)'

# ╔═╡ f79ed616-9958-4360-a5ef-cea51e103908
begin
	ϵ₁=max(norm(E), norm(F))
	β=svdvals(B₁)
	γ=svdvals(C)
	η₁=minimum(abs,svdvals(M).-svdvals(H)')
	display([β γ])
	maximum(abs,β-γ), 2*ϵ₁^2/(η₁+sqrt(η₁^2+4*ϵ₁^2))
end

# ╔═╡ 78e3a6ed-faa4-45c5-bb79-5230d6c018d5
md"""
# Relative perturbation theory

## Definitions

Matrix $A\in\mathbb{C}^{m\times n}$ is __multiplicatively pertubed__ to $\tilde A$ if
$\tilde A=D_L^* A D_R$ for some $D_L\in\mathbb{C}^{m\times m}$ and 
$D_R\in\mathbb{C}^{n\times n}$. 

Matrix $A$ is (highly) __graded__ if it can be scaled as $A=GS$ such that $\kappa_2(G)$ is of modest magnitude. The __scaling matrix__ $S$ is often diagonal. Interesting cases are when $\kappa_2(G)\ll \kappa_2(A)$.

__Relative distances__ between two complex numbers $\alpha$ and $\tilde \alpha$ are:

$$
\begin{aligned}
\zeta(\alpha,\tilde \alpha)&=\frac{|\alpha-\tilde\alpha|}{\sqrt{|\alpha\tilde \alpha|}}, \quad \textrm{for } \alpha\tilde\alpha\neq 0,\\
\varrho(\alpha,\tilde \alpha)&=\frac{|\alpha-\tilde\alpha|}
{\sqrt{|\alpha|^2 +  |\tilde \alpha|^2}}, \quad \textrm{for } |\alpha|+|\tilde\alpha|> 0.
\end{aligned}$$
"""

# ╔═╡ 69e1ae35-a5ec-4de1-9e81-ea5d71d7ef83
md"""

## Facts

1. If $D_L$ and $D_R$ are non-singular and $m\geq n$, then

$$
\begin{aligned}
\frac{\sigma_j}{\|D_L^{-1}\|_2\|D_R^{-1}\|_2}& \leq \tilde\sigma_j \leq
\sigma_j \|D_L\|_2\|D_R\|_2, \quad \textrm{for } i=1,\ldots,n, \\
\| \mathop{\mathrm{diag}}(\zeta(\sigma_1,\tilde \sigma_1),\ldots,
\zeta(\sigma_n,\tilde \sigma_n)\|_{2,F} & \leq
\frac{1}{2}\|D_L^*-D_L^{-1}\|_{2,F} + \frac{1}{2}\|D_R^*-D_R^{-1}\|_{2,F}.
\end{aligned}$$

2. Let $m\geq n$ and let

$$
\begin{bmatrix} U_1^*\\ U_2^* \end{bmatrix} A \begin{bmatrix} V_1 & V_2 \end{bmatrix}=
\begin{bmatrix} A_1 &  \\ & A_2 \end{bmatrix}, \quad 
\begin{bmatrix} \tilde U_1^*\\ \tilde U_2^* \end{bmatrix} \tilde A \begin{bmatrix} \tilde V_1 & \tilde V_2 \end{bmatrix}=
\begin{bmatrix} \tilde A_1 &  \\ & \tilde A_2 \end{bmatrix},$$

where $\begin{bmatrix} U_1 & U_2 \end{bmatrix}$, $\begin{bmatrix} V_1 & V_2 \end{bmatrix}$, $\begin{bmatrix} \tilde U_1 & \tilde U_2 \end{bmatrix}$, and $\begin{bmatrix} \tilde V_1 & \tilde V_2 \end{bmatrix}$ are unitary, and $U_1,\tilde U_1\in \mathbb{C}^{m\times k}$, $V_1,\tilde V_1\in \mathbb{C}^{n\times k}$. Set

$$
R=A\tilde V_1-\tilde U_1\tilde A_1,\quad 
S=A^*\tilde U_1-\tilde V_1 \tilde A_1.$$

Let $\eta=\min \varrho(\mu,\tilde \mu)$ over all $\mu\in sv(A_1)$ and $\tilde \mu\in sv_{ext}(A_2)$. If $\eta > 0$, then

$$
\begin{aligned}
& \sqrt{\|\sin\Theta(U_1,\tilde U_1)\|_F^2 +
\|\sin\Theta(V_1,\tilde V_1)\|_F^2} \\
& \leq \frac{1}{\eta}( \|(I-D_L^*)U_1\|_F^2+ \|(I-D_L^{-1})U_1\|_F^2 \\
& \quad +\|(I-D_R^*)V_1\|_F^2+ \|(I-D_R^{-1})V_1\|_F^2 )^{1/2}.
\end{aligned}$$

3. Let $A=GS$ and $\tilde A=\tilde GS$, and let $\Delta G=\tilde G-G$. Then $\tilde A=DA$, where $D=I+(\Delta G) G^{\dagger}$, and Fact 1 applies with $D_L=D$, $D_R=I$, and 

$$
\|D^*-D^{-1}\|_{2,F} \leq \bigg(1+\frac{1}{1-\|(\Delta G) G^{\dagger}\|_{2}}\bigg)
\frac{\|(\Delta G) G^{\dagger}\|_{2,F}}{2}.$$

According to the notebook on [Jacobi Method and High Relative Accuracy](https://ivanslapnicar.github.io/GIAN-Applied-NLA-Course/L4c%20Symmetric%20Eigenvalue%20Decomposition%20-%20Jacobi%20Method%20and%20High%20Relative%20Accuracy.jl.html), nearly optimal diagonal scaling is such that all columns of $G$ have unit norms, $S=\mathop{\mathrm{diag}} \big( \| A_{:,1}\|_2,\ldots,\|A_{:,n}\|_2 \big)$.

4. Let $A$ be an real upper-bidiagonal matrix with diagonal entries $a_1,a_2,\ldots,a_n$ and the super-diagonal entries $b_1,b_2, \ldots,b_{n-1}$. Let the diagonal entries of $\tilde A$ be $\alpha_1 a_1,\alpha_2 a_2,\ldots,\alpha_n a_n$, and its super-diagonal entries be $\beta_1 b_1,\beta_2 b_2,\ldots,\beta_{n-1} b_{n-1}$. Then $\tilde A=D_L^* A D_R$ with

$$
\begin{aligned}
D_L &=\mathop{\mathrm{diag}} \bigg(\alpha_1,\frac{\alpha_1 \alpha_2}{\beta_1},
\frac{\alpha_1 \alpha_2 \alpha_3}{\beta_1 \beta_2},\cdots\bigg),\\
D_R &=\mathop{\mathrm{diag}} \bigg(1, \frac{\beta_1}{\alpha_1},
\frac{\beta_1 \beta_2}{\alpha_1 \alpha_2},\cdots\bigg).
\end{aligned}$$

Let $\alpha=\prod\limits_{j=1}^n \max\{\alpha_j, 1/\alpha_j\}$ and $\beta=\prod\limits_{j=1}^{n-1} \max\{\beta_j, 1/\beta_j\}$. Then

$$
(\alpha\beta)^{-1}\leq \| D_L^{-1}\|_2 \|D_R^{-1}\|_2 \leq
\| D_L\|_2 \|D_R\|_2  \leq \alpha\beta,$$

and Fact 1 applies. This is a result by [Demmel and Kahan](http://www.netlib.org/lapack/lawnspdf/lawn03.pdf).
 
5. Consider the block partitioned matrices

$$
\begin{aligned}
A & =\begin{bmatrix} B & C \\ 0 & D\end{bmatrix}, \\
\tilde A & =  \begin{bmatrix} B & 0 \\ 0 & D\end{bmatrix}
=A \begin{bmatrix} I & -B^{-1} C \\ 0 & I \end{bmatrix}\equiv A D_R.
\end{aligned}$$

By Fact 1, $\zeta(\sigma_j,\tilde \sigma_j) \leq \frac{1}{2} \|B^{-1}C\|_2$. This is used as a deflation criterion in the SVD algorithm for bidiagonal matrices.
"""

# ╔═╡ 200a5b3d-4c30-4ea4-83de-ac89ce153e01
md"""
## Bidiagonal matrix

To illustrate Facts 1 to 3, we need an algorithm that computes the singular values with high relative accuracy. Such algorithm, the one-sided Jacobi method, is discussed in the notebook on [Jacobi and Lanczos Methods](L6b%20Singular%20Value%20Decomposition%20-%20Jacobi%20and%20Lanczos%20Methods.ipynb).

The algorithm actually used in the function `svdvals()` for `Bidiagonal` is the zero-shift bidiagonal QR algorithm, which attains the accuracy given by Fact 4: if all $1-\varepsilon \leq \alpha_i,\beta_j \leq 1+\varepsilon$, then

$$
(1-\varepsilon)^{2n-1} \leq (\alpha\beta)^{-1} \leq \alpha\beta \leq (1+\varepsilon)^{2n-1}.$$

In other words, $\varepsilon$ relative changes in diagonal and super-diagonal elements, cause at most $(2n-1)\varepsilon$ relative changes in the singular values.

__However__, if singular values and vectors are desired, the function `svd()` calls the standard algorithm, described in the next notebook, which __does not attain this accuracy__.
"""

# ╔═╡ 24db8b9d-9c8d-44a5-9ec1-16f664d1847a
begin
	Random.seed!(135)
	n₀=50
	δ=100000
	# The starting matrix
	a=exp.(50*(rand(n₀).-0.5))
	b=exp.(50*(rand(n₀-1).-0.5))
	A₀=Bidiagonal(a,b,'U')
	# Multiplicative perturbation
	DL=ones(n₀)+(rand(n₀).-0.5)/δ
	DR=ones(n₀)+(rand(n₀).-0.5)/δ
	# The perturbed matrix
	α₀=DL.*a.*DR
	β₀=DL[1:end-1].*b.*DR[2:end]
	B₀=Bidiagonal(α₀,β₀,'U')
	# B-Diagonal(DL)*A*Diagonal(DR)
	(A₀.dv-B₀.dv)./A₀.dv
end

# ╔═╡ a258164a-8877-48e3-8bb1-70829d0b64f6
A₀

# ╔═╡ 7f6b4544-3711-4188-a657-55057c682110
cond(A₀)

# ╔═╡ 748215b6-361e-4525-ac2a-f50c96b34047
(a-α₀)./a, (b-β₀)./b

# ╔═╡ 831a98d8-5df0-49ea-89fd-8a3db0e21e1e
@which svdvals(A₀)

# ╔═╡ ceeea545-7731-43f7-82fb-6141f9c98344
@which svdvals!(copy(A₀))

# ╔═╡ e2f56830-6169-43b6-8b73-70d0fee5a103
begin
	σ₀=svdvals(A₀)
	μ=svdvals(B₀)
	[σ₀ (σ₀-μ)./σ₀]
end

# ╔═╡ a20ee9c8-aba8-4dc9-9592-67b84b7ceb06
# The standard algorithm
S₀=svd(A₀)

# ╔═╡ 7223bef3-7da5-4988-a7c1-3190b12c84f8
(σ₀-S₀.S)./σ₀

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
# ╠═4d0ff51d-5cd6-4f47-9c48-66055deabde5
# ╠═7d4b46fd-de3f-4211-a27a-3b92b2a1e873
# ╟─b34474bc-3dd2-491d-a9aa-8d9402e2e880
# ╟─872db817-6bee-42dd-b50d-32c82534f993
# ╟─9f6cf19d-a3b4-44a9-b93c-294ba71526df
# ╟─e9f05c46-8d4c-4666-8ec4-05f6b7f986e5
# ╠═034bc13c-8d64-4deb-a6ee-8076880f8de9
# ╠═f03032b0-1617-4d8e-a7b6-ce4b70334166
# ╠═c8f1b7df-c164-4077-a7e2-0471d6db7660
# ╠═f487e102-294a-447c-a48e-316070c7f184
# ╠═83d33c37-acb2-4fbe-a116-3510e2ea7f60
# ╠═948a85ef-4fd8-4197-b916-c617369b7de9
# ╠═1cb8d9ff-7356-46a0-b932-b5d00d2bc3d0
# ╠═6c79949f-e690-49d0-90e8-5e8ac3eb0d98
# ╠═2f99ff26-feda-4aaf-b84b-63ff83824992
# ╠═4df6a5f9-6851-4938-9540-3b19af75fb93
# ╠═2fdcd958-67f8-431d-880a-f640ee01d071
# ╠═6265561d-2db5-4389-8cb5-b801223e49a2
# ╠═49c1a292-b554-4b65-a569-76beb97e9810
# ╠═a4fc10bb-e16a-475d-a634-0cfa269975df
# ╠═701183c4-c0a4-42ff-a7b2-739a97ff9f68
# ╠═83555dd6-0b4b-4baf-b648-e5a0d4fb52fb
# ╠═f79ed616-9958-4360-a5ef-cea51e103908
# ╟─78e3a6ed-faa4-45c5-bb79-5230d6c018d5
# ╟─69e1ae35-a5ec-4de1-9e81-ea5d71d7ef83
# ╟─200a5b3d-4c30-4ea4-83de-ac89ce153e01
# ╠═24db8b9d-9c8d-44a5-9ec1-16f664d1847a
# ╠═a258164a-8877-48e3-8bb1-70829d0b64f6
# ╠═7f6b4544-3711-4188-a657-55057c682110
# ╠═748215b6-361e-4525-ac2a-f50c96b34047
# ╠═831a98d8-5df0-49ea-89fd-8a3db0e21e1e
# ╠═ceeea545-7731-43f7-82fb-6141f9c98344
# ╠═e2f56830-6169-43b6-8b73-70d0fee5a103
# ╠═a20ee9c8-aba8-4dc9-9592-67b84b7ceb06
# ╠═7223bef3-7da5-4988-a7c1-3190b12c84f8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
