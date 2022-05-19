### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ c5dfc888-42a6-4fa5-b969-61ff6c9180fe
begin
    import Pkg
    # activate a temporary environment
    Pkg.activate(mktempdir())
    Pkg.add([Pkg.PackageSpec(name="Arrowhead", rev="master")])
    using PlutoUI, LinearAlgebra, Arrowhead
end

# ╔═╡ 51c330f6-7675-475e-b0eb-0fc8486fd094
PlutoUI.TableOfContents(aside=true)

# ╔═╡ c3caaf95-bef6-446d-b197-23ad1a7da450
md"""
# Updating the SVD

In many applications which are based on the SVD, arrival of new data requires SVD of the new matrix. Instead of computing from scratch, existing SVD can be updated.

__Prerequisites__

The reader should be familiar with concepts of singular values and singular vectors, related perturbation theory, and algorithms.

__Competences__

The reader should be able to recognise applications where SVD updating can be sucessfully applied and apply it.
"""

# ╔═╡ e070e205-b8fb-4608-93c1-f5c5beb075c3
md"""
# Facts

For more details see
[M. Gu and S. C. Eisenstat, A Stable and Fast Algorithm for Updating the Singular Value Decomposition](http://www.cs.yale.edu/publications/techreports/tr966.pdf) and [M. Brand, Fast low-rank modifications of the thin singular value decomposition](http://www.sciencedirect.com/science/article/pii/S0024379505003812) and the references therein.


1. Let $A\in\mathbb{R}^{m\times n}$ with $m\geq n$ and $\mathop{\mathrm{rank}}(A)=n$, and  let $A=U\Sigma V^T$ be its SVD. Let $a\in\mathbb{R}^{n}$ be a vector, and let $\tilde A=\begin{bmatrix} A \\ a^T\end{bmatrix}$. Then

$$
\begin{bmatrix} A \\ a^T\end{bmatrix} =\begin{bmatrix} U & \\ & 1 \end{bmatrix}
\begin{bmatrix} \Sigma \\ a^TV \end{bmatrix}  V^T.$$

Let $\begin{bmatrix} \Sigma \\ a^T V \end{bmatrix} = \bar U \bar \Sigma \bar V^T$ be the SVD of the half-arrowhead matrix. _This SVD can be computed in $O(n^2)$ operations._ Then

$$
\begin{bmatrix} A \\ a^T\end{bmatrix} =
\begin{bmatrix} U & \\ & 1 \end{bmatrix} \bar U \bar\Sigma \bar V^T V^T \equiv
\tilde U \bar \Sigma \tilde V^T$$

is the SVD of $\tilde A$.

2. Direct computation of $\tilde U$ and $\tilde V$ requires $O(mn^2)$ and $O(n^3)$ operations. However, these multiplications can be performed using Fast Multipole Method. This is not (yet) implemented in Julia and is "not for the timid" (quote by Steven G. Johnson).

3. If $m<n$ and $\mathop{\mathrm{rank}}(A)=n$, then

$$
\begin{bmatrix} A \\ a^T\end{bmatrix} =\begin{bmatrix} U & \\ & 1 \end{bmatrix}
\begin{bmatrix} \Sigma & 0 \\ a^T V & \beta\end{bmatrix} \begin{bmatrix} V^T \\ v^T \end{bmatrix},$$

where $\beta=\sqrt{\|a\|_2^2-\|V^T a\|_2^2}$ and $v=(I-VV^T)a$. Notice that $V^Tv=0$ by construction. Let $\begin{bmatrix} \Sigma & 0 \\ a^T V &  \beta\end{bmatrix} = \bar U \bar \Sigma \bar V^T$ be the SVD of the half-arrowhead matrix. Then

$$
\begin{bmatrix} A \\ a^T\end{bmatrix} =
\begin{bmatrix} U & \\ & 1 \end{bmatrix} \bar U \bar\Sigma \bar V^T \begin{bmatrix} V^T \\ v^T \end{bmatrix}
\equiv \tilde U \bar \Sigma \tilde V^T$$

is the SVD of $\tilde A$.

4. Adding a column $a$ to $A$ is equivalent to adding a row $a^T$ to $A^T$.

5. If $\mathop{\mathrm{rank}}(A)<\min\{m,n\}$ or if we are using SVD approximation of rank $r$, and if we want to keep the rank of the approximation (this is the common case in practice), then the formulas in Fact 1 hold approximately. More precisely, the updated rank $r$ approximation is __not__ what we would get by computing the approximation of rank $r$ of the updated matrix, but is sufficient in many applications.
"""

# ╔═╡ 0ff9f1d6-1ebf-4a42-b51a-5a67f6192fe7
md"""

# Examples

## Adding row to a tall matrix

If $m\geq n$, adding row does not increase the size of $\Sigma$.
"""

# ╔═╡ 1f7f9f7a-df70-40a3-aa06-b5621b729a0c
function SVDaddrow(svdA::SVD,a::Vector)
    # Create the transposed half-arrowhead
    m,r,n=size(svdA.U,1),length(svdA.S),size(svdA.V,1)
    T=typeof(a[1])
    b=svdA.Vt*a
    if m>=n || r<m
        M=HalfArrow(svdA.S,b)
    else
        β=√(norm(a)^2-norm(b)^2)
        M=HalfArrow(svdA.S,[b;β])
    end
    # From Arrowhead package
    svdM,info=svd(M)
    # Return the updated SVD
    if m>=n || r<m
        return SVD([svdA.U zeros(T,m); zeros(T,1,r) one(T)]*svdM.V,
            svdM.S, adjoint(svdA.V*svdM.U))
    else
        # Need one more row of svdA.V - v is an orthogonal projection
        v=a-svdA.V*b
        normalize!(v)
        return SVD([svdA.U zeros(T,m); zeros(T,1,r) one(T)]*svdM.V,
            svdM.S, adjoint([svdA.V v]*svdM.U))
    end
end

# ╔═╡ 445cacab-3bb2-42f6-8d34-73c77ceecd0a
methods(SVD)

# ╔═╡ 2d0c21e5-dc68-4434-94ca-f433355f0e5c
begin
	import Random
	Random.seed!(421)
	A=rand(10,6)
	a=rand(6)
end

# ╔═╡ 148c7157-3d93-4673-bd9b-26bef99627a7
A

# ╔═╡ c034e218-09d3-4403-a413-98932865c481
S=svd(A)

# ╔═╡ 485e8222-2bb7-490e-9a51-675f0baef274
# Residual
norm(A*S.V-S.U*Diagonal(S.S))

# ╔═╡ 3dd2734d-db44-4226-9163-e6a55dcc7667
Sa=SVDaddrow(S,a)

# ╔═╡ 2fb74892-bade-47fe-bcaa-ef5bfdb19445
begin
	Aa=[A;transpose(a)]
	size(Aa),size(Sa.U),size(S.V)
end

# ╔═╡ c074206d-ed7d-44fb-a0ee-22e5cdd6e0f8
[svdvals(Aa) Sa.S]

# ╔═╡ 89990d20-4752-4558-a328-a4ae6a428906
# Residual and orthogonality
norm(Aa*Sa.V-Sa.U*Diagonal(Sa.S)),
norm(Sa.U'*Sa.U-I), norm(Sa.Vt*Sa.V-I)

# ╔═╡ bd35d5e4-b454-4a02-a026-ded3bdf76cdf
md"""
## Adding row to a flat matrix
"""

# ╔═╡ 8f31ef4a-a5da-4837-9320-ec9578c5a277
begin
	# Now flat matrix
	Random.seed!(421)
	A₁=rand(6,10)
	a₁=rand(10)
	S₁=svd(A₁)
end

# ╔═╡ e2ff1d44-7f3c-4100-80d3-df97579bc56c
A₁

# ╔═╡ 2b055add-b46f-4a2e-8ae1-89e59f12c21c
Aa₁=[A₁;transpose(a₁)]

# ╔═╡ e68d3450-da63-4b3f-9244-632a4eba5aff
Sa₁=SVDaddrow(S₁,a₁);

# ╔═╡ fac291b0-cf08-46d6-a233-3e1e8675c7e0
size(Aa₁),size(Sa₁.U),size(S₁.V)

# ╔═╡ 9f2d9dcc-ee9c-49e5-a839-3dadfed371e8
# Residual and orthogonality
norm(Aa₁*Sa₁.V-Sa₁.U*Diagonal(Sa₁.S)),
norm(Sa₁.U'*Sa₁.U-I), norm(Sa₁.Vt*Sa₁.V-I)

# ╔═╡ 9727df4c-52f5-4527-84e6-dfc02820d583
md"""
## Adding columns

This can be viewed as adding rows to the transposed matrix, an elegant one-liner in Julia.
"""

# ╔═╡ 3b34587c-fc0c-4677-91a7-4efb8a9abdba
function SVDaddcol(svdA::SVD,a::Vector)
    X=SVDaddrow(SVD(svdA.V,svdA.S,adjoint(svdA.U)),a)
    SVD(X.V,X.S,adjoint(X.U))
end

# ╔═╡ d8807efc-0cff-4aee-8bc5-4a5403c092eb
begin
	# Tall matrix
	Random.seed!(897)
	A₂=rand(10,6)
	a₂=rand(10)
	S₂=svd(A₂)
	Sa₂=SVDaddcol(S₂,a₂)
end

# ╔═╡ 8ce644c3-f71a-4904-bf15-d3f602f44c05
Aa₂=[A₂ a₂];

# ╔═╡ ca5fd757-5ab8-4826-8636-59a29a8c5843
# Residual and orthogonality
norm(Aa₂*Sa₂.V-Sa₂.U*Diagonal(Sa₂.S)),
norm(Sa₂.U'*Sa₂.U-I), norm(Sa₂.Vt*Sa₂.V-I)

# ╔═╡ 5f5e6293-17b5-405d-96c6-d661151796dc
begin
	# Flat matrix
	Random.seed!(332)
	A₃=rand(6,10)
	a₃=rand(6)
	S₃=svd(A₃)
	Sa₃=SVDaddcol(S₃,a₃)
end

# ╔═╡ 3c5d82e1-81f4-4cff-9c2a-6cc9bfbefd58
Aa₃=[A₃ a₃]

# ╔═╡ a2f5ab12-5c09-4f2b-9665-d2d3d44abd17
# Residual and orthogonality
norm(Aa₃*Sa₃.V-Sa₃.U*Diagonal(Sa₃.S)),
norm(Sa₃.U'*Sa₃.U-I), norm(Sa₃.Vt*Sa₃.V-I)

# ╔═╡ 6563cdc2-8577-4fac-997d-413edc56e2ac
begin
	# Square matrix
	A₄=rand(10,10)
	a₄=rand(10)
	S₄=svd(A₄)
end

# ╔═╡ 0f656a69-2243-4273-8600-83a12b694497
begin
	Sa₄=SVDaddrow(S₄,a₄)
	Aa₄=[A₄;transpose(a₄)]
	norm(Aa₄*Sa₄.V-Sa₄.U*Diagonal(Sa₄.S)),
	norm(Sa₄.U'*Sa₄.U-I), norm(Sa₄.Vt*Sa₄.V-I)
end

# ╔═╡ 35b4c90e-830f-487e-b683-988952aa12af
begin
	Sa₅=SVDaddcol(S₄,a₄)
	Aa₅=[A₄ a₄]
	norm(Aa₅*Sa₅.V-Sa₅.U*Diagonal(Sa₅.S)),
	 norm(Sa₅.U'*Sa₅.U-I), norm(Sa₅.Vt*Sa₅.V-I)
end

# ╔═╡ 22b4dea8-79a0-438a-a3e4-d7c725d47051
md"""
## Updating a low rank approximation
"""

# ╔═╡ 0a5298eb-ed33-443b-b9e4-c0a32b408605
begin
	# Adding row to a tall matrix
	A₆=rand(10,6)
	S₆=svd(A₆)
	a₆=rand(6)
	# Rank of the approximation
	r=4
end

# ╔═╡ 99b2a284-4a24-492d-b677-4b08430a4399
# Low-rank approximation
Sr=SVD(S₆.U[:,1:r], S₆.S[1:r],adjoint(S₆.V[:,1:r]));

# ╔═╡ 3677f364-e828-4a42-84b0-e86f0e3fff3b
begin
	# Eckart, Young, Mirsky
	Ar=Sr.U*Diagonal(Sr.S)*Sr.Vt
	Δ=Ar-A₆
	opnorm(Δ),svdvals(A₆)[5]
end

# ╔═╡ 32880612-cdd1-4e16-8a05-025eb35110f9
Sa₆=SVDaddrow(Sr,a₆);

# ╔═╡ 65c83ee2-36ad-4388-b9b3-05830ab3ee9a
Aa₆=[A₆; transpose(a₆)];

# ╔═╡ 13deeb53-9670-4351-85b3-a366690a94a8
svdvals(Aa₆),svdvals([Ar;transpose(a₆)]),Sa₆.S

# ╔═╡ a02b7022-b775-4dbd-bb83-0d6f88a47312
begin
	# Adding row to a flat matrix
	A₇=rand(6,10)
	S₇=svd(A₇)
	a₇=rand(10)
	# Rank of the approximation
	r₇=4
end

# ╔═╡ 2f22e978-2fb4-4ec4-80af-519384e98f6c
begin
	Sr₇=SVD(S₇.U[:,1:r₇], S₇.S[1:r₇],adjoint(S₇.V[:,1:r₇]))
	Sa₇=SVDaddrow(Sr₇,a₇);
end

# ╔═╡ 2beba2fd-cb70-4da2-a9f3-d4c39815ff18
begin
	Ar₇=Sr₇.U*Diagonal(Sr₇.S)*Sr₇.Vt
	svdvals(Sa₇),svdvals([Ar₇;transpose(a₇)]),Sa₇.S
end

# ╔═╡ Cell order:
# ╠═c5dfc888-42a6-4fa5-b969-61ff6c9180fe
# ╠═51c330f6-7675-475e-b0eb-0fc8486fd094
# ╟─c3caaf95-bef6-446d-b197-23ad1a7da450
# ╟─e070e205-b8fb-4608-93c1-f5c5beb075c3
# ╟─0ff9f1d6-1ebf-4a42-b51a-5a67f6192fe7
# ╠═1f7f9f7a-df70-40a3-aa06-b5621b729a0c
# ╠═445cacab-3bb2-42f6-8d34-73c77ceecd0a
# ╠═2d0c21e5-dc68-4434-94ca-f433355f0e5c
# ╠═148c7157-3d93-4673-bd9b-26bef99627a7
# ╠═c034e218-09d3-4403-a413-98932865c481
# ╠═485e8222-2bb7-490e-9a51-675f0baef274
# ╠═3dd2734d-db44-4226-9163-e6a55dcc7667
# ╠═2fb74892-bade-47fe-bcaa-ef5bfdb19445
# ╠═c074206d-ed7d-44fb-a0ee-22e5cdd6e0f8
# ╠═89990d20-4752-4558-a328-a4ae6a428906
# ╟─bd35d5e4-b454-4a02-a026-ded3bdf76cdf
# ╠═8f31ef4a-a5da-4837-9320-ec9578c5a277
# ╠═e2ff1d44-7f3c-4100-80d3-df97579bc56c
# ╠═2b055add-b46f-4a2e-8ae1-89e59f12c21c
# ╠═e68d3450-da63-4b3f-9244-632a4eba5aff
# ╠═fac291b0-cf08-46d6-a233-3e1e8675c7e0
# ╠═9f2d9dcc-ee9c-49e5-a839-3dadfed371e8
# ╟─9727df4c-52f5-4527-84e6-dfc02820d583
# ╠═3b34587c-fc0c-4677-91a7-4efb8a9abdba
# ╠═d8807efc-0cff-4aee-8bc5-4a5403c092eb
# ╠═8ce644c3-f71a-4904-bf15-d3f602f44c05
# ╠═ca5fd757-5ab8-4826-8636-59a29a8c5843
# ╠═5f5e6293-17b5-405d-96c6-d661151796dc
# ╠═3c5d82e1-81f4-4cff-9c2a-6cc9bfbefd58
# ╠═a2f5ab12-5c09-4f2b-9665-d2d3d44abd17
# ╠═6563cdc2-8577-4fac-997d-413edc56e2ac
# ╠═0f656a69-2243-4273-8600-83a12b694497
# ╠═35b4c90e-830f-487e-b683-988952aa12af
# ╟─22b4dea8-79a0-438a-a3e4-d7c725d47051
# ╠═0a5298eb-ed33-443b-b9e4-c0a32b408605
# ╠═99b2a284-4a24-492d-b677-4b08430a4399
# ╠═3677f364-e828-4a42-84b0-e86f0e3fff3b
# ╠═32880612-cdd1-4e16-8a05-025eb35110f9
# ╠═65c83ee2-36ad-4388-b9b3-05830ab3ee9a
# ╠═13deeb53-9670-4351-85b3-a366690a94a8
# ╠═a02b7022-b775-4dbd-bb83-0d6f88a47312
# ╠═2f22e978-2fb4-4ec4-80af-519384e98f6c
# ╠═2beba2fd-cb70-4da2-a9f3-d4c39815ff18
