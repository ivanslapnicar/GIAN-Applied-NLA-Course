### A Pluto.jl notebook ###
# v0.14.4

using Markdown
using InteractiveUtils

# ╔═╡ d01a9b4f-8b55-4607-abb6-717d227fcd48
begin
	using PlutoUI, LinearAlgebra
	PlutoUI.TableOfContents(aside=true)
end

# ╔═╡ 479a40d9-b81e-442b-9962-f972b110a4dd
# Pkg.checkout("SpecialMatrices")
using SpecialMatrices

# ╔═╡ 574e86bb-159c-4141-8d8a-21bdcc9b5304
# Plot the eigenvalues (singular values) and left singular vectors
using Plots

# ╔═╡ 5eb73af5-f78a-4811-83a5-ac39063a4516
# pkg> add Arrowhead#master
using Arrowhead

# ╔═╡ 5d95dc2c-bf94-4b13-b9d5-b7b261e86cf6
md"""
# Algorithms for Structured Matrices


For matrices with some special structure, it is possible to derive versions of algorithms which are faster and/or more accurate than the standard algorithms.

__Prerequisites__

The reader should be familiar with concepts of eigenvalues and eigen vectors, singular values and singular vectors, related perturbation theory, and algorithms.
 
__Competences__

The reader should be able to recognise matrices which have rank-revealing decomposition and apply adequate algorithms, and to apply forward stable algorithms to arrowhead and diagonal-plus-rank-one matrices.
"""

# ╔═╡ e3e43840-d0a3-4cde-9b1e-5785759912b2
md"""
# Rank revealing decompositions

For more details, see [Z. Drmač, Computing Eigenvalues and Singular Values to High Relative Accuracy](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and [J. Demmel et al, Computing the singular value decomposition with high relative accuracy](http://www.sciencedirect.com/science/article/pii/S0024379599001342), and the references therein.

Let $A\in\mathbb{R}^{m\times n}$ with $\mathop{\mathrm{rank}}(A)=n$ (therefore, $m\geq n$) and $A=U\Sigma V^T$ its thin SVD.

## Definitions

Let $A\in\mathbb{R}^{m\times n}$. 

The singular values of $A$ are (__perfectly__) __well determined to high relative accuracy__ if changing any entry $A_{kl}$ to $\theta A_{kl}$, $\theta \neq 0$, causes perturbations in singular values bounded by 

$$
\min\{|\theta|,1/|\theta|\}\sigma_j \leq\tilde \sigma_j \leq  
\max\{|\theta|,1/|\theta|\}\sigma_j,\quad  \forall j.$$

The __sparsity pattern__ of $A$, $Struct(A)$, is the set of indices for which $A_{kl}$ is permitted to be non-zero.

The __bipartite graph__ of the sparsity pattern $S$, $\mathcal{G}(S)$, is the graph with vertices partitioned into row vertices $r_1,\ldots,r_m$ and column vertices $c_1,\ldots,c_n$, where $r_k$ and $c_l$ are connected if and only if $(k,l)\in S$. 

If $\mathcal{G}(S)$ is acyclic, matrices with sparsity pattern $S$ are __biacyclic__.

A decomposition $A=XDY^T$ with diagonal matrix $D$ is called a __rank revealing decomposition__ (RRD) if $X$ and $Y$ are full-column rank well-conditioned matrices. 

__Hilbert matrix__ is a square matrix $H$ with elements $H_{ij}=\displaystyle\frac{1}{i+j-1}$.

__Hankel matrix__ is a square matrix with constant elements along skew-diagonals.

__Cauchy matrix__ is an $m\times n$ matrix $C$ with elements $C_{ij}=\displaystyle\frac{1}{x_i+y_j}$ with $x_i+y_j\neq 0$ for all $i,j$.
"""

# ╔═╡ 03f797ac-8764-4688-9e7e-e144cafb3b4c
md"""
## Facts

1. The singular values of $A$ are perfectly well determined to high relative accuracy if and only if the bipartite graph $\mathcal{G}(S)$ is acyclic (forest of trees). Examples are bidiagonal and arrowhead matrices. Sparsity pattern $S$ of acyclic bipartite graph allows at most $m+n-1$ nonzero entries. A bisection algorithm computes all singular values of biacyclic matrices to high relative accuracy.

2. An RRD of $A$ can be given or computed to high accuracy by some method. Typical methods are Gaussian elimination with complete pivoting or QR factorization with complete pivoting.

3. Let $\hat X \hat D \hat Y^T$ be the computed RRD of $A$ satisfying $|D_{jj}-\hat D_{jj}| \leq O(\varepsilon)|D_{jj}|$, $\| X-\hat X\|\leq O(\varepsilon) \|X\|$, and $\| Y-\hat Y\|\leq O(\varepsilon) \|Y\|$. The following algorithm computes the EVD of $A$ with high relative accuracy:

    1. Perform QR factorization with pivoting to get $\hat X\hat D=QRP$, where $P$ is a permutation matrix. Thus $A=QRP\hat Y^T$.
    2. Multiply $W=RP\hat Y^T$ (_NOT_ Strassen's multiplication). Thus $A=QW$ and $W$ is well-scaled from the left.
    3. Compute the SVD of $W^T=V\Sigma^T \bar U^T$ using one-sided Jacobi method. Thus $A=Q\bar U \Sigma V^T$.
    4. Multiply $U=Q\bar U$. Thus $A=U\Sigma V^T$ is the computed SVD of $A$.

4. Let $R=D'R'$, where $D'$ is such that the _rows_ of $R'$ have unit norms.  Then the following error bounds hold:

$$
\frac{|\sigma_j-\tilde\sigma_j|}{\sigma_j}\leq O(\varepsilon \kappa(R')\cdot \max\{\kappa(X),\kappa(Y)\})\leq
O(\varepsilon n^{3/2}\kappa(X)\cdot \max\{\kappa(X),\kappa(Y)\}).$$

5. Hilbert matrix is Hankel matrix and Cauchy matrix, it is symmetric positive definite and _very_ ill-conditioned.

6. Every sumbatrix of a Cauchy matrix is itself a Cauchy matrix. 

7. Determinant of a square Cauchy matrix is

$$
\det(C)=\frac{\prod_{1\leq i<j\leq n}(x_j-x_i)(y_j-y_i)}
{\prod_{1\leq i,j\leq n} (x_i+y_j)}.$$

It is computed with elementwise high relative accuracy.

8. Let $A$ be square and nonsingular and let $A=LDR$ be its decomposition with diagonal $D$, lower unit-triangular $L$, and upper unit-triangular $R$. The closed formulas using quotients of minors are (see [A. S. Householder, The Theory of Matrices in Numerical Analysis](https://books.google.hr/books?id=hCre109IpRcC&printsec=frontcover&hl=hr&source=gbs_ge_summary_r&cad=0#v=onepage&q&f=false)):

$$
\begin{aligned}
D_{11}&=A_{11}, \\
D_{jj}&=\frac{\det(A_{1:j,1:j})}{\det(A_{1:j-1,1:j-1})}, \quad j=2,\ldots,n, \\
L_{jj}&=1, \\
L_{ij}&=\frac{\det(A_{[1,2,\ldots,j-1,i],[1:j]}\, )}
{\det(A_{1:j,1:j})}, \quad j < i, \\
R_{jj}&=1, \\
R_{ji}&=\frac{\det(A_{[1,2,\ldots,j],[1,2, \ldots,j-1,i]}\, )}
{\det(A_{1:j,1:j})}, \quad i > j, 
\end{aligned}$$

"""

# ╔═╡ f280b119-76a7-4ee8-b6fd-608d977af0c6
md"""
## Examples 

### Positive definite matrix

Let $A=DA_S D$ be strongly scaled symmetric positive definite matrix. Then Cholesky factorization with complete (diagonal) pivoting is an RRD. Consider the following three step algorithm:

1. Compute $P^T A P=LL^T$ (_Cholesky factorization with complete pivoting_). 
2. Compute the $L=\bar U\Sigma V^T$ (_one-sided Jacobi, V is not needed_).
3. Set $\Lambda=\Sigma^2$ and $U=P\bar U$. Thus $A=U\Lambda U^T$ is an EVD of $A$.

The Cholesky factorization with pivoting can be implemented very fast with block algorithm (see [C. Lucas, LAPack-Style Codes for Level 2 and 3 Pivoted Cholesky Factorizations](http://www.netlib.org/lapack/lawnspdf/lawn161.pdf)).

The eigenvalues $\tilde \lambda_j$ computed using the above algorithm satisfy relative error bounds:

$$
\frac{|\lambda_j-\tilde\lambda_j|}{\lambda_j} \leq O(n\varepsilon \|A_S\|_2^{-1}).$$

"""

# ╔═╡ 28c1a9e7-6c65-4184-b41b-b5cfd17645b5
function JacobiR(A₁::AbstractMatrix)
    A=deepcopy(A₁)
    m,n=size(A)
    T=typeof(A[1,1])
    V=Matrix{T}(I,n,n)
    # Tolerance for rotation
    tol=√n*eps(T)
    # Counters
    p=n*(n-1)/2
    sweep=0
    pcurrent=0
    # First criterion is for standard accuracy, second one is for relative accuracy
    # while sweep<30 && vecnorm(A-diagm(diag(A)))>tol
    while sweep<30 && pcurrent<p
        sweep+=1
        # Row-cyclic strategy
        for i = 1 : n-1 
            for j = i+1 : n
                # Compute the 2 x 2 sumbatrix of A'*A
                # F=A[:,[i,j]]'*A[:,[i,j]]
				F=view(A,:,[i,j])'*view(A,:,[i,j])
                # Check the tolerance - the first criterion is standard,
                # the second one is for relative accuracy               
                # if A[i,j]!=zero(T)
                # 
                if abs(F[1,2])>tol*√(F[1,1]*F[2,2])
                    # Compute c and s
                    τ=(F[2,2]-F[1,1])/(2*F[1,2])
                    t=sign(τ)/(abs(τ)+√(1+τ^2))
                    c=1/√(1+t^2)
                    s=c*t
                    G=LinearAlgebra.Givens(i,j,c,s)
                    # A*=G'
                    rmul!(A,G)
                    # V*=G'
                    rmul!(V,G)
                    pcurrent=0
                else
                    pcurrent+=1
                end
            end
        end
    end
    σ=[norm(A[:,k]) for k=1:n]
    for k=1:n
        A[:,k]./=σ[k]
    end
    # A, σ, V
    SVD(A,σ,adjoint(V))
end

# ╔═╡ bc8b94b7-7e20-4cd3-be68-7e9152fe6d7b
begin
	n=20
	import Random
	Random.seed!(421)
	B=randn(n,n)
	# Scaled matrix
	As=Matrix(Symmetric(B'*B))
	# Scaling
	D₀=exp.(50*(rand(n).-0.5))
	# Parentheses are necessary!
	A=[As[i,j]*(D₀[i]*D₀[j]) for i=1:n, j=1:n]
	issymmetric(A), cond(As), cond(A)
end

# ╔═╡ 66dedda0-24a0-48a0-9286-4ce688e5da72
# ?cholesky;

# ╔═╡ cc924252-137a-4241-b178-2eabf653ff71
md"""
We will not use the Cholesky factorization with complete pivoting. Instead, we will just sort the diagonal of $A$ in advance, which is sufficient for this example. 

_Write the function for Cholesky factorization with complete pivoting as an excercise._
"""

# ╔═╡ c41e4534-6b29-4cbe-9b17-2c69c89e5570
# ?sortperm;

# ╔═╡ bcee7b90-2dd4-47b0-a781-67d35962d5f2
begin
	p=sortperm(diag(A), rev=true)
	L=cholesky(A[p,p])
end

# ╔═╡ f471d60f-1c70-4bd5-bb67-2bb17c18f3f8
U,σ,V=JacobiR(Matrix(L.L));

# ╔═╡ c474702a-8af8-4640-9f45-bca748ab3952
begin
	λ=σ.^2
	U₁=U[invperm(p),:]
	λ
end

# ╔═╡ 25143fe5-3965-468a-8cb1-7c3e8e8027ea
U'*A[p,p]*U

# ╔═╡ 86dc734d-f45d-491f-9f80-7d958e642fbd
# Due to large condition number, this is not
# as accurate as expected
Ξ=U₁'*A*U₁

# ╔═╡ 001f801a-06d2-48be-a5ad-4c07f7a92890
# Orthogonality
norm(U₁'*U₁-I)

# ╔═╡ 5196b508-5aad-4811-bfc0-1c1fa00eec37
begin
	DΞ=sqrt.(diag(Ξ))
	Ξs=[Ξ[i,j]/(DΞ[i]*DΞ[j]) for i=1:n, j=1:n]
end

# ╔═╡ bcfdae20-040f-492e-bc67-60cf5f420cc0
begin
	K=U₁*Diagonal(σ)
	K'*K
end

# ╔═╡ 3ad5b60e-3fb9-443e-90eb-40b2fd78d9a3
# Explain why is the residual so large.
norm(A*U₁-U₁*Diagonal(λ))

# ╔═╡ 3e863b09-dc68-4560-adeb-e56ab22d9afd
# Relative residual is percfect
norm(A*U₁-U₁*Diagonal(λ))/norm(A)

# ╔═╡ a5555731-c229-43de-9d88-cc3b91d7db67
[λ sort(eigvals(A),rev=true)]

# ╔═╡ c3fcbaf1-b98f-401f-9244-aad632805ecb
md"""
### Hilbert matrix

We need the newest version of the package 
[SpecialMatrices.jl](https://github.com/JuliaMatrices/SpecialMatrices.jl).
"""

# ╔═╡ 8d7e4960-9b04-4f5e-b2e6-acce80a6812f
varinfo(SpecialMatrices)

# ╔═╡ 80d64fc6-33e7-4069-a6f4-8e85369dad9f
C=Cauchy([1,2,3,4,5],[0,1,2,3,4])

# ╔═╡ 804c4972-bf3d-4284-8426-b50b3ea5bb1b
H=Hilbert(5)

# ╔═╡ e9bf2b02-a60d-495b-bbdb-d87e31d8d9e0
Hf=Matrix(H)

# ╔═╡ c113b7fe-87b9-454c-aeb9-166af79bbe61
begin
	# Exact formula for the determinant of a Cauchy matrix from Fact 7.
	import LinearAlgebra.det
	function det(C::Cauchy{T}) where T
	    n=length(C.x)
	    F=triu([(C.x[j]-C.x[i])*(C.y[j]-C.y[i]) for i=1:n, j=1:n],1)
	    num=prod(F[findall(!iszero,F)])
	    den=prod([(C.x[i]+C.y[j]) for i=1:n, j=1:n])
	    if all(isinteger,C.x)&all(isinteger,C.y)
	        return num//den
	    else
	        return num/den
	    end
	end
end

# ╔═╡ c6c0e7b2-400c-478e-90af-496659fed9c4
# This is exact
det(Hf)

# ╔═╡ d057146d-4a5c-415f-9332-9b3bb7e82fc4
det(C)

# ╔═╡ 66115e0a-a5c8-4a66-958b-27d9a7865855
md"""
Compute componentwise highly accurate $A=LDL ^T$ factorization of a Hilbert (Cauchy) matrix. Using `Rational` numbers gives high accuracy.
"""

# ╔═╡ 43656da7-93db-4de3-b420-098f64dfba85
# Exact LDLT factorization from Fact 8, no pivoting.
function LDLT(C::Cauchy)
    n=length(C.x)
    T=typeof(C.x[1])
    D=Array{Rational{T}}(undef,n)
    L=Matrix{Rational{T}}(I,n,n)
    δ=[det(Cauchy(C.x[1:j],C.y[1:j])) for j=1:n]
    D[1]=map(Rational{T},C[1,1])
    D[2:n]=δ[2:n]./δ[1:n-1]
    for i=2:n
        for j=1:i-1
            L[i,j]=det(Cauchy( C.x[[1:j-1;i]], C.y[1:j])) / δ[j]
        end
    end
    L,D
end

# ╔═╡ d0b35d9b-8b87-4831-846c-b63fe8912ec7
L₁,D₁=LDLT(C)

# ╔═╡ d25dcaa1-2448-49ec-b863-b224edc4ec2c
L₁*Diagonal(D₁)*L₁' # -Matrix(H)

# ╔═╡ d668a5a2-8060-48ef-adf0-6be0b1c3859e
# L*D*L' is an RRD
cond(L₁)

# ╔═╡ a4a40d52-94fc-4ccc-8e35-09c75c5de755
cond(C)

# ╔═╡ dbfbbb00-ab00-4c85-a4ce-32468f18088a
md"""
We now compute the accurate EVD of the Hilbert matrix of order $n=100$. We cannot use the function `LDLT()` since the _computation of determinant causes overflow_ and _there is no pivoting_. Instead, we use Algorithm 3 from [J. Demmel, Computing the singular value decomposition with high relative accuracy, SIAM J. Matrix Anal. Appl, 21 (1999) 562-580](http://www.netlib.org/lapack/lawnspdf/lawn130.pdf).
"""

# ╔═╡ 0176739c-2846-4b07-a8d3-68454c1251a6
function GECP(C::Cauchy)
    n=length(C.x)
    G=Matrix(C)
    x=copy(C.x)
    y=copy(C.y)
    pr=collect(1:n)
    pc=collect(1:n)
    # Find the maximal element
    for k=1:n-1
        i,j=Tuple(argmax(abs.(G[k:n,k:n])))
        i+=k-1
        j+=k-1
        if i!=k || j!=k
            G[[i,k],:]=G[[k,i],:]
            G[:, [j,k]]=G[:, [k,j]]
            x[[k,i]]=x[[i,k]]
            y[[k,j]]=y[[j,k]]
            pr[[i,k]]=pr[[k,i]]
            pc[[j,k]]=pc[[k,j]]
        end
        for r=k+1:n
            for s=k+1:n
                G[r,s]=G[r,s]*(x[r]-x[k])*(y[s]-y[k])/
                ((x[k]+y[s])*(x[r]+y[k]))
            end
        end
        G=Matrix(Symmetric(G))
    end
    D=diag(G)
    X=tril(G,-1)*Diagonal(1.0./D)+I
    Y=Diagonal(1.0./D)*triu(G,1)+I
    X,D,Y', pr,pc
end

# ╔═╡ 8b844cf7-c574-45e9-b682-fa9cc6e9cb73
X,D,Y,pr,pc=GECP(C)

# ╔═╡ b3877ea1-d479-4d08-af8c-26e17de77106
# Check
norm(X*Diagonal(D)*Y'-Matrix(C)[pr,pc]),
norm(X[invperm(pr),:]*Diagonal(D)*Y[invperm(pc),:]'-C)

# ╔═╡ 9bff72d8-68b6-41b6-a954-ee39f90ec7b0
begin
	# Now the big test.
	n₂=100
	H₂=Hilbert(n₂)
	C₂=Cauchy(collect(1:n₂), collect(0:n₂-1))
end

# ╔═╡ 72390bf0-c5d6-46de-b8d0-0bee2cbb0af7
md"""
We need a function to compute RRD from `GECP()`
"""

# ╔═╡ 46656cee-2202-4eb9-9725-1e3e3af4df42
function RRD(C::Cauchy)
    X,D,Y,pr,pc=GECP(C)
    X[invperm(pr),:], D, Y[invperm(pc),:]
end

# ╔═╡ 6b544eaf-858a-40e5-b59b-75cfa2237a6f
X₂,D₂,Y₂=RRD(C₂);

# ╔═╡ 8eb2d15e-53c0-40ab-98ce-7b0bbf4d6cd1
# Check
norm((X₂*Diagonal(D₂)*Y₂')-C₂)

# ╔═╡ c7fd604c-9e63-4ac0-8839-82371f968ba7
cond(C)

# ╔═╡ 23df1242-1b43-4d52-a8ed-1b12b0d5d4b9
# Is this RRD? here X=Y
cond(X₂), cond(Y₂)

# ╔═╡ 14374962-2eea-4c8f-9390-73ef886c25d9
# Algorithm from Fact 3
function RRDSVD(X,D,Y)
    Q,R,p=qr(X*Diagonal(D),Val(true))
    W=R[:,p]*Y'
    V,σ,U₁=JacobiR(W')
    U=Q*U₁
    U,σ,V
end

# ╔═╡ d4a86d03-0afe-4b1d-8828-52eae055aa9f
U₂,σ₂,V₂=RRDSVD(X₂,D₂,Y₂);

# ╔═╡ 1dfe86bf-7553-444b-a94e-340acccf5375
# Residual and orthogonality
norm(Matrix(C₂)*V₂-U₂*Diagonal(σ₂)), norm(U₂'*U₂-I), norm(V₂'*V₂-I)

# ╔═╡ cc123376-b657-4d8f-88e0-0f7577058c2b
# Observe the differences!!
[sort(σ₂) sort(svdvals(C₂)) sort(eigvals(Matrix(C₂)))]

# ╔═╡ 337ade70-accb-417b-b655-9640ed61b375
plot(σ₂,yscale = :log10,legend=false, title="Singular values of Hilbert matrix")

# ╔═╡ f7556414-693d-4d46-889b-ed6b091a235e
begin
	# Better spy
	# some options :bluesreds,clim=(-1.0,1.0)
	import Plots.spy
	spy(A)=heatmap(A, yflip=true, color=:bluesreds, aspectratio=1) 
end

# ╔═╡ 9ee70a5d-59f7-460c-92af-8872966cec40
spy(U₂)

# ╔═╡ abb64f41-6817-49e9-9d44-81ef0ce32934
md"""
# Symmetric arrowhead and DPR1 matrices

For more details, see 
[N. Jakovčević Stor, I. Slapničar and J. Barlow, Accurate eigenvalue decomposition of real symmetric arrowhead matrices and applications](https://arxiv.org/abs/1302.7203) and [N. Jakovčević Stor, I. Slapničar and J. Barlow, Forward stable eigenvalue decomposition of rank-one modifications of diagonal matrices](https://arxiv.org/abs/1405.7537).
"""

# ╔═╡ b24b1c96-c73e-4c8e-83af-f35e2b9df304
md"""
## Definitions

An __arrowhead matrix__ is a real symmetric matrix of order $n$ of the form $A=\begin{bmatrix} D & z \\ z^{T} & \alpha \end{bmatrix}$, where $D=\mathop{\mathrm{diag}}(d_{1},d_{2},\ldots ,d_{n-1})$, $z=\begin{bmatrix} \zeta _{1} & \zeta _{2} & \cdots & \zeta _{n-1} \end{bmatrix}^T$ is a vector, and $\alpha$ is a scalar.

An arrowhead matrix is __irreducible__ if $\zeta _{i}\neq 0$ for all $i$ and $d_{i}\neq d_{j}$ for all $i\neq j$.

A __diagonal-plus-rank-one matrix__ (DPR1 matrix) is a real symmetric matrix of order $n$ of the form $A= D +\rho z z^T$, where $D=\mathop{\mathrm{diag}}(d_{1},d_{2},\ldots ,d_{n})$, $z=\begin{bmatrix} \zeta _{1} & \zeta _{2} & \cdots & \zeta _{n} \end{bmatrix}^T$ is a vector, and $\rho \neq 0$ is a scalar.

A DPR1 matrix is __irreducible__ if $\zeta _{i}\neq 0$ for all $i$ and $d_{i}\neq d_{j}$ for all $i\neq j$.
"""

# ╔═╡ 5e068291-cb05-4fd5-acaa-8e5d766e375a
md"""
## Facts on arrowhead matrices

Let $A$ be an arrowhead matrix of order $n$ and let $A=U\Lambda U^T$ be its EVD.

1. If $d_i$ and $\lambda_i$ are nonincreasingy ordered, the Cauchy Interlace Theorem implies 

$$\lambda _{1}\geq d_{1}\geq \lambda _{2}\geq d_{2}\geq \cdots \geq d_{n-2}\geq\lambda
_{n-1}\geq d_{n-1}\geq \lambda _{n}.$$

2. If $\zeta _{i}=0$ for some $i$, then $d_{i}$ is an eigenvalue whose corresponding eigenvector is the $i$-th unit vector, and we can reduce the size of the problem by deleting the $i$-th row and column of the matrix. If $d_{i}=d_{j}$, then $d_{i}$ is an eigenvalue of $A$ (this follows from the interlacing property) and we can reduce the size of the problem by annihilating $\zeta_j$ with a Givens rotation in the $(i,j)$-plane.

3. If $A$ is irreducible, the interlacing property holds with strict inequalities. 

4. The eigenvalues of $A$ are the zeros of the __Pick function__ (also, _secular equation_)

$$
f(\lambda )=\alpha -\lambda -\sum_{i=1}^{n-1}\frac{\zeta _{i}^{2}}{%
d_{i}-\lambda }=\alpha -\lambda -z^{T}(D-\lambda I)^{-1}z,$$

and the corresponding eigenvectors are 

$$
U_{:,i}=\frac{x_{i}}{\left\Vert x_{i}\right\Vert _{2}},\quad 
x_{i}=\begin{bmatrix}
\left( D-\lambda _{i}I\right) ^{-1}z \\ 
-1%
\end{bmatrix}, 
\quad i=1,\ldots ,n.$$

5. Let $A$ be irreducible and nonsingular. If $d_i\neq 0$ for all $i$, then $A^{-1}$ is a DPR1 matrix

$$
A^{-1}=\begin{bmatrix} D^{-1} &  \\ & 0 \end{bmatrix} + \rho uu^{T},$$

where $u=\begin{bmatrix} D^{-1}z  \\ -1 \end{bmatrix}$, and $\rho =\displaystyle\frac{1}{\alpha-z^{T}D^{-1}z}$. If $d_i=0$, then $A^{-1}$ is a permuted arrowhead matrix,

$$
A^{-1}\equiv 
\begin{bmatrix}
D_{1} & 0 & 0 & z_{1} \\ 
0 & 0 & 0 & \zeta _{i} \\ 
0 & 0 & D_{2} & z_{2} \\ 
z_{1}^{T} & \zeta _{i} & z_{2}^{T} & \alpha
\end{bmatrix}^{-1}
= \begin{bmatrix}
D_{1}^{-1} & w_{1} & 0 & 0 \\ 
w_{1}^{T} & b & w_{2}^{T} & 1/\zeta _{i} \\ 
0 & w_{2} & D_{2}^{-1} & 0 \\ 
0 & 1/\zeta _{i} & 0 & 0
\end{bmatrix},$$

where

$$
\begin{aligned}
w_{1}&=-D_{1}^{-1}z_{1}\displaystyle\frac{1}{\zeta _{i}},\\ 
w_{2}&=-D_{2}^{-1}z_{2}\displaystyle\frac{1}{\zeta _{i}},\\
b&= \displaystyle\frac{1}{\zeta _{i}^{2}}\left(-\alpha +z_{1}^{T}D_{1}^{-1}z_{1}+z_{2}^{T}D_{2}^{-1}z_{2}\right).
\end{aligned}$$

6. The algorithm based on the following approach computes all eigenvalues and _all components_ of the corresponding eigenvectors in a forward stable manner to almost full accuracy in $O(n)$ operations per eigenpair:

    1. Shift the irreducible $A$ to $d_i$ which is closer to $\lambda_i$ (one step of bisection on $f(\lambda)$).
    2. Invert the shifted matrix.
    3. Compute the absolutely largest eigenvalue of the inverted shifted matrix and the corresponding eigenvector.

7. The algorithm is implemented in the package [Arrowhead.jl](https://github.com/ivanslapnicar/Arrowhead.jl). In certain cases, $b$ or $\rho$ need to be computed with extended precision. For this, we use the functions from file [DoubleDouble.jl](https://github.com/ivanslapnicar/Arrowhead.jl/blob/master/src/DoubleDouble.jl), originally from the the package [DoubleDouble.jl](https://github.com/simonbyrne/DoubleDouble.jl).
"""

# ╔═╡ 6d7f330f-3c88-44bd-aff9-65daa7f1ea1c
md"
## Examples

### Extended precision arithmetic
"

# ╔═╡ eea30449-2410-448d-885f-e27b7aa657c0
begin
	# Extended precision arithmetic
	a=2.0
	b=3.0
	√a
end

# ╔═╡ 21f8e11a-09fe-49a9-91cd-b118876910a8
√BigFloat(a)

# ╔═╡ 6d484719-e545-4742-9e46-515525af8244
# Double numbers according to Dekker, 1971
ad=Arrowhead.Double(a)

# ╔═╡ 310ace86-6193-43ee-bdac-fe1a79a2cb26
bd=Arrowhead.Double(b)

# ╔═╡ be91a6f1-2594-4084-8b23-312c51f35553
roota=√ad

# ╔═╡ 5762ccae-2a28-4041-8a02-4bec918e192a
rootb=√bd

# ╔═╡ 4c1a73a1-bdfe-4939-8282-54b7fa193e5a
# 30 digits should match
BigFloat(roota.hi)+BigFloat(roota.lo)

# ╔═╡ f5b83025-64e7-44d1-a841-f9a7fc8614fb
√BigFloat(a)*√BigFloat(b)

# ╔═╡ 89e578d9-7608-42f2-a2f0-0ec2c89bd510
rootab=roota*rootb

# ╔═╡ 24f1cb57-ed20-4399-996f-80294c7b1bb2
BigFloat(rootab.hi)+BigFloat(rootab.lo)

# ╔═╡ 8339adb3-b0fa-45e4-b107-62954547345b
md"""
### Random arrowhead matrix
"""

# ╔═╡ 321070d6-c0c9-482d-9a5f-0e5a4f7b522f
varinfo(Arrowhead)

# ╔═╡ 9c207955-174d-474f-be95-061c71761023
methods(GenSymArrow)

# ╔═╡ bd8ea662-f4c2-4f2d-943b-b1a5f59cbe74
begin
	n₃=10
	A₃=GenSymArrow(n₃,n₃)
end

# ╔═╡ 9fe07467-4033-4374-b4f2-b0ceecf91a03
# Elements of the type SymArrow
A₃.D, A₃.z, A₃.a, A₃.i

# ╔═╡ a3ba9c15-1d55-4e2b-b32e-4202fbeca671
E₃,info₃=eigen(A₃)

# ╔═╡ 8677d8b8-f68b-4013-864c-89902ffff8fd
# Residual and orthogonality
norm(A₃*E₃.vectors-E₃.vectors*Diagonal(E₃.values)), 
norm(E₃.vectors'*E₃.vectors-I)

# ╔═╡ 360829ce-2790-4d70-8699-687650fc51b4
begin
	# Timings - notice the O(n^2)
	@time eigen(GenSymArrow(1000,1000))
	@time eigen(GenSymArrow(2000,2000))
	1
end

# ╔═╡ 9f2a5e5d-a705-4d65-8ebf-6b6d9d548999
md"""
### Numerically demanding matrix
"""

# ╔═╡ 19d95a91-8c03-4b12-a8d3-46a6861991a1
A₄=SymArrow( [ 1e10+1.0/3.0, 4.0, 3.0, 2.0, 1.0 ], 
    [ 1e10 - 1.0/3.0, 1.0, 1.0, 1.0, 1.0 ], 1e10, 6 )

# ╔═╡ 4a86007c-1137-4964-be3c-da51ab7fca3c
begin
	E₄,info₄=eigen(A₄)
	[sort(E₄.values) sort(eigvals(Matrix(A₄))) sort(E₄.values)-sort(eigvals(Matrix(A₄)))]
end

# ╔═╡ 33e70165-1fab-4996-99e6-a5d27bf976c5
# Residual and orthogonality
norm(A₄*E₄.vectors-E₄.vectors*Diagonal(E₄.values)),
norm(E₄.vectors'*E₄.vectors-I)

# ╔═╡ 65e00890-d4d7-4d36-a359-4606378401b7
md"""
## Facts on DPR1 matrices

The properties of DPR1 matrices are very similar to those of arrowhead matrices. Let $A$ be a DPR1 matrix of order $n$ and let $A=U\Lambda U^T$ be its EVD.

1. If $d_i$ and $\lambda_i$ are nonincreasingy ordered and $\rho>0$, then 

$$\lambda _{1}\geq d_{1}\geq \lambda _{2}\geq d_{2}\geq \cdots \geq d_{n-2}\geq\lambda
_{n-1}\geq d_{n-1}\geq \lambda _{n}\geq d_n.$$

If $A$ is irreducible, the inequalities are strict.

2. Fact 2 on arrowhead matrices holds.

3. The eigenvalues of $A$ are the zeros of the __secular equation__ 

$$
f(\lambda )=1+\rho\sum_{i=1}^{n}\frac{\zeta _{i}^{2}}{d_{i}-\lambda }
=1 +\rho z^{T}(D-\lambda I)^{-1}z=0,$$

and the corresponding eigenvectors are 

$$
U_{:,i}=\frac{x_{i}}{\left\Vert x_{i}\right\Vert _{2}},\quad
x_{i}=( D-\lambda _{i}I) ^{-1}z.$$

4. Let $A$ be irreducible and nonsingular. If $d_i\neq 0$ for all $i$, then

$$
A^{-1}=D^{-1} +\gamma uu^{T},\quad  u=D^{-1}z, \quad \gamma =-\frac{\rho}{1+\rho z^{T}D^{-1}z},$$

is also a DPR1 matrix. If $d_i=0$, then $A^{-1}$ is a permuted arrowhead matrix,

$$
A^{-1}\equiv \left(\begin{bmatrix} D_{1} & 0 & 0 \\  0 & 0 & 0  \\  0 & 0 & D_{2} \end{bmatrix}
+\rho \begin{bmatrix} z_{1} \\ \zeta _{i} \\ z_{2}
\end{bmatrix}
\begin{bmatrix}
z_{1}^{T} & \zeta _{i} & z_{2}^{T}
\end{bmatrix}\right)^{-1}=
\begin{bmatrix}
D_{1}^{-1} & w_{1} & 0 \\ 
w_{1}^{T} & b & w_{2}^{T} \\ 
0 & w_{2} & D_{2}^{-1} 
\end{bmatrix},$$

where

$$
\begin{aligned}
w_{1}&=-D_{1}^{-1}z_{1}\displaystyle\frac{1}{\zeta _{i}},\\
w_{2}&=-D_{2}^{-1}z_{2}\displaystyle\frac{1}{\zeta _{i}},\\
b &=\displaystyle\frac{1}{\zeta _{i}^{2}}\left(
\frac{1}{\rho}+z_{1}^{T}D_{1}^{-1}z_{1}+z_{2}^{T}D_{2}^{-1}z_{2}\right).
\end{aligned}$$

5. The algorithm based on the same approach as above, computes all eigenvalues and all components of the corresponding eigenvectors in a forward stable manner to almost full accuracy in $O(n)$ operations per eigenpair. The algorithm is implemented in the package `Arrowhead.jl`. In certain cases, $b$ or $\gamma$ need to be computed with extended precision.
"""

# ╔═╡ 6b3c7336-f1fc-428b-8b1e-94546b6622e8
md"""

## Examples

### Random DPR1 matrix
"""

# ╔═╡ 71d0022d-3cd8-488c-80b8-4c3c958ed8fa
begin
	n₅=10
	A₅=GenSymDPR1(n₅)
end

# ╔═╡ 4655eb27-0ae8-46db-9deb-575c89d08e7e
# Elements of the type SymDPR1
A₅.D, A₅.u, A₅.r

# ╔═╡ 6642fa72-8b1b-40b2-87f2-91199a96d7f9
begin
	E₅,info₅=eigen(A₅)
	norm(A₅*E₅.vectors-E₅.vectors*Diagonal(E₅.values)), 
	norm(E₅.vectors'*E₅.vectors-I)
end

# ╔═╡ 827bb865-c361-4a98-a6d6-90dca425eddc
md"""
### Numerically demanding DPR1 matrix
"""

# ╔═╡ 56c0c2ec-220f-4e04-a0cf-a144d89dd225
# Choose one
A₆=SymDPR1( [ 1e10, 5.0, 4e-3, 0.0, -4e-3,-5.0 ], [ 1e10, 1.0, 1.0, 1e-7, 1.0,1.0 ], 1.0 )

# ╔═╡ 31114bf7-450b-458d-9a1d-c22123f5f291
begin
	E₆,info₆=eigen(A₆)
	[sort(E₆.values) sort(eigvals(Matrix(A₆)))]
end

# ╔═╡ ad305a98-6cdc-424e-8f06-386b801b53e5
# Residual and orthogonality
norm(A₆*E₆.vectors-E₆.vectors*Diagonal(E₆.values)),
norm(E₆.vectors'*E₆.vectors-I)

# ╔═╡ Cell order:
# ╟─d01a9b4f-8b55-4607-abb6-717d227fcd48
# ╟─5d95dc2c-bf94-4b13-b9d5-b7b261e86cf6
# ╟─e3e43840-d0a3-4cde-9b1e-5785759912b2
# ╟─03f797ac-8764-4688-9e7e-e144cafb3b4c
# ╟─f280b119-76a7-4ee8-b6fd-608d977af0c6
# ╠═28c1a9e7-6c65-4184-b41b-b5cfd17645b5
# ╠═bc8b94b7-7e20-4cd3-be68-7e9152fe6d7b
# ╠═66dedda0-24a0-48a0-9286-4ce688e5da72
# ╟─cc924252-137a-4241-b178-2eabf653ff71
# ╠═c41e4534-6b29-4cbe-9b17-2c69c89e5570
# ╠═bcee7b90-2dd4-47b0-a781-67d35962d5f2
# ╠═f471d60f-1c70-4bd5-bb67-2bb17c18f3f8
# ╠═c474702a-8af8-4640-9f45-bca748ab3952
# ╠═25143fe5-3965-468a-8cb1-7c3e8e8027ea
# ╠═86dc734d-f45d-491f-9f80-7d958e642fbd
# ╠═001f801a-06d2-48be-a5ad-4c07f7a92890
# ╠═5196b508-5aad-4811-bfc0-1c1fa00eec37
# ╠═bcfdae20-040f-492e-bc67-60cf5f420cc0
# ╠═3ad5b60e-3fb9-443e-90eb-40b2fd78d9a3
# ╠═3e863b09-dc68-4560-adeb-e56ab22d9afd
# ╠═a5555731-c229-43de-9d88-cc3b91d7db67
# ╟─c3fcbaf1-b98f-401f-9244-aad632805ecb
# ╠═479a40d9-b81e-442b-9962-f972b110a4dd
# ╠═8d7e4960-9b04-4f5e-b2e6-acce80a6812f
# ╠═80d64fc6-33e7-4069-a6f4-8e85369dad9f
# ╠═804c4972-bf3d-4284-8426-b50b3ea5bb1b
# ╠═e9bf2b02-a60d-495b-bbdb-d87e31d8d9e0
# ╠═c6c0e7b2-400c-478e-90af-496659fed9c4
# ╠═c113b7fe-87b9-454c-aeb9-166af79bbe61
# ╠═d057146d-4a5c-415f-9332-9b3bb7e82fc4
# ╟─66115e0a-a5c8-4a66-958b-27d9a7865855
# ╠═43656da7-93db-4de3-b420-098f64dfba85
# ╠═d0b35d9b-8b87-4831-846c-b63fe8912ec7
# ╠═d25dcaa1-2448-49ec-b863-b224edc4ec2c
# ╠═d668a5a2-8060-48ef-adf0-6be0b1c3859e
# ╠═a4a40d52-94fc-4ccc-8e35-09c75c5de755
# ╟─dbfbbb00-ab00-4c85-a4ce-32468f18088a
# ╠═0176739c-2846-4b07-a8d3-68454c1251a6
# ╠═8b844cf7-c574-45e9-b682-fa9cc6e9cb73
# ╠═b3877ea1-d479-4d08-af8c-26e17de77106
# ╠═9bff72d8-68b6-41b6-a954-ee39f90ec7b0
# ╟─72390bf0-c5d6-46de-b8d0-0bee2cbb0af7
# ╠═46656cee-2202-4eb9-9725-1e3e3af4df42
# ╠═6b544eaf-858a-40e5-b59b-75cfa2237a6f
# ╠═8eb2d15e-53c0-40ab-98ce-7b0bbf4d6cd1
# ╠═c7fd604c-9e63-4ac0-8839-82371f968ba7
# ╠═23df1242-1b43-4d52-a8ed-1b12b0d5d4b9
# ╠═14374962-2eea-4c8f-9390-73ef886c25d9
# ╠═d4a86d03-0afe-4b1d-8828-52eae055aa9f
# ╠═1dfe86bf-7553-444b-a94e-340acccf5375
# ╠═cc123376-b657-4d8f-88e0-0f7577058c2b
# ╠═574e86bb-159c-4141-8d8a-21bdcc9b5304
# ╠═337ade70-accb-417b-b655-9640ed61b375
# ╠═f7556414-693d-4d46-889b-ed6b091a235e
# ╠═9ee70a5d-59f7-460c-92af-8872966cec40
# ╟─abb64f41-6817-49e9-9d44-81ef0ce32934
# ╟─b24b1c96-c73e-4c8e-83af-f35e2b9df304
# ╟─5e068291-cb05-4fd5-acaa-8e5d766e375a
# ╟─6d7f330f-3c88-44bd-aff9-65daa7f1ea1c
# ╠═5eb73af5-f78a-4811-83a5-ac39063a4516
# ╠═eea30449-2410-448d-885f-e27b7aa657c0
# ╠═21f8e11a-09fe-49a9-91cd-b118876910a8
# ╠═6d484719-e545-4742-9e46-515525af8244
# ╠═310ace86-6193-43ee-bdac-fe1a79a2cb26
# ╠═be91a6f1-2594-4084-8b23-312c51f35553
# ╠═5762ccae-2a28-4041-8a02-4bec918e192a
# ╠═4c1a73a1-bdfe-4939-8282-54b7fa193e5a
# ╠═f5b83025-64e7-44d1-a841-f9a7fc8614fb
# ╠═89e578d9-7608-42f2-a2f0-0ec2c89bd510
# ╠═24f1cb57-ed20-4399-996f-80294c7b1bb2
# ╟─8339adb3-b0fa-45e4-b107-62954547345b
# ╠═321070d6-c0c9-482d-9a5f-0e5a4f7b522f
# ╠═9c207955-174d-474f-be95-061c71761023
# ╠═bd8ea662-f4c2-4f2d-943b-b1a5f59cbe74
# ╠═9fe07467-4033-4374-b4f2-b0ceecf91a03
# ╠═a3ba9c15-1d55-4e2b-b32e-4202fbeca671
# ╠═8677d8b8-f68b-4013-864c-89902ffff8fd
# ╠═360829ce-2790-4d70-8699-687650fc51b4
# ╟─9f2a5e5d-a705-4d65-8ebf-6b6d9d548999
# ╠═19d95a91-8c03-4b12-a8d3-46a6861991a1
# ╠═4a86007c-1137-4964-be3c-da51ab7fca3c
# ╠═33e70165-1fab-4996-99e6-a5d27bf976c5
# ╟─65e00890-d4d7-4d36-a359-4606378401b7
# ╟─6b3c7336-f1fc-428b-8b1e-94546b6622e8
# ╠═71d0022d-3cd8-488c-80b8-4c3c958ed8fa
# ╠═4655eb27-0ae8-46db-9deb-575c89d08e7e
# ╠═6642fa72-8b1b-40b2-87f2-91199a96d7f9
# ╟─827bb865-c361-4a98-a6d6-90dca425eddc
# ╠═56c0c2ec-220f-4e04-a0cf-a144d89dd225
# ╠═31114bf7-450b-458d-9a1d-c22123f5f291
# ╠═ad305a98-6cdc-424e-8f06-386b801b53e5
