### A Pluto.jl notebook ###
# v0.19.20

using Markdown
using InteractiveUtils

# ╔═╡ 56fea874-cdd8-46c2-babf-10c5bd6440d2
begin
	using PlutoUI
	PlutoUI.TableOfContents(aside=true)
end

# ╔═╡ b089f3ea-ac9d-48c8-ae6d-769e6d3efd07
begin
	import Random
	Random.seed!(421)
	using LinearAlgebra
	n=6
	T=SymTridiagonal(randn(n),randn(n-1))
end

# ╔═╡ fee10c7e-f1f4-41d4-86a9-c980b5b9b8a5
md"""
# Algorithms for Symmetric Tridiagonal Matrices


Due to their importance, there is plethora of excellent algorithms for symmetric tridiagonal matrices.

For more details, see 
[I. Slapničar, Symmetric Matrix Eigenvalue Techniques, pp. 55.1-55.25](https://www.routledge.com/Handbook-of-Linear-Algebra/Hogben/p/book/9781138199897) and the references therein.




__Prerequisites__

The reader should be familiar with concepts of eigenvalues and eigenvectors, related perturbation theory, and algorithms. 

__Competences__

The reader should be able to apply adequate algorithm to a given symmetric tridiagonal matrix and to assess the speed of the algorithm and the accuracy of the solution.
"""

# ╔═╡ dd5b2908-1dc7-4e2a-8b11-c5686a120c36
md"""
# Bisection and inverse iteration

The bisection method is convenient if only part of the spectrum is needed.
If the eigenvectors are needed, as well, they can be efficiently computed by
the inverse iteration method. 

## Facts

 $A$ is a real symmetric $n\times n$ matrix and $T$ is a real symmetric tridiagonal $n\times n$ matrix.
  
1. __Application of Sylvester's Theorem.__ Let $\alpha,\beta\in\mathbb{R}$ with $\alpha<\beta$. The number of eigenvalues of $A$ in the interval $[\alpha,\beta)$ is equal to $\nu (A- \beta I)-\nu(A-\alpha I)$. By systematically choosing the intervals $[\alpha,\beta)$, the bisection method pinpoints each eigenvalue of $A$ to any desired accuracy.

2. The factorization $T-\mu I=LDL^T$, where $D=\mathop{\mathrm{diag}}(d_1,\dots,d_n)$ and $L$ is the unit lower bidiagonal matrix, is computed as:

$$\begin{aligned}
& d_{1}=T_{11}-\mu, \quad    d_{i}=(T_{ii}-\mu)-
\frac{T_{i,i-1}^2}{d_{i-1}}, \quad i=2,\ldots n, \\
& L_{i+1,i}=\frac{T_{i+1,i}}{d_{i}}, \quad i=1,\ldots,n-1.
\end{aligned}$$ 

Since the matrices $T$ and $D$ have the same inertia, this recursion enables an efficient implementation of the bisection method for $T$.

3. The factorization from Fact 2 is essentially Gaussian elimination without pivoting. Nevertheless, if $d_i\neq 0$ for all $i$, the above recursion is very stable. Even when $d_{i-1}=0$ for some $i$, if the IEEE arithmetic is used, the computation will continue and the inertia will be computed correctly. Namely, in that case, we would have $d_i=-\infty$, $L_{i+1,i}=0$, and $d_{i+1}=T_{i+1.i+1}-\mu$.

4. Computing one eigenvalue of $T$ by using the recursion from Fact 2 and bisection requires $O(n)$ operations. The corresponding eigenvector is computed by inverse iteration. The convergence is very fast, so the cost of computing each eigenvector is also $O(n)$ operations. Therefore, the overall cost for computing all eigenvalues and eigenvectors is $O(n^2)$ operations.  

5. Both, bisection and inverse iteration are highly parallel since each eigenvalue and eigenvector can be computed independently.  

6. If some of the eigenvalues are too close, the corresponding eigenvectors computed by inverse iteration may not be sufficiently orthogonal. In this case, it is necessary to orthogonalize these eigenvectors  (for example, by the modified Gram--Schmidt procedure). If the number of close eigenvalues is too large, the overall operation count can increase to $O(n^3)$.

7. The EVD computed by bisection and inverse iteration satisfies the error bounds from previous notebook.
 
8. The bisection method for tridiagonal matrices is implemented in the  LAPACK subroutine [DSTEBZ](http://www.netlib.org/lapack/explore-html/d4/d48/dstebz_8f.html). This routine can compute all eigenvalues in a given interval or the eigenvalues from $\lambda_l$ to $\lambda_k$, where $l<k$, and the eigenvalues are ordered from smallest to largest. Inverse iteration (with reorthogonalization) is implemented in the LAPACK subroutine DSTEIN. 

"""

# ╔═╡ 219aea92-5737-42f5-8bed-a214a287851c
md"
## Examples
"

# ╔═╡ 2694c3c7-a3a5-4302-9d1a-7a7a9fbdbf34
Λ,U=eigen(T)

# ╔═╡ 0140996f-3c6e-4ed9-9df2-cebd9b6fbd7d
# Fact 2
function myLDLt(T::SymTridiagonal{S},μ::S) where S<:Real
    n=length(T.dv)
    D=Diagonal(Vector{S}(undef,n))
    L=Bidiagonal(fill(one(S),n),Vector{S}(undef,n-1),'L')
    D.diag[1]=T.dv[1]-μ
    for i=2:n
        D.diag[i]=(T.dv[i]-μ)-T.ev[i-1]^2/D.diag[i-1]
    end
    for i=1:n-1
        L.ev[i]=T.ev[i]/D.diag[i]
    end
    return D,L
end

# ╔═╡ 9a185d60-6431-4e1e-a729-268a767bad85
σ=1.0

# ╔═╡ b7889ca1-c79d-43d7-9d65-11f0fed35791
D,L=myLDLt(T,σ)

# ╔═╡ 174dd76d-4221-4496-b990-7ca726f87f65
Matrix(L*D*transpose(L))-(T-σ*I)

# ╔═╡ 009d63ef-b5f3-4f60-9d27-c12688282bed
# Inertias are the same
[Λ.-σ D.diag]

# ╔═╡ f4e77574-7d83-4cf9-9d94-2fe24e27395e
# Fact 8
methods(LAPACK.stebz!)

# ╔═╡ 9c07effc-e1b0-4b31-b228-e256469add5f
# ?LAPACK.stebz!

# ╔═╡ f53d1a63-739d-4a57-9216-6101a2a3412f
λ₁, =LAPACK.stebz!('A','E',1.0,1.0,1,1,2*eps(),copy(T.dv),copy(T.ev))

# ╔═╡ 88bfbb99-a597-4ccc-89e0-b3f4b23e1949
Λ-λ₁

# ╔═╡ c7663867-20b4-4dd3-b173-456f283aa629
U₁=LAPACK.stein!(copy(T.dv),copy(T.ev),λ₁)

# ╔═╡ 7639eeec-d31d-42d9-b55a-6bc60176d5ad
# Residual
norm(T*U₁-U₁*Diagonal(λ₁))

# ╔═╡ 1052bcf0-1f11-4da8-b8d8-cabe87f5c6e4
# Orthogonality
norm(U₁'*U₁-I)

# ╔═╡ 298753db-7ef3-46e8-b81a-7bcc9641572d
# Let us compute just some eigenvalues - from 2nd to 4th
λ₂,=LAPACK.stebz!('V','E',0.0,1.0,2,4,2*eps(),copy(T.dv),copy(T.ev))

# ╔═╡ f12f2f13-098f-443b-a517-c896f8cb38f1
# And the corresponding eigenvectors
LAPACK.stein!(copy(T.dv),copy(T.ev),λ₂)

# ╔═╡ 9582de91-3ac0-40c8-a160-a52b2d785e2c
md"""
# Divide-and-conquer

This is currently the fastest method for computing the EVD of a real symmetric tridiagonal matrix $T$. It is based on splitting the given tridiagonal matrix into two matrices, then computing the EVDs of the smaller matrices and computing the final EVD from the two EVDs.

 $T$ is a real symmetric tridiagonal matrix of order $n$ and $T=U\Lambda U^T$ is its EVD.

## Facts

1. Let $T$ be partitioned as

$$T=\begin{bmatrix} T_1 & \alpha_k e_k e_1^T \\
\alpha_k e_1 e_k^T & T_2
\end{bmatrix}.$$

We assume that $T$ is unreduced, that is, $\alpha_i\neq 0$ for all $i$. Further, we assume that $\alpha_i>0$ for all $i$, which can be easily be attained by diagonal similarity with a diagonal matrix of signs. Let

$$\hat T_1=T_1- \alpha_k e_k e_k^T,\qquad
\hat T_2=T_2- \alpha_k e_1 e_1^T.$$

In other words, $\hat T_1$ is equal to $T_1$ except that $T_{kk}$ is replaced by $T_{kk}-\alpha_k$, and $\hat T_2$ is equal to $T_2$ except that $T_{k+1,k+1}$ is replaced by $T_{k+1,k+1}-\alpha_k$. 

Let 

$$\hat T_i= \hat U_i \hat \Lambda_i \hat U_i^T,\qquad i=1,2,$$ be the respective EVDs
and let  

$$
v=\begin{bmatrix} \hat U_1^T e_k   \\ \hat U_2^T e_1 \end{bmatrix}$$ 
In other words, $v$ consists of the last column of $\hat U_1^T$ and the first column of $\hat U_2^T$). 

Set $\hat U=\hat U_1\oplus \hat U_2$ and $\hat \Lambda=\hat \Lambda_1 \oplus \hat \Lambda_2$. Then

$$T=\begin{bmatrix}\hat U_1 & \\ & \hat U_2 \end{bmatrix}
\left[\begin{bmatrix} \hat \Lambda_1 & \\ & \hat \Lambda_2 \end{bmatrix} + \alpha_k v v^T\right] 
\begin{bmatrix} \hat U_1^T & \\ & \hat U_2^T \end{bmatrix}=
\hat U (\hat \Lambda + \alpha_k v v^T) \hat U^T.$$

If $\hat \Lambda + \alpha_k v v^T=X\Lambda X^T$ is the EVD of the rank-one modification of the diagonal matrix $\hat \Lambda$, then $T=U\Lambda U^T$, where $U=\hat U X$  is the EVD of $T$. Thus, the original tridiagonal eigenvalue problem is reduced to two smaller tridiagonal eigenvalue problems and one eigenvalue problem for the diagonal-plus-rank-one matrix.

2. If all $\hat \lambda_i$ are different, then the eigenvalues $\lambda_i$ of $\hat \Lambda + \alpha_k v v^T$ are solutions of the so-called secular equation,

$$1+\alpha_k \sum_{i=1}^n \frac{v_i^2}{\hat \lambda_i-\lambda}=0.$$

The eigenvalues can be computed by bisection, or by some faster  zero finder of the Newton type, and they need to be computed as accurately as possible. The corresponding eigenvectors are 

$$x_i=(\hat \Lambda-\lambda_i I)^{-1}v.$$

3. Each $\lambda_i$ and $x_i$ is computed in in $O(n)$ operations, respectively, so the overall computational cost for computing the EVD of $\hat \Lambda + \alpha_k v v^T$ is $O(n^2)$ operations. 
  
4. The method can be implemented so that the accuracy of the computed EVD is given by the bound from the previous notebook.

5. Tridiagonal Divide-and-conquer method is implemented in the LAPACK subroutine [DSTEDC](http://www.netlib.org/lapack/explore-html/d7/d82/dstedc_8f.html). This routine can compute just the eigenvalues or both, eigenvalues and eigenvectors. 

6. The file [lapack.jl](https://github.com/JuliaLang/julia/blob/master/stdlib/LinearAlgebra/src/lapack.jl) contains wrappers for a selection of LAPACK routines needed in the current Julia `Base`. However, _all_ LAPACK routines are in the compiled library, so  additional wrappers can be easily written. Notice that arrays are passed directly and scalars as passed as pointers. The wrapper for `DSTEDC`, similar to the ones from the file `lapack.jl` follows.

"""

# ╔═╡ 131ae1b7-a946-4e3f-b260-d037ee7ee6ce
md"
## Examples
"

# ╔═╡ e1288577-8a5b-4b21-8533-68be1619a5e6
# Our matrix with positive 	off-diagonal
T₀=SymTridiagonal(T.dv,abs.(T.ev))

# ╔═╡ a13052ab-297a-4e5a-8654-e65e1fde2e41
eigvals(T)-eigvals(T₀)

# ╔═╡ 04206e97-57c2-41fe-938f-9108cab56740
begin
	T₁=T₀[1:3,1:3]
	T₂=T₀[4:6,4:6]
	α=T₀[3,4]
	T₁[3,3]-=α
	T₂[1,1]-=α
	D₁,X₁=eigen(Matrix(T₁))
	D₂,X₂=eigen(Matrix(T₂))
	x=zeros(6)
	x[3]=√(α)
	x[4]=√(α)
	# X=[X₁ zeros(3,3);zeros(3,3) X₂]
	X=cat(X₁,X₂,dims=(1,2))
end

# ╔═╡ 7cbc4972-f604-4d1f-865d-0d9c5439b437
T₁

# ╔═╡ 06c885f9-f73c-448e-9ba7-7664ba1ad105
T₂

# ╔═╡ e2c2803c-e709-43c1-9b79-524bf6dc03e3
X'*T₀*X

# ╔═╡ 6b848adb-1b45-44c8-9be2-e451ef5d21d7
v=X'*x

# ╔═╡ 2abbf839-847a-4d94-8a4b-f25f57a91540
D₀=Diagonal([D₁;D₂])+v*v'

# ╔═╡ 903e7308-e7be-4821-82e5-105aca75aca2
eigvals(D₀)

# ╔═╡ c000c311-8262-4cdf-ace6-13106a792aad
eigvals(T₀)

# ╔═╡ 574c1653-a2c2-4365-a768-e447bab2d805
md"
### Importing Divide & Conquer from LAPACK
"

# ╔═╡ cb7ac40c-b1fb-4b67-8a76-6c0df47fb270
# See the list of imported LAPACK routines - D&C is not among them
LAPACK.stebz!

# ╔═╡ bc5ea97a-70ef-4a95-b59e-96790ba0346f
begin
	# Part of the preamble of lapack.jl
	const liblapack = Base.liblapack_name
	import LinearAlgebra.BLAS.@blasfunc
	# import ..LinAlg: BlasFloat, Char, BlasInt, LAPACKException,
	    # DimensionMismatch, SingularException, PosDefException, chkstride1, chksquare
	import LinearAlgebra.BlasInt
	function chklapackerror(ret::BlasInt)
	    if ret == 0
	        return
	    elseif ret < 0
	        throw(ArgumentError("invalid argument #$(-ret) to LAPACK call"))
	    else # ret > 0
	        throw(LAPACKException(ret))
	    end
	end
end

# ╔═╡ b6804e98-b5bc-4f42-a864-25018886dc65
for (stedc, elty) in
    ((:dstedc_,:Float64),
    (:sstedc_,:Float32))
    @eval begin
        """
        COMPZ is CHARACTER*1
          = 'N':  Compute eigenvalues only.
          = 'I':  Compute eigenvectors of tridiagonal matrix also.
          = 'V':  Compute eigenvectors of original dense symmetric
                  matrix also.  On entry, Z contains the orthogonal
                  matrix used to reduce the original matrix to
                  tridiagonal form.
        """
        function stedc!(compz::Char, dv::Vector{$elty}, ev::Vector{$elty}, 
                Z::Array{$elty})
            n = length(dv)
            ldz=n
            if length(ev) != n - 1
                throw(DimensionMismatch("ev has length $(length(ev)) 
                        but needs one less than dv's length, $n)"))
            end
            w = deepcopy(dv)
            u = deepcopy(ev)
            lwork=5*n^2
            work = Array{$elty}(undef,lwork)
            liwork=6+6*n+5*n*round(Int,ceil(log(n)/log(2)))
            iwork = Array{BlasInt}(undef,liwork)
            # info = Array{BlasInt}(undef,5)
            info = Ref{BlasInt}()
            ccall((@blasfunc($stedc), liblapack), Cvoid,
                (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, 
                    Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, 
                    Ptr{BlasInt}), compz, n, w, u, Z, ldz, work, 
                    lwork, iwork, liwork, info) 
                chklapackerror(info[])
            return w,Z
        end
    end
end

# ╔═╡ 93aadb04-e1bd-4031-aef4-6246141a9fc3
μ,Q=stedc!('I',copy(T.dv),copy(T.ev),Matrix{Float64}(I,n,n))

# ╔═╡ 79a282fc-a91d-486b-8620-6fad38f00e0b
norm(Λ-μ)

# ╔═╡ 8c988e82-1c05-4e3a-a30f-277cb462cd5e
norm(Q'*Q-I)

# ╔═╡ 85eae2ff-8d8f-4567-95a5-4b87efb711fa
Q'*T*Q

# ╔═╡ bd7472cd-2ede-4c57-af89-4a4325395a2f
# Simple version
function DivideConquer(T::SymTridiagonal{S}) where S
    n=length(T.dv)
    U=Matrix{S}(undef,n,n)
    Λ=Vector{S}(undef,n)
    if n==1
        Λ=T.dv[1]
        U=[one(S)]
    else
        k=div(n,2)
        T₁=SymTridiagonal(T.dv[1:k],T.ev[1:k-1])
        T₁.dv[k]-=T.ev[k]
        T₂=SymTridiagonal(T.dv[k+1:n],T.ev[k+1:n-1])
        T₂.dv[1]-=T.ev[k]
        Λ₁,U₁=DivideConquer(T₁)
        Λ₂,U₂=DivideConquer(T₂)
        v=vcat(transpose(U₁)[:,k],transpose(U₂)[:,1])
        D=Diagonal(vcat(Λ₁,Λ₂))
        Λ=eigvals(D+T.ev[k]*v*transpose(v))
        for i=1:n
            U[:,i]=(D-Λ[i]*I)\v
            normalize!(view(U,:,i))
        end
		# This can be done using Cauchy-like matrices
        U[1:k,:]=U₁*U[1:k,:]
        U[k+1:n,:]=U₂*U[k+1:n,:]
    end
    return Λ,U
end

# ╔═╡ 1cdb29eb-b325-4352-8ad6-f23c9698a3c2
Λ₀,U₀=DivideConquer(T)

# ╔═╡ 772f61aa-faa0-4ce3-a2f6-b1ac5b46630b
Λ-Λ₀

# ╔═╡ 43d8bd1f-7d97-4de6-bdcf-9b5d551000a8
norm(U₀'*U₀-I)

# ╔═╡ 4338efe4-e3a6-43ca-8466-b1d51450748d
md"
### Timings for a large matrix
"

# ╔═╡ ab72f31d-6753-4728-a8f5-f66aaff618fd
begin
	# Timings
	n₃=1500
	T₃=SymTridiagonal(randn(n₃),randn(n₃-1));
end

# ╔═╡ 3a194020-2c77-4112-b102-f76ea10beda5
@time eigen(T₃);

# ╔═╡ ab5fef7b-b2e5-41b7-a3e4-8cf8134aad7e
@time a,b=stedc!('I',copy(T₃.dv),T₃.ev,Matrix{Float64}(I,n₃,n₃))

# ╔═╡ 487a91b4-6697-4e01-a711-2d0b20b63dbb
md"""
# MRRR

The method of Multiple Relatively Robust Representations

The computation of the tridiagonal EVD which satisfies the
error standard error bounds such that the eigenvectors are orthogonal to
working precision, all in $O(n^2)$ operations, has been the _holy
grail_ of numerical linear algebra for a long time. 
The method of Multiple Relatively Robust Representations does the job, except in some
exceptional cases. The key idea is to implement inverse iteration more
carefully. The practical algorithm is quite
elaborate and the reader is advised to consider references.

The MRRR method is implemented in the LAPACK subroutine 
[DSTEGR](http://www.netlib.org/lapack/explore-html/d0/d3b/dstegr_8f.html). 
This routine can compute just the
eigenvalues, or both  eigenvalues and eigenvectors.
"""

# ╔═╡ 4a2db4fd-514d-40d3-b396-341ede59e1dd
methods(LAPACK.stegr!)

# ╔═╡ 145aae34-7b07-421c-aee7-40091783f7a9
# LAPACK.stegr!

# ╔═╡ 02e88a87-da41-4fe0-9cde-457a325b0f74
LAPACK.stegr!('V',copy(T.dv),copy(T.ev))

# ╔═╡ fd9c471e-aa0b-413e-83f3-2f63b02eb3c4
# Timings
@time LAPACK.stegr!('V',copy(T₃.dv),copy(T₃.ev));

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
PlutoUI = "~0.7.38"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "23a7138a3fc77fba614979696a6f861d251c9afb"

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

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

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
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

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

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "85b5da0fa43588c75bb1ff986493443f821c70b7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "670e559e5c8e191ded66fa9ea89c97f10376bb4c"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.38"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
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
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

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

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

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
# ╠═56fea874-cdd8-46c2-babf-10c5bd6440d2
# ╟─fee10c7e-f1f4-41d4-86a9-c980b5b9b8a5
# ╟─dd5b2908-1dc7-4e2a-8b11-c5686a120c36
# ╟─219aea92-5737-42f5-8bed-a214a287851c
# ╠═b089f3ea-ac9d-48c8-ae6d-769e6d3efd07
# ╠═2694c3c7-a3a5-4302-9d1a-7a7a9fbdbf34
# ╠═0140996f-3c6e-4ed9-9df2-cebd9b6fbd7d
# ╠═9a185d60-6431-4e1e-a729-268a767bad85
# ╠═b7889ca1-c79d-43d7-9d65-11f0fed35791
# ╠═174dd76d-4221-4496-b990-7ca726f87f65
# ╠═009d63ef-b5f3-4f60-9d27-c12688282bed
# ╠═f4e77574-7d83-4cf9-9d94-2fe24e27395e
# ╠═9c07effc-e1b0-4b31-b228-e256469add5f
# ╠═f53d1a63-739d-4a57-9216-6101a2a3412f
# ╠═88bfbb99-a597-4ccc-89e0-b3f4b23e1949
# ╠═c7663867-20b4-4dd3-b173-456f283aa629
# ╠═7639eeec-d31d-42d9-b55a-6bc60176d5ad
# ╠═1052bcf0-1f11-4da8-b8d8-cabe87f5c6e4
# ╠═298753db-7ef3-46e8-b81a-7bcc9641572d
# ╠═f12f2f13-098f-443b-a517-c896f8cb38f1
# ╟─9582de91-3ac0-40c8-a160-a52b2d785e2c
# ╟─131ae1b7-a946-4e3f-b260-d037ee7ee6ce
# ╠═e1288577-8a5b-4b21-8533-68be1619a5e6
# ╠═a13052ab-297a-4e5a-8654-e65e1fde2e41
# ╠═7cbc4972-f604-4d1f-865d-0d9c5439b437
# ╠═04206e97-57c2-41fe-938f-9108cab56740
# ╠═06c885f9-f73c-448e-9ba7-7664ba1ad105
# ╠═e2c2803c-e709-43c1-9b79-524bf6dc03e3
# ╠═6b848adb-1b45-44c8-9be2-e451ef5d21d7
# ╠═2abbf839-847a-4d94-8a4b-f25f57a91540
# ╠═903e7308-e7be-4821-82e5-105aca75aca2
# ╠═c000c311-8262-4cdf-ace6-13106a792aad
# ╟─574c1653-a2c2-4365-a768-e447bab2d805
# ╠═cb7ac40c-b1fb-4b67-8a76-6c0df47fb270
# ╠═bc5ea97a-70ef-4a95-b59e-96790ba0346f
# ╠═b6804e98-b5bc-4f42-a864-25018886dc65
# ╠═93aadb04-e1bd-4031-aef4-6246141a9fc3
# ╠═79a282fc-a91d-486b-8620-6fad38f00e0b
# ╠═8c988e82-1c05-4e3a-a30f-277cb462cd5e
# ╠═85eae2ff-8d8f-4567-95a5-4b87efb711fa
# ╠═bd7472cd-2ede-4c57-af89-4a4325395a2f
# ╠═1cdb29eb-b325-4352-8ad6-f23c9698a3c2
# ╠═772f61aa-faa0-4ce3-a2f6-b1ac5b46630b
# ╠═43d8bd1f-7d97-4de6-bdcf-9b5d551000a8
# ╟─4338efe4-e3a6-43ca-8466-b1d51450748d
# ╠═ab72f31d-6753-4728-a8f5-f66aaff618fd
# ╠═3a194020-2c77-4112-b102-f76ea10beda5
# ╠═ab5fef7b-b2e5-41b7-a3e4-8cf8134aad7e
# ╟─487a91b4-6697-4e01-a711-2d0b20b63dbb
# ╠═4a2db4fd-514d-40d3-b396-341ede59e1dd
# ╠═145aae34-7b07-421c-aee7-40091783f7a9
# ╠═02e88a87-da41-4fe0-9cde-457a325b0f74
# ╠═fd9c471e-aa0b-413e-83f3-2f63b02eb3c4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
