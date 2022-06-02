module ModuleB

using LinearAlgebra

export myPower, myDeflation, myPowerMethod, myTridiag, myTridiagX, myTridiagG, myTridEigQR,
mySymEigQR, stedc!, myJacobi, myLanczos, myBidiagX, myBidiagY, myBidiagG, myJacobiR

function myPower(A::Matrix,x::Vector,tol::Real)
    y=A*x
    ν=x⋅y
    steps=1
    while norm(y-ν*x)>tol
        x=y/norm(y)
        y=A*x
        ν=x⋅y
        steps+=1
    end
    ν, y/norm(y), steps
end

function myDeflation(A::Matrix,x::Vector)
    X,R=qr(x[:,:])
    # To make sure the returned matrix symmetric use
    # full(Symmetric(X[:,2:end]'*A*X[:,2:end]))
    X[:,2:end]'*A*X[:,2:end]
end

function myPowerMethod(A::Matrix{T}, tol::Real) where T
    n=size(A,1)
    S =  T==Int ? Float64 : T
    λ=Vector{S}(undef,n)
    for i=1:n
        λ[i],x,steps=myPower(A,ones(n-i+1),tol)
        A=myDeflation(A,x)
    end
    λ
end

function myTridiag(A::Matrix)
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

# Extract X
function myTridiagX(H::Matrix)
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

# Tridiagonalization using Givens rotations
function myTridiagG(A::Matrix)
    n=size(A,1)
    T=Float64
    X=Matrix{T}(I,n,n)
    for j = 1 : n-2
        for i = j+2 : n
            G,r=givens(A,j+1,i,j)
            A=(G*A)*G'
            X*=G'
        end
    end
    SymTridiagonal(diag(A),diag(A,1)), X
end

function myTridEigQR(A1::SymTridiagonal)
    A=deepcopy(A1)
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
    end
    λ[1:k]=myTridEigQR(SymTridiagonal(A.dv[1:k],A.ev[1:k-1]))
    λ[k+1:n]=myTridEigQR(SymTridiagonal(A.dv[k+1:n],A.ev[k+1:n-1]))
    λ
end

function myTridEigQR(A1::SymTridiagonal,U::Matrix)
    # U is either the identity matrix or the output from myTridiagX()
    A=deepcopy(A1)
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
    λ[1:k], U[:,1:k]=myTridEigQR(SymTridiagonal(A.dv[1:k],A.ev[1:k-1]),U[:,1:k])
    λ[k+1:n], U[:,k+1:n]=myTridEigQR(SymTridiagonal(A.dv[k+1:n],A.ev[k+1:n-1]),U[:,k+1:n])
    λ, U
end

function mySymEigQR(A::Matrix)
    T,H=myTridiag(A)
    X=myTridiagX(H)
    myTridEigQR(T,X)
end

### DSTEDC
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

function myJacobi(A1::Array{T}) where T<:Real
    A=deepcopy(A1)
    n=size(A,1)
    U=Matrix{T}(I,n,n)
    # Tolerance for rotation
    tol=sqrt(map(T,n))*eps(T)
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
                # Check for the tolerance - the first criterion is standard,
                # the second one is for relative accuracy for PD matrices               
                # if A[i,j]!=zero(T)
                if abs(A[i,j])>tol*sqrt(abs(A[i,i]*A[j,j]))
                    # Compute c and s
                    τ=(A[i,i]-A[j,j])/(2*A[i,j])
                    t=sign(τ)/(abs(τ)+sqrt(1+τ^2))
                    c=1/sqrt(1+t^2)
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
    diag(A), U
end

function myLanczos(A::Array{T}, x::Vector{T}, k::Int) where T
    n=size(A,1)
    X=Array{T}(undef,n,k)
    dv=Array{T}(undef,k)
    ev=Array{T}(undef,k-1)
    X[:,1]=x/norm(x)
    for i=1:k-1
        z=A*X[:,i]
        dv[i]=X[:,i]⋅z
        # Three-term recursion
        if i==1
            z=z-dv[i]*X[:,i]
        else
            # z=z-dv[i]*X[:,i]-ev[i-1]*X[:,i-1]
            # Full reorthogonalization - once or even twice
            z=z-sum([(z⋅X[:,j])*X[:,j] for j=1:i])
            z=z-sum([(z⋅X[:,j])*X[:,j] for j=1:i])
        end
        μ=norm(z)
        if μ==0
            Tr=SymTridiagonal(dv[1:i-1],ev[1:i-2])
            return eigvals(Tr), X[:,1:i-1]*eigvecs(Tr), X[:,1:i-1], μ
        else
            ev[i]=μ
            X[:,i+1]=z/μ
        end
    end
    # Last step
    z=A*X[:,end]
    dv[end]=X[:,end]⋅z
    z=z-dv[end]*X[:,end]-ev[end]*X[:,end-1]
    μ=norm(z)
    Tr=SymTridiagonal(dv,ev)
    eigvals(Tr), X*eigvecs(Tr), X, μ
end

# Extract X
function myBidiagX(H::Matrix)
    m,n=size(H)
    T=typeof(H[1,1])
    X = Matrix{T}(I,m,n)
    v = Array{T}(undef,m)
    for j = n : -1 : 1
        v[j] = one(T)
        v[j+1:m] = H[j+1:m, j]
        γ = -2 / (v[j:m]⋅v[j:m])
        w = γ * X[j:m, j:n]'*v[j:m]
        X[j:m, j:n] = X[j:m, j:n] + v[j:m]*w'
    end
    X
end

# Extract Y
function myBidiagY(H::AbstractMatrix)
    n,m=size(H)
    T=typeof(H[1,1])
    Y = Matrix{T}(I,n,n)
    v = Array{T}(undef,n)
    for j = n-2 : -1 : 1
        v[j+1] = one(T)
        v[j+2:n] = H[j+2:n, j]
        γ = -2 / (v[j+1:n]⋅v[j+1:n])
        w = γ * Y[j+1:n, j+1:n]'*v[j+1:n]
        Y[j+1:n, j+1:n] = Y[j+1:n, j+1:n] + v[j+1:n]*w'
    end
    Y
end

# Bidiagonalization using Givens rotations
function myBidiagG(A1::Matrix)
    A=deepcopy(A1)
    m,n=size(A)
    T=typeof(A[1,1])
    X=Matrix{T}(I,m,m)
    Y=Matrix{T}(I,n,n)
    for j = 1 : n        
        for i = j+1 : m
            G,r=givens(A,j,i,j)
            # Use the faster in-place variant
            # A=G*A
            lmul!(G,A)
            # X=G*X
            lmul!(G,X)
        end
        for i=j+2:n
            G,r=givens(A',j+1,i,j)
            # A*=adjoint(G)
            rmul!(A,adjoint(G))
            # Y*=adjoint(G)
            rmul!(Y,adjoint(G)) 
        end
    end
    X',Bidiagonal(diag(A),diag(A,1),'U'), Y
end

function myJacobiR(A1::AbstractMatrix)
    A=deepcopy(A1)
    m,n=size(A)
    T=typeof(A[1,1])
    V=Matrix{T}(I,n,n)
    # Tolerance for rotation
    tol=sqrt(map(T,n))*eps(T)
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
                F=A[:,[i,j]]'*A[:,[i,j]]
                # Check the tolerance - the first criterion is standard,
                # the second one is for relative accuracy               
                # if A[i,j]!=zero(T)
                # 
                if abs(F[1,2])>tol*sqrt(F[1,1]*F[2,2])
                    # Compute c and s
                    τ=(F[1,1]-F[2,2])/(2*F[1,2])
                    t=sign(τ)/(abs(τ)+sqrt(1+τ^2))
                    c=1/sqrt(1+t^2)
                    s=c*t
                    G=LinearAlgebra.Givens(i,j,c,s)
                    A*=G'
                    # rmul!(A,adjoint(G))
                    V*=G'
                    # rmul!(A,adjoint(G))
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
    A, σ, V
end

end
