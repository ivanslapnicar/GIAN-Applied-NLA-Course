### A Pluto.jl notebook ###
# v0.19.46

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

# ╔═╡ b4423c48-1d43-4887-9f96-a2f6ac8eaa45
using PlutoUI, FFTW, ToeplitzMatrices, SpecialMatrices, Random, LinearAlgebra, WAV, Arpack, LinearMaps, Plots

# ╔═╡ b6058fe0-72a9-4be3-9d09-5f35109198a2
plotly()

# ╔═╡ 5b8f2611-cdf9-4761-8768-4b014f02b842
PlutoUI.TableOfContents(aside=true)

# ╔═╡ 9e73d94e-c26e-4513-8142-1058738f103b
md"""
# Signal Decomposition Using EVD of Hankel Matrices


Suppose we are given a data signal which consists of several nearly mono-components.

_Can we recover the mono-components?_

The answer is _YES, with an efficient algorithm using EVDs of Hankel matrices._ 

Mono-component recovery can be successfully applied to audio signals.

__Prerequisites__

The reader should be familiar with elementary concepts about signals and linear algebra concepts, particularly eigenvalue decomposition and its properties and algorithms. The latter can be learned from any textbook on linear algebra or from this [notebook](https://ivanslapnicar.github.io/GIAN-Applied-NLA-Course/L3a%20Eigenvalue%20Decomposition%20-%20Definitions%20and%20Facts.html).


__Competences__

The reader should be able to decompose given signal into mono-components using EVD methods.

__References__

For more details see [P. Jain and R. B. Pachori, An iterative approach for decomposition of multi-component non-stationary signals based on eigenvalue decomposition of the Hankel matrix](http://www.sciencedirect.com/science/article/pii/S0016003215002288).
"""

# ╔═╡ d1b6461e-6e87-4cab-af2d-f28e09c336ac
md"""
# Extraction  of stationary mono-components

## Definitions

Let $x\in\mathbb{R}^{m}$, denote a __signal__ with $N$ samples.

Assume $x$ is composed of $L$ __stationary mono-components__:

$$
x=\sum\limits_{k=1}^L x^{(k)},$$

where

$$
x^{(k)}_i=A_k \cos(2\pi f_k i +\theta_k)\quad i=1,2,\ldots,m.$$

Here:

 $f_k=\displaystyle\frac{F_k}{F}$ is the __normalized frequency__ of $x^{(k)}$,

 $p_k=\displaystyle\frac{1}{f_k}$ is the period of $x^{(k)}$,

 $F$ is the __sampling frequency__ of $x$ in Hz, 

 $F_k$ is the sampling frequency of $x^{(k)}$,

 $A_k$ is the __amplitude__ of $x^{(k)}$, and 

 $\theta_k$ is the __phase__ of $x^{(k)}$.

We assume that $F_k< F_{k+1}$ for $k=1,2,\ldots,n-1$, and $F>2F_n$.

A __Hankel matrix__ is a (real) square matrix with constant values along the skew-diagonals. More precisely, let $a\in\mathbb{R}^{2n-1}$. An $n\times n$ matrix $H\equiv H(a)$ for which $H_{ij}=H_{i+1,j-1}=a_{i+j-1}$ is a Hankel matrix.
"""

# ╔═╡ 545776f8-e826-430b-8744-05a53ef708b6
begin
	# Small Hankel matrix
	a=collect(1:9)
	Hankel(a)
end

# ╔═╡ a8bff246-a387-43ec-91b9-de97d366c79d
md"""
## Facts

Let $x$ be a signal with $2n-1$ samples composed of $L$ stationary mono-components.

Let $H$ be an $n\times n$ Hankel matrix corresponding to $x$ and let $H=U\Lambda U^T$ be its EVD (Hankel matrix is symmetric) with $\lambda_1\leq \lambda_2 \leq \cdots \leq \lambda_n$.

Smilarly, let $H_k$ be the $n\times n$ Hankel matrix corresponding to the $k$-th component $x^{(k)}$ and let $H_k=U_k\Lambda_k U_k^T$ be its EVD. Then

1.  $H=\sum\limits_{k=1}^{L} H_k$.

2.  $H_k=\lambda_k U_{:,k}U_{:,k}^T + \lambda_{n-k+1} U_{:,n-k+1}U_{:,n-k+1}^T$.
"""

# ╔═╡ 9ad716ab-57d8-4b17-b40c-93f6e14903b2
md"""
## Example - Signal with three mono-components
"""

# ╔═╡ 2206c20c-9e67-49a4-bc26-b203477b872f
begin
	# Create the signal
	n=160
	N=2*n-1
	F = 6400
	L = 3
	A = [3, 2, 1]
	Fk= [200, 320, 160]
	# Normalized frequencies
	fk=Fk./F
	θ = [pi/2, pi/4, 0]
	x = zeros(N)
	for k=1:3
	    for i=1:N
	        x[i]+=A[k]*cos(2*pi*Fk[k]*i/F+θ[k])
	    end
	end
	x, fk
end

# ╔═╡ a781a4c1-d0b1-4bab-a08b-724697d617f9
# Plot the signal
plot(x,xlabel="Number of samples N", ylabel="Amplitude",leg=false)

# ╔═╡ a03492f5-c172-46f3-839f-4d7417015acb
# Periods
F./Fk

# ╔═╡ e6ad6e6e-b0ba-42d4-8677-f1149aed08bc
# FFT indicates that there are three components with (approximate) 
# sampling frequencies 160,200, and 320
scatter(range(-F/2,stop=F/2,length=length(x)),abs.(fftshift(fft(x))), title="FFT of a signal", legend=false, xlabel="Angular frequency",xlims=(-500,500))

# ╔═╡ e0054a3b-2e14-4bb8-81b0-6763e963dfaa
# Decompose the signal 
H=Hankel(x)

# ╔═╡ 7b8eeeb2-bae5-46ed-9dee-0a8efb2d31c9
λ,U=eigen(Matrix(H))

# ╔═╡ 771a6712-36a4-4e1c-8355-210e4f3787ec
md"""
The three smallest and the three largest eigenvalues come in $\pm$ pairs and define the three mono-components. 

The ratios of the moduli of the eigenvalues correspond to the ratios of the amplitudes of the mono-components. 
"""

# ╔═╡ 407c19d9-d67f-4e98-a8e6-8bb26c24e46f
begin
	# Form the three matrices
	Hcomp=Array{Any}(undef,3)
	for k=1:L
	    Hcomp[k]=λ[k]*U[:,k]*U[:,k]' + λ[end-k+1]*U[:,end-k+1]*U[:,end-k+1]'
	end
end

# ╔═╡ 2369b267-cfbc-4092-9822-24582c10af13
begin
	# For comparison, create the first mono-component and its Hankel matrix
	x₁ = zeros(N)
	c=1
	for i=1:N
	    x₁[i]+=A[c]*cos(2*pi*Fk[c]*i/F+θ[c])
	end
	H₁=Hankel(x₁)
end

# ╔═╡ 16f56d88-d9e4-4ad5-9431-3a993408d26d
# Compare
Hcomp[1]

# ╔═╡ 4a35f8fa-33d8-481f-8423-d5af7e32dc8f
# Check norm
norm(Hcomp[1]-H₁)

# ╔═╡ 2d0f04ca-fb8d-4dc9-81e6-1f2600fb0fab
begin
	# Reconstruct the mono-component xcomp[k] using the border elements of Hcomp[k]
	# (first column and last row)
	xcomp=Array{Vector{Float64}}(undef,L)
	for k=1:L
		xcomp[k]=[Hcomp[k][:,1];Hcomp[k][end,2:end]]
	end
end

# ╔═╡ 79db0e46-1685-4c90-b51c-b158eac11f03
# xaxis=collect(1:N)
plot([xcomp[1],xcomp[2],xcomp[3]],title="Extracted mono-components", label=["First" "Second" "Third"], xlabel="Sample")

# ╔═╡ 97b56eb0-df22-4877-8320-6440a1806c10
md"""
# Fast Hankel × Vector using FFT

Several outer eigenvalues pairs of Hankel matrices can be computed using Lanczos method. If the multiplication $Hx$ is performed using Fast Fourier Transform, this EVD computation is very fast.

## Definitions

A __Toeplitz matrix__ is a (real) square matrix with constant values along the diagonals. More precisely, let 

$$
a=(a_{-(n-1)},a_{-(n-2)},\ldots,a_{-1},a_0,a_1,\ldots,a_{n-1})\in\mathbb{R}^{2n-1}.$$ 

An $n\times n$ matrix $T\equiv T(a)$ for which $T_{ij}=T_{i+1,j+1}=a_{i-j}$ is a Toeplitz matrix.
"""

# ╔═╡ fd85e7bd-b950-4f0b-909c-cbdc85216a61
md"""
Notice different meanings of vector $a$: in `C=Circulant(a)`, $a$ is the first column, in 
`T=Toeplitz(a,b)`, $a_i$ are the elements in the first column and $b_i$ are the elements in the first row, and in `H=Hankel(a)`, $a_i$ is the element of the $i$-th skew-diagonal starting from $H_{11}$.
"""

# ╔═╡ 26beaaa8-923b-4b03-ae6d-af18996eb398
# T
Toeplitz([5,6,7,8,9],[5,4,3,2,1])

# ╔═╡ afd8681e-3169-44f1-aece-599ca9998531
md"""
A __circulant matrix__ is a Toeplitz matrix where each column is rotated one element downwards relative to the preceding column. 

More precisely, let $a\in\mathbb{R}^{n}$. An $n\times n$ matrix $C\equiv C(a)=T(a,a_{1:n-1})$ is a Circulant matrix.
"""

# ╔═╡ a2570d4b-101c-4120-aa6c-8bbf8e42decd
Circulant([1,2,3,4,5])

# ╔═╡ 31498e3b-94eb-4fe6-a814-f69fc9e5bb4c
# Notice the different indexing
Toeplitz([1,2,3,4,5],[1,5,4,3,2])

# ╔═╡ c07030fd-47e4-4ee7-b409-8591771f61c7
md"""
A __rotation matrix__ is an identity matrix rotated 90 degrees to the right (or left).

Hankel matrix is the product of a Toeplitz matrix and the rotation matrix.
"""

# ╔═╡ 86985c1c-c4a2-4b38-88e5-d1488d903ea8
begin
	J=rotl90(Matrix{Int64}(I,5,5))
end

# ╔═╡ cce7dba2-2bee-474d-b17a-4d091e4a1fd6
Matrix(Toeplitz([5,6,7,8,9],[5,4,3,2,1]))*J, Hankel([1,2,3,4,5,6,7,8,9])

# ╔═╡ 9dbdb70b-758d-49e7-b0b1-74fb339a9a8d
rotl90(Matrix(Toeplitz([5,6,7,8,9],[5,4,3,2,1])))

# ╔═╡ 23c239f3-4cd2-48fc-9391-41d361460f98
md"""

Given vector $x$ of length $n$, a __Vandermonde matrix__ is a $n\times n$ matrix:

$$
V(x)=\begin{bmatrix} x.^0 & x.^1 & x.^2 & \cdots & x.^n\end{bmatrix}.$$

A __Fourier matrix__ is the Vandermonde matrix:

$$
F_n=V(1,\omega_n,\omega_n^2,\ldots, \omega_n^{n-1}),$$

where $\omega_n=\exp(2\pi i/n)$ is the $n$-th root of unity.
"""

# ╔═╡ faf54ef7-7f3b-4f16-a18f-b21c0f00c2c9
Vandermonde([6,2,3,4,5])

# ╔═╡ f17cf45a-738e-4d23-8bfb-c06990ebd1fe
md"""
## Facts 

For more details see [G. H. Golub and C. F. Van Loan, Matrix Computations] (https://epubs.siam.org/doi/book/10.1137/1.9781421407944), and the references therein

1. Circulant matrix is normal and, thus, unitarily diagonalizable, with the eigenvalue decomposition
$$
C(a)=U\mathop{\mathrm{diag}}(F_n^* a)U^*,$$
where $U=\displaystyle\frac{1}{\sqrt{n}} F_n$. The product $F_n^* a$ can be computed by the Fast Fourier Transform (FFT).
2. Given $a,x\in\mathbb{R}^n$, the product $y=C(a)x$ can be computed using FFT as follows:
$$\begin{aligned}
\tilde x&=F_n^*x\\
\tilde a&=F_n^*a\\
\tilde y&=\tilde x.* \tilde a\\
y&= F_n^{-*} \tilde y.
\end{aligned}$$
3. Toeplitz matrix of order $n$ can be embedded in a circulant matrix of order $2n-1$: if $a\in\mathbb{R}^{2n-1}$, then 
$$
T(a)=[C([a_{n+1:2n-1};a_{1:n}])]_{1:n,1:n}.$$
4. Further, let $x\in\mathbb{R}^{n}$ and let $\bar x\in\mathbb{R}^{2n-1}$ be equal to $x$ padded with $n-1$ zeros. Then
$$
\begin{bmatrix} T & A \\ B & C\end{bmatrix}
\begin{bmatrix} x \\ 0 \end{bmatrix}$$
or

$$
T(a)x=[C([a_{n+1:2n-1};a_{1:n}])\bar x]_{1:n}.$$
5. Since Hankel = Toeplitz × Rotation, we have
$$
H(a)x=(T(a)J)x=T(a)(Jx).$$
"""

# ╔═╡ 9782d6f9-285c-46d8-a826-017f4a5bcf53
md"
## Examples
"

# ╔═╡ 94171cfa-1e8e-4aba-a5b8-faad0104cf80
begin
	# Fact 1 - EVD of Circulant
	Random.seed!(467)
	n₀=5
	a₀=rand(-8:8,n₀)
	C₀=Circulant(a₀)
	ω=exp(2*pi*im/n₀)
	v=[ω^k for k=0:n₀-1]
	F₀=Vandermonde(v)
	U₀=F₀/√n₀
	λ₀=Matrix(F₀)'*a₀
	λ₀,eigvals(C₀)
end

# ╔═╡ 8a1efef7-8903-426c-b8d6-a99bc5288981
C₀

# ╔═╡ 41b7458c-dc6c-4774-8991-f74519a850ed
# Residual
norm(C₀*U₀-U₀*Diagonal(λ₀))

# ╔═╡ 8fbdea3e-2c49-4068-8b2e-6339737554f2
md"""

## Fast Circulant × Vector

Fact 2 - Circulant() × vector, as implemented in the (oudated) package `SpecialMatrices.jl`, use `add SpecialMatrices#withToeplitz`.

```
function *(C::Circulant{T},x::Vector{T}) where T
    xt=fft(x)
    vt=fft(C.c)
    yt=vt.*xt
    real(ifft(yt))
end
```

Similarly, `mul!()`

```
function mul!(y::StridedVector{T},C::Circulant{T},x::StridedVector{T}) where T
    xt=fft(x)
    vt=fft(C.c)
    yt=ifft(vt.*xt)
    if T<: Int
        map!(round,y,yt) 
    elseif T<: Real
        map!(real,y,yt)
    else
        copy!(y,yt)
    end
    return y
end
```
"""

# ╔═╡ 5c76963a-5cd6-4797-bc28-536a565bf4fe
md"""

```
function mul!(y::StridedVector{T},A::Toeplitz{T},x::StridedVector{T}) where T
    n=length(A.c)
    k=div(n+1,2)
    C=Circulant([A.c[k:n];A.c[1:k-1]])
    xx=[x;zeros(T,k-1)]
    yy=mul!(similar(xx),C,xx)
    copyto!(y, 1, yy, 1, length(y))
    return y
end
```

```
function *(A::Hankel{T},x::Vector{T}) where T
    Toeplitz(A.c)*reverse(x)
end
```
"""

# ╔═╡ 561faa84-f4e3-4712-92b0-68e6edabae65
begin
	# Fact 2 - Fast Circulant × Vector
	x₀=rand(-9.0:9,1000)
	M=Circulant(x₀)
	y₀=rand(-9.0:9,1000)
end

# ╔═╡ 0c6a11eb-e4b2-4daa-83c9-97872ca150ce
@which mul!(similar(x₀),M,x₀,1,0)

# ╔═╡ c8d73641-e8ba-46ff-b368-981d3c288d48
@which M*y₀

# ╔═╡ b69942f4-34b8-47f9-b33a-9776719feac0
@time mul!(similar(y₀),M,y₀,1,0);

# ╔═╡ 72f3a029-32f2-4dbd-b8c3-f5794ea85404
# For comparison of timing
@time Matrix(M)*y₀;

# ╔═╡ 319a6376-3172-4f9b-9137-c8a3809920c3
norm(M*y₀-Matrix(M)*y₀)

# ╔═╡ 61d7911a-ea4d-4ef1-9778-6a2e8eff10e1
begin
	# Fact 3 - Embedding Toeplitz into Circulant
	a₂=rand(-6:6,5)
	b₂=[a₂[1];rand(-6:6,4)]
	T=Toeplitz(a₂,b₂)
end

# ╔═╡ b2fc20f1-faf1-4e08-9ccd-6b175cff0066
C=Circulant([a₂;reverse(b₂[2:end])])

# ╔═╡ d132b9c0-8101-4c14-89d7-5fa1462ea71e
# Fact 4 - Fast Toeplitz() × vector
x₂=rand(-6:6,n₀)

# ╔═╡ 32f20ec2-bf15-467f-98ef-6dbe6a568bca
[Matrix(T)*x₂ T*x₂ mul!(similar(x₂),T,x₂,1,0)]

# ╔═╡ 825e7090-8196-4165-853c-57237f5e05c9
begin
	# Fact 6 - Fast Hankel() x vector
	h₂=rand(-9:9,9)
	H₂=Hankel(h₂)
end

# ╔═╡ 8f361246-8941-47a6-98c1-2b92dea2c74b
[Matrix(H₂)*x₂ H₂*x₂ mul!(similar(x₂),H₂,x₂,1,0)]

# ╔═╡ 84d96895-e4ad-4bf4-8df7-6e81b983bb3e
md"""
# Fast EVD of a Hankel matrix

Given a Hankel matrix $H$, the Lanczos method can be applied by defining a function (linear map) which returns the product $Hx$ for any vector $x$. This approach uses the package [LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl) and is described in the this
[notebook](https://ivanslapnicar.github.io/GIAN-Applied-NLA-Course/L4d%20Symmetric%20Eigenvalue%20Decomposition%20-%20Lanczos%20Method.html). 

__The computation is high-speed and allocates little extra space.__
"""

# ╔═╡ cc41548c-674a-4f67-bdf7-95ce16a6a5d8
# Define the function using the Hankel matrix H of the three-component signal
f(x)=mul!(similar(x),H,x,1,0)

# ╔═╡ eb1daded-ad36-41d3-9088-a9f8cf6bf63f
# Define the linear map using the function f
A₁=LinearMap(f,size(H,1),issymmetric=true)

# ╔═╡ eef3c2ad-3619-49b1-a612-63c234314dfd
size(A₁)

# ╔═╡ aac0a39f-0c49-4009-974e-852c7e8e2b17
# This is the standard O(n^3) algorithm
@time eigvals(Matrix(H));

# ╔═╡ 38b8e9f0-2546-4113-83b7-85599faa6992
# This is the Lanczos algorithm using fast multiplication
@time λA,UA=eigs(A₁, nev=6, which=:LM)

# ╔═╡ 146871c4-e014-4ee4-9933-1d21c2504635
md"""
# Extraction of non-stationary mono-components

## Definitions

Let $x\in\mathbb{R}^{m}$, denote a __signal__ with $N$ samples.

Assume $x$ is composed of $L$ __non-stationary mono-components__:

$$
x=\sum\limits_{k=1}^L x^{(k)},$$

where

$$
x^{(k)}_i=A_k \cos(2\pi f_k i +\theta_k),\quad i=1,2,\ldots,m.$$

Assume that the normalized frequencies $f_k=\displaystyle\frac{F_k}{F}$, the sampling frequencies $F_k$, the amplitudes  $A_k$, and the phases $\theta_k$, all _sightly_ change in time.

Let $H\equiv H(x)$ be the Hankel matrix of $x$. The eigenpair of $(\lambda,u)$ of $H$ is __significant__ if $|\lambda|> \tau  \cdot \sigma(H)$. Here $\sigma(H)$ is the spectral radius of $H$, and $\tau$ is the __significant threshold percentage__ chosen by the user depending on the type of the problem.
"""

# ╔═╡ e09da774-b32d-487f-97a9-2195d3306224
md"""
## Fact

The following algorithm decomposes the signal $x$:
1. Choose the threshold $\tau$
2. Form the Hankel matrix $H$
2. Compute the EVD of $H$
3. Choose the significant eigenpairs of $H$
4. For each significant eigenpair $(\lambda,u)$
    1. Form the rank one matrix $M=\lambda uu^T$
    2. Define a new signal $y$ consisting of averages of the skew-diagonals of $M$ (__or take the first column and last row, which is much faster!__)
    3. Form the Hankel matrix $H(y)$
    3. Compute the EVD of $H(y)$
    4. Choose the significant eigenpairs of $H(y)$
    5. __If__ $H(y)$ has only two significant eigenpairs, declare $y$ a mono-component, __else__ go to step 4.
"""

# ╔═╡ e47f1bd9-89b5-4f67-96bf-d6b4a50a721c
md"""
## Note A⁴ (440 Hz)

Each tone has its fundamental frequency (mono-component). However, musical instruments produce different overtones (harmonics) which are near integer multiples of the fundamental frequency.
Due to construction of resonant boxes, these frequencies slightly vary in time, and their amplitudes are contained in a time varying envelope.

Tones produces by musical instruments  are nice examples of non-stationary signals. We shall decompose the note A4 played on piano.

For manipulation of recordings, we are using package [WAV.jl](https://github.com/dancasimiro/WAV.jl). Another package with similar functionality is the package [LibSndFile.jl](https://github.com/JuliaAudio/LibSndFile.jl).
"""

# ╔═╡ 309e82f8-2ea4-4de8-a27b-92844948c579
varinfo(WAV)

# ╔═╡ 0499ad09-e8a8-41e8-99f3-4ca309ceb9d9
# Load a signal
Signalₐ = wavread("files/piano_A41.wav")

# ╔═╡ e3399220-d1f9-48c1-9863-edd307ca7d4e
typeof(Signalₐ)

# ╔═╡ 59afc24f-fdf0-4694-a723-426352299629
begin
	# Data
	sₐ=Signalₐ[1]
	# Sampling frequency
	Fs=Signalₐ[2]
end

# ╔═╡ ce4d893e-3470-4e7e-8b03-c22ce5f9f434
# Play the signal
wavplay(sₐ,Fs)

# ╔═╡ e52f9b5c-12eb-4b9b-9d60-9b421d5b7fe2
# Plot the signal
plot(sₐ,title="Note A⁴", legend=false, xlabel="sample")

# ╔═╡ ec8ae004-b945-4fe7-a32f-172ff5f6e82a
begin
	# Plot in time scale
	tₐ=range(0,stop=length(sₐ)/Fs,length=length(sₐ))
	plot(tₐ,sₐ,title="Note A⁴", legend=false,xlabel="time (s)")
end

# ╔═╡ 72e0fc55-4c34-473a-b78b-b6910530b1e9
# Total time and number of samples
tₐ[end], length(sₐ)

# ╔═╡ b9523247-9c65-4278-8e10-1050783ae73a
md"
Second half of the signal is not interesting, so we create a shorter signal. $N$ must be odd.
"

# ╔═╡ 527f72cf-16de-4c16-bd96-4e6581687527
begin
	# Signal = wavread("files/piano_A41.wav",100001)
	s=sₐ[1:100001]
	t=tₐ[1:100001]
end

# ╔═╡ 0e8c82c6-ca02-4684-b319-ce35c2cf19cb
# Play the shorter signal
wavplay(s,Fs)

# ╔═╡ 5f3c0dd6-8ec8-432c-a230-786cf3d8a73a
# Plot the shorter signal
plot(t,s,title="Note a", legend=false,xlabel="time (s)")

# ╔═╡ 368e2830-3152-40eb-9795-5ea0ec69d8a5
md"""
Let us visualize the signal in detail:

k = $(@bind k Slider(1:1000:100001-1000,show_value=true))
"""

# ╔═╡ 533165d8-3c49-4609-8f69-1237c43b6946
plot(tₐ[k:k+1000],s[k:k+1000], title="Note A⁴",label=false,xlabel="time (s)")

# ╔═╡ 3873b054-5005-4b17-bae8-a51c44dca506
# Save the short signal
wavwrite(s,"files/piano_A41_short.wav",Fs=Fs)

# ╔═╡ 8d59bb1c-458c-4d51-a735-2587c84e0f2c
begin
	# Check the signal with FFT
	# Notice 3 stronger harmonics and six weaker ones
	fs=abs.(fft(s))
	plot(Fs/length(fs)*(1:length(fs)),fs, title="FFT of the note A⁴",xlabel="Frequency", leg=false, xlims=(0,4000))
end

# ╔═╡ 3f257922-9321-4b2a-84b3-ab3e5f57253e
# Form the Hankel matrix
# IMPORTANT - Do not try to display H - it is a 50001 x 50001 matrix.
Hₐ=Hankel(s);

# ╔═╡ b41a985a-c81a-467d-b137-4a0cde1c4a73
size(Hₐ), Hₐ[100,200]

# ╔═╡ d539f8c0-2f35-4a88-9646-07aedff40bda
# Get the idea about the time to compute EVD
@time fft(s);

# ╔═╡ 9575d673-a0e5-47d1-b109-9be6ee241623
begin
	# We are looking for 20 eigenvalue pairs
	nₐ=size(Hₐ,1)
	fₐ(x)=mul!(similar(x),Hₐ,x,1.0,0.0)
	Aₐ=LinearMap(fₐ,nₐ,issymmetric=true)
	size(Aₐ)
end

# ╔═╡ b0ac9727-2842-43a5-acee-f2ea74a1115e
@time λₐ,Uₐ=eigs(Aₐ, nev=40, which=:LM)

# ╔═╡ 25375eed-4226-42e2-ae65-f1418443732a
# Count the eigenvalue pairs (+-) larger than the 10% of the maximum
τ=0.1

# ╔═╡ a630247b-0505-43ff-b94a-468ef8887728
Lₐ=round(Int,(sum(abs.(λₐ).>(τ*maximum(abs,λₐ)))/2))

# ╔═╡ 1b3e2e57-7c3f-4365-8616-e4b46b046102
md"""
At this point, the implementation using full matrices is rather obvious. However, we cannot do that, due to the large dimension. Recall, ideally we should define the signal by averaging skew-diagonals of the Hankel matrices $H_k$ for $k=1,\ldots,L$,

$$
H_k=\lambda_k U_{:,k}U_{:,k}^T + \lambda_{n-k+1} U_{:,n-k+1}U_{:,n-k+1}^T,$$

This can be done without forming the matrices:
"""

# ╔═╡ 93248879-4486-4059-a363-6c7b6a0015d8
function averages(λ::T, u::Vector{T}) where T
    n=length(u)
    x=Array{Float64}(undef,2*n-1)
    # Average lower diagonals
    for i=1:n
        x[i]=dot(u[1:i],reverse(u[1:i]))/i
    end
    for i=2:n
        x[n+i-1]=dot(u[i:n],reverse(u[i:n]))/(n-i+1)
    end
    λ*x
end

# ╔═╡ 46f8977f-fe22-4efc-9724-6b8ca588414d
#=
xcompₐ=Array(Array{Float64},Lₐ)
for k=1:Lₐ
    xcompₐ[k]=averages(λₐ[2*k-1],Uₐ[:,2*k-1])+averages(λₐ[2*k],Uₐ[:,2*k])
end
=#

# ╔═╡ 949954b4-663d-4ef3-99b5-0df3c74a31e7
md"""
N.B. `eigs()` returns the eigenvalues arranged by the absoulte value, so the consecutive pairs define the $i$-th signal. 
"""

# ╔═╡ 8fc35997-f124-423e-b384-0f2369ecaa35
md"""
However, function `averages()` is very slow - it requires $O(n^2)$ operations and takes 7 minutes, compared to 5 seconds for the eigenvalue computation.

The simplest option is to disregard the averages and use the first column and the last row of the underlying matrix, as in the definition of Hankel matrices, which we do next. 

(A smarter approach might be to approximate averages using small random samples.)
"""

# ╔═╡ 84b53076-c26b-445a-a458-fe71cca242dc
begin
	xcompₐ=Array{Array{Float64}}(undef,Lₐ)
	for k=1:Lₐ
	    k₁=2*k-1
	    k₂=2*k
		xsimple=[(λₐ[k₁]*Uₐ[1,k₁])*Uₐ[:,k₁]; (λₐ[k₁]*Uₐ[nₐ,k₁])*Uₐ[2:nₐ,k₁]]
	    xsimple+=[(λₐ[k₂]*Uₐ[1,k₂])*Uₐ[:,k₂]; (λₐ[k₂]*Uₐ[nₐ,k₂])*Uₐ[2:nₐ,k₂]]
	    xcompₐ[k]=xsimple
	end
end

# ╔═╡ 16f2dc1f-30d2-4335-87e2-afb32235f1dc
md"""
Let us look and listen to what we got:

Mono-component number $(@bind kₐ Slider(1:Lₐ,show_value=true))
"""

# ╔═╡ d33b3243-058c-446f-975a-0aee5b5426ac
plot(t,xcompₐ[kₐ],title="Mono-component $(kₐ)",leg=false,xlabel="time (s)")

# ╔═╡ 0d3e2f32-19fe-435d-96b6-83f047ecd8ef
begin
	# FFT of a mono-component and computed frequency
	l₁=10000
	fsₐ=abs.(fft(xcompₐ[kₐ]))
	m,ind=findmax(fsₐ[1:l₁])
	"Frequency of mono-component $(kₐ) = ", ind*Fs/length(fsₐ)  ," Hz, Amplitude = ", m
end

# ╔═╡ cf6f1ff9-6439-4ef9-af20-f278495eb239
# Plot the FFT
plot(Fs/length(fsₐ)*(1:l₁),fsₐ[1:l₁], title="FFT of mono-component $(kₐ)",leg=false,xlabel="Frequency")

# ╔═╡ 847bb094-a2b8-4459-9d7b-f39bd3db2101
# Listen to individual mono-components
wavplay(5*xcompₐ[kₐ],Fs)

# ╔═╡ e03267b6-1320-435a-818a-c2018556c25b
md"""
We see and hear that all `xcompₐ[k]` are (almost 😀) clean mono-components - see 
[Fundamental Frequencies of Notes ..](http://auditoryneuroscience.com/index.php/pitch/fundamental-frequencies-notes-western-music):

```
1 = 440 Hz (A4)
2 = 880 Hz (2*440,+octave,A5)
3 = 1320 Hz (3*440,+quint,E6)
4 = 440 Hz  
5 = 880 Hz
6 = 2200 Hz (5*440,++major terza, C#7) 
7 = 2640 Hz (6*440,++quint,E7)
8 = 440 Hz
9 = 2200 Hz
10 = 1760 Hz (4*440,++octave,A6)
11 = 2640 Hz
```

__N.B.__ Some mono-components are repeated, and they should be grouped by adding components with absolute weighted 
correlation larger than some prescribed threshold. 
"""

# ╔═╡ 662b0be2-9f82-4980-83de-bb0143c28736
# Store the mono-components
for i=1:Lₐ
    wavwrite(xcompₐ[i],"files/comp$i.wav",Fs=Fs)
end

# ╔═╡ 60b8f0cc-7e28-4787-be5c-e2b779e655c4
# Listen to the sum of mono-components
wavplay(sum([xcompₐ[i] for i=1:Lₐ]),Fs)

# ╔═╡ 1fba26bb-4d17-4df8-8ce2-ca4185101681
# Store the sum of mono-components
wavwrite(sum([xcompₐ[i] for i=1:Lₐ]),"files/compsum.wav",Fs=Fs)

# ╔═╡ 13e65ea8-e0c4-45ee-ae57-460310380097
md"""
## C-minor chord

Let us decompose the first chord of Beethoven's Piano Piano Sonata No. 8 in C minor, Op. 13 (Pathétique), recorded by Arthur Rubinstein. 

The original `mp3` file's name is `Beethoven__Piano_Sonata_Pathetique__Arthur_Rubenstein_64kb.mp3`.
It is converted to mono `wav` file using Linux command:

```
> ffmpeg -i Beethoven__Piano_Sonata_Pathetique__Arthur_Rubenstein_64kb.mp3 -ac 1 Pathetique_mono.wav
```

Then, the first 2 seconds (the C minor chord consisting of 7 notes, see [link](https://tonic-chord.com/beethoven-piano-sonata-no-8-in-c-minor-pathetique-analysis/)) were extracted as above, and saved to the file `Pathetique_mono_2sec.wav`. Here is the code used:

```
begin
	# Load the signal and make a short 2seconds file
	Pat = wavread("files/Pathetique_mono.wav")
	wavwrite(Pat[1][1:48001],"files/Pathetique_mono_2sec.wav",Fs=Pat[2])
end
```

Here we are interested in basic notes and not the overtones, but we keep the threshold at $0.1$.
"""

# ╔═╡ f83e5917-ee9e-46b4-a11e-35a6a2c4a16e
Pat2 = wavread("files/Pathetique_mono_2sec.wav")

# ╔═╡ 5c3bd9b8-128d-4966-9f25-740650ac174a
wavplay(5*Pat2[1],Pat2[2])

# ╔═╡ cd88c2ef-6754-456d-a06a-7bf525c4cc14
# Signal and sampling frequency
p=Pat2[1][:]; Fp=Pat2[2]

# ╔═╡ a205b9df-6dbc-4273-a562-a140114250fc
length(p)

# ╔═╡ dc48d7b5-65ec-45e4-9611-9a3f589d9463
p

# ╔═╡ 1449ca3a-5d15-4e96-8ae7-ce2b96df9747
begin
	# Plot in time scale
	tₚ=range(0,stop=length(p)/Fp,length=length(p))
	plot(tₚ,p,title="C minor chord", legend=false,xlabel="time (s)")
end

# ╔═╡ 18369292-fa64-4f23-9f47-6d7087f2913f
begin
	# Check the signal with FFT
	# Notice that many overtones show up
	fp=abs.(fft(p))
	plot(Fp/length(fp)*(1:length(fp)),fp, title="FFT of the C minor chord",xlabel="Frequency", leg=false, xlims=(0,2000))
end

# ╔═╡ 564c7262-fe98-444e-a1a4-6d04a39fb013
# Form the Hankel matrix
# IMPORTANT - Do not try to display H - it is a 24001 x 24001 matrix.
Hₚ=Hankel(p);

# ╔═╡ d67ac882-dfb3-4469-a6f5-bd7562a9030c
nₚ=size(Hₚ)[1]

# ╔═╡ ffda42db-43e7-4c52-a1ec-bd967f98cfb9
begin
	# We are looking for 20-40 eigenvalue pairs, try different values of nev
	fₚ(x)=mul!(similar(x),Hₚ,x,1.0,0.0)
	Aₚ=LinearMap(fₚ,nₚ,issymmetric=true)
	@time λₚ,Uₚ=eigs(Aₚ, nev=80, which=:LM)
end

# ╔═╡ 925818b5-d07f-4b0b-913e-84ef7ab0a27e
# Eigenvalues are in pairs
λₚ

# ╔═╡ cc992d38-8ef4-4b1a-8c11-2ba00010abf5
# Count the eigenvalue pairs (+-) larger than the 10% of the maximum
τₚ=0.1

# ╔═╡ e9a1ce96-4b7c-4502-884e-daabb567417f
Lₚ=round(Int,(sum(abs.(λₚ).>(τₚ*maximum(abs,λₚ)))/2))

# ╔═╡ 7704fedd-98aa-4f3e-ba27-81292ead309e
begin
	xcompₚ=Array{Array{Float64}}(undef,Lₚ)
	for kp=1:Lₚ
	    kp₁=2*kp-1
	    kp₂=2*kp
		xsimpleₚ=[(λₚ[kp₁]*Uₚ[1,kp₁])*Uₚ[:,kp₁]; (λₚ[kp₁]*Uₚ[nₚ,kp₁])*Uₚ[2:nₚ,kp₁]]
	    xsimpleₚ+=[(λₚ[kp₂]*Uₚ[1,kp₂])*Uₚ[:,kp₂]; (λₚ[kp₂]*Uₚ[nₚ,kp₂])*Uₚ[2:nₚ,kp₂]]
	    xcompₚ[kp]=xsimpleₚ
	end
end

# ╔═╡ 6c16f8ea-9d17-4ed6-9586-7d387170f6cb
md"""
Let us look and listen to what we got:

Mono-component number $(@bind kₚ Slider(1:Lₚ,show_value=true))
"""

# ╔═╡ 2e89cddd-cbbc-4523-946c-f02758318be8
begin
	# FFT of a mono-component and computed frequency
	lp₁=1000
	fsₚ=abs.(fft(xcompₚ[kₚ]))
	mₚ,indₚ=findmax(fsₚ[1:lp₁])
	"Frequency of mono-component $(kₚ) = ", indₚ*Fp/length(fsₚ)  ," Hz, Amplitude = ", mₚ
end

# ╔═╡ 4dc7a0ca-b421-4975-9c53-bd5ba95f9a6b
# Plot the FFT
plot(Fp/length(fsₚ)*(1:lp₁),fsₚ[1:lp₁], title="FFT of mono-component $(kₚ)",leg=false,xlabel="Frequency")

# ╔═╡ f8392ada-3896-4ec0-bf43-b99f58ec714a
# Listen to individual mono-components
wavplay(xcompₚ[kₚ],Fp)

# ╔═╡ 2171aa3d-e86e-4dfc-a3f2-2800707f78c3
# Listen to the sum of mono-components
wavplay(sum([xcompₚ[i] for i=1:Lₚ]),Fp)

# ╔═╡ 35eb8042-1fbd-4416-85aa-39e569c21148
md"""
How to recognize the notes? First we need table of frequencies of individual notes, `Df`, which we create using `DataFrames.jl`. DataFrame 

`Df` is constructed using the table of rounded frequencies `Nf`, where each row represents half-tones from `C` to `B` and each column represents an octave from `Oct0` to `Oct8`.
"""

# ╔═╡ 2ced9f0e-a769-4955-97c2-120d08910420
# Table of rounded frequencies
Nf=[16 33 65 131 262 523 1047 2093 4186; 
	17 35 69 139 277 554 1109 2217 4435;
	18 37 73 147 294 587 1175 2349 4699;
	19 39 78 156 311 622 1245 2489 4978;
	21 41 82 165 330 659 1319 2637 5274;
	22 44 87 175 349 698 1397 2794 5588;
	23 46 93 185 370 740 1480 2960 5920;
	25 49 98 196 392 784 1568 3136 6272;
	26 52 104 208 415 831 1661 3322 6645;
	28 55 110 220 440 880 1760 3520 7040;
	29 58 117 233 466 932 1865 3729 7459;
	31 62 123 247 494 988 1976 3951 7902]

# ╔═╡ ae4e5a29-73a8-45d2-81d2-c3c656ddbfef
Notes=["C", "C♯/D♭", "D", "D♯/E♭", "E", "F", "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"]

# ╔═╡ 283b8d85-d42c-486c-9430-4d0e1111fadb
Octaves=string.(collect(0:8))

# ╔═╡ a1b27036-352d-4f68-a629-44691120f8e6
Chord=Vector{Any}(undef,Lₚ);

# ╔═╡ 63a0bb76-0098-4575-bd2c-fddf3a81a466
begin
	# Compute again the frequencies of all mono-components
	Frequency=Vector{Float64}(undef,Lₚ)
	Amplitude=Vector{Float64}(undef,Lₚ)
	for i=1:Lₚ
		# FFT of a mono-component and computed frequency
		l0=1000
		fs0=abs.(fft(xcompₚ[i]))
		Amplitude[i],ind0=findmax(fs0[1:l0])
		Frequency[i]=ind0*Fp/length(fs0)
	end
end

# ╔═╡ 0b5d7813-d4d8-4e62-9e9a-2dd9eab5adb6
 [Frequency Amplitude]

# ╔═╡ 5adaf3aa-3256-4076-85fc-e10439d8be63
sort(Amplitude,rev=true)

# ╔═╡ 08912314-bbf1-404e-bcdc-05623f48b173
for i=1:Lₚ
	Cind=findfirst(isapprox.(Nf,Frequency[i],rtol=0.03))
	Chord[i]=Notes[Cind[1]]*Octaves[Cind[2]]
end

# ╔═╡ 469b3d33-b616-4e48-998a-d9476263ab74
[Chord Amplitude]

# ╔═╡ 174995db-f16b-4a6c-867e-744f3b2a265c
sort(Amplitude,rev=true)

# ╔═╡ 9d8c7bf8-67b0-45e1-98a5-312dbe032081
begin
	# Indices of notes with Abig largest amplitudes
	Abig=10
	Aind=sortperm(Amplitude,rev=true)[1:Abig]
end

# ╔═╡ 20ac7f13-df71-415d-ad1b-4df08ff7fb3f
# Notes with Abig largest amplitudes
Chord[Aind]

# ╔═╡ 04ae92c1-8609-4fc0-9939-a5963b5ecc59
unique(Chord[Aind])

# ╔═╡ 3463f0b0-60b0-4d10-810a-c53b07914380
md"""
__Nota bene:__ D4 is somewhat unexpected. The frequency (with the largest amplitude) of the 17-th mono-component is 291.5 Hz, close to 294 Hz.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Arpack = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LinearMaps = "7a12625a-238d-50fd-b39a-03d52299707e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialMatrices = "928aab9d-ef52-54ac-8ca1-acd7ca42c160"
ToeplitzMatrices = "c751599d-da0a-543b-9d20-d0a503d91d24"
WAV = "8149f6b0-98f6-5db9-b78f-408fbbb8ef88"

[compat]
Arpack = "~0.5.4"
FFTW = "~1.8.0"
LinearMaps = "~3.11.2"
Plots = "~1.40.3"
PlutoUI = "~0.7.58"
SpecialMatrices = "~3.0.0"
ToeplitzMatrices = "~0.8.3"
WAV = "~1.2.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "60f5ac1832d0de849086c7df37cfd805421a279f"

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
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+2"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DSP]]
deps = ["Compat", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "0df00546373af8eee1598fb4b2ba480b1ebe895c"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.7.10"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc5231d52eb1771251fbd37171dbc408bcc8a1b6"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+1"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "2dd20384bf8c6d411b5c7370865b1e9b26cb2ea3"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.6"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"

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
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "532f9126ad901533af1d4f5c198867227a7bb077"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "ee28ddcd5517d54e417182fec3886e7412d3926f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f31929b9e67066bee48eec8b03c0df47d31a74b3"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.8+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "674ff0db93fffcd11a3573986e550d66cd4fd71f"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "1336e07ba2eb75614c99496501a8f4b233e9fafe"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.10"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "401e4f3f30f43af2c8478fc008da50096ea5240f"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.3.1+0"

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
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "39d64b09147620f5ffbf6b2d3255be3c901bec63"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.8"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "25ee0be4d43d0269027024d75a24c24d6c6e590c"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.4+0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "36bdbc52f13a7d1dcb0f3cd694e01677a515655b"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6ce1e19f3aec9b59186bdf06cdf3c4fc5f5f3e6"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.50.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "61dfdba58e585066d8bce214c5a51eaa0539f269"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "b404131d06f7886402758c9ce2214b636eb4d54a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ee79c3208e55786de58f8dcccca098ced79f743f"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.3"

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

    [deps.LinearMaps.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

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
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e127b609fb9ecba6f201ba7ab753d5a605d53801"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.54.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "dae01f8c2e069a683d3a6e17bbae5070ab94786f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.9"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

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
version = "1.11.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

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
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.SpecialMatrices]]
deps = ["LinearAlgebra", "Polynomials"]
git-tree-sha1 = "8fd75ee3d16a83468a96fd29a1913fb170d2d2fd"
uuid = "928aab9d-ef52-54ac-8ca1-acd7ca42c160"
version = "3.0.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ToeplitzMatrices]]
deps = ["AbstractFFTs", "DSP", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "05a042dcb3dabaedb4f1c20de0932c34a0fcee76"
uuid = "c751599d-da0a-543b-9d20-d0a503d91d24"
version = "0.8.4"
weakdeps = ["StatsBase"]

    [deps.ToeplitzMatrices.extensions]
    ToeplitzMatricesStatsBaseExt = "StatsBase"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d95fe458f26209c66a187b1114df96fd70839efd"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.WAV]]
deps = ["Base64", "FileIO", "Libdl", "Logging"]
git-tree-sha1 = "7e7e1b4686995aaf4ecaaf52f6cd824fa6bd6aa5"
uuid = "8149f6b0-98f6-5db9-b78f-408fbbb8ef88"
version = "1.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "15e637a697345f6743674f1322beefbc5dcd5cfc"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "936081b536ae4aa65415d869287d43ef3cb576b2"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.53.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╠═b4423c48-1d43-4887-9f96-a2f6ac8eaa45
# ╠═b6058fe0-72a9-4be3-9d09-5f35109198a2
# ╠═5b8f2611-cdf9-4761-8768-4b014f02b842
# ╟─9e73d94e-c26e-4513-8142-1058738f103b
# ╟─d1b6461e-6e87-4cab-af2d-f28e09c336ac
# ╠═545776f8-e826-430b-8744-05a53ef708b6
# ╟─a8bff246-a387-43ec-91b9-de97d366c79d
# ╟─9ad716ab-57d8-4b17-b40c-93f6e14903b2
# ╠═2206c20c-9e67-49a4-bc26-b203477b872f
# ╠═a781a4c1-d0b1-4bab-a08b-724697d617f9
# ╠═a03492f5-c172-46f3-839f-4d7417015acb
# ╠═e6ad6e6e-b0ba-42d4-8677-f1149aed08bc
# ╠═e0054a3b-2e14-4bb8-81b0-6763e963dfaa
# ╠═7b8eeeb2-bae5-46ed-9dee-0a8efb2d31c9
# ╟─771a6712-36a4-4e1c-8355-210e4f3787ec
# ╠═407c19d9-d67f-4e98-a8e6-8bb26c24e46f
# ╠═2369b267-cfbc-4092-9822-24582c10af13
# ╠═16f56d88-d9e4-4ad5-9431-3a993408d26d
# ╠═4a35f8fa-33d8-481f-8423-d5af7e32dc8f
# ╠═2d0f04ca-fb8d-4dc9-81e6-1f2600fb0fab
# ╠═79db0e46-1685-4c90-b51c-b158eac11f03
# ╟─97b56eb0-df22-4877-8320-6440a1806c10
# ╟─fd85e7bd-b950-4f0b-909c-cbdc85216a61
# ╠═26beaaa8-923b-4b03-ae6d-af18996eb398
# ╟─afd8681e-3169-44f1-aece-599ca9998531
# ╠═a2570d4b-101c-4120-aa6c-8bbf8e42decd
# ╠═31498e3b-94eb-4fe6-a814-f69fc9e5bb4c
# ╟─c07030fd-47e4-4ee7-b409-8591771f61c7
# ╠═86985c1c-c4a2-4b38-88e5-d1488d903ea8
# ╠═cce7dba2-2bee-474d-b17a-4d091e4a1fd6
# ╠═9dbdb70b-758d-49e7-b0b1-74fb339a9a8d
# ╟─23c239f3-4cd2-48fc-9391-41d361460f98
# ╠═faf54ef7-7f3b-4f16-a18f-b21c0f00c2c9
# ╟─f17cf45a-738e-4d23-8bfb-c06990ebd1fe
# ╟─9782d6f9-285c-46d8-a826-017f4a5bcf53
# ╠═94171cfa-1e8e-4aba-a5b8-faad0104cf80
# ╠═8a1efef7-8903-426c-b8d6-a99bc5288981
# ╠═41b7458c-dc6c-4774-8991-f74519a850ed
# ╟─8fbdea3e-2c49-4068-8b2e-6339737554f2
# ╟─5c76963a-5cd6-4797-bc28-536a565bf4fe
# ╠═561faa84-f4e3-4712-92b0-68e6edabae65
# ╠═0c6a11eb-e4b2-4daa-83c9-97872ca150ce
# ╠═c8d73641-e8ba-46ff-b368-981d3c288d48
# ╠═b69942f4-34b8-47f9-b33a-9776719feac0
# ╠═72f3a029-32f2-4dbd-b8c3-f5794ea85404
# ╠═319a6376-3172-4f9b-9137-c8a3809920c3
# ╠═61d7911a-ea4d-4ef1-9778-6a2e8eff10e1
# ╠═b2fc20f1-faf1-4e08-9ccd-6b175cff0066
# ╠═d132b9c0-8101-4c14-89d7-5fa1462ea71e
# ╠═32f20ec2-bf15-467f-98ef-6dbe6a568bca
# ╠═825e7090-8196-4165-853c-57237f5e05c9
# ╠═8f361246-8941-47a6-98c1-2b92dea2c74b
# ╟─84d96895-e4ad-4bf4-8df7-6e81b983bb3e
# ╠═cc41548c-674a-4f67-bdf7-95ce16a6a5d8
# ╠═eb1daded-ad36-41d3-9088-a9f8cf6bf63f
# ╠═eef3c2ad-3619-49b1-a612-63c234314dfd
# ╠═aac0a39f-0c49-4009-974e-852c7e8e2b17
# ╠═38b8e9f0-2546-4113-83b7-85599faa6992
# ╟─146871c4-e014-4ee4-9933-1d21c2504635
# ╟─e09da774-b32d-487f-97a9-2195d3306224
# ╟─e47f1bd9-89b5-4f67-96bf-d6b4a50a721c
# ╠═309e82f8-2ea4-4de8-a27b-92844948c579
# ╠═0499ad09-e8a8-41e8-99f3-4ca309ceb9d9
# ╠═e3399220-d1f9-48c1-9863-edd307ca7d4e
# ╠═59afc24f-fdf0-4694-a723-426352299629
# ╠═ce4d893e-3470-4e7e-8b03-c22ce5f9f434
# ╠═e52f9b5c-12eb-4b9b-9d60-9b421d5b7fe2
# ╠═ec8ae004-b945-4fe7-a32f-172ff5f6e82a
# ╠═72e0fc55-4c34-473a-b78b-b6910530b1e9
# ╟─b9523247-9c65-4278-8e10-1050783ae73a
# ╠═527f72cf-16de-4c16-bd96-4e6581687527
# ╠═0e8c82c6-ca02-4684-b319-ce35c2cf19cb
# ╠═5f3c0dd6-8ec8-432c-a230-786cf3d8a73a
# ╟─368e2830-3152-40eb-9795-5ea0ec69d8a5
# ╠═533165d8-3c49-4609-8f69-1237c43b6946
# ╠═3873b054-5005-4b17-bae8-a51c44dca506
# ╠═8d59bb1c-458c-4d51-a735-2587c84e0f2c
# ╠═3f257922-9321-4b2a-84b3-ab3e5f57253e
# ╠═b41a985a-c81a-467d-b137-4a0cde1c4a73
# ╠═d539f8c0-2f35-4a88-9646-07aedff40bda
# ╠═9575d673-a0e5-47d1-b109-9be6ee241623
# ╠═b0ac9727-2842-43a5-acee-f2ea74a1115e
# ╠═25375eed-4226-42e2-ae65-f1418443732a
# ╠═a630247b-0505-43ff-b94a-468ef8887728
# ╟─1b3e2e57-7c3f-4365-8616-e4b46b046102
# ╠═93248879-4486-4059-a363-6c7b6a0015d8
# ╠═46f8977f-fe22-4efc-9724-6b8ca588414d
# ╟─949954b4-663d-4ef3-99b5-0df3c74a31e7
# ╟─8fc35997-f124-423e-b384-0f2369ecaa35
# ╠═84b53076-c26b-445a-a458-fe71cca242dc
# ╟─16f2dc1f-30d2-4335-87e2-afb32235f1dc
# ╠═d33b3243-058c-446f-975a-0aee5b5426ac
# ╠═0d3e2f32-19fe-435d-96b6-83f047ecd8ef
# ╠═cf6f1ff9-6439-4ef9-af20-f278495eb239
# ╠═847bb094-a2b8-4459-9d7b-f39bd3db2101
# ╟─e03267b6-1320-435a-818a-c2018556c25b
# ╠═662b0be2-9f82-4980-83de-bb0143c28736
# ╠═60b8f0cc-7e28-4787-be5c-e2b779e655c4
# ╠═1fba26bb-4d17-4df8-8ce2-ca4185101681
# ╟─13e65ea8-e0c4-45ee-ae57-460310380097
# ╠═f83e5917-ee9e-46b4-a11e-35a6a2c4a16e
# ╠═5c3bd9b8-128d-4966-9f25-740650ac174a
# ╠═cd88c2ef-6754-456d-a06a-7bf525c4cc14
# ╠═a205b9df-6dbc-4273-a562-a140114250fc
# ╠═dc48d7b5-65ec-45e4-9611-9a3f589d9463
# ╠═1449ca3a-5d15-4e96-8ae7-ce2b96df9747
# ╠═18369292-fa64-4f23-9f47-6d7087f2913f
# ╠═564c7262-fe98-444e-a1a4-6d04a39fb013
# ╠═d67ac882-dfb3-4469-a6f5-bd7562a9030c
# ╠═ffda42db-43e7-4c52-a1ec-bd967f98cfb9
# ╠═925818b5-d07f-4b0b-913e-84ef7ab0a27e
# ╠═cc992d38-8ef4-4b1a-8c11-2ba00010abf5
# ╠═e9a1ce96-4b7c-4502-884e-daabb567417f
# ╠═7704fedd-98aa-4f3e-ba27-81292ead309e
# ╟─6c16f8ea-9d17-4ed6-9586-7d387170f6cb
# ╠═2e89cddd-cbbc-4523-946c-f02758318be8
# ╠═4dc7a0ca-b421-4975-9c53-bd5ba95f9a6b
# ╠═f8392ada-3896-4ec0-bf43-b99f58ec714a
# ╠═2171aa3d-e86e-4dfc-a3f2-2800707f78c3
# ╟─35eb8042-1fbd-4416-85aa-39e569c21148
# ╠═2ced9f0e-a769-4955-97c2-120d08910420
# ╠═ae4e5a29-73a8-45d2-81d2-c3c656ddbfef
# ╠═283b8d85-d42c-486c-9430-4d0e1111fadb
# ╠═a1b27036-352d-4f68-a629-44691120f8e6
# ╠═63a0bb76-0098-4575-bd2c-fddf3a81a466
# ╠═0b5d7813-d4d8-4e62-9e9a-2dd9eab5adb6
# ╠═5adaf3aa-3256-4076-85fc-e10439d8be63
# ╠═08912314-bbf1-404e-bcdc-05623f48b173
# ╠═469b3d33-b616-4e48-998a-d9476263ab74
# ╠═174995db-f16b-4a6c-867e-744f3b2a265c
# ╠═9d8c7bf8-67b0-45e1-98a5-312dbe032081
# ╠═20ac7f13-df71-415d-ad1b-4df08ff7fb3f
# ╠═04ae92c1-8609-4fc0-9939-a5963b5ecc59
# ╟─3463f0b0-60b0-4d10-810a-c53b07914380
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
