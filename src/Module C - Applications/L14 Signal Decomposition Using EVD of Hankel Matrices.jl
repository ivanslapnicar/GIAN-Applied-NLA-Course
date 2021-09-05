### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 00980e71-789d-4930-bd5a-b49bf6ef59e0
begin
	using PlutoUI
	PlutoUI.TableOfContents(aside=true)
end

# ╔═╡ 2b83a78d-6e87-4417-93dc-416751fb9d4d
# Install with ] add SpecialMatrices#withToeplitz
using Plots, SpecialMatrices, LinearAlgebra

# ╔═╡ ad66c567-a293-4375-a294-bdb593c065fd
using FFTW

# ╔═╡ 02e5af9c-14d9-4d2e-ad50-0e73aed1e111
using LinearMaps

# ╔═╡ 944a0269-6d52-4472-9a50-933012e2fc7c
using Arpack

# ╔═╡ a0c08d02-078b-46c9-a746-b9770dd9df02
using WAV

# ╔═╡ 9e73d94e-c26e-4513-8142-1058738f103b
md"""
# Signal Decomposition Using EVD of Hankel Matrices


Suppose we are given a data signal which consists of several nearly mono-components.

_Can we recover the mono-components?_

The answer is _YES, with an efficient algorithm using EVDs of Hankel matrices._ 

Mono-component recovery can be successfully applied to audio signals.

__Prerequisites__

The reader should be familiar to elementary concepts about signals, and with linear algebra concepts, particularly EVD and its properties and algorithms.
 
__Competences__

The reader should be able to decompose given signal into mono-components using EVD methods.

__References__

For more details see [P. Jain and R. B. Pachori, An iterative approach for decomposition of multi-component non-stationary signals based on eigenvalue decomposition of the Hankel matrix](http://www.sciencedirect.com/science/article/pii/S0016003215002288).

__Credits__: The first Julia implementation was derived in [A. M. Bačak, Master's Thesis]().
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

 $F$ is the __sampling frequency__ of $x$ in Hz, 

 $F_k$ is the sampling frequency of $x^{(k)}$,

 $A_k$ is the __amplitude__ of $x^{(k)}$, and 

 $\theta_k$ is the __phase__ of $x^{(k)}$.

We assume that $F_k< F_{k+1}$ for $k=1,2,\ldots,n-1$, and $F>2F_n$.

A __Hankel matrix__ is a (real) square matrix with constant values along the skew-diagonals. More precisely, let $a\in\mathbb{R}^{2n-1}$. An $n\times n$ matrix $H\equiv H(a)$ for which $H_{ij}=H_{i+1,j-1}=a_{i+j-1}$ is a Hankel matrix.

## Facts

Let $x$ be a signal with $2n-1$ samples composed of $L$ stationary mono-components.

Let $H$ be an $n\times n$ Hankel matrix corresponding to $x$ and let $H=U\Lambda U^T$ be its EVD (Hankel matrix is symmetric) with $\lambda_1\leq \lambda_2 \leq \cdots \leq \lambda_n$.

Smilarly, let $H_k$ be the $n\times n$ Hankel matrix corresponding to the $k$-th component $x^{(k)}$ and let $H_k=U_k\Lambda_k U_k^T$ be its EVD.

1.  $H=\sum\limits_{k=1}^{L} H_k$.

2.  $H_k=\lambda_k U_{:,k}U_{:,k}^T + \lambda_{n-k+1} U_{:,n-k+1}U_{:,n-k+1}^T$.
"""

# ╔═╡ 9ad716ab-57d8-4b17-b40c-93f6e14903b2
md"""
## Example - Signal with three mono-components
"""

# ╔═╡ 2e6d75f8-43a4-47b1-87e6-be95d6ee1f6e
plotly()

# ╔═╡ 545776f8-e826-430b-8744-05a53ef708b6
begin
	# Small Hankel matrix
	a=collect(1:11)
	Hankel(a)
end

# ╔═╡ 2206c20c-9e67-49a4-bc26-b203477b872f
begin
	# Create the signal
	n=160
	N=2*n-1
	F = 6400
	L = 3
	A = [3, 2, 1]
	Fk= [200, 320, 160]
	θ = [pi/2, pi/4, 0]
	x = zeros(N)
	for k=1:3
	    for i=1:N
	        x[i]+=A[k]*cos(2*pi*Fk[k]*i/F+θ[k])
	    end
	end
	x
end

# ╔═╡ a781a4c1-d0b1-4bab-a08b-724697d617f9
# Plot the signal
plot(x,xlabel="Number of samples N", ylabel="Amplitude",leg=false)

# ╔═╡ a03492f5-c172-46f3-839f-4d7417015acb
# Periods
F./Fk

# ╔═╡ e6ad6e6e-b0ba-42d4-8677-f1149aed08bc
# FFT indicates that there are three components with (approximate) 
# angular frequencies 160,200, and 320
plot(range(-F/2,stop=F/2,length=length(x)),abs.(fftshift(fft(x))), title="FFT of a signal", legend=false, xlabel="Angular frequency")

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
	# Compare the first matrix with the Hankel matrix of the first mono-component
	x₁ = zeros(N)
	c=1
	for i=1:N
	    x₁[i]+=A[c]*cos(2*pi*Fk[c]*i/F+θ[c])
	end
end

# ╔═╡ 4a35f8fa-33d8-481f-8423-d5af7e32dc8f
begin
	H₁=Hankel(x₁)
	eigvals(Matrix(H₁)), norm(Hcomp[1]-H₁)
end

# ╔═╡ 2d0f04ca-fb8d-4dc9-81e6-1f2600fb0fab
begin
	# Now we reconstruct the mono-components from the skew-diagonal elements of Hcomp
	xcomp=Array{Array{Float64}}(undef,L)
	z=Array{Float64}(undef,N)
	for k=1:L
	    z[1:2:N]=diag(Hcomp[k])
	    z[2:2:N]=diag(Hcomp[k],1)
	    xcomp[k]=copy(z)
	end
end

# ╔═╡ 79db0e46-1685-4c90-b51c-b158eac11f03
# xaxis=collect(1:N)
plot([xcomp[1],xcomp[2],xcomp[3]],title="Extracted mono-components", label=["First" "Second" "Third"],xlabel="Sample")

# ╔═╡ 97b56eb0-df22-4877-8320-6440a1806c10
md"""
# Fast EVD of Hankel matrices

Several outer eigenvalues pairs of Hankel matrices can be computed using Lanczos method. If the multiplication $Hx$ is performed using Fast Fourier Transform, this EVD computation is very fast.

## Definitions

A __Toeplitz matrix__ is a (real) square matrix with constant values along the diagonals. More precisely, let 

$$
a=(a_{-(n-1)},a_{-(n-2)},\ldots,a_{-1},a_0,a_1,\ldots,a_{n-1})\in\mathbb{R}^{2n-1}.$$ 

An $n\times n$ matrix $T\equiv T(a)$ for which $T_{ij}=T_{i+1,j+1}=a_{i-j}$ is a Toeplitz matrix.

A __circulant matrix__ is a Toeplitz matrix where each column is rotated one element downwards relative to preceeding column. 

More precisely, let $a\in\mathbb{R}^{n}$. An $n\times n$ matrix $C\equiv C(a)=T(a,a_{1:n-1})$ is a Circulant matrix.

A __rotation matrix__ is an identity matrix rotated 90 degrees to the right (or left).

A __Fourier matrix__ is Vandermonde matrix:

$$
F_n=V(1,\omega_n,\omega_n^2,\ldots, \omega_n^{n-1}),$$

where 
$\omega_n=exp(2\pi i/n)$ is the $n$-th root of unity (see the notebook 
[Eigenvalue Decomposition - Definitions and Facts](../Module+B+-+Eigenvalue+and+Singular+Value+Decompositions/L3a+Eigenvalue+Decomposition+-+Definitions+and+Facts.ipynb)).
"""

# ╔═╡ fd85e7bd-b950-4f0b-909c-cbdc85216a61
md"""
Notice different meanings of vector $a$: in `C=Circulant(a)`, $a$ is the first column, in 
`T=Toeplitz(a)`, $a_i$ is the diagonal element of the $i$-th diagonal starting from $T_{1n}$, and in `H=Hankel(a)`, $a_i$ is the element of the $i$-th skew-diagonal starting from $H_{11}$.
"""

# ╔═╡ a2570d4b-101c-4120-aa6c-8bbf8e42decd
# C
Circulant([1,2,3,4,5])

# ╔═╡ 31498e3b-94eb-4fe6-a814-f69fc9e5bb4c
# TC
Toeplitz([2,3,4,5,1,2,3,4,5])

# ╔═╡ 26beaaa8-923b-4b03-ae6d-af18996eb398
# T
Toeplitz([1,2,3,4,5,6,7,8,9])

# ╔═╡ 55c51a2a-d99b-4e39-b2d5-4d82d8742a78
# H₁
Hankel([1,2,3,4,5,6,7,8,9])

# ╔═╡ faf54ef7-7f3b-4f16-a18f-b21c0f00c2c9
Vandermonde([6,2,3,4,5])

# ╔═╡ f17cf45a-738e-4d23-8bfb-c06990ebd1fe
md"""
## Facts 

For more details see [G. H. Golub and C. F. Van Loan, Matrix Computations, p. 202] (http://web.mit.edu/ehliu/Public/sclark/Golub%20G.H.,%20Van%20Loan%20C.F.-%20Matrix%20Computations.pdf), and the references therein

1. Hankel matrix is the product of a Toeplitz matrix and the rotation matrix.

2. Circulant matrix is normal and, thus, unitarily diagonalizable, with the eigenvalue decomposition

$$
C(a)=U\mathop{\mathrm{diag}}(F_n^* a)U^*,$$

where $U=\displaystyle\frac{1}{\sqrt{n}} F_n$. The product $F_n^* a$ can be computed by the _Fast Fourier Transform_(FFT).

3. Given $a,x\in\mathbb{R}^n$, the product $y=C(a)x$ can be computed using FFT as follows:

$$\begin{aligned}
\tilde x&=F_n^*x\\
\tilde a&=F_n^*a\\
\tilde y&=\tilde x.* \tilde a\\
y&= F_n^{-*} \tilde y.
\end{aligned}$$

4. Toeplitz matrix of order $n$ can be embedded in a circulant matrix of order $2n-1$: if $a\in\mathbb{R}^{2n-1}$, then 

$$
T(a)=[C([a_{n+1:2n-1};a_{1:n}])]_{1:n,1:n}.$$

5. Further, let $x\in\mathbb{R}^{n}$ and let $\bar x\in\mathbb{R}^{2n-1}$ be equal to $x$ padded with $n-1$ zeros.Then

$$
T(a)x=[C([a_{n+1:2n-1};a_{1:n}])\bar x]_{1:n}.$$

6. Fact 1 implies $H(a)x=(T(a)J)x=T(a)(Jx)$.

"""

# ╔═╡ 9782d6f9-285c-46d8-a826-017f4a5bcf53
md"
## Examples

### Facts 1 and 2
"

# ╔═╡ 86985c1c-c4a2-4b38-88e5-d1488d903ea8
begin
	J=rotl90(Matrix{Int64}(I,5,5))
end

# ╔═╡ cce7dba2-2bee-474d-b17a-4d091e4a1fd6
Toeplitz([1,2,3,4,5,6,7,8,9])*J, Hankel([1,2,3,4,5,6,7,8,9])

# ╔═╡ 9dbdb70b-758d-49e7-b0b1-74fb339a9a8d
rotl90(Toeplitz([1,2,3,4,5,6,7,8,9]))

# ╔═╡ b9c3585a-e4cb-45a0-a864-a1739f4a47ed
begin
	# Fact 2
	import Random
	Random.seed!(467)
	n₀=6
	a₀=rand(-8:8,n₀)
	a₀,fft(a₀)
end

# ╔═╡ 94171cfa-1e8e-4aba-a5b8-faad0104cf80
begin	
	C₀=Circulant(a₀)
	ω=exp(2*pi*im/n₀)
	v=[ω^k for k=0:n₀-1]
	F₀=Vandermonde(v)
	U₀=F₀/sqrt(n₀)
	λ₀=Matrix(F₀)'*a₀
	λ₀,eigvals(Matrix(C₀))
end

# ╔═╡ 41b7458c-dc6c-4774-8991-f74519a850ed
# Residual
norm(Matrix(C₀)*U₀-U₀*Diagonal(λ₀))

# ╔═╡ e7e58aaf-af48-4eac-a875-b64c569e435c
#?fft

# ╔═╡ 955fa4c1-50a8-4b61-be12-b4adb557ea82
# Check fft
norm(λ₀-fft(a₀))

# ╔═╡ 8fbdea3e-2c49-4068-8b2e-6339737554f2
md"""

### Fast multiplication using FFT

Fact 3 - Circulant() x vector, as implemented in the package `SpecialMatrices.jl`

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

# ╔═╡ 3c96b257-8c18-4e7b-9677-c8a703d5d21f
x₀=rand(-9:9,n₀)

# ╔═╡ c8d73641-e8ba-46ff-b368-981d3c288d48
@which C₀*x₀

# ╔═╡ 5be3c4c2-dcae-4a57-8bcc-9a6d1f580110
[Matrix(C₀)*x₀ C₀*x₀ mul!(similar(x₀),C₀,x₀)]

# ╔═╡ fbe40971-cd45-4196-9577-cc8faab29af0
begin
	# Fact 4 - Embedding Toeplitz() into Circulant()
	Random.seed!(317)
	a₂=rand(-6:6,2*n₀-1)
	T=Toeplitz(a₂)
end

# ╔═╡ a598c1a7-9788-44bb-85ee-24aa4f14d9f9
C=Circulant([a₂[n₀:2*n₀-1];a₂[1:n₀-1]])

# ╔═╡ d132b9c0-8101-4c14-89d7-5fa1462ea71e
# Fact 5 - Toeplitz() x vector
x₂=rand(-6:6,n₀)

# ╔═╡ f3fce6ec-b9c8-4aae-bdfa-2387bb2cc38a
md"""
$$
\begin{bmatrix} T & A \\ B & C\end{bmatrix} 
\begin{bmatrix} x \\ 0 \end{bmatrix}$$
"""

# ╔═╡ 32f20ec2-bf15-467f-98ef-6dbe6a568bca
[Matrix(T)*x₂ T*x₂ mul!(similar(x₂),T,x₂)]

# ╔═╡ 825e7090-8196-4165-853c-57237f5e05c9
# Fact 6 - Hankel() x vector
H₂=Hankel(a₂)

# ╔═╡ 8f361246-8941-47a6-98c1-2b92dea2c74b
[Matrix(H₂)*x₂ H₂*x₂ mul!(similar(x₂),H₂,x₂)]

# ╔═╡ 84d96895-e4ad-4bf4-8df7-6e81b983bb3e
md"""
### Fast EVD of a Hankel matrix

Given a Hankel matrix $H$, the Lanczos method can be applied by defining a function (linear map) which returns the product $Hx$ for any vector $x$. This approach uses the package [LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl) and is described in the notebook
[Symmetric Eigenvalue Decomposition - Lanczos Method](../Module+B+-+Eigenvalue+and+Singular+Value+Decompositions/L4d+Symmetric+Eigenvalue+Decomposition+-+Lanczos+Method.ipynb) notebook. 

_The computation is very fast and allocates little extra space._
"""

# ╔═╡ cc41548c-674a-4f67-bdf7-95ce16a6a5d8
f(x)=mul!(similar(x),H,x)

# ╔═╡ 40e51d70-0e20-48c8-9156-b1475870a604
H

# ╔═╡ eb1daded-ad36-41d3-9088-a9f8cf6bf63f
A₁=LinearMap(f,size(H,1),issymmetric=true)

# ╔═╡ eef3c2ad-3619-49b1-a612-63c234314dfd
size(A₁)

# ╔═╡ aac0a39f-0c49-4009-974e-852c7e8e2b17
@time eigvals(Matrix(H));

# ╔═╡ 38b8e9f0-2546-4113-83b7-85599faa6992
# Run twice
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


## Fact

The following algorithm decomposes the signal $x$:
1. Choose $\tau$ and form the Hankel matrix $H$
2. Compute the EVD of $H$
3. Choose the significant eigenpairs of $H$
4. For each significant eigenpair $(\lambda,u)$
    1. Form the rank one matrix $M=\lambda uu^T$
    2. Define a new signal $y$ consisting of averages of the skew-diagonals of $M$
    3. Form the Hankel matrix $H(y)$
    3. Compute the EVD of $H(y)$
    4. Choose the significant eigenpairs of $H(y)$
    5. __If__ $H(y)$ has only two significant eigenpairs, declare $y$ a mono-component, __else__ go to step 4.
"""

# ╔═╡ e47f1bd9-89b5-4f67-96bf-d6b4a50a721c
md"""
## Example -  Note A

Each tone has its fundamental frequency (mono-component). However, musical instruments produce different overtones (harmonics) which are near integer multiples of the fundamental frequency.
Due to construction of resonant boxes, these frequencies slightly vary in time, and their amplitudes are contained in a time varying envelope.

Tones produces by musical instruments  are nice examples of non-stationary signals. We shall decompose the note A4 played on piano.

For manipulation of recordings, we are using package [WAV.jl](https://github.com/dancasimiro/WAV.jl). Another package with similar functionality is the package [AudioIO.jl](https://github.com/ssfrr/AudioIO.jl).
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
	sₐ=Signalₐ[1]
	Fs=Signalₐ[2]
end

# ╔═╡ ce4d893e-3470-4e7e-8b03-c22ce5f9f434
# Play the signal
wavplay(sₐ,Fs)

# ╔═╡ e52f9b5c-12eb-4b9b-9d60-9b421d5b7fe2
# Plot the signal
plot(sₐ,title="Note a", legend=false, xlabel="sample")

# ╔═╡ ec8ae004-b945-4fe7-a32f-172ff5f6e82a
begin
	# Plot in time scale
	tₐ=range(0,stop=length(sₐ)/Fs,length=length(sₐ))
	plot(tₐ,sₐ,title="Note a", legend=false,xlabel="time (s)")
end

# ╔═╡ 72e0fc55-4c34-473a-b78b-b6910530b1e9
# Total time and number of samples
tₐ[end], length(sₐ)

# ╔═╡ b9523247-9c65-4278-8e10-1050783ae73a
md"
Last part of the signal is just noise, so we create (or read) a shorter signal. $N$ must be odd.
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
Let us visualize the signal in detail.
"""

# ╔═╡ 6945a13f-7554-40b9-9e65-75c18ce5ec1a
@bind k Slider(1:1000:100001-1000,show_value=true)

# ╔═╡ 533165d8-3c49-4609-8f69-1237c43b6946
plot(tₐ[k:k+1000],s[k:k+1000], title="Note a",label=false,xlabel="time (s)")

# ╔═╡ 3873b054-5005-4b17-bae8-a51c44dca506
# Save the short signal
wavwrite(s,"files/piano_A41_short.wav",Fs=Fs)

# ╔═╡ 8d59bb1c-458c-4d51-a735-2587c84e0f2c
begin
	# Check the signal with FFT
	# Notice 3 stronger harmonics and six weaker ones
	fs=abs.(fft(s))
	plot(Fs/length(fs)*(1:length(fs)),fs, title="FFT of note a",xlabel="Frequency", leg=false)
end

# ╔═╡ a1cf574f-db23-424f-aa0b-301770768323
@bind l Slider(1:1000:100001-1000,show_value=true)

# ╔═╡ 3f257922-9321-4b2a-84b3-ab3e5f57253e
# Form the Hankel matrix
# IMPORTANT - Do not try to display H - it is a 50001 x 50001 matrix.
Hₐ=Hankel(vec(s));

# ╔═╡ b41a985a-c81a-467d-b137-4a0cde1c4a73
size(Hₐ), Hₐ[100,200]

# ╔═╡ d539f8c0-2f35-4a88-9646-07aedff40bda
# Get the idea about time to compute EVD
@time fft(s);

# ╔═╡ 9575d673-a0e5-47d1-b109-9be6ee241623
begin
	# We are looking for 20 eigenvalue pairs
	nₐ=size(Hₐ,1)
	fₐ(x)=mul!(similar(x),Hₐ,x)
	Aₐ=LinearMap(fₐ,nₐ,issymmetric=true)
	size(Aₐ)
end

# ╔═╡ b0ac9727-2842-43a5-acee-f2ea74a1115e
@time λₐ,Uₐ=eigs(Aₐ, nev=40, which=:LM)

# ╔═╡ ef8de8e6-ad76-4d67-a9ef-5b99dafe411d
begin
	# Count the eigenvalue pairs (+-) larger than the 10% of the maximum
	τ=0.1
	Lₐ=round(Int,(sum(abs.(λₐ).>(τ*maximum(abs,λₐ)))/2))
end

# ╔═╡ 1b3e2e57-7c3f-4365-8616-e4b46b046102
md"""
At this point, the implementation using full matrices is rather obvious. However, we cannot do that, due to large dimension. Recall, the task is to define Hankel matrices $H_k$ for $k=1,\ldots,L$, from the signal obtained by averaging the skew-diagonals of the matrices

$$
H_k=\lambda_k U_{:,k}U_{:,k}^T + \lambda_{n-k+1} U_{:,n-k+1}U_{:,n-k+1}^T,$$

__without actually forming the matrices__.

This is a nice programming excercise which can be solved using $\cdot$ products.
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

# ╔═╡ 251b6304-4314-479e-aa8a-d9e573d29c69
begin
	# A small test
	u=[1,2,3,4,5]
	u*u'
end

# ╔═╡ 3d552383-bf50-422d-9d8a-e096cdb521c4
averages(1,u)

# ╔═╡ 949954b4-663d-4ef3-99b5-0df3c74a31e7
md"""
We now execute the first step of the algorithm from the above Fact.

Notice that `eigs()` returns the eigenvalues arranged by the absoulte value, so the consecutive 
pairs define the $i$-th signal. The computation of averages is long - it requires $O(n^2)$ 
operations and takes several minutes.
"""

# ╔═╡ 46f8977f-fe22-4efc-9724-6b8ca588414d
# This step takes 7 minutes, so we skip it

# xcompₐ=Array(Array{Float64},Lₐ)
# for k=1:Lₐ
#     xcompₐ[k]=averages(λₐ[2*k-1],Uₐ[:,2*k-1])+averages(λₐ[2*k],Uₐ[:,2*k])
# end

# ╔═╡ 8fc35997-f124-423e-b384-0f2369ecaa35
md"""
__Can we do without averaging?__

The function `averages()` is very slow - 7 minutes, compared to 5 seconds for the eigenvalue computation.

The simplest option is to disregard the averages, and use the first column and the last row of the underlying matrix, as in definition of Hankel matrices, which we do next. 
Smarter approach might be to use small random samples 
to compute the averages.

Let us try the simple approach for the fundamental frequency. (See also the notebook [Examples in Signal Decomposition.ipynb](S8%20Examples%20in%20Signal%20Decomposition.ipynb).)
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
"""

# ╔═╡ 62c83b97-32ed-4c48-987e-bbbd95afbd20
typeof(xcompₐ[1])

# ╔═╡ 400fa7e6-6090-46fb-9610-ab3c816177c5
@bind kₐ Slider(1:Lₐ,show_value=true)

# ╔═╡ d33b3243-058c-446f-975a-0aee5b5426ac
plot(t,xcompₐ[kₐ],title="Mono-component $(kₐ)",leg=false,xlabel="time (s)")

# ╔═╡ 08f567b2-769a-498d-9a49-4fb82eae8639
@bind lₐ Slider(1:1000:100001-1000,show_value=true)

# ╔═╡ 9c452b24-0e8a-4b4e-a408-ea60da97a831
# Details of a mono-component
plot(t[lₐ:lₐ+1000],xcompₐ[kₐ][lₐ:lₐ+1000], title="Mono-component $(kₐ)",leg=false,xlabel="time (s)")

# ╔═╡ 87f1dd44-5127-4c94-b0ef-4aa018029c18
@bind k₂ Slider(1:Lₐ,show_value=true)

# ╔═╡ 0d3e2f32-19fe-435d-96b6-83f047ecd8ef
begin
	# FFT of a mono-component and computed frequency
	l₁=10000
	fsₐ=abs.(fft(xcompₐ[k₂]))
	m,ind=findmax(fsₐ[1:l₁])
	"Frequency of mono-component $(k₂) = ", ind*Fs/length(fsₐ)  ," Hz, Amplitude = ", m
end

# ╔═╡ cf6f1ff9-6439-4ef9-af20-f278495eb239
# Plot the FFT
plot(Fs/length(fsₐ)*(1:l₁),fsₐ[1:l₁], title="FFT of mono-component $(k₂)",leg=false,xlabel="Frequency")

# ╔═╡ e03267b6-1320-435a-818a-c2018556c25b
md"""
We see that all `xcompₐ[k]` are clean mono-components - see 
[Physics of Music - Notes](http://www.phy.mtu.edu/~suits/notefreqs.html):

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

# ╔═╡ 0f3840e7-0ca5-4d57-ac12-cd4df3c9caf3
@bind k₃ Slider(1:Lₐ,show_value=true)

# ╔═╡ 847bb094-a2b8-4459-9d7b-f39bd3db2101
# Listen to individual mono-components
wavplay(xcompₐ[k₃],Fs)

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

# ╔═╡ Cell order:
# ╟─00980e71-789d-4930-bd5a-b49bf6ef59e0
# ╟─9e73d94e-c26e-4513-8142-1058738f103b
# ╟─d1b6461e-6e87-4cab-af2d-f28e09c336ac
# ╟─9ad716ab-57d8-4b17-b40c-93f6e14903b2
# ╠═2b83a78d-6e87-4417-93dc-416751fb9d4d
# ╠═2e6d75f8-43a4-47b1-87e6-be95d6ee1f6e
# ╠═545776f8-e826-430b-8744-05a53ef708b6
# ╠═2206c20c-9e67-49a4-bc26-b203477b872f
# ╠═a781a4c1-d0b1-4bab-a08b-724697d617f9
# ╠═a03492f5-c172-46f3-839f-4d7417015acb
# ╠═ad66c567-a293-4375-a294-bdb593c065fd
# ╠═e6ad6e6e-b0ba-42d4-8677-f1149aed08bc
# ╠═e0054a3b-2e14-4bb8-81b0-6763e963dfaa
# ╠═7b8eeeb2-bae5-46ed-9dee-0a8efb2d31c9
# ╟─771a6712-36a4-4e1c-8355-210e4f3787ec
# ╠═407c19d9-d67f-4e98-a8e6-8bb26c24e46f
# ╠═2369b267-cfbc-4092-9822-24582c10af13
# ╠═4a35f8fa-33d8-481f-8423-d5af7e32dc8f
# ╠═2d0f04ca-fb8d-4dc9-81e6-1f2600fb0fab
# ╠═79db0e46-1685-4c90-b51c-b158eac11f03
# ╟─97b56eb0-df22-4877-8320-6440a1806c10
# ╟─fd85e7bd-b950-4f0b-909c-cbdc85216a61
# ╠═a2570d4b-101c-4120-aa6c-8bbf8e42decd
# ╠═31498e3b-94eb-4fe6-a814-f69fc9e5bb4c
# ╠═26beaaa8-923b-4b03-ae6d-af18996eb398
# ╠═55c51a2a-d99b-4e39-b2d5-4d82d8742a78
# ╠═faf54ef7-7f3b-4f16-a18f-b21c0f00c2c9
# ╟─f17cf45a-738e-4d23-8bfb-c06990ebd1fe
# ╟─9782d6f9-285c-46d8-a826-017f4a5bcf53
# ╠═86985c1c-c4a2-4b38-88e5-d1488d903ea8
# ╠═cce7dba2-2bee-474d-b17a-4d091e4a1fd6
# ╠═9dbdb70b-758d-49e7-b0b1-74fb339a9a8d
# ╠═b9c3585a-e4cb-45a0-a864-a1739f4a47ed
# ╠═94171cfa-1e8e-4aba-a5b8-faad0104cf80
# ╠═41b7458c-dc6c-4774-8991-f74519a850ed
# ╠═e7e58aaf-af48-4eac-a875-b64c569e435c
# ╠═955fa4c1-50a8-4b61-be12-b4adb557ea82
# ╟─8fbdea3e-2c49-4068-8b2e-6339737554f2
# ╠═3c96b257-8c18-4e7b-9677-c8a703d5d21f
# ╠═c8d73641-e8ba-46ff-b368-981d3c288d48
# ╠═5be3c4c2-dcae-4a57-8bcc-9a6d1f580110
# ╠═fbe40971-cd45-4196-9577-cc8faab29af0
# ╠═a598c1a7-9788-44bb-85ee-24aa4f14d9f9
# ╠═d132b9c0-8101-4c14-89d7-5fa1462ea71e
# ╟─f3fce6ec-b9c8-4aae-bdfa-2387bb2cc38a
# ╠═32f20ec2-bf15-467f-98ef-6dbe6a568bca
# ╠═825e7090-8196-4165-853c-57237f5e05c9
# ╠═8f361246-8941-47a6-98c1-2b92dea2c74b
# ╟─84d96895-e4ad-4bf4-8df7-6e81b983bb3e
# ╠═02e5af9c-14d9-4d2e-ad50-0e73aed1e111
# ╠═cc41548c-674a-4f67-bdf7-95ce16a6a5d8
# ╠═40e51d70-0e20-48c8-9156-b1475870a604
# ╠═eb1daded-ad36-41d3-9088-a9f8cf6bf63f
# ╠═eef3c2ad-3619-49b1-a612-63c234314dfd
# ╠═aac0a39f-0c49-4009-974e-852c7e8e2b17
# ╠═944a0269-6d52-4472-9a50-933012e2fc7c
# ╠═38b8e9f0-2546-4113-83b7-85599faa6992
# ╟─146871c4-e014-4ee4-9933-1d21c2504635
# ╟─e47f1bd9-89b5-4f67-96bf-d6b4a50a721c
# ╠═a0c08d02-078b-46c9-a746-b9770dd9df02
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
# ╠═6945a13f-7554-40b9-9e65-75c18ce5ec1a
# ╠═533165d8-3c49-4609-8f69-1237c43b6946
# ╠═3873b054-5005-4b17-bae8-a51c44dca506
# ╠═8d59bb1c-458c-4d51-a735-2587c84e0f2c
# ╠═a1cf574f-db23-424f-aa0b-301770768323
# ╠═3f257922-9321-4b2a-84b3-ab3e5f57253e
# ╠═b41a985a-c81a-467d-b137-4a0cde1c4a73
# ╠═d539f8c0-2f35-4a88-9646-07aedff40bda
# ╠═9575d673-a0e5-47d1-b109-9be6ee241623
# ╠═b0ac9727-2842-43a5-acee-f2ea74a1115e
# ╠═ef8de8e6-ad76-4d67-a9ef-5b99dafe411d
# ╟─1b3e2e57-7c3f-4365-8616-e4b46b046102
# ╠═93248879-4486-4059-a363-6c7b6a0015d8
# ╠═251b6304-4314-479e-aa8a-d9e573d29c69
# ╠═3d552383-bf50-422d-9d8a-e096cdb521c4
# ╟─949954b4-663d-4ef3-99b5-0df3c74a31e7
# ╠═46f8977f-fe22-4efc-9724-6b8ca588414d
# ╟─8fc35997-f124-423e-b384-0f2369ecaa35
# ╠═84b53076-c26b-445a-a458-fe71cca242dc
# ╟─16f2dc1f-30d2-4335-87e2-afb32235f1dc
# ╠═62c83b97-32ed-4c48-987e-bbbd95afbd20
# ╠═400fa7e6-6090-46fb-9610-ab3c816177c5
# ╠═d33b3243-058c-446f-975a-0aee5b5426ac
# ╠═08f567b2-769a-498d-9a49-4fb82eae8639
# ╠═9c452b24-0e8a-4b4e-a408-ea60da97a831
# ╟─87f1dd44-5127-4c94-b0ef-4aa018029c18
# ╠═0d3e2f32-19fe-435d-96b6-83f047ecd8ef
# ╠═cf6f1ff9-6439-4ef9-af20-f278495eb239
# ╟─e03267b6-1320-435a-818a-c2018556c25b
# ╟─0f3840e7-0ca5-4d57-ac12-cd4df3c9caf3
# ╠═847bb094-a2b8-4459-9d7b-f39bd3db2101
# ╠═662b0be2-9f82-4980-83de-bb0143c28736
# ╠═60b8f0cc-7e28-4787-be5c-e2b779e655c4
# ╠═1fba26bb-4d17-4df8-8ce2-ca4185101681
