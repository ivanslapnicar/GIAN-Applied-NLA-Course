### A Pluto.jl notebook ###
# v0.12.21

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

# ╔═╡ 80456803-2a40-40a3-adf5-b412cf132143
begin
	using Polynomials
	p=fromroots([1,2,3,4])
end

# ╔═╡ 9d75f84a-dfda-43c2-9878-9f6bdf047adb
using Plots

# ╔═╡ 086c2da0-7c4f-11eb-1e62-99242ca4db07
using LinearAlgebra

# ╔═╡ 85afa530-7c54-11eb-333a-1b15e02d03f6
using PlutoUI

# ╔═╡ 2bd4e64a-85a4-4726-aad7-01938b4d7edc
md"""
# Solutions 1 - Examples in Julia



## Assignment 1

The function `eps()` return the smallest real number larger than 1.0. It can be called for each of the `AbstractFloat` types. 

Functions `floatmin()` and `floatmax()` return the largest and the smallest positive numbers representable in the given type.
"""

# ╔═╡ 976a9e66-ad6c-42b3-939a-afa33c4dcfb8
subtypes(AbstractFloat)

# ╔═╡ 45e49a4e-f595-4d50-9bba-3d350962ee71
# Default values are for Float64
eps(), floatmax(), floatmin()

# ╔═╡ c0b0f670-7ce5-11eb-3b0a-4f234cd20b49
1.0+eps()/2>1.0

# ╔═╡ 5e133d50-7c4c-11eb-1d2b-d951910f16e2
for T in (Float16, Float32, Float64, BigFloat)
    println(eps(T)," ", floatmin(T)," ", floatmax(T))
end

# ╔═╡ acc500d1-3cdf-4c3f-86c6-985a0176ef7c
md"""
We see that `BigFloat` has approximately 77 significant decimal digits (actually 256 bits) and very large exponents. This makes the format ideal for Greaffe's method. 

Precision of `BigFloat` can be increased, but exponents do not change.
"""

# ╔═╡ 25b172d5-8ee1-479a-9621-dc9b02054fee
precision(BigFloat)

# ╔═╡ bb3f06e0-db79-4905-b23c-4d3eae8ec83a
begin
	setprecision(512)
	eps(BigFloat), floatmax(BigFloat)
end

# ╔═╡ 71652918-d91f-451b-80e8-5f59a2e6ce10
setprecision(256)

# ╔═╡ db19eb66-2ecb-4b47-929c-4369718373e1
md"""
Here is the function for the Graeffe's method. We also define small test polynomial with all real simple zeros.
"""

# ╔═╡ dc4cfc80-7ce5-11eb-0a0e-852f1c12ce11
# varinfo(Polynomials)

# ╔═╡ c1531fcb-5e7f-4994-baf0-c12d1028ad2e
roots(p)

# ╔═╡ eeca4340-7ce5-11eb-191c-79230c225529
derivative(p)

# ╔═╡ 0285883e-7ce6-11eb-0b3b-e708c1fd8980
integrate(p)

# ╔═╡ 50532280-7ce6-11eb-1034-bd8a269fbd56
companion(p)

# ╔═╡ 71e04ea0-7ce6-11eb-2232-a7680fccc403
eigvals(companion(p))

# ╔═╡ 60d365fa-907a-47f7-89cd-034215bd91e5
function Graeffe(p::Polynomial{T},steps::Int64) where T
    # map the polynomial to BigFloat
    pbig=Polynomial(map(BigFloat,coeffs(p)))
    px=Polynomial([zero(BigFloat),one(BigFloat)])
    n=degree(p)
    σ=map(BigFloat,2^steps)
    for k=1:steps
        peven=Polynomial(coeffs(pbig)[1:2:end])
        podd=Polynomial(coeffs(pbig)[2:2:end])
        pbig=peven^2-podd^2*px 
    end
    # @show p[end]
    y=Array{BigFloat}(undef,n)
    # Normalize if p is not monic
    y[1]=-pbig[end-1]/pbig[end]
    for k=2:n
        y[k]=-pbig[end-k]/pbig[end-(k-1)]
    end
    # Extract the roots
    for k=1:n
        y[k]=exp(log(y[k])/σ)
    end
    # Return root in Float64
    map(Float64,y)
end

# ╔═╡ b0e22007-d80a-4db0-8796-f3d28e735ddd
Graeffe(p,8)

# ╔═╡ a5ea266d-2406-4bd3-b5b6-0007f06fda60
md"""
Now the Wilkinson's polynomial:
"""

# ╔═╡ 28003bac-9ae8-4a6d-9073-830e0c47f8fa
ω=fromroots(collect(one(BigFloat):20))

# ╔═╡ 58b8dda8-c7f9-4f28-86c1-c7977e7418fa
r=Graeffe(ω,8)

# ╔═╡ d53d270d-063f-4e9d-9865-e746b5ed6012
r[2]

# ╔═╡ 19dbd540-f872-4f74-a1f6-f742c2ac52f6
Graeffe(ω,16)[2]

# ╔═╡ 4d810eb7-d825-4f4a-a819-f00f38109eca
md"""
We now generate the Chebyshev polynomial $T_{50}(x)$ using the three term recurence.
"""

# ╔═╡ 729b1a27-da12-4158-a4a2-17767e7b272b
begin
	n=50
	T₀=Polynomial([BigInt(1)])
	T₁=Polynomial([0,1])
	Tₓ=Polynomial([0,1])
	Tₙ=Polynomial([0,1])
	for i=3:n+1
	    Tₙ=2*Tₓ*T₁-T₀
	    T₀=T₁
	    T₁=Tₙ
	end
end

# ╔═╡ ef196b97-81c1-458b-8725-4864b770ff2c
Tₙ

# ╔═╡ 3247ea24-7e87-47da-a762-95bf28e07ee2
f(x)=Tₙ(x)

# ╔═╡ e32f5d08-fd9d-4a84-b209-f101df96586d
plot(f,-1,1)

# ╔═╡ 84d4bbf0-dffc-4e45-9759-abbaf2093525
md"""
In order to use Graeffe's method, we need to shift $T$ to the right by one, so that all roots have simple moduli, that is, we compute $T(1-x)$:
"""

# ╔═╡ e0dde918-8672-4124-ac42-aad9d3c3b2a9
Tₛ=Tₙ(Polynomial([BigFloat(1),-1]));

# ╔═╡ b2b74778-6ea2-4279-be80-7c3796e4e5f0
begin
	fₛ(x)=Tₛ(x)
	plot(fₛ,0,2)
end

# ╔═╡ 5b793dcd-d176-48f8-bf1a-8c7c67b7acc4
# Computed roots, 16 steps are fine
y=Graeffe(Tₛ,16).-1

# ╔═╡ bc80699f-4b0a-4e11-9380-c22d9b79addd
# Exact roots
z=map(Float64,[cos((2*k-1)*pi/(2*n)) for k=1:n]);

# ╔═╡ 23227215-f509-436a-8bd1-4ebf530750dc
# Relative error
maximum(abs,(z-y)./z)

# ╔═╡ de733944-8938-4864-836e-501324fab73c
md"""
## Assignment 2

The key is that `/` works for block matrices, too. $A$ is overwritten and must therefore be copied at the beggining of the function, so that the original matrix is not overwritten.
"""

# ╔═╡ 73e29784-73c3-4eab-8e0c-05f87a135cd5
# Strang's book, page 100
function mylu(A₀::Array{T}) where T 
    A=copy(A₀)
    n,m=size(A)
    for k=1:n-1
        for rho=k+1:n
            A[rho,k]=A[rho,k]/A[k,k]
            for l=k+1:n
                A[rho,l]=A[rho,l]-A[rho,k]*A[k,l]
            end
        end
    end
    # We return L and U
    L=tril(A,-1)
    U=triu(A)
    # This is the only difference for the block case
    for i=1:maximum(size(L))
        L[i,i]=one(L[1,1])
    end
    L,U
end

# ╔═╡ cefd4484-62a7-4640-ac97-b69b0122338c
A=rand(5,5)

# ╔═╡ 1af22159-7662-4259-9e74-50f2fbdfa104
L,U=mylu(A)

# ╔═╡ 0fbcbbe2-6461-4cb4-ba0a-8fb76b6ad8df
# Residual
L*U-A

# ╔═╡ cd8ca823-9027-423b-a171-a7e84cdaf022
L

# ╔═╡ 09b776de-ad89-4503-8a45-3d6b635cb413
U

# ╔═╡ f9e9697b-702b-45ba-911c-409ff5a070ad
md"""
We now try block-matrices. We need a convenience function to unblock the block-matrix. First, a small example:
"""

# ╔═╡ 22f0e9a0-e8c7-4d9c-ba82-710031783c33
unblock(A) = mapreduce(identity, hcat, 
    [mapreduce(identity, vcat, A[:,i]) for i = 1:size(A,2)])

# ╔═╡ 843bccd9-6546-4c91-9e94-ef9e894901bd
begin
	# Try k,l=32,16 i k,l=64,8
	k,l=2,4
	# k,l=32,16
	# k,l=64,8
	Ab=[rand(k,k) for i=1:l, j=1:l]
	Ar=unblock(Ab)
end

# ╔═╡ 7312297c-7515-4d49-a774-bfe01a77a126
Ab[1,1]

# ╔═╡ 3a51917a-d73c-4299-a79c-93ffee0a279f
Lb,Ub=mylu(Ab)

# ╔═╡ a9837f40-7c4f-11eb-2d1d-4726be2375ac
# Built-in LAPack function with pivoting
lu(Ar)

# ╔═╡ 5a4e9399-c48d-42aa-bd61-80f2e5011ddc
L[1,1]

# ╔═╡ f230dbc0-6cea-436a-a16e-c1e10dadfb8c
# Residual
R=Lb*Ub-Ab

# ╔═╡ d5bdff29-d7e3-4c7b-a60c-eea6d0d60ded
norm(R)

# ╔═╡ fc06fbd7-56e0-4ec5-af90-52626de2ee4a
md"""
## Assignment 3
"""

# ╔═╡ 66409e20-7c54-11eb-1925-613427e9efa4
md"
k = $(@bind kₘ Slider(10:30,show_value=true))
n = $(@bind nₘ Slider(10:30,show_value=true))
"

# ╔═╡ 022d3ee4-0c3d-467a-a64b-5d9ada7b489f
begin
	E=Array{Any}(undef,nₘ,kₘ)
	for i=1:kₘ
		# Unsymmetric uniform distribution
	    # A=rand(nₘ,nₘ)
		# Unsymmetric normal distribution
		# A=randn(nₘ,nₘ)
		# Symmetric uniform distribution
		# B=rand(nₘ,nₘ); A=B'+B
		# Symmetric normal distribution
		B=randn(nₘ,nₘ); A=B'+B
	    E[:,i]=eigvals(A)
	end
	# We need this since plot cannot handle `Any`
	E=map(eltype(E[1,1]),E)
end

# ╔═╡ 92e27997-3df3-44b7-ad5d-154f4e7bb413
scatter(E,legend=false)

# ╔═╡ c3edd1f6-87a9-462d-8fe2-88b971a62c7a
md"""
_Mathematics is about spotting patterns!_ (Alan Edelman)
"""

# ╔═╡ b069b405-53b4-45e5-ad2c-b29114a8bddd


# ╔═╡ Cell order:
# ╟─2bd4e64a-85a4-4726-aad7-01938b4d7edc
# ╠═976a9e66-ad6c-42b3-939a-afa33c4dcfb8
# ╠═45e49a4e-f595-4d50-9bba-3d350962ee71
# ╠═c0b0f670-7ce5-11eb-3b0a-4f234cd20b49
# ╠═5e133d50-7c4c-11eb-1d2b-d951910f16e2
# ╟─acc500d1-3cdf-4c3f-86c6-985a0176ef7c
# ╠═25b172d5-8ee1-479a-9621-dc9b02054fee
# ╠═bb3f06e0-db79-4905-b23c-4d3eae8ec83a
# ╠═71652918-d91f-451b-80e8-5f59a2e6ce10
# ╟─db19eb66-2ecb-4b47-929c-4369718373e1
# ╠═80456803-2a40-40a3-adf5-b412cf132143
# ╠═dc4cfc80-7ce5-11eb-0a0e-852f1c12ce11
# ╠═c1531fcb-5e7f-4994-baf0-c12d1028ad2e
# ╠═eeca4340-7ce5-11eb-191c-79230c225529
# ╠═0285883e-7ce6-11eb-0b3b-e708c1fd8980
# ╠═50532280-7ce6-11eb-1034-bd8a269fbd56
# ╠═71e04ea0-7ce6-11eb-2232-a7680fccc403
# ╠═60d365fa-907a-47f7-89cd-034215bd91e5
# ╠═b0e22007-d80a-4db0-8796-f3d28e735ddd
# ╟─a5ea266d-2406-4bd3-b5b6-0007f06fda60
# ╠═28003bac-9ae8-4a6d-9073-830e0c47f8fa
# ╠═58b8dda8-c7f9-4f28-86c1-c7977e7418fa
# ╠═d53d270d-063f-4e9d-9865-e746b5ed6012
# ╠═19dbd540-f872-4f74-a1f6-f742c2ac52f6
# ╟─4d810eb7-d825-4f4a-a819-f00f38109eca
# ╠═729b1a27-da12-4158-a4a2-17767e7b272b
# ╠═ef196b97-81c1-458b-8725-4864b770ff2c
# ╠═9d75f84a-dfda-43c2-9878-9f6bdf047adb
# ╠═3247ea24-7e87-47da-a762-95bf28e07ee2
# ╠═e32f5d08-fd9d-4a84-b209-f101df96586d
# ╟─84d4bbf0-dffc-4e45-9759-abbaf2093525
# ╠═e0dde918-8672-4124-ac42-aad9d3c3b2a9
# ╠═b2b74778-6ea2-4279-be80-7c3796e4e5f0
# ╠═5b793dcd-d176-48f8-bf1a-8c7c67b7acc4
# ╠═bc80699f-4b0a-4e11-9380-c22d9b79addd
# ╠═23227215-f509-436a-8bd1-4ebf530750dc
# ╟─de733944-8938-4864-836e-501324fab73c
# ╠═086c2da0-7c4f-11eb-1e62-99242ca4db07
# ╠═73e29784-73c3-4eab-8e0c-05f87a135cd5
# ╠═cefd4484-62a7-4640-ac97-b69b0122338c
# ╠═1af22159-7662-4259-9e74-50f2fbdfa104
# ╠═0fbcbbe2-6461-4cb4-ba0a-8fb76b6ad8df
# ╠═cd8ca823-9027-423b-a171-a7e84cdaf022
# ╠═09b776de-ad89-4503-8a45-3d6b635cb413
# ╟─f9e9697b-702b-45ba-911c-409ff5a070ad
# ╠═22f0e9a0-e8c7-4d9c-ba82-710031783c33
# ╠═843bccd9-6546-4c91-9e94-ef9e894901bd
# ╠═7312297c-7515-4d49-a774-bfe01a77a126
# ╠═3a51917a-d73c-4299-a79c-93ffee0a279f
# ╠═a9837f40-7c4f-11eb-2d1d-4726be2375ac
# ╠═5a4e9399-c48d-42aa-bd61-80f2e5011ddc
# ╠═f230dbc0-6cea-436a-a16e-c1e10dadfb8c
# ╠═d5bdff29-d7e3-4c7b-a60c-eea6d0d60ded
# ╟─fc06fbd7-56e0-4ec5-af90-52626de2ee4a
# ╠═85afa530-7c54-11eb-333a-1b15e02d03f6
# ╠═022d3ee4-0c3d-467a-a64b-5d9ada7b489f
# ╟─66409e20-7c54-11eb-1925-613427e9efa4
# ╠═92e27997-3df3-44b7-ad5d-154f4e7bb413
# ╟─c3edd1f6-87a9-462d-8fe2-88b971a62c7a
# ╠═b069b405-53b4-45e5-ad2c-b29114a8bddd
